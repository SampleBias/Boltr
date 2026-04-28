//! RunPod SSH status and remote prediction execution.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::predict_job::{build_predict_argv, push_log, PredictCliOptions, PredictJob};

#[derive(Debug, Clone)]
pub struct RunPodConfig {
    pub host: String,
    pub user: String,
    pub port: u16,
    pub key: Option<PathBuf>,
    pub workdir: String,
    pub boltr: String,
    pub cache: String,
}

#[derive(Debug, Serialize)]
pub struct RunPodGpu {
    pub name: String,
    pub memory_free_mb: Option<u64>,
    pub memory_total_mb: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct RunPodStatus {
    pub configured: bool,
    pub connected: bool,
    pub target: Option<String>,
    pub workdir: Option<String>,
    pub boltr: Option<String>,
    pub cache: Option<String>,
    pub gpus: Vec<RunPodGpu>,
    pub boltr_doctor: Option<serde_json::Value>,
    pub remote_cache_ready: Option<bool>,
    pub warnings: Vec<String>,
    pub error: Option<String>,
}

pub struct RemotePredictRequest {
    pub config: RunPodConfig,
    pub local_input_root: PathBuf,
    pub local_input_path: PathBuf,
    pub local_out_dir: PathBuf,
    pub opts: PredictCliOptions,
}

fn env_trimmed(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn default_local_cache() -> String {
    if let Some(cache) = env_trimmed("BOLTR_RUNPOD_CACHE").or_else(|| env_trimmed("BOLTZ_CACHE")) {
        return cache;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!("{home}/.cache/boltr")
}

fn default_local_workdir() -> String {
    env_trimmed("BOLTR_RUNPOD_WORKDIR")
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|p| p.display().to_string())
        })
        .unwrap_or_else(|| ".".to_string())
}

fn default_local_boltr() -> String {
    env_trimmed("BOLTR_RUNPOD_BOLTR")
        .or_else(|| env_trimmed("BOLTR"))
        .unwrap_or_else(|| "boltr".to_string())
}

impl RunPodConfig {
    pub fn from_env() -> Option<Self> {
        let host = env_trimmed("BOLTR_RUNPOD_HOST")?;
        let user = env_trimmed("BOLTR_RUNPOD_USER").unwrap_or_else(|| "root".to_string());
        let port = env_trimmed("BOLTR_RUNPOD_PORT")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(22);
        let key = env_trimmed("BOLTR_RUNPOD_KEY").map(PathBuf::from);
        let workdir =
            env_trimmed("BOLTR_RUNPOD_WORKDIR").unwrap_or_else(|| "/workspace/boltr".to_string());
        let boltr = env_trimmed("BOLTR_RUNPOD_BOLTR").unwrap_or_else(|| "boltr".to_string());
        let cache = env_trimmed("BOLTR_RUNPOD_CACHE").unwrap_or_else(|| format!("{workdir}/cache"));
        Some(Self {
            host,
            user,
            port,
            key,
            workdir,
            boltr,
            cache,
        })
    }

    fn target(&self) -> String {
        format!("{}@{}", self.user, self.host)
    }

    fn ssh_args(&self) -> Vec<String> {
        let mut args = vec![
            "-p".to_string(),
            self.port.to_string(),
            "-o".to_string(),
            "BatchMode=yes".to_string(),
            "-o".to_string(),
            "ConnectTimeout=10".to_string(),
            "-o".to_string(),
            "StrictHostKeyChecking=accept-new".to_string(),
        ];
        if let Some(ref key) = self.key {
            args.push("-i".to_string());
            args.push(key.display().to_string());
        }
        args.push(self.target());
        args
    }
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn local_shell_quote(path: &Path) -> String {
    shell_quote(&path.to_string_lossy())
}

fn remote_join(base: &str, child: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        child.trim_start_matches('/')
    )
}

fn parse_gpu_line(line: &str) -> RunPodGpu {
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next().unwrap_or("").to_string();
    let free = parts
        .next()
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u64>().ok());
    let total = parts
        .next()
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u64>().ok());
    RunPodGpu {
        name,
        memory_free_mb: free,
        memory_total_mb: total,
    }
}

async fn ssh_output(
    cfg: &RunPodConfig,
    remote_cmd: &str,
    timeout_secs: u64,
) -> Result<String, String> {
    let mut args = cfg.ssh_args();
    args.push(remote_cmd.to_string());
    let fut = Command::new("ssh").args(args).output();
    let out = tokio::time::timeout(Duration::from_secs(timeout_secs), fut)
        .await
        .map_err(|_| format!("timeout after {timeout_secs}s"))?
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "ssh exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

async fn local_output(program: &str, args: &[&str], timeout_secs: u64) -> Result<String, String> {
    let fut = Command::new(program).args(args).output();
    let out = tokio::time::timeout(Duration::from_secs(timeout_secs), fut)
        .await
        .map_err(|_| format!("timeout after {timeout_secs}s"))?
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "{program} exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub async fn local_cuda_available() -> bool {
    local_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        5,
    )
    .await
    .map(|out| out.lines().any(|l| !l.trim().is_empty()))
    .unwrap_or(false)
}

async fn local_cuda_status() -> Option<RunPodStatus> {
    let gpu_out = local_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        5,
    )
    .await
    .ok()?;
    let gpus: Vec<_> = gpu_out
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(parse_gpu_line)
        .collect();
    if gpus.is_empty() {
        return None;
    }

    let workdir = default_local_workdir();
    let boltr = default_local_boltr();
    let cache = default_local_cache();
    let remote_cache_ready = {
        let c = Path::new(&cache);
        Some(
            c.join("boltz2_conf.safetensors").is_file()
                && c.join("ccd.pkl").is_file()
                && c.join("mols.tar").is_file(),
        )
    };
    let boltr_doctor = match local_output(&boltr, &["doctor", "--json"], 15).await {
        Ok(out) => serde_json::from_str::<serde_json::Value>(&out).ok(),
        Err(_) => None,
    };

    let mut warnings = vec![
        "BOLTR_RUNPOD_HOST is not set; using the GPU attached to this boltr-web server instead of SSH."
            .to_string(),
    ];
    if boltr_doctor.is_none() {
        warnings.push(
            "Local boltr doctor did not return JSON; set BOLTR to the tch-enabled boltr binary if needed."
                .to_string(),
        );
    }

    Some(RunPodStatus {
        configured: true,
        connected: true,
        target: Some("local CUDA GPU (no SSH)".to_string()),
        workdir: Some(workdir),
        boltr: Some(boltr),
        cache: Some(cache),
        gpus,
        boltr_doctor,
        remote_cache_ready,
        warnings,
        error: None,
    })
}

pub async fn status_from_env() -> RunPodStatus {
    let Some(cfg) = RunPodConfig::from_env() else {
        if let Some(status) = local_cuda_status().await {
            return status;
        }
        return RunPodStatus {
            configured: false,
            connected: false,
            target: None,
            workdir: None,
            boltr: None,
            cache: None,
            gpus: Vec::new(),
            boltr_doctor: None,
            remote_cache_ready: None,
            warnings: vec![
                "Set BOLTR_RUNPOD_HOST to enable RunPod SSH status, or launch boltr-web on a CUDA/RunPod host for local GPU auto-detect. Optional: BOLTR_RUNPOD_USER, PORT, KEY, WORKDIR, BOLTR, CACHE."
                    .to_string(),
            ],
            error: None,
        };
    };

    let mut status = RunPodStatus {
        configured: true,
        connected: false,
        target: Some(cfg.target()),
        workdir: Some(cfg.workdir.clone()),
        boltr: Some(cfg.boltr.clone()),
        cache: Some(cfg.cache.clone()),
        gpus: Vec::new(),
        boltr_doctor: None,
        remote_cache_ready: None,
        warnings: Vec::new(),
        error: None,
    };

    let gpu_cmd =
        "nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits";
    match ssh_output(&cfg, gpu_cmd, 12).await {
        Ok(out) => {
            status.connected = true;
            status.gpus = out
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(parse_gpu_line)
                .collect();
            if status.gpus.is_empty() {
                status
                    .warnings
                    .push("SSH works, but nvidia-smi returned no GPUs.".to_string());
            }
        }
        Err(e) => {
            status.error = Some(format!("GPU probe failed: {e}"));
            return status;
        }
    }

    let cache_cmd = format!(
        "test -f {c}/boltz2_conf.safetensors && test -f {c}/ccd.pkl && test -f {c}/mols.tar",
        c = shell_quote(&cfg.cache)
    );
    status.remote_cache_ready = Some(ssh_output(&cfg, &cache_cmd, 8).await.is_ok());

    let doctor_cmd = format!("{} doctor --json", shell_quote(&cfg.boltr));
    match ssh_output(&cfg, &doctor_cmd, 15).await {
        Ok(out) => match serde_json::from_str::<serde_json::Value>(&out) {
            Ok(v) => status.boltr_doctor = Some(v),
            Err(e) => status
                .warnings
                .push(format!("Remote boltr doctor did not return JSON: {e}")),
        },
        Err(e) => status
            .warnings
            .push(format!("Remote boltr doctor failed: {e}")),
    }

    status
}

async fn run_shell_logged(command: String, logs: Arc<Mutex<Vec<String>>>) -> bool {
    push_log(&logs, format!("[runpod] shell: {command}")).await;
    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            push_log(&logs, format!("[runpod] spawn failed: {e}")).await;
            return false;
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_logs = Arc::clone(&logs);
    let err_logs = Arc::clone(&logs);
    let t1 = stdout.map(|s| tokio::spawn(pipe_remote_lines(s, "runpod-stdout", out_logs)));
    let t2 = stderr.map(|s| tokio::spawn(pipe_remote_lines(s, "runpod-stderr", err_logs)));
    let status = child.wait().await;
    if let Some(t) = t1 {
        let _ = t.await;
    }
    if let Some(t) = t2 {
        let _ = t.await;
    }
    matches!(status, Ok(s) if s.success())
}

async fn pipe_remote_lines<R: tokio::io::AsyncRead + Unpin>(
    reader: R,
    name: &'static str,
    logs: Arc<Mutex<Vec<String>>>,
) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        push_log(&logs, format!("[{name}] {line}")).await;
    }
}

fn ssh_shell_prefix(cfg: &RunPodConfig) -> String {
    let mut parts = vec!["ssh".to_string()];
    for arg in cfg.ssh_args() {
        parts.push(shell_quote(&arg));
    }
    parts.join(" ")
}

pub async fn run_remote_predict_job(
    job_id: String,
    job: Arc<PredictJob>,
    req: RemotePredictRequest,
    sem: Arc<tokio::sync::Semaphore>,
) {
    let _permit = match sem.acquire_owned().await {
        Ok(p) => p,
        Err(_) => return,
    };

    let logs = Arc::clone(&job.logs);
    let cfg = req.config;
    let remote_job_dir = remote_join(&cfg.workdir, &format!("jobs/{job_id}"));
    let remote_in_dir = remote_join(&remote_job_dir, "input");
    let remote_out_dir = remote_join(&remote_job_dir, "out");

    let rel_input = req
        .local_input_path
        .strip_prefix(&req.local_input_root)
        .unwrap_or(&req.local_input_path)
        .to_string_lossy()
        .replace('\\', "/");
    let remote_input = remote_join(&remote_in_dir, &rel_input);
    let remote_cache = PathBuf::from(&cfg.cache);
    let mut opts = req.opts;
    opts.device = "cuda".to_string();
    let remote_argv = build_predict_argv(
        Path::new(&remote_input),
        Path::new(&remote_out_dir),
        &remote_cache,
        &opts,
    );

    push_log(
        &logs,
        format!(
            "[runpod] starting remote job on {} with input {}",
            cfg.target(),
            remote_input
        ),
    )
    .await;

    let mkdir_cmd = format!(
        "{} {}",
        ssh_shell_prefix(&cfg),
        shell_quote(&format!(
            "rm -rf {job} && mkdir -p {input} {out}",
            job = shell_quote(&remote_job_dir),
            input = shell_quote(&remote_in_dir),
            out = shell_quote(&remote_out_dir)
        ))
    );
    if !run_shell_logged(mkdir_cmd, Arc::clone(&logs)).await {
        finish_remote_job(&job, &logs, -1, false).await;
        return;
    }

    let upload_cmd = format!(
        "tar -C {} -cf - . | {} {}",
        local_shell_quote(&req.local_input_root),
        ssh_shell_prefix(&cfg),
        shell_quote(&format!("tar -C {} -xf -", shell_quote(&remote_in_dir)))
    );
    if !run_shell_logged(upload_cmd, Arc::clone(&logs)).await {
        finish_remote_job(&job, &logs, -1, false).await;
        return;
    }

    let remote_cmd = format!(
        "cd {workdir} && {boltr} {argv}",
        workdir = shell_quote(&cfg.workdir),
        boltr = shell_quote(&cfg.boltr),
        argv = remote_argv
            .iter()
            .map(|a| shell_quote(a))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let mut ssh_args = cfg.ssh_args();
    ssh_args.push(remote_cmd);
    push_log(
        &logs,
        format!(
            "[runpod] remote command: {} {}",
            cfg.boltr,
            remote_argv.join(" ")
        ),
    )
    .await;
    let mut child = match Command::new("ssh")
        .args(ssh_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            push_log(&logs, format!("[runpod] remote spawn failed: {e}")).await;
            finish_remote_job(&job, &logs, -1, false).await;
            return;
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let t1 = stdout.map(|s| tokio::spawn(pipe_remote_lines(s, "remote-stdout", Arc::clone(&logs))));
    let t2 = stderr.map(|s| tokio::spawn(pipe_remote_lines(s, "remote-stderr", Arc::clone(&logs))));
    let status = child.wait().await;
    if let Some(t) = t1 {
        let _ = t.await;
    }
    if let Some(t) = t2 {
        let _ = t.await;
    }

    let (code, remote_ok) = match status {
        Ok(s) => (s.code().unwrap_or(-1), s.success()),
        Err(e) => {
            push_log(&logs, format!("[runpod] remote wait failed: {e}")).await;
            (-1, false)
        }
    };

    if remote_ok {
        let _ = tokio::fs::create_dir_all(&req.local_out_dir).await;
        let download_cmd = format!(
            "{} {} | tar -C {} -xzf -",
            ssh_shell_prefix(&cfg),
            shell_quote(&format!(
                "if test -d {out}; then tar -C {out} -czf - .; else exit 2; fi",
                out = shell_quote(&remote_out_dir)
            )),
            local_shell_quote(&req.local_out_dir)
        );
        if !run_shell_logged(download_cmd, Arc::clone(&logs)).await {
            finish_remote_job(&job, &logs, -1, false).await;
            return;
        }
    }

    finish_remote_job(&job, &logs, code, remote_ok).await;
}

async fn finish_remote_job(job: &PredictJob, logs: &Arc<Mutex<Vec<String>>>, code: i32, ok: bool) {
    job.exit_code.store(code, Ordering::SeqCst);
    job.success.store(ok, Ordering::SeqCst);
    push_log(
        logs,
        format!(
            "[runpod] exit code {code} ({})",
            if ok { "success" } else { "failure" }
        ),
    )
    .await;
    job.done.store(true, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv() {
        let gpu = parse_gpu_line("NVIDIA A40, 45123, 46068");
        assert_eq!(gpu.name, "NVIDIA A40");
        assert_eq!(gpu.memory_free_mb, Some(45123));
        assert_eq!(gpu.memory_total_mb, Some(46068));
    }

    #[test]
    fn shell_quote_handles_single_quotes() {
        assert_eq!(shell_quote("a'b"), "'a'\"'\"'b'");
    }

    #[tokio::test]
    #[ignore = "requires BOLTR_RUNPOD_HOST and SSH access to a live pod"]
    async fn runpod_status_probe_integration() {
        let status = status_from_env().await;
        assert!(status.configured, "set BOLTR_RUNPOD_HOST before running");
        assert!(
            status.connected,
            "RunPod SSH/GPU probe failed: {:?}",
            status.error
        );
    }
}

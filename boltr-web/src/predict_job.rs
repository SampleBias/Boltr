//! Spawn `boltr predict` subprocess, log capture, tarball download, TTL cleanup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use flate2::write::GzEncoder;
use flate2::Compression;
use tar::Builder;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{error, info};

use crate::prereq::resolve_cache_dir;

/// `BOLTR_WEB_ENABLE_PREDICT=0` disables predict endpoints (default: enabled).
#[must_use]
pub fn predict_enabled() -> bool {
    match std::env::var("BOLTR_WEB_ENABLE_PREDICT") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

/// CLI flags collected from the predict form (empty optional = omit).
#[derive(Debug, Clone, Default)]
pub struct PredictCliOptions {
    pub device: String,
    pub use_msa_server: bool,
    pub msa_server_url: Option<String>,
    pub msa_pairing_strategy: Option<String>,
    pub affinity: bool,
    pub use_potentials: bool,
    pub recycling_steps: Option<i64>,
    pub sampling_steps: Option<i64>,
    pub diffusion_samples: Option<i64>,
    pub max_parallel_samples: Option<i64>,
    pub step_scale: Option<f64>,
    pub output_format: Option<String>,
    pub max_msa_seqs: Option<usize>,
    pub num_samples: Option<usize>,
    pub checkpoint: Option<PathBuf>,
    pub affinity_checkpoint: Option<PathBuf>,
    pub affinity_mw_correction: bool,
    pub sampling_steps_affinity: Option<i64>,
    pub diffusion_samples_affinity: Option<i64>,
    pub preprocessing_threads: Option<usize>,
    pub r#override: bool,
    pub write_full_pae: bool,
    pub write_full_pde: bool,
    pub spike_only: bool,
}

/// Build `boltr predict` argv: `predict`, `<input>`, `--output`, …
#[must_use]
pub fn build_predict_argv(
    input: &Path,
    output: &Path,
    cache: &Path,
    opts: &PredictCliOptions,
) -> Vec<String> {
    let mut args = vec![
        "predict".to_string(),
        input.display().to_string(),
        "--output".to_string(),
        output.display().to_string(),
        "--device".to_string(),
        opts.device.clone(),
        "--cache-dir".to_string(),
        cache.display().to_string(),
    ];

    if opts.use_msa_server {
        args.push("--use-msa-server".to_string());
    }
    if let Some(ref u) = opts.msa_server_url {
        if !u.is_empty() {
            args.push("--msa-server-url".to_string());
            args.push(u.clone());
        }
    }
    if let Some(ref s) = opts.msa_pairing_strategy {
        if !s.is_empty() {
            args.push("--msa-pairing-strategy".to_string());
            args.push(s.clone());
        }
    }
    if opts.affinity {
        args.push("--affinity".to_string());
    }
    if opts.use_potentials {
        args.push("--use-potentials".to_string());
    }
    if let Some(n) = opts.recycling_steps {
        args.push("--recycling-steps".to_string());
        args.push(n.to_string());
    }
    if let Some(n) = opts.sampling_steps {
        args.push("--sampling-steps".to_string());
        args.push(n.to_string());
    }
    if let Some(n) = opts.diffusion_samples {
        args.push("--diffusion-samples".to_string());
        args.push(n.to_string());
    }
    if let Some(n) = opts.max_parallel_samples {
        args.push("--max-parallel-samples".to_string());
        args.push(n.to_string());
    }
    if let Some(x) = opts.step_scale {
        args.push("--step-scale".to_string());
        args.push(format!("{x}"));
    }
    if let Some(ref f) = opts.output_format {
        let f = f.trim().to_lowercase();
        if f == "mmcif" || f == "pdb" {
            args.push("--output-format".to_string());
            args.push(f);
        }
    }
    if let Some(n) = opts.max_msa_seqs {
        args.push("--max-msa-seqs".to_string());
        args.push(n.to_string());
    }
    if let Some(n) = opts.num_samples {
        args.push("--num-samples".to_string());
        args.push(n.to_string());
    }
    if let Some(ref p) = opts.checkpoint {
        if p.as_os_str().len() > 0 {
            args.push("--checkpoint".to_string());
            args.push(p.display().to_string());
        }
    }
    if let Some(ref p) = opts.affinity_checkpoint {
        if p.as_os_str().len() > 0 {
            args.push("--affinity-checkpoint".to_string());
            args.push(p.display().to_string());
        }
    }
    if opts.affinity_mw_correction {
        args.push("--affinity-mw-correction".to_string());
    }
    if let Some(n) = opts.sampling_steps_affinity {
        args.push("--sampling-steps-affinity".to_string());
        args.push(n.to_string());
    }
    if let Some(n) = opts.diffusion_samples_affinity {
        args.push("--diffusion-samples-affinity".to_string());
        args.push(n.to_string());
    }
    if let Some(n) = opts.preprocessing_threads {
        args.push("--preprocessing-threads".to_string());
        args.push(n.to_string());
    }
    if opts.r#override {
        args.push("--override".to_string());
    }
    if opts.write_full_pae {
        args.push("--write-full-pae".to_string());
    }
    if opts.write_full_pde {
        args.push("--write-full-pde".to_string());
    }
    if opts.spike_only {
        args.push("--spike-only".to_string());
    }

    args
}

async fn push_log(logs: &Arc<Mutex<Vec<String>>>, line: String) {
    let line = line.trim_end_matches('\n').to_string();
    if line.is_empty() {
        return;
    }
    let mut g = logs.lock().await;
    g.push(line);
    const MAX: usize = 50_000;
    if g.len() > MAX {
        let drain = g.len() - MAX;
        g.drain(0..drain);
    }
}

/// One prediction job.
pub struct PredictJob {
    pub workspace: PathBuf,
    pub out_dir: PathBuf,
    pub logs: Arc<Mutex<Vec<String>>>,
    pub done: Arc<AtomicBool>,
    pub exit_code: Arc<AtomicI32>,
    pub success: Arc<AtomicBool>,
    pub created_at: Instant,
}

impl PredictJob {
    pub fn new(workspace: PathBuf, out_dir: PathBuf) -> Self {
        Self {
            workspace,
            out_dir,
            logs: Arc::new(Mutex::new(Vec::new())),
            done: Arc::new(AtomicBool::new(false)),
            exit_code: Arc::new(AtomicI32::new(-1)),
            success: Arc::new(AtomicBool::new(false)),
            created_at: Instant::now(),
        }
    }
}

/// Shared job registry + single-flight semaphore.
pub struct JobStore {
    inner: Arc<Mutex<HashMap<String, Arc<PredictJob>>>>,
    sem: Arc<tokio::sync::Semaphore>,
}

impl JobStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            sem: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    pub async fn get(&self, id: &str) -> Option<Arc<PredictJob>> {
        self.inner.lock().await.get(id).cloned()
    }

    pub async fn insert(&self, id: String, job: Arc<PredictJob>) {
        self.inner.lock().await.insert(id, job);
    }

    pub async fn remove(&self, id: &str) -> Option<Arc<PredictJob>> {
        self.inner.lock().await.remove(id)
    }

    pub fn semaphore(&self) -> Arc<tokio::sync::Semaphore> {
        Arc::clone(&self.sem)
    }

    /// Remove jobs older than `ttl` and delete their workspace directories.
    pub async fn cleanup_expired(&self, ttl: Duration) {
        let now = Instant::now();
        let mut to_remove: Vec<String> = Vec::new();
        {
            let guard = self.inner.lock().await;
            for (id, job) in guard.iter() {
                if now.duration_since(job.created_at) > ttl && job.done.load(Ordering::SeqCst) {
                    to_remove.push(id.clone());
                }
            }
        }
        for id in to_remove {
            if let Some(job) = self.remove(&id).await {
                let _ = tokio::fs::remove_dir_all(&job.workspace).await;
                info!(job_id = %id, path = %job.workspace.display(), "cleaned predict job workspace");
            }
        }
    }
}

/// Create `out_dir` as `.tar.gz` bytes (top-level folder `output/` in archive).
pub fn tarball_out_dir(out_dir: &Path) -> anyhow::Result<Vec<u8>> {
    let enc = GzEncoder::new(Vec::new(), Compression::default());
    let mut tar = Builder::new(enc);
    tar.append_dir_all("output", out_dir)
        .context("tar append_dir_all")?;
    tar.finish().context("tar finish")?;
    let gz = tar.into_inner().context("tar into_inner")?;
    let bytes = gz.finish().context("gzip finish")?;
    Ok(bytes)
}

async fn pipe_lines<R: tokio::io::AsyncRead + Unpin>(
    reader: R,
    name: &'static str,
    logs: Arc<Mutex<Vec<String>>>,
) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        push_log(&logs, format!("[{name}] {line}")).await;
    }
}

/// Spawn `boltr predict` in the background; updates `job` logs and exit status.
pub async fn run_predict_job(
    job_id: String,
    job: Arc<PredictJob>,
    boltr: PathBuf,
    argv: Vec<String>,
    sem: Arc<tokio::sync::Semaphore>,
) {
    let _permit = match sem.acquire_owned().await {
        Ok(p) => p,
        Err(_) => return,
    };

    let logs = Arc::clone(&job.logs);
    push_log(
        &logs,
        format!(
            "[boltr-web] starting: {} {}",
            boltr.display(),
            argv.join(" ")
        ),
    )
    .await;

    let mut child = match Command::new(&boltr)
        .args(&argv)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            push_log(&logs, format!("[boltr-web] spawn failed: {e}"))
                .await;
            job.exit_code.store(-1, Ordering::SeqCst);
            job.success.store(false, Ordering::SeqCst);
            job.done.store(true, Ordering::SeqCst);
            return;
        }
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let logs_o = Arc::clone(&logs);
    let logs_e = Arc::clone(&logs);
    let t1 = stdout.map(|s| tokio::spawn(pipe_lines(s, "stdout", logs_o)));
    let t2 = stderr.map(|s| tokio::spawn(pipe_lines(s, "stderr", logs_e)));

    let status = child.wait().await;
    if let Some(t) = t1 {
        let _ = t.await;
    }
    if let Some(t) = t2 {
        let _ = t.await;
    }

    match status {
        Ok(s) => {
            let code = s.code().unwrap_or(-1);
            job.exit_code.store(code, Ordering::SeqCst);
            let ok = s.success();
            job.success.store(ok, Ordering::SeqCst);
            push_log(
                &logs,
                format!(
                    "[boltr-web] exit code {code} ({})",
                    if ok { "success" } else { "failure" }
                ),
            )
            .await;
        }
        Err(e) => {
            error!("wait: {e}");
            push_log(&logs, format!("[boltr-web] wait error: {e}"))
                .await;
            job.exit_code.store(-1, Ordering::SeqCst);
            job.success.store(false, Ordering::SeqCst);
        }
    }
    job.done.store(true, Ordering::SeqCst);
    info!(job_id = %job_id, "predict job finished");
}

/// Ensure `job_dir` (canonicalized) contains `path` (canonicalized).
pub fn path_under_job_dir(job_dir: &std::path::Path, path: &std::path::Path) -> anyhow::Result<()> {
    let base = job_dir
        .canonicalize()
        .with_context(|| format!("canonicalize job_dir {}", job_dir.display()))?;
    let p = path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", path.display()))?;
    if !p.starts_with(&base) {
        anyhow::bail!(
            "path escapes job_dir: {} not under {}",
            p.display(),
            base.display()
        );
    }
    Ok(())
}

/// Resolve cache path for predict from optional override string.
pub fn cache_from_form_override(cache_override: Option<&str>, default_cache: &Path) -> PathBuf {
    cache_override
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| resolve_cache_dir(Some(Path::new(s))))
        .unwrap_or_else(|| default_cache.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_predict_argv_minimal() {
        let opts = PredictCliOptions {
            device: "cpu".to_string(),
            ..Default::default()
        };
        let args = build_predict_argv(
            Path::new("/in/a.yaml"),
            Path::new("/out"),
            Path::new("/cache"),
            &opts,
        );
        assert_eq!(
            args,
            vec![
                "predict",
                "/in/a.yaml",
                "--output",
                "/out",
                "--device",
                "cpu",
                "--cache-dir",
                "/cache",
            ]
        );
    }

    #[test]
    fn build_predict_argv_flags() {
        let opts = PredictCliOptions {
            device: "cuda:0".to_string(),
            use_msa_server: true,
            recycling_steps: Some(3),
            r#override: true,
            spike_only: false,
            ..Default::default()
        };
        let args = build_predict_argv(
            Path::new("/x.yaml"),
            Path::new("/o"),
            Path::new("/c"),
            &opts,
        );
        assert!(args.contains(&"--use-msa-server".to_string()));
        assert!(args.contains(&"--recycling-steps".to_string()));
        assert!(args.contains(&"3".to_string()));
        assert!(args.contains(&"--override".to_string()));
        assert!(args.contains(&"--device".to_string()));
        assert!(args.contains(&"cuda:0".to_string()));
    }
}

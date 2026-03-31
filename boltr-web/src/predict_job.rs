//! Spawn `boltr predict` subprocess, log capture, tarball download, TTL cleanup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Serialize;
use tar::Builder;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{error, info};

use crate::paths::prepend_torch_wheel_lib_to_ld_path;
use crate::prereq::{find_venv_python, resolve_cache_dir};

/// Matches `boltr predict --preprocess` (`boltr-cli` `PreprocessCli`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PreprocessMode {
    /// No bundle generation.
    #[default]
    Off,
    /// Rust native when eligible, else Boltz if available.
    Auto,
    /// Upstream Boltz subprocess + copy bundle next to YAML.
    Boltz,
    /// Rust-only protein-only bundle.
    Native,
}

impl PreprocessMode {
    fn as_cli_value(self) -> Option<&'static str> {
        match self {
            PreprocessMode::Off => None,
            PreprocessMode::Native => Some("native"),
            PreprocessMode::Boltz => Some("boltz"),
            PreprocessMode::Auto => Some("auto"),
        }
    }
}

/// Parse multipart `preprocess` field; empty trims to [`PreprocessMode::Off`].
pub fn parse_preprocess_mode(s: &str) -> Result<PreprocessMode, &'static str> {
    let t = s.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("off") {
        return Ok(PreprocessMode::Off);
    }
    if t.eq_ignore_ascii_case("native") {
        return Ok(PreprocessMode::Native);
    }
    if t.eq_ignore_ascii_case("boltz") {
        return Ok(PreprocessMode::Boltz);
    }
    if t.eq_ignore_ascii_case("auto") {
        return Ok(PreprocessMode::Auto);
    }
    Err("invalid preprocess (use off, native, boltz, auto)")
}

/// True if `cmd` is an existing file path, or resolves via [`which::which`] (matches `boltr-cli`).
fn command_is_runnable(cmd: &str) -> bool {
    let p = Path::new(cmd);
    if p.is_absolute() || cmd.contains('/') {
        return p.is_file();
    }
    which::which(cmd).is_ok()
}

/// Repo root from `BOLTR_REPO` or from `BOLTR` pointing at `…/target/release/boltr`.
fn repo_root_hint() -> Option<PathBuf> {
    if let Ok(r) = std::env::var("BOLTR_REPO") {
        let p = PathBuf::from(r);
        if p.join("Cargo.toml").is_file() {
            return Some(p);
        }
    }
    if let Ok(b) = std::env::var("BOLTR") {
        let p = PathBuf::from(b);
        let repo = p.parent()?.parent()?.parent()?;
        if repo.join("Cargo.toml").is_file() {
            return Some(repo.to_path_buf());
        }
    }
    None
}

/// Run `shutil.which('boltz')` with `venv/bin` prepended to `PATH` (finds console scripts when boltr-web's PATH omits the venv).
fn shutil_which_boltz_with_venv_bin(py: &Path) -> Option<PathBuf> {
    let bin_dir = py.parent()?;
    let path = std::env::var("PATH").unwrap_or_default();
    let combined = format!("{}:{}", bin_dir.display(), path);
    let out = std::process::Command::new(py)
        .env("PATH", combined)
        .args(["-c", "import shutil; print(shutil.which('boltz') or '')"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        return None;
    }
    let p = PathBuf::from(&s);
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

/// Search common install locations when `boltz` is not on `PATH` (repo `.venv`, pip user, conda, etc.).
fn discover_boltz_executable() -> Option<PathBuf> {
    if let Some(h) = dirs::home_dir() {
        let p = h.join(".local/bin/boltz");
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(v) = std::env::var("CONDA_PREFIX") {
        let p = PathBuf::from(v).join("bin/boltz");
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(v) = std::env::var("VIRTUAL_ENV") {
        let p = PathBuf::from(v).join("bin/boltz");
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(repo) = repo_root_hint() {
        for sub in [".venv/bin/boltz", "venv/bin/boltz"] {
            let p = repo.join(sub);
            if p.is_file() {
                return Some(p);
            }
        }
        let py = repo.join(".venv/bin/python");
        if py.is_file() {
            if let Some(b) = shutil_which_boltz_with_venv_bin(&py) {
                return Some(b);
            }
        }
    }
    let mut d = std::env::current_dir().ok()?;
    for _ in 0..12 {
        for sub in [".venv/bin/boltz", "venv/bin/boltz"] {
            let p = d.join(sub);
            if p.is_file() {
                return Some(p);
            }
        }
        if !d.pop() {
            break;
        }
    }
    if let Some(py) = find_venv_python() {
        if let Some(b) = shutil_which_boltz_with_venv_bin(&py) {
            return Some(b);
        }
    }
    None
}

/// Ensure upstream Boltz CLI can be spawned: use `PATH`, explicit `--bolt-command`, or auto-discover.
fn resolve_boltz_for_preprocess(opts: &mut PredictCliOptions) -> Result<(), String> {
    let cmd = opts.bolt_command.as_deref().unwrap_or("boltz");
    if command_is_runnable(cmd) {
        return Ok(());
    }
    if opts.bolt_command.is_some() {
        return Err(format!(
            "Boltz preprocess: nothing executable at {cmd:?} (from form or BOLTR_BOLTZ_COMMAND). Fix the path or install upstream Boltz."
        ));
    }
    if let Some(p) = discover_boltz_executable() {
        let s = p.display().to_string();
        opts.bolt_command = Some(s.clone());
        if command_is_runnable(&s) {
            info!(path = %s, "boltr-web: auto-selected boltz for --bolt-command");
            return Ok(());
        }
    }
    Err(
        "Boltz preprocess: could not find the upstream `boltz` executable. Install: `pip install boltz` (or conda) into your dev venv, then ensure `BOLTR` points at `…/target/release/boltr` so the server can locate `…/repo/.venv/bin/boltz`, or set `BOLTR_REPO` to the repo root, or set `BOLTR_BOLTZ_COMMAND` / Web UI Bolt command to the full path to `boltz`.".to_string(),
    )
}

/// Validate preprocess choices before spawning `boltr predict` (clear 400 instead of failing mid-job).
pub fn preprocess_preflight(input_path: &Path, opts: &mut PredictCliOptions) -> Result<(), String> {
    match opts.preprocess {
        PreprocessMode::Off => Ok(()),
        PreprocessMode::Native => {
            let input =
                boltr_io::parse_input_path(input_path).map_err(|e| format!("parse YAML: {e}"))?;
            boltr_io::validate_native_eligible(&input).map_err(|e| {
                format!(
                    "{e}. For YAML with templates/constraints (or ligands/DNA/RNA), use preprocess \"boltz\" or \"auto\" and install the Python `boltz` CLI (set \"Bolt command\" if it is not on PATH)."
                )
            })
        }
        PreprocessMode::Boltz => resolve_boltz_for_preprocess(opts),
        PreprocessMode::Auto => {
            let input =
                boltr_io::parse_input_path(input_path).map_err(|e| format!("parse YAML: {e}"))?;
            if boltr_io::validate_native_eligible(&input).is_ok() {
                return Ok(());
            }
            resolve_boltz_for_preprocess(opts)
        }
    }
}

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
    pub preprocess: PreprocessMode,
    pub bolt_command: Option<String>,
    pub preprocess_staging: Option<PathBuf>,
    pub preprocess_keep_staging: bool,
    pub preprocess_symlink: bool,
    pub preprocess_bolt_arg: Vec<String>,
    pub preprocess_record_id: Option<String>,
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

    if let Some(v) = opts.preprocess.as_cli_value() {
        args.push("--preprocess".to_string());
        args.push(v.to_string());
    }
    if let Some(ref c) = opts.bolt_command {
        let t = c.trim();
        if !t.is_empty() {
            args.push("--bolt-command".to_string());
            args.push(t.to_string());
        }
    }
    if let Some(ref p) = opts.preprocess_staging {
        if !p.as_os_str().is_empty() {
            args.push("--preprocess-staging".to_string());
            args.push(p.display().to_string());
        }
    }
    if opts.preprocess_keep_staging {
        args.push("--preprocess-keep-staging".to_string());
    }
    if opts.preprocess_symlink {
        args.push("--preprocess-symlink".to_string());
    }
    for a in &opts.preprocess_bolt_arg {
        let t = a.trim();
        if !t.is_empty() {
            args.push("--preprocess-bolt-arg".to_string());
            args.push(t.to_string());
        }
    }
    if let Some(ref id) = opts.preprocess_record_id {
        let t = id.trim();
        if !t.is_empty() {
            args.push("--preprocess-record-id".to_string());
            args.push(t.to_string());
        }
    }

    args
}

async fn push_log(logs: &Arc<Mutex<Vec<String>>>, line: String) {
    // SSE `data:` fields and some clients reject embedded CR/LF; `lines()` keeps `\r` from CRLF.
    let line: String = line
        .chars()
        .map(|c| match c {
            '\n' | '\r' => ' ',
            _ => c,
        })
        .collect();
    let line = line.trim_end().to_string();
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

/// Result of scanning a finished job's `--output` directory for structures + completion metadata.
#[derive(Debug, Clone, Serialize)]
pub struct PredictOutputInspect {
    /// Canonical absolute paths to each `.cif` / `.pdb` found under `out_dir`.
    pub structure_paths: Vec<String>,
    /// One short paragraph for UI: success summary, or why no structure file was found.
    pub structure_message: String,
    /// `status` from `boltr_predict_complete.txt` when present.
    pub completion_status: Option<String>,
    /// `note` from `boltr_predict_complete.txt` when present.
    pub completion_note: Option<String>,
}

impl PredictOutputInspect {
    /// Placeholder while the subprocess is still running.
    pub fn job_running() -> Self {
        Self {
            structure_paths: Vec::new(),
            structure_message:
                "Job still running; structure paths and completion details appear when the job finishes."
                    .to_string(),
            completion_status: None,
            completion_note: None,
        }
    }
}

fn collect_structure_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_structure_files(&p, out);
        } else if let Some(ext) = p.extension().and_then(|x| x.to_str()) {
            if ext.eq_ignore_ascii_case("cif") || ext.eq_ignore_ascii_case("pdb") {
                out.push(p);
            }
        }
    }
}

fn collect_named_files(dir: &Path, filename: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_named_files(&p, filename, out);
        } else if p.file_name().and_then(|n| n.to_str()) == Some(filename) {
            out.push(p);
        }
    }
}

fn parse_completion_json(path: &Path) -> Option<(String, String)> {
    let txt = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    let status = v
        .get("status")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let note = v
        .get("note")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    if status.is_empty() && note.is_empty() {
        return None;
    }
    Some((status, note))
}

fn build_structure_message(
    paths: &[String],
    completion_status: Option<&str>,
    completion_note: Option<&str>,
) -> String {
    if !paths.is_empty() {
        let mut s = format!(
            "Found {} structure file(s) (.cif / .pdb) under the job output directory.",
            paths.len()
        );
        if paths.len() == 1 {
            s.push_str(" Full path is listed in structure_paths.");
        } else {
            s.push_str(" All paths are listed in structure_paths.");
        }
        return s;
    }

    let mut s = "No .cif (mmCIF) or .pdb file was found under the output directory.".to_string();

    match completion_status {
        Some("predict_step_complete") => {
            s.push_str(
                " Logs report diffusion completed, but no structure files are present (unexpected).",
            );
        }
        Some("preprocess_reference_structure") => {
            s.push_str(
                " A preprocess-only reference export was expected; if files are missing, check permissions or logs.",
            );
        }
        Some("pipeline_complete") => {
            s.push_str(" Boltr exited without writing a sampled structure: ");
            if let Some(n) = completion_note.filter(|x| !x.is_empty()) {
                s.push_str(n);
            } else {
                s.push_str(
                    "usually `manifest.json` + preprocess `.npz` must sit next to the input YAML (enable preprocess auto/boltz/native), and you need a tch-enabled `boltr` build for diffusion.",
                );
            }
        }
        Some(other) => {
            s.push_str(&format!(" Completion status: {other}."));
            if let Some(n) = completion_note.filter(|x| !x.is_empty()) {
                s.push(' ');
                s.push_str(n);
            }
        }
        None => {
            s.push_str(
                " No `boltr_predict_complete.txt` JSON was parsed, or the run did not reach the structure writer (see log_tail).",
            );
        }
    }
    s
}

/// Scan `out_dir` after a predict run for mmCIF/PDB outputs and `boltr_predict_complete.txt`.
pub fn inspect_predict_output(out_dir: &Path) -> PredictOutputInspect {
    if !out_dir.is_dir() {
        return PredictOutputInspect {
            structure_paths: Vec::new(),
            structure_message: "Output directory is missing or not a directory.".to_string(),
            completion_status: None,
            completion_note: None,
        };
    }

    let mut raw_paths = Vec::new();
    collect_structure_files(out_dir, &mut raw_paths);
    raw_paths.sort();
    let structure_paths: Vec<String> = raw_paths
        .into_iter()
        .filter_map(|p| {
            p.canonicalize()
                .ok()
                .map(|c| c.to_string_lossy().into_owned())
        })
        .collect();

    let mut markers = Vec::new();
    collect_named_files(out_dir, "boltr_predict_complete.txt", &mut markers);
    markers.sort();

    let mut completion_status = None;
    let mut completion_note = None;
    for m in &markers {
        if let Some((st, note)) = parse_completion_json(m) {
            completion_status = Some(st);
            completion_note = Some(note);
            break;
        }
    }

    let structure_message = build_structure_message(
        &structure_paths,
        completion_status.as_deref(),
        completion_note.as_deref(),
    );

    PredictOutputInspect {
        structure_paths,
        structure_message,
        completion_status,
        completion_note,
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

    let py = find_venv_python().unwrap_or_else(|| PathBuf::from("python3"));
    let mut cmd = Command::new(&boltr);
    prepend_torch_wheel_lib_to_ld_path(&mut cmd, &py);
    let mut child = match cmd
        .args(&argv)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            push_log(&logs, format!("[boltr-web] spawn failed: {e}")).await;
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
            push_log(&logs, format!("[boltr-web] wait error: {e}")).await;
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

    #[test]
    fn build_predict_argv_preprocess_off_omits_flag() {
        let opts = PredictCliOptions {
            device: "cpu".to_string(),
            ..Default::default()
        };
        let args = build_predict_argv(
            Path::new("/in.yaml"),
            Path::new("/out"),
            Path::new("/cache"),
            &opts,
        );
        assert!(!args.iter().any(|a| a == "--preprocess"));
    }

    #[test]
    fn build_predict_argv_preprocess_native() {
        let opts = PredictCliOptions {
            device: "cpu".to_string(),
            preprocess: PreprocessMode::Native,
            ..Default::default()
        };
        let args = build_predict_argv(
            Path::new("/in.yaml"),
            Path::new("/out"),
            Path::new("/cache"),
            &opts,
        );
        let i = args.iter().position(|a| a == "--preprocess").unwrap();
        assert_eq!(args.get(i + 1).map(String::as_str), Some("native"));
    }

    #[test]
    fn parse_preprocess_mode_accepts_aliases() {
        assert_eq!(parse_preprocess_mode("").unwrap(), PreprocessMode::Off);
        assert_eq!(parse_preprocess_mode("off").unwrap(), PreprocessMode::Off);
        assert_eq!(parse_preprocess_mode("AUTO").unwrap(), PreprocessMode::Auto);
        assert!(parse_preprocess_mode("nope").is_err());
    }

    #[test]
    fn preprocess_preflight_off_skips_checks() {
        let mut opts = PredictCliOptions {
            preprocess: PreprocessMode::Off,
            ..Default::default()
        };
        assert!(preprocess_preflight(Path::new("/no/such/file.yaml"), &mut opts).is_ok());
    }

    #[test]
    fn inspect_predict_output_finds_cif() {
        let tmp = tempfile::tempdir().unwrap();
        let rec = tmp.path().join("rec1");
        std::fs::create_dir_all(&rec).unwrap();
        std::fs::write(rec.join("rec1_model_0.cif"), b"data_x").unwrap();
        let i = inspect_predict_output(tmp.path());
        assert_eq!(i.structure_paths.len(), 1);
        assert!(i.structure_message.contains("Found 1"));
    }

    #[test]
    fn inspect_predict_output_pipeline_complete_explains_missing_structure() {
        let tmp = tempfile::tempdir().unwrap();
        let rec = tmp.path().join("rec1");
        std::fs::create_dir_all(&rec).unwrap();
        std::fs::write(
            rec.join("boltr_predict_complete.txt"),
            r#"{"status":"pipeline_complete","note":"test note"}"#,
        )
        .unwrap();
        let i = inspect_predict_output(tmp.path());
        assert!(i.structure_paths.is_empty());
        assert_eq!(i.completion_status.as_deref(), Some("pipeline_complete"));
        assert!(i.structure_message.contains("test note"));
    }
}

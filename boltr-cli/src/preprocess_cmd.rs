//! `boltr preprocess` — Tier 1 (Boltz subprocess) and Tier 2 (native protein-only) bundle generation.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use tracing::info;

#[cfg(feature = "tch")]
use tch;

/// Environment applied only to the upstream `boltz predict` subprocess.
#[derive(Debug, Default, Clone)]
pub struct BoltzChildEnv {
    /// `CUDA_VISIBLE_DEVICES` for the Boltz child (e.g. `"1"` so Boltz uses GPU 1 while LibTorch uses `cuda:0` on GPU 0).
    pub cuda_visible_devices: Option<String>,
    /// `PYTORCH_CUDA_ALLOC_CONF` for the Boltz child (reduces fragmentation before LibTorch loads).
    pub pytorch_cuda_alloc_conf: Option<String>,
    /// Thread cap for CPU-heavy BLAS/token preprocessing in the upstream Boltz process.
    pub preprocess_threads: Option<usize>,
    /// PyTorch float32 matmul precision for the Boltz child when the wrapper is used.
    pub matmul_precision: Option<String>,
}

/// True if `device` selects a CUDA device for LibTorch (`cpu` / `cuda` / `cuda:N`).
pub fn predict_device_is_cuda(device: &str) -> bool {
    let s = device.trim().to_lowercase();
    s == "cuda" || s == "gpu" || s.starts_with("cuda:")
}

/// True when the user already passed `--accelerator …` to the Boltz subprocess.
fn user_set_boltz_accelerator(bolt_extra: &[String]) -> bool {
    bolt_extra
        .iter()
        .any(|arg| arg == "--accelerator" || arg.starts_with("--accelerator="))
}

fn user_set_boltz_kernel_mode(bolt_extra: &[String]) -> bool {
    bolt_extra
        .iter()
        .any(|arg| arg == "--no_kernels" || arg == "--no-kernels")
}

fn should_disable_boltz_kernels_by_default() -> bool {
    !boltz_kernel_probe_allows_enable()
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

fn boltz_kernel_probe_allows_enable() -> bool {
    if !env_truthy("BOLTR_BOLTZ_USE_KERNELS") {
        return false;
    }
    let py = std::env::var("BOLTR_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = r#"
import importlib.util
import sys

try:
    import torch
    ok = torch.cuda.is_available() and importlib.util.find_spec("cuequivariance_torch") is not None
except Exception:
    ok = False
sys.exit(0 if ok else 1)
"#;
    let ok = Command::new(&py)
        .args(["-c", script])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        tracing::info!(%py, "preprocess: enabling upstream Boltz GPU kernels after CUDA/cuEquivariance probe");
    } else {
        tracing::warn!(
            %py,
            "BOLTR_BOLTZ_USE_KERNELS requested, but CUDA/cuEquivariance probe failed; keeping --no_kernels"
        );
    }
    ok
}

/// Merge CLI (`--preprocess-cuda-visible-devices`) with `BOLTR_BOLTZ_CUDA_VISIBLE_DEVICES` (CLI wins).
pub fn resolve_preprocess_cuda_visible_devices(cli: Option<&str>) -> Option<String> {
    let from_cli = cli
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);
    if from_cli.is_some() {
        return from_cli;
    }
    std::env::var("BOLTR_BOLTZ_CUDA_VISIBLE_DEVICES")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// When LibTorch uses CUDA, default `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the Boltz child
/// unless `BOLTR_BOLTZ_PYTORCH_CUDA_ALLOC_CONF` is set (empty string disables).
pub fn resolve_boltz_pytorch_alloc_conf(predict_cuda: bool) -> Option<String> {
    match std::env::var("BOLTR_BOLTZ_PYTORCH_CUDA_ALLOC_CONF") {
        Ok(s) => {
            let t = s.trim().to_string();
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        }
        Err(_) => {
            if predict_cuda {
                Some("expandable_segments:True".to_string())
            } else {
                None
            }
        }
    }
}

/// `BOLTR_PREPROCESS_THREADS`: cap CPU threads used by upstream Boltz preprocessing.
pub fn resolve_preprocess_threads() -> Option<usize> {
    std::env::var("BOLTR_PREPROCESS_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
}

/// `BOLTR_BOLTZ_MATMUL_PRECISION`: Tensor Core-friendly matmul precision for the Python wrapper.
pub fn resolve_boltz_matmul_precision() -> Option<String> {
    let v = std::env::var("BOLTR_BOLTZ_MATMUL_PRECISION").unwrap_or_else(|_| "high".to_string());
    let t = v.trim().to_lowercase();
    match t.as_str() {
        "highest" | "high" | "medium" => Some(t),
        "" | "off" | "none" => None,
        _ => Some("high".to_string()),
    }
}

/// `--preprocess-boltz-cpu` or `BOLTR_PREPROCESS_BOLTZ_CPU=1`.
pub fn resolve_force_boltz_cpu(cli_flag: bool) -> bool {
    cli_flag
        || std::env::var("BOLTR_PREPROCESS_BOLTZ_CPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

/// After Boltz preprocess, run `torch.cuda.empty_cache()` when LibTorch uses CUDA.
/// Defaults to **on** for GPU predict (`predict_cuda`); set `BOLTR_PREPROCESS_POST_BOLTZ_EMPTY_CACHE=0` to disable.
/// `--preprocess-post-boltz-empty-cache` still forces on (redundant when default is on).
pub fn resolve_post_boltz_empty_cache(cli_flag: bool, predict_cuda: bool) -> bool {
    if !predict_cuda {
        return false;
    }
    if let Ok(s) = std::env::var("BOLTR_PREPROCESS_POST_BOLTZ_EMPTY_CACHE") {
        let t = s.trim().to_lowercase();
        if t == "0" || t == "false" || t == "no" || t == "off" {
            return false;
        }
        if t == "1" || t == "true" || t == "yes" || t == "on" {
            return true;
        }
    }
    cli_flag || true
}

/// Number of CUDA devices visible to this process (`CUDA_VISIBLE_DEVICES` token count, or LibTorch count when unset).
///
/// Used for `--device auto` memory policy: when exactly **one** GPU is visible, Boltz preprocess defaults to CPU
/// so its peak does not stack with LibTorch on the same card (unless opted out).
pub fn parent_visible_cuda_device_count() -> usize {
    if let Ok(s) = std::env::var("CUDA_VISIBLE_DEVICES") {
        let t = s.trim();
        if t.is_empty() {
            return 0;
        }
        return t.split(',').filter(|x| !x.trim().is_empty()).count();
    }
    #[cfg(feature = "tch")]
    {
        tch::maybe_init_cuda();
        if tch::Cuda::is_available() {
            return tch::Cuda::device_count().max(0) as usize;
        }
    }
    0
}

/// Opt out of `--device auto` defaulting Boltz to CPU on a single visible GPU (restore Boltz GPU for speed).
///
/// CLI `--preprocess-auto-boltz-gpu` or env `BOLTR_AUTO_BOLTZ_GPU=1` / `true`.
pub fn resolve_auto_boltz_gpu_opt_out(cli_flag: bool) -> bool {
    cli_flag
        || std::env::var("BOLTR_AUTO_BOLTZ_GPU")
            .map(|v| {
                let t = v.trim();
                t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
            })
            .unwrap_or(false)
}

/// When true, Boltz subprocess should use `--accelerator cpu` for memory: only for `--device auto` with LibTorch on CUDA,
/// a single visible GPU, and no opt-out.
#[must_use]
pub fn auto_default_boltz_cpu_for_memory(
    device_requested: Option<&str>,
    predict_cuda: bool,
    auto_boltz_gpu_opt_out: bool,
) -> bool {
    if let Ok(s) = std::env::var("BOLTR_AUTO_PREPROCESS_BOLTZ_CPU") {
        match s.trim().to_lowercase().as_str() {
            "0" | "false" | "no" | "off" => return false,
            "1" | "true" | "yes" | "on" => return true,
            _ => {}
        }
    }
    if auto_boltz_gpu_opt_out || !predict_cuda {
        return false;
    }
    if device_requested.map(str::trim) != Some("auto") {
        return false;
    }
    if parent_visible_cuda_device_count() != 1 {
        return false;
    }
    if auto_boltz_gpu_allowed_by_free_vram() {
        tracing::info!(
            "preprocess: auto allows GPU Boltz on the single visible GPU because free VRAM is above BOLTR_AUTO_BOLTZ_GPU_MIN_FREE_VRAM_MB"
        );
        return false;
    }
    true
}

#[cfg(feature = "tch")]
fn auto_boltz_min_free_vram_mb() -> u64 {
    std::env::var("BOLTR_AUTO_BOLTZ_GPU_MIN_FREE_VRAM_MB")
        .or_else(|_| std::env::var("BOLTR_AUTO_MIN_FREE_VRAM_MB"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(40_000)
}

#[cfg(feature = "tch")]
fn auto_boltz_gpu_allowed_by_free_vram() -> bool {
    let min_mb = auto_boltz_min_free_vram_mb();
    if min_mb == 0 {
        return true;
    }
    match crate::gpu_mem::query_free_vram_bytes() {
        Some(free) => {
            let free_mib = free / (1024 * 1024);
            let ok = free_mib >= min_mb;
            tracing::info!(
                free_mib,
                min_mib = min_mb,
                ok,
                "preprocess: auto Boltz GPU free VRAM check"
            );
            ok
        }
        None => {
            tracing::info!(
                min_mib = min_mb,
                "preprocess: auto Boltz GPU free VRAM probe failed; keeping CPU fallback"
            );
            return false;
        }
    }
}

#[cfg(not(feature = "tch"))]
fn auto_boltz_gpu_allowed_by_free_vram() -> bool {
    false
}

/// Extra args for `boltz predict` when invoked from `boltr predict --preprocess …`.
///
/// - With **`--preprocess-cuda-visible-devices`** (or env), Boltz runs on that GPU; LibTorch uses `--device` (typically another physical GPU).
/// - On a **single** GPU, Boltz defaults to **`--accelerator gpu`** (upstream default); set **`--preprocess-boltz-cpu`** to force CPU Boltz if LibTorch OOMs after Boltz.
/// - When **`--device auto`** resolves to CUDA and exactly one GPU is visible, **`auto_default_boltz_cpu`** defaults Boltz to CPU unless **`--preprocess-auto-boltz-gpu`** / `BOLTR_AUTO_BOLTZ_GPU`.
/// - When **`--device cpu`** for LibTorch, upstream Boltz must also get **`--accelerator cpu`**, or it keeps the default GPU path and can fail (e.g. cuSOLVER) despite the user choosing CPU.
pub fn bolt_preprocess_args_for_predict(
    device: &str,
    bolt_extra: &[String],
    force_boltz_cpu: bool,
    auto_default_boltz_cpu: bool,
) -> Vec<String> {
    let mut out = bolt_extra.to_vec();
    if should_disable_boltz_kernels_by_default() && !user_set_boltz_kernel_mode(&out) {
        out.push("--no_kernels".to_string());
    }
    if user_set_boltz_accelerator(bolt_extra) {
        return out;
    }
    let need_boltz_cpu =
        !predict_device_is_cuda(device) || force_boltz_cpu || auto_default_boltz_cpu;
    if need_boltz_cpu {
        if !predict_device_is_cuda(device) {
            tracing::info!("preprocess: Boltz subprocess --accelerator cpu (boltr --device cpu)");
        } else {
            tracing::info!(
                "preprocess: Boltz subprocess --accelerator cpu (--preprocess-boltz-cpu or BOLTR_PREPROCESS_BOLTZ_CPU)"
            );
        }
        out.push("--accelerator".to_string());
        out.push("cpu".to_string());
        return out;
    }
    out
}

/// `python` / `python3` next to a `boltz` executable (e.g. `…/venv/bin/boltz` → `…/venv/bin/python`).
fn sibling_python_candidates(bolt_exe: &str) -> Vec<PathBuf> {
    let p = Path::new(bolt_exe.trim());
    if !p.is_file() {
        return vec![];
    }
    let Some(parent) = p.parent() else {
        return vec![];
    };
    vec![parent.join("python"), parent.join("python3")]
}

fn python_imports_torch(py: &str) -> bool {
    Command::new(py)
        .args(["-c", "import torch"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn push_unique_python(candidates: &mut Vec<String>, s: String) {
    let t = s.trim().to_string();
    if t.is_empty() || candidates.iter().any(|x| x == &t) {
        return;
    }
    candidates.push(t);
}

/// Resolve a Python that can `import torch` for post-Boltz helpers.
///
/// Tries in order: `BOLTR_PYTHON`, `BOLTR_REPO/.venv/bin/python`, siblings of `BOLTR_BOLTZ_COMMAND`,
/// siblings of `bolt_command` (from `--bolt-command`), then `python3`.
pub fn resolve_python_for_torch_scripts(bolt_command: Option<&str>) -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();

    if let Ok(p) = std::env::var("BOLTR_PYTHON") {
        push_unique_python(&mut candidates, p);
    }
    if let Ok(repo) = std::env::var("BOLTR_REPO") {
        let v = PathBuf::from(repo).join(".venv/bin/python");
        if v.is_file() {
            if let Some(s) = v.to_str() {
                push_unique_python(&mut candidates, s.to_string());
            }
        }
    }
    if let Ok(b) = std::env::var("BOLTR_BOLTZ_COMMAND") {
        for s in sibling_python_candidates(&b) {
            if s.is_file() {
                if let Some(ss) = s.to_str() {
                    push_unique_python(&mut candidates, ss.to_string());
                }
            }
        }
    }
    if let Some(bc) = bolt_command {
        for s in sibling_python_candidates(bc) {
            if s.is_file() {
                if let Some(ss) = s.to_str() {
                    push_unique_python(&mut candidates, ss.to_string());
                }
            }
        }
    }
    push_unique_python(&mut candidates, "python3".to_string());

    for py in candidates {
        if python_imports_torch(&py) {
            tracing::debug!(%py, "using Python for torch subprocess helpers");
            return Some(py);
        }
    }
    None
}

/// After Boltz exits, run a tiny Python stub to `torch.cuda.empty_cache()` on visible CUDA devices (single-GPU fragmentation mitigation).
///
/// Skips quietly when no interpreter can `import torch` (avoids `ModuleNotFoundError` noise from bare `python3`).
/// Prefer `BOLTR_PYTHON`, or a venv next to the `boltz` executable (`--bolt-command` / `BOLTR_BOLTZ_COMMAND`).
pub fn maybe_post_boltz_empty_cache(bolt_command: Option<&str>) -> Result<()> {
    let Some(py) = resolve_python_for_torch_scripts(bolt_command) else {
        tracing::info!(
            "post-Boltz empty_cache skipped: no Python with PyTorch found (set BOLTR_PYTHON to a venv that has torch, or use a full path to boltz so we can use the same venv's python)"
        );
        return Ok(());
    };
    let status = Command::new(&py)
        .args([
            "-c",
            "import torch; \
             torch.cuda.empty_cache() if torch.cuda.is_available() else None",
        ])
        .status()
        .with_context(|| format!("spawn {py} for post-Boltz empty_cache"))?;
    if !status.success() {
        tracing::warn!(
            ?status,
            py = %py,
            "post-Boltz torch.cuda.empty_cache() subprocess exited non-zero (continuing)"
        );
    } else {
        tracing::info!(%py, "post-Boltz: torch.cuda.empty_cache() ok");
    }
    Ok(())
}

fn collect_prediction_structures(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_prediction_structures(&path, out)?;
            continue;
        }
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        let ext = ext.to_ascii_lowercase();
        if (ext == "cif" || ext == "pdb")
            && path
                .components()
                .any(|c| c.as_os_str().to_string_lossy() == "predictions")
        {
            out.push(path);
        }
    }
    Ok(())
}

fn copy_upstream_prediction_structures(staging: &Path, dest_dir: &Path) -> Result<usize> {
    let mut structures = Vec::new();
    collect_prediction_structures(staging, &mut structures)?;
    if structures.is_empty() {
        return Ok(0);
    }
    let pred_dir = dest_dir.join(".boltr_upstream_predictions");
    fs::create_dir_all(&pred_dir).with_context(|| format!("mkdir {}", pred_dir.display()))?;
    let mut copied = 0_usize;
    for src in structures {
        let Some(name) = src.file_name() else {
            continue;
        };
        let dst = pred_dir.join(name);
        fs::copy(&src, &dst).with_context(|| {
            format!(
                "copy upstream prediction {} -> {}",
                src.display(),
                dst.display()
            )
        })?;
        copied += 1;
    }
    Ok(copied)
}

/// Run upstream `boltz predict` and copy preprocess artifacts next to `yaml_path`.
pub fn run_boltz_preprocess(
    yaml_path: &Path,
    bolt_command: &str,
    staging_dir: Option<PathBuf>,
    use_msa_server: bool,
    bolt_extra: &[String],
    use_symlink: bool,
    keep_staging: bool,
    child_env: &BoltzChildEnv,
) -> Result<()> {
    let yaml_path = yaml_path
        .canonicalize()
        .with_context(|| format!("yaml {}", yaml_path.display()))?;
    let yaml_stem = yaml_path
        .file_stem()
        .and_then(|s| s.to_str())
        .context("yaml file stem")?;
    let dest_dir = yaml_path
        .parent()
        .context("yaml has no parent")?
        .to_path_buf();

    let staging = staging_dir.unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "boltr-preprocess-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ))
    });
    if staging.exists() {
        fs::remove_dir_all(&staging).ok();
    }
    fs::create_dir_all(&staging)?;

    let mut cmd = Command::new(bolt_command);
    if let Some(ref vis) = child_env.cuda_visible_devices {
        cmd.env("CUDA_VISIBLE_DEVICES", vis);
        tracing::info!(
            %vis,
            "Boltz subprocess CUDA_VISIBLE_DEVICES (LibTorch uses parent process device index)"
        );
    }
    if let Some(ref conf) = child_env.pytorch_cuda_alloc_conf {
        cmd.env("PYTORCH_CUDA_ALLOC_CONF", conf);
        tracing::info!(%conf, "Boltz subprocess PYTORCH_CUDA_ALLOC_CONF");
    }
    if let Some(n) = child_env.preprocess_threads {
        let n = n.to_string();
        for key in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BOLTR_PREPROCESS_THREADS",
        ] {
            cmd.env(key, &n);
        }
        tracing::info!(threads = %n, "Boltz subprocess CPU thread cap");
    }
    if let Some(ref precision) = child_env.matmul_precision {
        cmd.env("BOLTR_BOLTZ_MATMUL_PRECISION", precision);
        tracing::info!(%precision, "Boltz subprocess matmul precision");
    }
    cmd.arg("predict");
    cmd.arg(&yaml_path);
    cmd.arg("--out_dir");
    cmd.arg(&staging);
    cmd.arg("--override");
    if use_msa_server {
        cmd.arg("--use_msa_server");
    }
    for a in bolt_extra {
        cmd.arg(a);
    }

    info!(?cmd, "spawning Boltz preprocess");
    let status = cmd
        .status()
        .with_context(|| format!(
            "failed to spawn {bolt_command} — install Boltz on PATH or pass --bolt-command with the full path to the boltz executable"
        ))?;
    if !status.success() {
        bail!("{bolt_command} predict exited with {status}");
    }

    let manifest_path =
        boltr_io::find_boltz_manifest_path(&staging, yaml_stem).with_context(|| {
            format!(
                "find manifest under {} (check Boltz version / output layout)",
                staging.display()
            )
        })?;

    boltr_io::copy_flat_preprocess_bundle(&manifest_path, &dest_dir, use_symlink)
        .context("copy preprocess bundle")?;
    let copied_predictions = copy_upstream_prediction_structures(&staging, &dest_dir)
        .context("copy upstream Boltz prediction structures")?;

    info!(
        dest = %dest_dir.display(),
        copied_predictions,
        "preprocess bundle materialized next to YAML"
    );

    if !keep_staging {
        fs::remove_dir_all(&staging).ok();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{bolt_preprocess_args_for_predict, resolve_post_boltz_empty_cache};
    use std::path::Path;

    #[test]
    fn sibling_python_candidates_beside_boltz() {
        let tmp = tempfile::tempdir().unwrap();
        let boltz = tmp.path().join("boltz");
        std::fs::write(&boltz, b"x").unwrap();
        let c = super::sibling_python_candidates(boltz.to_str().unwrap());
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], Path::new(tmp.path()).join("python"));
        assert_eq!(c[1], Path::new(tmp.path()).join("python3"));
    }

    #[test]
    fn boltz_gpu_default_when_boltr_cuda_no_force() {
        let out = bolt_preprocess_args_for_predict("cuda", &[], false, false);
        assert_eq!(out, vec!["--no_kernels".to_string()]);
    }

    #[test]
    fn boltz_cpu_when_force() {
        let out = bolt_preprocess_args_for_predict("cuda", &[], true, false);
        assert_eq!(
            out,
            vec![
                "--no_kernels".to_string(),
                "--accelerator".to_string(),
                "cpu".to_string()
            ]
        );
    }

    #[test]
    fn boltz_cpu_when_boltr_cpu() {
        let out = bolt_preprocess_args_for_predict("cpu", &[], false, false);
        assert_eq!(
            out,
            vec![
                "--no_kernels".to_string(),
                "--accelerator".to_string(),
                "cpu".to_string()
            ]
        );
    }

    #[test]
    fn user_bolt_arg_accelerator_untouched() {
        let extra = vec!["--accelerator".to_string(), "gpu".to_string()];
        let out = bolt_preprocess_args_for_predict("cuda", &extra, true, false);
        assert_eq!(
            out,
            vec![
                "--accelerator".to_string(),
                "gpu".to_string(),
                "--no_kernels".to_string()
            ]
        );
    }

    #[test]
    fn auto_default_boltz_cpu_flag_adds_accelerator() {
        let out = bolt_preprocess_args_for_predict("cuda", &[], false, true);
        assert_eq!(
            out,
            vec![
                "--no_kernels".to_string(),
                "--accelerator".to_string(),
                "cpu".to_string()
            ]
        );
    }

    #[test]
    fn auto_default_boltz_cpu_requires_auto_request() {
        assert!(!super::auto_default_boltz_cpu_for_memory(
            Some("gpu"),
            true,
            false
        ));
        assert!(!super::auto_default_boltz_cpu_for_memory(None, true, false));
    }

    #[test]
    fn auto_default_boltz_cpu_respects_opt_out() {
        assert!(!super::auto_default_boltz_cpu_for_memory(
            Some("auto"),
            true,
            true
        ));
    }

    #[test]
    fn post_empty_cache_never_when_cpu_predict() {
        assert!(!resolve_post_boltz_empty_cache(false, false));
        assert!(!resolve_post_boltz_empty_cache(true, false));
    }

    #[test]
    fn auto_default_boltz_cpu_disabled_by_env() {
        std::env::set_var("BOLTR_AUTO_PREPROCESS_BOLTZ_CPU", "0");
        let r = super::auto_default_boltz_cpu_for_memory(Some("auto"), true, false);
        std::env::remove_var("BOLTR_AUTO_PREPROCESS_BOLTZ_CPU");
        assert!(!r);
    }

    #[test]
    fn auto_default_boltz_cpu_only_for_auto_cuda() {
        std::env::remove_var("BOLTR_AUTO_PREPROCESS_BOLTZ_CPU");
        assert!(!super::auto_default_boltz_cpu_for_memory(
            Some("gpu"),
            true,
            false
        ));
        assert!(!super::auto_default_boltz_cpu_for_memory(None, true, false));
    }
}

/// Write native protein-only bundle next to `yaml_path`.
pub fn run_native_preprocess(
    yaml_path: &Path,
    record_id: Option<&str>,
    max_msa_seqs: Option<usize>,
    fetched_msa_dir: Option<&std::path::Path>,
) -> Result<()> {
    let yaml_path = yaml_path
        .canonicalize()
        .with_context(|| format!("yaml {}", yaml_path.display()))?;
    let dest_dir = yaml_path
        .parent()
        .context("yaml has no parent")?
        .to_path_buf();
    boltr_io::write_native_preprocess_bundle(
        &yaml_path,
        &dest_dir,
        record_id,
        max_msa_seqs,
        fetched_msa_dir,
    )?;
    info!(dest = %dest_dir.display(), "native preprocess bundle written");
    Ok(())
}

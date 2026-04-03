//! `boltr preprocess` — Tier 1 (Boltz subprocess) and Tier 2 (native protein-only) bundle generation.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use tracing::info;

/// Environment applied only to the upstream `boltz predict` subprocess.
#[derive(Debug, Default, Clone)]
pub struct BoltzChildEnv {
    /// `CUDA_VISIBLE_DEVICES` for the Boltz child (e.g. `"1"` so Boltz uses GPU 1 while LibTorch uses `cuda:0` on GPU 0).
    pub cuda_visible_devices: Option<String>,
    /// `PYTORCH_CUDA_ALLOC_CONF` for the Boltz child (reduces fragmentation before LibTorch loads).
    pub pytorch_cuda_alloc_conf: Option<String>,
}

/// True if `device` selects a CUDA device for LibTorch (`cpu` / `cuda` / `cuda:N`).
pub fn predict_device_is_cuda(device: &str) -> bool {
    let s = device.trim().to_lowercase();
    s == "cuda" || s == "gpu" || s.starts_with("cuda:")
}

/// True when the user already passed `--accelerator …` to the Boltz subprocess.
fn user_set_boltz_accelerator(bolt_extra: &[String]) -> bool {
    bolt_extra.iter().any(|arg| {
        arg == "--accelerator" || arg.starts_with("--accelerator=")
    })
}

/// Merge CLI (`--preprocess-cuda-visible-devices`) with `BOLTR_BOLTZ_CUDA_VISIBLE_DEVICES` (CLI wins).
pub fn resolve_preprocess_cuda_visible_devices(cli: Option<&str>) -> Option<String> {
    let from_cli = cli.map(str::trim).filter(|s| !s.is_empty()).map(String::from);
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

/// `--preprocess-boltz-cpu` or `BOLTR_PREPROCESS_BOLTZ_CPU=1`.
pub fn resolve_force_boltz_cpu(cli_flag: bool) -> bool {
    cli_flag
        || std::env::var("BOLTR_PREPROCESS_BOLTZ_CPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

/// `--preprocess-post-boltz-empty-cache` or `BOLTR_PREPROCESS_POST_BOLTZ_EMPTY_CACHE=1`.
pub fn resolve_post_boltz_empty_cache(cli_flag: bool) -> bool {
    cli_flag
        || std::env::var("BOLTR_PREPROCESS_POST_BOLTZ_EMPTY_CACHE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

/// Extra args for `boltz predict` when invoked from `boltr predict --preprocess …`.
///
/// - With **`--preprocess-cuda-visible-devices`** (or env), Boltz runs on that GPU; LibTorch uses `--device` (typically another physical GPU).
/// - On a **single** GPU, Boltz defaults to **`--accelerator gpu`** (upstream default); set **`--preprocess-boltz-cpu`** to force CPU Boltz if LibTorch OOMs after Boltz.
pub fn bolt_preprocess_args_for_predict(
    device: &str,
    bolt_extra: &[String],
    force_boltz_cpu: bool,
) -> Vec<String> {
    if !predict_device_is_cuda(device) || user_set_boltz_accelerator(bolt_extra) {
        return bolt_extra.to_vec();
    }
    if force_boltz_cpu {
        let mut out = bolt_extra.to_vec();
        tracing::info!(
            "preprocess: Boltz subprocess --accelerator cpu (--preprocess-boltz-cpu or BOLTR_PREPROCESS_BOLTZ_CPU)"
        );
        out.push("--accelerator".to_string());
        out.push("cpu".to_string());
        return out;
    }
    bolt_extra.to_vec()
}

/// After Boltz exits, run a tiny Python stub to `torch.cuda.empty_cache()` on visible CUDA devices (single-GPU fragmentation mitigation).
pub fn maybe_post_boltz_empty_cache() -> Result<()> {
    let py = std::env::var("BOLTR_PYTHON").unwrap_or_else(|_| "python3".to_string());
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
            "post-Boltz torch.cuda.empty_cache() subprocess exited non-zero (continuing)"
        );
    } else {
        tracing::info!("post-Boltz: torch.cuda.empty_cache() ok");
    }
    Ok(())
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

    info!(
        dest = %dest_dir.display(),
        "preprocess bundle materialized next to YAML"
    );

    if !keep_staging {
        fs::remove_dir_all(&staging).ok();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::bolt_preprocess_args_for_predict;

    #[test]
    fn boltz_gpu_default_when_boltr_cuda_no_force() {
        let out = bolt_preprocess_args_for_predict("cuda", &[], false);
        assert!(out.is_empty());
    }

    #[test]
    fn boltz_cpu_when_force() {
        let out = bolt_preprocess_args_for_predict("cuda", &[], true);
        assert_eq!(
            out,
            vec!["--accelerator".to_string(), "cpu".to_string()]
        );
    }

    #[test]
    fn boltz_default_when_boltr_cpu() {
        let out = bolt_preprocess_args_for_predict("cpu", &[], false);
        assert!(out.is_empty());
    }

    #[test]
    fn user_bolt_arg_accelerator_untouched() {
        let extra = vec!["--accelerator".to_string(), "gpu".to_string()];
        let out = bolt_preprocess_args_for_predict("cuda", &extra, true);
        assert_eq!(out, extra);
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

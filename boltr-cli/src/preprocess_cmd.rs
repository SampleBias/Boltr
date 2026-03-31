//! `boltr preprocess` — Tier 1 (Boltz subprocess) and Tier 2 (native protein-only) bundle generation.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use tracing::info;

/// Run upstream `boltz predict` and copy preprocess artifacts next to `yaml_path`.
pub fn run_boltz_preprocess(
    yaml_path: &Path,
    bolt_command: &str,
    staging_dir: Option<PathBuf>,
    use_msa_server: bool,
    bolt_extra: &[String],
    use_symlink: bool,
    keep_staging: bool,
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

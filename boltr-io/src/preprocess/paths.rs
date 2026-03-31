//! Canonical directory resolution for preprocess bundles (must match [`super::native::write_native_preprocess_bundle`] /
//! [`super::bundle::copy_flat_preprocess_bundle`] which use [`std::path::Path::canonicalize`]).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::inference_dataset::{msa_id_for_path, msa_id_is_active, parse_manifest_path};

/// Parent directory of the input YAML after [`Path::canonicalize`].
///
/// The predict bridge and preprocess writers both rely on this directory so `manifest.json` and
/// `{id}.npz` resolve to the same files regardless of relative vs absolute CLI paths.
pub fn canonical_yaml_parent(yaml_path: &Path) -> Result<PathBuf> {
    let p = yaml_path
        .canonicalize()
        .with_context(|| format!("canonicalize input YAML {}", yaml_path.display()))?;
    p.parent()
        .map(Path::to_path_buf)
        .context("input YAML has no parent directory")
}

/// Returns true when a Boltz-style preprocess bundle beside the YAML is complete enough for
/// [`crate::load_input`] (non-affinity path): `manifest.json` plus structure and MSA `.npz` files.
///
/// When `affinity` is true, checks for `{target_dir}/{id}/pre_affinity_{id}.npz` instead of a flat
/// `{id}.npz`.
pub fn preprocess_bundle_ready(yaml_path: &Path, affinity: bool) -> Result<bool> {
    let dir = canonical_yaml_parent(yaml_path)?;
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Ok(false);
    }
    let manifest = parse_manifest_path(&manifest_path)?;
    let Some(rec) = manifest.records.first() else {
        return Ok(false);
    };

    let structure_ok = if affinity {
        dir.join(&rec.id)
            .join(format!("pre_affinity_{}.npz", rec.id))
            .is_file()
    } else {
        dir.join(format!("{}.npz", rec.id)).is_file()
    };
    if !structure_ok {
        return Ok(false);
    }

    let mut seen_msa: HashSet<String> = HashSet::new();
    for ch in &rec.chains {
        if !msa_id_is_active(&ch.msa_id) {
            continue;
        }
        let fname = msa_id_for_path(&ch.msa_id)?;
        if !seen_msa.insert(fname.clone()) {
            continue;
        }
        if !dir.join(format!("{fname}.npz")).is_file() {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Copy `preprocess_dir/msa/*.a3m` into `out_dir/msa/` (for prediction run artifacts).
pub fn copy_msa_a3m_to_output(preprocess_dir: &Path, out_dir: &Path) -> Result<()> {
    let src = preprocess_dir.join("msa");
    if !src.is_dir() {
        return Ok(());
    }
    let dst = out_dir.join("msa");
    fs::create_dir_all(&dst).with_context(|| format!("mkdir {}", dst.display()))?;
    for ent in fs::read_dir(&src).with_context(|| format!("read_dir {}", src.display()))? {
        let ent = ent?;
        let p = ent.path();
        if p.extension().and_then(|e| e.to_str()) != Some("a3m") {
            continue;
        }
        let dest = dst.join(ent.file_name());
        fs::copy(&p, &dest)
            .with_context(|| format!("copy {} -> {}", p.display(), dest.display()))?;
    }
    Ok(())
}

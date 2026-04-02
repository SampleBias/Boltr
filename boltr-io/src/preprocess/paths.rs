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

/// Resolve optional `constraints_dir` / `extra_mols_dir` for [`crate::load_input`].
///
/// When `auto_extras` is true and a CLI path is omitted, looks under `preprocess_dir` for
/// `mols/` or `extra_mols/` (must contain at least one `*.json`) and `constraints/` or
/// `residue_constraints/` (directory exists).
#[must_use]
pub fn resolve_preprocess_load_dirs(
    preprocess_dir: &Path,
    extra_mols_cli: Option<&Path>,
    constraints_cli: Option<&Path>,
    auto_extras: bool,
) -> (Option<PathBuf>, Option<PathBuf>) {
    let mut extra = extra_mols_cli.map(Path::to_path_buf);
    let mut cons = constraints_cli.map(Path::to_path_buf);
    if auto_extras {
        if extra.is_none() {
            for name in ["mols", "extra_mols"] {
                let p = preprocess_dir.join(name);
                if dir_has_json_files(&p) {
                    tracing::info!(
                        dir = %p.display(),
                        "preprocess auto-extras: using extra mols directory"
                    );
                    extra = Some(p);
                    break;
                }
            }
        }
        if cons.is_none() {
            for name in ["constraints", "residue_constraints"] {
                let p = preprocess_dir.join(name);
                if p.is_dir() {
                    tracing::info!(
                        dir = %p.display(),
                        "preprocess auto-extras: using residue constraints directory"
                    );
                    cons = Some(p);
                    break;
                }
            }
        }
    }
    (extra, cons)
}

fn dir_has_json_files(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    rd.filter_map(|e| e.ok()).any(|e| {
        e.path()
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    })
}

#[cfg(test)]
mod resolve_tests {
    use super::*;
    use std::fs;

    #[test]
    fn resolve_respects_explicit_cli_paths() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mols = tmp.path().join("my_mols");
        fs::create_dir_all(&mols).unwrap();
        let c = tmp.path().join("my_cons");
        fs::create_dir_all(&c).unwrap();
        let (e, co) = resolve_preprocess_load_dirs(tmp.path(), Some(&mols), Some(&c), false);
        assert_eq!(e.as_ref().map(|p| p.as_path()), Some(mols.as_path()));
        assert_eq!(co.as_ref().map(|p| p.as_path()), Some(c.as_path()));
    }

    #[test]
    fn auto_extras_finds_mols_with_json() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mols = tmp.path().join("mols");
        fs::create_dir_all(&mols).unwrap();
        fs::write(mols.join("X.json"), "[]").unwrap();
        let (e, _) = resolve_preprocess_load_dirs(tmp.path(), None, None, true);
        assert_eq!(
            e.as_ref().map(|p| p.file_name().unwrap().to_str().unwrap()),
            Some("mols")
        );
    }
}

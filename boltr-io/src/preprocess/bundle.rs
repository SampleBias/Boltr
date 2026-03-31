//! Copy Boltz preprocess artifacts into a flat directory layout for [`crate::inference_dataset::load_input`].
//!
//! Expected layout next to the input YAML: `manifest.json`, `{record_id}.npz`, `{msa_id}.npz` per chain.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::inference_dataset::{msa_id_for_path, msa_id_is_active, parse_manifest_path};

/// Copy (or symlink) preprocess files from `manifest_parent` (directory containing `manifest.json`)
/// into `dest_dir`. Includes structure `{id}.npz`, MSA `{msa_id}.npz`, optional `{id}_{template}.npz`,
/// optional constraints `{id}.npz`.
pub fn copy_flat_preprocess_bundle(
    manifest_path: &Path,
    dest_dir: &Path,
    use_symlink: bool,
) -> Result<()> {
    let manifest_parent = manifest_path
        .parent()
        .context("manifest has no parent directory")?;
    fs::create_dir_all(dest_dir).with_context(|| format!("mkdir {}", dest_dir.display()))?;

    let manifest = parse_manifest_path(manifest_path)?;
    if manifest.records.is_empty() {
        bail!("manifest has no records");
    }

    let dest_manifest = dest_dir.join("manifest.json");
    copy_or_link(manifest_path, &dest_manifest, use_symlink)?;

    for rec in &manifest.records {
        let id = &rec.id;
        let structure = manifest_parent.join(format!("{id}.npz"));
        if structure.is_file() {
            copy_or_link(
                &structure,
                &dest_dir.join(format!("{id}.npz")),
                use_symlink,
            )?;
        }

        for chain in &rec.chains {
            if !msa_id_is_active(&chain.msa_id) {
                continue;
            }
            let fname = msa_id_for_path(&chain.msa_id)?;
            let msa_src = manifest_parent.join(format!("{fname}.npz"));
            if msa_src.is_file() {
                copy_or_link(
                    &msa_src,
                    &dest_dir.join(format!("{fname}.npz")),
                    use_symlink,
                )?;
            }
        }

        if let Some(templates) = &rec.templates {
            for t in templates {
                let tmpl = manifest_parent.join(format!("{}_{}.npz", rec.id, t.name));
                if tmpl.is_file() {
                    copy_or_link(
                        &tmpl,
                        &dest_dir.join(format!("{}_{}.npz", rec.id, t.name)),
                        use_symlink,
                    )?;
                }
            }
        }

    }

    Ok(())
}

fn copy_or_link(src: &Path, dst: &Path, symlink: bool) -> Result<()> {
    if dst.exists() {
        fs::remove_file(dst).ok();
    }
    if symlink {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(src, dst).with_context(|| {
                format!(
                    "symlink {} -> {}",
                    src.display(),
                    dst.display()
                )
            })?;
        }
        #[cfg(not(unix))]
        {
            fs::copy(src, dst).with_context(|| {
                format!(
                    "copy {} -> {} (symlink not supported)",
                    src.display(),
                    dst.display()
                )
            })?;
        }
    } else {
        fs::copy(src, dst)
            .with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
    }
    Ok(())
}

/// Search `staging_dir` for a `manifest.json` that belongs to `yaml_stem` (input file stem).
/// Preference order: `processed/{stem}/manifest.json`, then any `processed/**/manifest.json` whose parent ends with `stem`,
/// then first `manifest.json` found under `staging_dir` with a sibling `{record_id}.npz` for the first record.
pub fn find_boltz_manifest_path(staging_dir: &Path, yaml_stem: &str) -> Result<PathBuf> {
    let direct = staging_dir.join("processed").join(yaml_stem).join("manifest.json");
    if direct.is_file() {
        return Ok(direct);
    }

    let processed = staging_dir.join("processed");
    if processed.is_dir() {
        if let Some(p) = find_manifest_with_structure_npz(&processed, 6)? {
            return Ok(p);
        }
    }

    if let Some(p) = find_manifest_with_structure_npz(staging_dir, 8)? {
        return Ok(p);
    }

    bail!(
        "could not find manifest.json with sibling structure npz under {}",
        staging_dir.display()
    );
}

fn find_manifest_with_structure_npz(dir: &Path, max_depth: usize) -> Result<Option<PathBuf>> {
    find_manifest_with_structure_npz_inner(dir, max_depth, 0)
}

fn find_manifest_with_structure_npz_inner(
    dir: &Path,
    max_depth: usize,
    depth: usize,
) -> Result<Option<PathBuf>> {
    if depth > max_depth {
        return Ok(None);
    }
    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    for ent in read.filter_map(|e| e.ok()) {
        let p = ent.path();
        if p.is_dir() {
            if let Some(found) = find_manifest_with_structure_npz_inner(&p, max_depth, depth + 1)? {
                return Ok(Some(found));
            }
        } else if p.file_name().and_then(|n| n.to_str()) == Some("manifest.json") {
            if let Ok(m) = parse_manifest_path(&p) {
                if let Some(r) = m.records.first() {
                    let parent = p.parent().unwrap();
                    if parent.join(format!("{}.npz", r.id)).is_file() {
                        return Ok(Some(p));
                    }
                }
            }
        }
    }
    Ok(None)
}

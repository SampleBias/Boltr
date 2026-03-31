//! Copy Boltz preprocess artifacts into a flat directory layout for [`crate::inference_dataset::load_input`].
//!
//! Expected layout next to the input YAML: `manifest.json`, `{record_id}.npz`, `{msa_id}.npz` per chain.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::inference_dataset::{msa_id_for_path, msa_id_is_active, parse_manifest_path};

/// Structure `.npz` next to `manifest.json` (legacy) or under `structures/` (current upstream Boltz).
fn structure_npz_path(manifest_parent: &Path, record_id: &str) -> Option<PathBuf> {
    let flat = manifest_parent.join(format!("{record_id}.npz"));
    if flat.is_file() {
        return Some(flat);
    }
    let nested = manifest_parent
        .join("structures")
        .join(format!("{record_id}.npz"));
    if nested.is_file() {
        return Some(nested);
    }
    None
}

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
        if let Some(structure) = structure_npz_path(manifest_parent, id) {
            copy_or_link(&structure, &dest_dir.join(format!("{id}.npz")), use_symlink)?;
        }

        for chain in &rec.chains {
            if !msa_id_is_active(&chain.msa_id) {
                continue;
            }
            let fname = msa_id_for_path(&chain.msa_id)?;
            let flat = manifest_parent.join(format!("{fname}.npz"));
            let msa_src = if flat.is_file() {
                flat
            } else {
                manifest_parent.join("msa").join(format!("{fname}.npz"))
            };
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
                let name = format!("{}_{}.npz", rec.id, t.name);
                let flat = manifest_parent.join(&name);
                let tmpl = if flat.is_file() {
                    flat
                } else {
                    manifest_parent.join("templates").join(&name)
                };
                if tmpl.is_file() {
                    copy_or_link(&tmpl, &dest_dir.join(&name), use_symlink)?;
                }
            }
        }
    }

    copy_msa_a3m_sidecars(manifest_parent, dest_dir, use_symlink)?;

    Ok(())
}

/// Copy ColabFold/Boltz `.a3m` files from `manifest_parent/msa/` into `dest_dir/msa/` so the YAML
/// directory has both `.npz` tensors and human-readable MSAs for inspection and re-runs.
fn copy_msa_a3m_sidecars(manifest_parent: &Path, dest_dir: &Path, use_symlink: bool) -> Result<()> {
    let src_msa = manifest_parent.join("msa");
    if !src_msa.is_dir() {
        return Ok(());
    }
    let dst_msa = dest_dir.join("msa");
    fs::create_dir_all(&dst_msa).with_context(|| format!("mkdir {}", dst_msa.display()))?;
    for ent in fs::read_dir(&src_msa).with_context(|| format!("read_dir {}", src_msa.display()))? {
        let ent = ent?;
        let p = ent.path();
        if p.extension().and_then(|e| e.to_str()) != Some("a3m") {
            continue;
        }
        let name = ent.file_name();
        copy_or_link(&p, &dst_msa.join(name), use_symlink)?;
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
            std::os::unix::fs::symlink(src, dst)
                .with_context(|| format!("symlink {} -> {}", src.display(), dst.display()))?;
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
/// Preference order: `boltz_results_{stem}/processed/manifest.json` (upstream Boltz ≥2 layout),
/// `processed/{stem}/manifest.json`, then any `processed/**/manifest.json` with a structure `.npz`,
/// then first `manifest.json` found under `staging_dir` with structure npz for the first record.
pub fn find_boltz_manifest_path(staging_dir: &Path, yaml_stem: &str) -> Result<PathBuf> {
    let boltz_results = staging_dir.join(format!("boltz_results_{yaml_stem}"));
    let br_manifest = boltz_results.join("processed").join("manifest.json");
    if br_manifest.is_file() {
        return Ok(br_manifest);
    }

    let direct = staging_dir
        .join("processed")
        .join(yaml_stem)
        .join("manifest.json");
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
                    if structure_npz_path(parent, &r.id).is_some() {
                        return Ok(Some(p));
                    }
                }
            }
        }
    }
    Ok(None)
}

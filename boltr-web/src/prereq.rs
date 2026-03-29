//! Cache + YAML checks aligned with `boltr predict` / `predict_tch` expectations.

use std::path::{Path, PathBuf};

use boltr_io::config::BoltzInput;
use boltr_io::parse_manifest_path;
use serde::Serialize;

/// Resolve cache directory: same rules as `boltr` CLI.
pub fn resolve_cache_dir(cli_cache: Option<&Path>) -> PathBuf {
    if let Some(p) = cli_cache {
        return p.to_path_buf();
    }
    if let Ok(p) = std::env::var("BOLTZ_CACHE") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("boltr")
}

#[derive(Debug, Clone, Serialize)]
pub struct FileCheck {
    pub label: String,
    pub path: String,
    pub present: bool,
    /// Whether this file is strictly required for native `predict` with `--features tch`.
    pub required: bool,
    /// Human-readable note (e.g. export script hint).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub cache_dir: String,
    pub files: Vec<FileCheck>,
    /// `true` when `boltz2_conf.safetensors` exists (Rust-native load path).
    pub native_structure_checkpoint_ok: bool,
    /// `true` when preprocess bridge inputs could work (manifest + npz checks are per-upload).
    pub notes: Vec<String>,
}

pub fn gather_status(cache: &Path) -> StatusResponse {
    let mut notes = Vec::new();
    let sf = cache.join("boltz2_conf.safetensors");
    let ckpt = cache.join("boltz2_conf.ckpt");
    let hparams = cache.join("boltz2_hparams.json");
    let ccd = cache.join("ccd.pkl");
    let mols = cache.join("mols.tar");
    let aff_sf = cache.join("boltz2_aff.safetensors");
    let aff_ckpt = cache.join("boltz2_aff.ckpt");

    let native_structure_checkpoint_ok = sf.is_file();

    if ckpt.is_file() && !sf.is_file() {
        notes.push(
            "Have .ckpt but not .safetensors — run scripts/export_checkpoint_to_safetensors.py for native predict."
                .to_string(),
        );
    }

    if !cache.is_dir() {
        notes.push(format!("Cache folder missing: {}", cache.display()));
    }

    let files = vec![
        FileCheck {
            label: "boltz2_conf.safetensors".to_string(),
            path: sf.display().to_string(),
            present: sf.is_file(),
            required: true,
            note: None,
        },
        FileCheck {
            label: "boltz2_conf.ckpt".to_string(),
            path: ckpt.display().to_string(),
            present: ckpt.is_file(),
            required: false,
            note: Some("Export to .safetensors for Rust load.".to_string()),
        },
        FileCheck {
            label: "boltz2_hparams.json".to_string(),
            path: hparams.display().to_string(),
            present: hparams.is_file(),
            required: false,
            note: Some("Optional.".to_string()),
        },
        FileCheck {
            label: "ccd.pkl".to_string(),
            path: ccd.display().to_string(),
            present: ccd.is_file(),
            required: true,
            note: None,
        },
        FileCheck {
            label: "mols.tar".to_string(),
            path: mols.display().to_string(),
            present: mols.is_file(),
            required: true,
            note: None,
        },
        FileCheck {
            label: "boltz2_aff.safetensors".to_string(),
            path: aff_sf.display().to_string(),
            present: aff_sf.is_file(),
            required: false,
            note: Some("Affinity only.".to_string()),
        },
        FileCheck {
            label: "boltz2_aff.ckpt".to_string(),
            path: aff_ckpt.display().to_string(),
            present: aff_ckpt.is_file(),
            required: false,
            note: Some("Affinity only.".to_string()),
        },
    ];

    notes.push("Inference needs LibTorch and `boltr` with `--features tch`.".to_string());

    StatusResponse {
        cache_dir: cache.display().to_string(),
        files,
        native_structure_checkpoint_ok,
        notes,
    }
}

#[derive(Debug, Serialize)]
pub struct PathProbe {
    pub kind: String,
    pub chain_or_hint: String,
    pub path: String,
    pub exists: bool,
}

#[derive(Debug, Serialize)]
pub struct PreprocessProbe {
    pub manifest_present: bool,
    pub manifest_ok: Option<bool>,
    pub manifest_error: Option<String>,
    pub record_id: Option<String>,
    pub npz_present: bool,
    pub npz_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ValidateResponse {
    pub yaml_ok: bool,
    pub parse_error: Option<String>,
    pub chain_ids: Vec<String>,
    pub path_probes: Vec<PathProbe>,
    pub proteins_missing_msa: Vec<String>,
    pub preprocess: PreprocessProbe,
    pub cache: StatusResponse,
    /// Overall: YAML parses, required paths exist, cache has safetensors + ccd + mols, preprocess bridge complete when manifest exists.
    pub ready_to_run_native: bool,
    pub blockers: Vec<String>,
    /// Non-fatal issues (e.g. MSA not inlined — `boltr predict --use-msa-server` may still work).
    pub warnings: Vec<String>,
}

fn push_blocker(blockers: &mut Vec<String>, msg: String) {
    if !blockers.iter().any(|b| b == &msg) {
        blockers.push(msg);
    }
}

/// Validate YAML bytes as if saved at `yaml_path` (parent dir used for relative paths).
pub fn validate_yaml_at(
    yaml_path: &Path,
    yaml_text: &str,
    cache: &Path,
) -> ValidateResponse {
    let cache_status = gather_status(cache);
    let mut blockers: Vec<String> = Vec::new();

    let parsed: Result<BoltzInput, _> = serde_yaml::from_str(yaml_text);
    let (yaml_ok, parse_error, input) = match parsed {
        Ok(i) => (true, None, Some(i)),
        Err(e) => {
            push_blocker(
                &mut blockers,
                format!("YAML parse error: {e}"),
            );
            (
                false,
                Some(e.to_string()),
                None,
            )
        }
    };

    let mut path_probes = Vec::new();
    let mut chain_ids = Vec::new();
    let mut proteins_missing_msa = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let base = yaml_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    if let Some(ref input) = input {
        chain_ids = input.summary_chain_ids();

        for (chain, msa_rel) in input.protein_msa_paths() {
            let p = base.join(msa_rel.trim_start_matches("./"));
            let exists = p.is_file();
            path_probes.push(PathProbe {
                kind: "msa".to_string(),
                chain_or_hint: chain,
                path: p.display().to_string(),
                exists,
            });
            if !exists {
                push_blocker(
                    &mut blockers,
                    format!("MSA file missing: {}", p.display()),
                );
            }
        }

        for t in input.template_entries() {
            if let Some(tp) = t.path() {
                let p = base.join(tp.trim_start_matches("./"));
                let exists = p.is_file();
                path_probes.push(PathProbe {
                    kind: "template".to_string(),
                    chain_or_hint: if t.is_cif() { "cif".to_string() } else { "pdb".to_string() },
                    path: p.display().to_string(),
                    exists,
                });
                if !exists {
                    push_blocker(
                        &mut blockers,
                        format!("Template file missing: {}", p.display()),
                    );
                }
            }
        }

        for id in input.protein_sequences_for_msa().into_iter().map(|(a, _)| a) {
            proteins_missing_msa.push(id);
        }
        if !proteins_missing_msa.is_empty() {
            warnings.push(format!(
                "Some protein chains have no `msa:` path — use local .a3m files or run `boltr predict --use-msa-server` for those chains: {}",
                proteins_missing_msa.join(", ")
            ));
        }
    }

    let manifest_path = base.join("manifest.json");
    let manifest_present = manifest_path.is_file();
    let mut manifest_ok = None;
    let mut manifest_error = None;
    let mut record_id = None;
    let mut npz_present = false;
    let mut npz_path = None;

    if manifest_present {
        match parse_manifest_path(&manifest_path) {
            Ok(m) => {
                manifest_ok = Some(true);
                if let Some(rec) = m.records.first() {
                    record_id = Some(rec.id.clone());
                    let np = base.join(format!("{}.npz", rec.id));
                    npz_present = np.is_file();
                    npz_path = Some(np.display().to_string());
                    if !npz_present {
                        push_blocker(
                            &mut blockers,
                            format!(
                                "Preprocess npz missing for record {:?}: {}",
                                rec.id,
                                np.display()
                            ),
                        );
                    }
                } else {
                    manifest_ok = Some(false);
                    manifest_error = Some("manifest.json has no records".to_string());
                    push_blocker(&mut blockers, "manifest.json has no records".to_string());
                }
            }
            Err(e) => {
                manifest_ok = Some(false);
                manifest_error = Some(e.to_string());
                push_blocker(
                    &mut blockers,
                    format!("manifest.json invalid: {e}"),
                );
            }
        }
    }

    for f in &cache_status.files {
        if f.required && !f.present {
            push_blocker(
                &mut blockers,
                format!("Cache missing required file: {} ({})", f.label, f.path),
            );
        }
    }

    let ready_to_run_native = yaml_ok
        && blockers.is_empty();

    ValidateResponse {
        yaml_ok,
        parse_error,
        chain_ids,
        path_probes,
        proteins_missing_msa,
        preprocess: PreprocessProbe {
            manifest_present,
            manifest_ok,
            manifest_error,
            record_id,
            npz_present,
            npz_path,
        },
        cache: cache_status,
        ready_to_run_native,
        blockers,
        warnings,
    }
}

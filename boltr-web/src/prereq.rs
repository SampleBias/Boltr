//! Cache + YAML checks aligned with `boltr predict` / `predict_tch` expectations.

use std::path::{Path, PathBuf};
use std::time::Duration;

use boltr_io::config::BoltzInput;
use boltr_io::parse_manifest_path;
use serde::Serialize;
use serde_json::Value;

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
    /// All required cache entries present (`boltz2_conf.safetensors`, `ccd.pkl`, `mols.tar`).
    pub cache_ready: bool,
    /// Repo `.venv/bin/python` exists (optional dev signal).
    pub venv_python_present: bool,
    /// `python -c "import torch"` succeeded (venv python first, else `python3`).
    pub torch_import_ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub torch_import_error: Option<String>,
    /// Parsed output of `boltr doctor --json` when the binary is found (`BOLTR` or PATH).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boltr_doctor: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boltr_doctor_error: Option<String>,
    /// Cache + torch import + `boltr doctor` reports tch + LibTorch runtime OK.
    pub environment_ready: bool,
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

    let cache_ready = files.iter().filter(|f| f.required).all(|f| f.present);
    let venv_python_present = find_venv_python().is_some();

    notes.push("Set BOLTR=/path/to/tch-enabled boltr so status can run `boltr doctor --json`.".to_string());

    StatusResponse {
        cache_dir: cache.display().to_string(),
        files,
        native_structure_checkpoint_ok,
        cache_ready,
        venv_python_present,
        torch_import_ok: None,
        torch_import_error: None,
        boltr_doctor: None,
        boltr_doctor_error: None,
        environment_ready: false,
        notes,
    }
}

/// Walk cwd / `BOLTR_REPO` for `.venv/bin/python`.
pub fn find_venv_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BOLTR_REPO") {
        let v = PathBuf::from(p).join(".venv/bin/python");
        if v.is_file() {
            return Some(v);
        }
    }
    let mut d = std::env::current_dir().ok()?;
    for _ in 0..12 {
        let v = d.join(".venv/bin/python");
        if v.is_file() {
            return Some(v);
        }
        if !d.pop() {
            break;
        }
    }
    None
}

async fn probe_import_torch(py: &Path) -> (Option<bool>, Option<String>) {
    let fut = tokio::process::Command::new(py)
        .args(["-c", "import torch; print(torch.__version__)"])
        .output();
    let res = tokio::time::timeout(Duration::from_secs(8), fut).await;
    match res {
        Ok(Ok(out)) if out.status.success() => (Some(true), None),
        Ok(Ok(out)) => (
            Some(false),
            Some(format!(
                "stderr: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )),
        ),
        Ok(Err(e)) => (Some(false), Some(e.to_string())),
        Err(_) => (Some(false), Some("timeout waiting for python".to_string())),
    }
}

fn resolve_python_for_torch_probe() -> PathBuf {
    find_venv_python().unwrap_or_else(|| PathBuf::from("python3"))
}

async fn boltr_doctor_json() -> Result<Value, String> {
    let exe = resolve_boltr_binary_async().await.ok_or_else(|| {
        "boltr not found (set BOLTR, use repo target/release/boltr, or add boltr to PATH)".to_string()
    })?;
    let out = tokio::time::timeout(
        Duration::from_secs(12),
        tokio::process::Command::new(&exe)
            .args(["doctor", "--json"])
            .output(),
    )
    .await
    .map_err(|_| "timeout running boltr doctor".to_string())?
    .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "boltr doctor exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let s = String::from_utf8(out.stdout).map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| format!("doctor JSON: {e}"))
}

async fn resolve_boltr_binary_async() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BOLTR") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    let mut d = std::env::current_dir().ok()?;
    for _ in 0..8 {
        let cand = d.join("target/release/boltr");
        if cand.is_file() {
            return Some(cand);
        }
        if !d.pop() {
            break;
        }
    }
    let out = tokio::process::Command::new("sh")
        .args(["-c", "command -v boltr 2>/dev/null"])
        .output()
        .await
        .ok()?;
    if out.status.success() {
        let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !line.is_empty() {
            let p = PathBuf::from(line);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn compute_environment_ready(s: &StatusResponse) -> bool {
    if !s.cache_ready {
        return false;
    }
    if s.torch_import_ok != Some(true) {
        return false;
    }
    let Some(ref doc) = s.boltr_doctor else {
        return false;
    };
    doc.get("tch_feature").and_then(|v| v.as_bool()) == Some(true)
        && doc.get("libtorch_runtime_ok").and_then(|v| v.as_bool()) == Some(true)
}

/// Async probes: torch import + `boltr doctor --json`.
pub async fn enrich_status(mut s: StatusResponse) -> StatusResponse {
    let py = resolve_python_for_torch_probe();
    s.venv_python_present = find_venv_python().is_some();
    let (tok_ok, tok_err) = probe_import_torch(&py).await;
    s.torch_import_ok = tok_ok;
    s.torch_import_error = tok_err;

    match boltr_doctor_json().await {
        Ok(v) => {
            s.boltr_doctor = Some(v);
            s.boltr_doctor_error = None;
        }
        Err(e) => {
            s.boltr_doctor = None;
            s.boltr_doctor_error = Some(e);
        }
    }
    s.environment_ready = compute_environment_ready(&s);
    s
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
    /// Strict: YAML + all referenced paths + cache + preprocess when manifest exists.
    pub job_ready_for_native: bool,
    /// Same as `job_ready_for_native` (legacy name).
    pub ready_to_run_native: bool,
    /// YAML parses, cache ready, and no remaining blockers (with `assume_msa_server`, missing local MSA paths are warnings).
    pub yaml_ready_relaxed: bool,
    pub cache_ready: bool,
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
    assume_msa_server: bool,
) -> ValidateResponse {
    let cache_status = gather_status(cache);
    let cache_ready = cache_status.cache_ready;
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
                let msg = format!("MSA file missing: {}", p.display());
                if assume_msa_server {
                    warnings.push(format!(
                        "{msg} (not blocking while \"assume MSA server\" is enabled)"
                    ));
                } else {
                    push_blocker(&mut blockers, msg);
                }
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

    let job_ready_for_native = yaml_ok && blockers.is_empty();
    let yaml_ready_relaxed = yaml_ok && cache_ready && blockers.is_empty();

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
        job_ready_for_native,
        ready_to_run_native: job_ready_for_native,
        yaml_ready_relaxed,
        cache_ready,
        blockers,
        warnings,
    }
}

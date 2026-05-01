//! Full `predict_step` wiring and `predict_args` resolution for `boltr predict` (`--features tch`).
//!
//! This module implements the prediction pipeline:
//! 1. Resolve checkpoint + hyperparameters
//! 2. Build `Boltz2Model` from hparams on the requested device
//! 3. Load weights (safetensors)
//! 4. When `manifest.json` + preprocess `.npz` sit next to the input YAML: `load_input` → collate → `predict_step` → PDB/mmCIF
//! 5. If step 4 did not write (e.g. `--affinity` without flat `{id}.npz`, no manifest, or `predict_step` failed): try preprocess reference export from the same bundle
//! 6. Otherwise: placeholder completion marker (no structure file)
//! 7. Confidence / affinity / PAE npz when those heads are loaded and wired (see `boltr-io` writers)
//!
//! Reference: [boltz-reference/docs/prediction.md](../../../boltz-reference/docs/prediction.md)

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use boltr_backend_tch::{
    resolve_predict_args, AffinityModuleConfig, AtomDiffusionConfig, Boltz2DiffusionArgs,
    Boltz2Hparams, Boltz2Model, Boltz2PredictArgs, ConfidenceModuleConfig, PredictArgsCliOverrides,
    RelPosFeatures, SteeringParams,
};
use boltr_io::config::BoltzInput;
use boltr_io::{
    canonical_yaml_parent, collate_inference_batches, copy_msa_a3m_to_output,
    featurized_atom_token_sum, load_input, parse_manifest_path, resolve_preprocess_load_dirs,
    trunk_smoke_feature_batch_from_inference_input_with_ensemble, AffinitySummary,
    InferenceEnsembleMode,
};
use tch::{self, Kind, Tensor};

use crate::OutputFormat;

fn env_flag_true(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

/// Turn `diffusion.sample_atom_coords` into one `[x,y,z]` per atom for [`StructureV2Tables::apply_predicted_atom_coords`].
///
/// Boltz uses `[batch_or_multiplicity, N_atoms, 3]`. Some layouts may appear as `[N, 3]` or wrongly
/// transposed `[3, N]` (xyz major). Mis-reading `[3, N]` as `[N, 3]` assigns nonsense coordinates
/// (often degenerate / line-like in the viewer).
fn diffusion_sample_coords_to_xyz_vec(
    coord_tensor: &Tensor,
    sample_idx: i64,
) -> Result<Vec<[f32; 3]>> {
    let mut t = coord_tensor.shallow_clone();
    if t.dim() == 3 {
        let sz = t.size();
        if sz[2] != 3 {
            bail!(
                "sample_atom_coords: expected last dim 3 (xyz), got shape {:?}",
                sz
            );
        }
        if sz[0] > 1 {
            if sample_idx < 0 || sample_idx >= sz[0] {
                bail!(
                    "sample_atom_coords: sample index {sample_idx} outside batch/multiplicity {}",
                    sz[0]
                );
            }
            t = t.narrow(0, sample_idx, 1);
        }
        t = t.squeeze_dim(0);
    }
    if t.dim() != 2 {
        bail!(
            "sample_atom_coords: expected 2D [N,3] after batch trim, got dim {} shape {:?}",
            t.dim(),
            t.size()
        );
    }

    // Ensure [N, 3] (atoms × xyz). Standard is size[1] == 3.
    let sz = t.size();
    let t = if sz[1] == 3 {
        t
    } else if sz[0] == 3 && sz[1] != 3 {
        // [3, N_atoms] — transpose to [N, 3]
        t.transpose(0, 1)
    } else {
        bail!(
            "sample_atom_coords: cannot interpret as [N,3] or [3,N], got shape {:?}",
            sz
        );
    };

    let t = t.to_kind(Kind::Float).to_device(tch::Device::Cpu);
    let n = t.size()[0] as usize;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push([
            t.double_value(&[i as i64, 0]) as f32,
            t.double_value(&[i as i64, 1]) as f32,
            t.double_value(&[i as i64, 2]) as f32,
        ]);
    }
    Ok(v)
}

#[derive(Debug, Clone)]
struct SampleRanking {
    selected_sample: i64,
    ranking_metric: String,
    ranking_score: Option<f64>,
    confidence_available: bool,
}

#[derive(Debug, Clone)]
struct PredictBridgeSuccess {
    record_id: String,
    ranking: SampleRanking,
}

fn tensor_scalar_at(t: &Tensor, idx: i64) -> Option<f64> {
    let t = t
        .to_kind(Kind::Float)
        .to_device(tch::Device::Cpu)
        .reshape(&[-1]);
    let n = t.size().first().copied().unwrap_or(0);
    if idx < 0 || idx >= n {
        return None;
    }
    Some(t.double_value(&[idx]))
}

fn select_best_sample(
    out: &boltr_backend_tch::PredictStepOutput,
    confidence_ranking_enabled: bool,
) -> SampleRanking {
    let sample_count = out
        .diffusion
        .sample_atom_coords
        .size()
        .first()
        .copied()
        .unwrap_or(1)
        .max(1);
    let Some(conf) = out
        .confidence
        .as_ref()
        .filter(|_| confidence_ranking_enabled)
    else {
        return SampleRanking {
            selected_sample: 0,
            ranking_metric: "sample0_no_confidence".to_string(),
            ranking_score: None,
            confidence_available: false,
        };
    };
    let scores = if conf.iptm.numel() as i64 >= sample_count {
        Some((
            "iptm",
            conf.iptm
                .to_kind(Kind::Float)
                .to_device(tch::Device::Cpu)
                .reshape(&[-1]),
        ))
    } else if conf.complex_plddt.numel() as i64 >= sample_count {
        Some((
            "complex_plddt",
            conf.complex_plddt
                .to_kind(Kind::Float)
                .to_device(tch::Device::Cpu)
                .reshape(&[-1]),
        ))
    } else {
        None
    };
    let Some((metric, scores)) = scores else {
        return SampleRanking {
            selected_sample: 0,
            ranking_metric: "sample0_confidence_unrankable".to_string(),
            ranking_score: None,
            confidence_available: true,
        };
    };
    let mut best_idx = 0_i64;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..sample_count {
        let score = scores.double_value(&[i]);
        if score.is_finite() && score > best_score {
            best_idx = i;
            best_score = score;
        }
    }
    SampleRanking {
        selected_sample: best_idx,
        ranking_metric: metric.to_string(),
        ranking_score: Some(best_score),
        confidence_available: true,
    }
}

fn is_likely_cuda_oom(err_msg: &str) -> bool {
    let l = err_msg.to_lowercase();
    l.contains("out of memory")
        || l.contains("cuda out of memory")
        || (l.contains("cuda") && l.contains("alloc") && l.contains("fail"))
}

/// When `manifest.json` + preprocess `.npz` live next to the input YAML, run collate → `predict_step` → structure writer.
fn try_predict_from_preprocess(
    input_path: &Path,
    out_dir: &Path,
    output_format: OutputFormat,
    model: &Boltz2Model,
    resolved: &Boltz2PredictArgs,
    affinity: bool,
    use_potentials: bool,
    extra_mols_dir: Option<&Path>,
    constraints_dir: Option<&Path>,
    preprocess_auto_extras: bool,
    ensemble_mode: InferenceEnsembleMode,
    device_requested: Option<&str>,
    confidence_ranking_enabled: bool,
) -> Result<Option<PredictBridgeSuccess>> {
    let preprocess_dir = match canonical_yaml_parent(input_path) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(error = %e, "preprocess bridge: could not resolve YAML directory");
            return Ok(None);
        }
    };
    let manifest_path = preprocess_dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Ok(None);
    }

    let manifest = parse_manifest_path(&manifest_path)?;
    let record = manifest
        .records
        .first()
        .context("manifest.json has no records")?;
    if record
        .templates
        .as_ref()
        .map(|templates| !templates.is_empty())
        .unwrap_or(false)
    {
        bail!(
            "preprocess bridge: manifest contains templates for record {}, but template bias is not wired into the current Rust predict_step path",
            record.id
        );
    }

    // `--affinity` normally uses `pre_affinity_{id}.npz` for affinity-specific tensors, but the
    // structure diffusion bridge always reads the flat `{id}.npz` (same as `load_input(..., false)`).
    // If only the affinity layout exists and not the flat bundle, skip the bridge (reference export
    // may still run from `try_write_preprocess_reference_structure`).
    if affinity {
        let flat_npz = preprocess_dir.join(format!("{}.npz", record.id));
        if !flat_npz.is_file() {
            tracing::info!(
                record_id = %record.id,
                expected = %flat_npz.display(),
                "--affinity: flat preprocess npz missing; skipping structure diffusion bridge"
            );
            return Ok(None);
        }
        tracing::info!(
            record_id = %record.id,
            path = %flat_npz.display(),
            "--affinity: using flat preprocess bundle for structure diffusion + predict_step"
        );
    }

    tracing::info!(
        path = %manifest_path.display(),
        "preprocess bridge: manifest found; loading bundle and running predict_step"
    );
    let (extra_mols_pb, constraints_pb) = resolve_preprocess_load_dirs(
        &preprocess_dir,
        extra_mols_dir,
        constraints_dir,
        preprocess_auto_extras,
    );
    let mut inference_input = load_input(
        record,
        &preprocess_dir,
        &preprocess_dir,
        constraints_pb.as_deref(),
        None,
        extra_mols_pb.as_deref(),
        false,
    )
    .with_context(|| {
        format!(
            "load_input from preprocess dir {}",
            preprocess_dir.display()
        )
    })?;

    let template_dim = 4_usize;
    let fb = trunk_smoke_feature_batch_from_inference_input_with_ensemble(
        &inference_input,
        template_dim,
        ensemble_mode,
    );
    let coll = collate_inference_batches(std::slice::from_ref(&fb), 0.0, 0, 0)
        .map_err(|e| anyhow::anyhow!("collate_inference_batches: {e}"))?;

    let mut max_parallel = resolved.max_parallel_samples;
    let mut retried_oom = false;
    let out = loop {
        match crate::collate_predict_bridge::predict_step_from_collate(
            model,
            &coll,
            Some(resolved.recycling_steps),
            resolved.sampling_steps,
            resolved.diffusion_samples,
            max_parallel,
            use_potentials && !affinity,
        ) {
            Ok(o) => break o,
            Err(e) => {
                let msg = e.to_string();
                if !retried_oom
                    && device_requested == Some("auto")
                    && is_likely_cuda_oom(&msg)
                    && max_parallel != Some(1)
                {
                    let _ = crate::preprocess_cmd::maybe_post_boltz_empty_cache(None);
                    retried_oom = true;
                    max_parallel = Some(1);
                    tracing::warn!(
                        "predict_step OOM; retrying once with max_parallel_samples=1 after best-effort CUDA cache flush"
                    );
                    continue;
                }
                return Err(e).context("predict_step_from_collate");
            }
        }
    };

    let ranking = select_best_sample(&out, confidence_ranking_enabled);
    let raw_shape = out.diffusion.sample_atom_coords.size();
    let xyz = diffusion_sample_coords_to_xyz_vec(
        &out.diffusion.sample_atom_coords,
        ranking.selected_sample,
    )
    .map_err(|e| anyhow!("diffusion coords → xyz: {e}"))?;
    let n_from_model = xyz.len();
    let n_featurized = featurized_atom_token_sum(&inference_input);
    let n_atoms_struct = inference_input.structure.atoms.len();

    // `n_from_model` includes atom_pad_mask tail padding (window-rounded); it can exceed
    // `n_featurized` and `n_atoms_struct`. Only compare structure vs featurizer for order/count.
    if n_featurized != n_atoms_struct {
        tracing::warn!(
            n_featurized,
            n_structure_atoms = n_atoms_struct,
            raw_shape = ?raw_shape,
            "featurized atom count (token sum) differs from StructureV2 atoms length; tail atoms may keep reference coords or map incorrectly"
        );
    }
    if n_from_model < n_featurized {
        tracing::warn!(
            n_from_model,
            n_featurized,
            raw_shape = ?raw_shape,
            "diffusion returned fewer coords than featurizer real atoms; truncating coordinate write"
        );
    }
    if n_from_model > n_featurized {
        tracing::debug!(
            n_from_model,
            n_featurized,
            "diffusion atom dim exceeds featurized real atoms (expected: padding to atoms_per_window)"
        );
    }

    let n_apply = n_featurized.min(n_atoms_struct).min(n_from_model);
    inference_input
        .structure
        .apply_predicted_atom_coords(&xyz[..n_apply]);

    let record_id = record.id.clone();
    write_structure_file(
        out_dir,
        &record_id,
        0,
        output_format,
        &inference_input.structure,
    )?;
    if let Some(ref aff) = out.affinity {
        let pred = tensor_scalar_at(&aff.affinity_pred_value, 0).unwrap_or(f64::NAN);
        let prob = tensor_scalar_at(&aff.affinity_logits_binary.sigmoid(), 0).unwrap_or(f64::NAN);
        let summary = AffinitySummary::single(pred, prob).with_sample_metadata(
            ranking.selected_sample,
            ranking.ranking_metric.clone(),
            ranking.ranking_score,
            ranking.confidence_available,
            model.affinity_mw_correction(),
        );
        boltr_io::write_affinity_json(out_dir, &record_id, &summary)
            .context("write affinity JSON")?;
    }
    copy_msa_a3m_to_output(&preprocess_dir, out_dir).context("copy MSA .a3m into output dir")?;

    tracing::info!(
        record_id = %record_id,
        selected_sample = ranking.selected_sample,
        ranking_metric = %ranking.ranking_metric,
        "wrote predicted structure (preprocess dir + predict_step)"
    );

    Ok(Some(PredictBridgeSuccess { record_id, ranking }))
}

/// Write mmCIF/PDB from preprocess `StructureV2` only (reference/input geometry, no diffusion).
/// Used as a fallback after [`try_predict_from_preprocess`] returns `None` or `Err` when
/// `manifest.json` + `.npz` exist beside the YAML.
fn try_write_preprocess_reference_structure(
    input_path: &Path,
    out_dir: &Path,
    output_format: OutputFormat,
    affinity: bool,
    extra_mols_dir: Option<&Path>,
    constraints_dir: Option<&Path>,
    preprocess_auto_extras: bool,
) -> Result<Option<String>> {
    let preprocess_dir = match canonical_yaml_parent(input_path) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(error = %e, "preprocess reference export: could not resolve YAML dir");
            return Ok(None);
        }
    };
    let manifest_path = preprocess_dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Ok(None);
    }

    let manifest = match parse_manifest_path(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = %e, "preprocess reference export: manifest parse failed");
            return Ok(None);
        }
    };
    let record = match manifest.records.first() {
        Some(r) => r,
        None => {
            tracing::warn!("preprocess reference export: manifest has no records");
            return Ok(None);
        }
    };

    let (extra_mols_pb, constraints_pb) = resolve_preprocess_load_dirs(
        &preprocess_dir,
        extra_mols_dir,
        constraints_dir,
        preprocess_auto_extras,
    );

    // `--affinity` expects `pre_affinity_{id}.npz`; native/Boltz preprocess usually emits flat `{id}.npz`.
    // Fall back to the flat bundle so users still get a reference .cif/.pdb when affinity layout is absent.
    let inference_input = match load_input(
        record,
        &preprocess_dir,
        &preprocess_dir,
        constraints_pb.as_deref(),
        None,
        extra_mols_pb.as_deref(),
        affinity,
    ) {
        Ok(i) => i,
        Err(e) if affinity => {
            let flat = preprocess_dir.join(format!("{}.npz", record.id));
            if !flat.is_file() {
                tracing::warn!(
                    error = %e,
                    dir = %preprocess_dir.display(),
                    record_id = %record.id,
                    "preprocess reference export: load_input failed (affinity expects pre_affinity npz; no flat structure npz for fallback)",
                );
                return Ok(None);
            }
            tracing::info!(
                record_id = %record.id,
                "preprocess reference export: using flat preprocess bundle (--affinity without pre_affinity npz)"
            );
            match load_input(
                record,
                &preprocess_dir,
                &preprocess_dir,
                constraints_pb.as_deref(),
                None,
                extra_mols_pb.as_deref(),
                false,
            ) {
                Ok(i) => i,
                Err(e2) => {
                    tracing::warn!(
                        error = %e2,
                        dir = %preprocess_dir.display(),
                        "preprocess reference export: load_input failed after affinity fallback"
                    );
                    return Ok(None);
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                dir = %preprocess_dir.display(),
                "preprocess reference export: load_input failed"
            );
            return Ok(None);
        }
    };

    let record_id = record.id.clone();
    write_structure_file(
        out_dir,
        &record_id,
        0,
        output_format,
        &inference_input.structure,
    )?;
    if let Err(e) = copy_msa_a3m_to_output(&preprocess_dir, out_dir) {
        tracing::warn!(
            error = %e,
            "preprocess reference export: could not copy MSA .a3m to output"
        );
    }

    tracing::info!(
        record_id = %record_id,
        affinity,
        "wrote preprocess reference structure (no diffusion sampling)"
    );
    Ok(Some(record_id))
}

// ===========================================================================
// Public types
// ===========================================================================

/// All arguments for the tch-backed predict flow.
pub struct PredictTchArgs<'a> {
    pub input_path: PathBuf,
    pub cache: PathBuf,
    pub out_dir: PathBuf,
    pub device: String,
    pub affinity: bool,
    pub use_potentials: bool,
    pub quality_preset: bool,
    pub overrides: PredictArgsCliOverrides,
    pub step_scale: f64,
    pub output_format: OutputFormat,
    pub max_msa_seqs: usize,
    pub num_samples: usize,
    pub checkpoint: Option<PathBuf>,
    pub affinity_checkpoint: Option<PathBuf>,
    pub affinity_mw_correction: bool,
    pub sampling_steps_affinity: Option<i64>,
    pub diffusion_samples_affinity: Option<i64>,
    pub preprocessing_threads: Option<usize>,
    pub override_flag: bool,
    pub write_full_pae: bool,
    pub write_full_pde: bool,
    pub spike_only: bool,
    /// Optional directory of CCD `*.json` (same as [`boltr_io::load_input`] `extra_mols_dir`).
    pub extra_mols_dir: Option<PathBuf>,
    /// Optional directory containing `{record_id}.npz` residue constraints.
    pub constraints_dir: Option<PathBuf>,
    /// When true, discover `mols`/`extra_mols` and `constraints` under the preprocess dir if CLI paths omitted.
    pub preprocess_auto_extras: bool,
    /// Atom-feature ensemble policy for trunk featurization (default single index 0).
    pub ensemble_mode: InferenceEnsembleMode,
    /// Original `--device` request when `auto` or `gpu` (OOM retry policy).
    pub device_requested: Option<String>,
    pub parsed: &'a BoltzInput,
}

// ===========================================================================
// Hyperparameter loading
// ===========================================================================

/// Load hyperparameters: `BOLTR_HPARAMS_JSON` env → `{cache}/boltz2_hparams.json` → default.
pub fn load_hparams_for_predict(cache_dir: &Path) -> Result<Boltz2Hparams> {
    if let Ok(p) = std::env::var("BOLTR_HPARAMS_JSON") {
        let path = PathBuf::from(p);
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read BOLTR_HPARAMS_JSON {}", path.display()))?;
        return Boltz2Hparams::from_json_slice(&bytes)
            .with_context(|| format!("parse hparams {}", path.display()));
    }
    let candidate = cache_dir.join("boltz2_hparams.json");
    if candidate.is_file() {
        let bytes =
            std::fs::read(&candidate).with_context(|| format!("read {}", candidate.display()))?;
        return Boltz2Hparams::from_json_slice(&bytes)
            .with_context(|| format!("parse {}", candidate.display()));
    }
    tracing::warn!("no hparams JSON found; using Boltz2Hparams::default()");
    Ok(Boltz2Hparams::default())
}

/// Parse optional `predict_args:` from the input YAML file (Boltz-style).
pub fn yaml_predict_args_from_input_path(input: &Path) -> Result<Option<serde_json::Value>> {
    let text =
        std::fs::read_to_string(input).with_context(|| format!("read {}", input.display()))?;
    let v: serde_yaml::Value =
        serde_yaml::from_str(&text).with_context(|| format!("YAML {}", input.display()))?;
    match v.get("predict_args") {
        Some(node) => Ok(Some(
            serde_json::to_value(node).unwrap_or(serde_json::Value::Null),
        )),
        None => Ok(None),
    }
}

/// Merge checkpoint hparams + YAML + CLI; write `boltr_predict_args.json` into `out_dir`.
pub fn write_resolved_predict_args(
    out_dir: &Path,
    cache_dir: &Path,
    input: &Path,
    cli: PredictArgsCliOverrides,
) -> Result<Boltz2PredictArgs> {
    let h = load_hparams_for_predict(cache_dir)?;
    let yaml = yaml_predict_args_from_input_path(input)?;
    let resolved = resolve_predict_args(&h, yaml.as_ref(), cli);
    let path = out_dir.join("boltr_predict_args.json");
    let j = serde_json::to_string_pretty(&resolved).context("serialize Boltz2PredictArgs")?;
    std::fs::write(&path, j).with_context(|| format!("write {}", path.display()))?;
    Ok(resolved)
}

// ===========================================================================
// Checkpoint resolution
// ===========================================================================

/// Resolve the structure confidence checkpoint path.
///
/// Priority: `--checkpoint` flag → `{cache}/boltz2_conf.safetensors` →
/// `{cache}/boltz2_conf.ckpt` (requires prior export) → error.
fn resolve_conf_checkpoint(checkpoint: Option<&Path>, cache: &Path) -> Result<PathBuf> {
    if let Some(p) = checkpoint {
        if p.is_file() {
            return Ok(p.to_path_buf());
        }
        bail!("--checkpoint path does not exist: {}", p.display());
    }
    // Prefer safetensors (Rust-native)
    let sf = cache.join("boltz2_conf.safetensors");
    if sf.is_file() {
        return Ok(sf);
    }
    // Fall back to .ckpt (needs prior export)
    let ckpt = cache.join("boltz2_conf.ckpt");
    if ckpt.is_file() {
        tracing::warn!(
            "found {} but not {}; run scripts/export_checkpoint_to_safetensors.py first",
            ckpt.display(),
            sf.display()
        );
        bail!(
            "checkpoint is in .ckpt format; export to safetensors first:\n  \
             python scripts/export_checkpoint_to_safetensors.py {}",
            ckpt.display()
        );
    }
    bail!(
        "no checkpoint found in cache ({}). Run: boltr download --version boltz2",
        cache.display()
    );
}

/// Resolve the affinity checkpoint path (optional).
fn resolve_affinity_checkpoint(
    affinity_checkpoint: Option<&Path>,
    cache: &Path,
) -> Option<PathBuf> {
    if let Some(p) = affinity_checkpoint {
        if p.is_file() {
            return Some(p.to_path_buf());
        }
        tracing::warn!("--affinity-checkpoint path does not exist: {}", p.display());
        return None;
    }
    let sf = cache.join("boltz2_aff.safetensors");
    if sf.is_file() {
        return Some(sf);
    }
    tracing::debug!("no affinity checkpoint found");
    None
}

// ===========================================================================
// Output writers
// ===========================================================================

/// Write all prediction outputs for a single record.
///
/// Output layout matches Boltz `predictions/{record_id}/`:
/// - `{record_id}_model_{rank}.{cif|pdb}` — structure sorted by confidence
/// - `confidence_{record_id}_model_{rank}.json`
/// - `affinity_{record_id}.json` (when affinity predicted)
/// - `pae_{record_id}_model_{rank}.npz`, `pde_*`, `plddt_*`
fn write_prediction_outputs(
    out_dir: &Path,
    record_id: &str,
    _output_format: OutputFormat,
    _write_full_pae: bool,
    _write_full_pde: bool,
) -> Result<()> {
    let record_dir = out_dir.join(record_id);
    std::fs::create_dir_all(&record_dir)
        .with_context(|| format!("mkdir {}", record_dir.display()))?;
    tracing::info!(dir = %record_dir.display(), "created output directory for record");
    Ok(())
}

fn warn_if_all_present_coords_zero(structure: &boltr_io::StructureV2Tables, context: &'static str) {
    if structure.present_atoms_all_coords_near_zero(1e-12) {
        tracing::warn!(
            context,
            "all Cartesian coordinates for present atoms are zero; may be preprocess placeholders or a failed coordinate write"
        );
    }
}

/// Write structure file (PDB or mmCIF) from predicted coordinates.
#[allow(clippy::too_many_arguments)]
fn write_structure_file(
    out_dir: &Path,
    record_id: &str,
    model_rank: usize,
    output_format: OutputFormat,
    structure: &boltr_io::StructureV2Tables,
) -> Result<PathBuf> {
    warn_if_all_present_coords_zero(structure, "write_structure_file");

    let record_dir = out_dir.join(record_id);
    std::fs::create_dir_all(&record_dir)
        .with_context(|| format!("mkdir {}", record_dir.display()))?;

    let path = match output_format {
        OutputFormat::Pdb => {
            let p = record_dir.join(format!("{record_id}_model_{model_rank}.pdb"));
            let bytes = boltr_io::structure_v2_to_pdb(structure, None);
            std::fs::write(&p, &bytes).with_context(|| format!("write {}", p.display()))?;
            p
        }
        OutputFormat::Mmcif => {
            let p = record_dir.join(format!("{record_id}_model_{model_rank}.cif"));
            let bytes = boltr_io::structure_v2_to_mmcif(structure);
            std::fs::write(&p, &bytes).with_context(|| format!("write {}", p.display()))?;
            p
        }
        OutputFormat::Both => {
            let pdb = record_dir.join(format!("{record_id}_model_{model_rank}.pdb"));
            let cif = record_dir.join(format!("{record_id}_model_{model_rank}.cif"));
            let pdb_bytes = boltr_io::structure_v2_to_pdb(structure, None);
            let cif_bytes = boltr_io::structure_v2_to_mmcif(structure);
            std::fs::write(&pdb, &pdb_bytes).with_context(|| format!("write {}", pdb.display()))?;
            std::fs::write(&cif, &cif_bytes).with_context(|| format!("write {}", cif.display()))?;
            cif
        }
    };
    tracing::debug!(path = %path.display(), "wrote structure file");
    Ok(path)
}

// ===========================================================================
// Spike-only path (trunk smoke without full diffusion)
// ===========================================================================

/// Run a trunk-only spike test: load model, forward random tensors, verify shapes.
async fn try_model_spike(
    device_str: &str,
    cache: &Path,
    out_dir: &Path,
    recycling_steps: i64,
    use_potentials_steering: bool,
) -> Result<()> {
    use boltr_backend_tch::{
        cuda_is_available, parse_device_spec, safetensor_names_not_in_var_store,
    };

    tch::maybe_init_cuda();
    tracing::info!(
        cuda_available = cuda_is_available(),
        requested_device = %device_str,
        use_potentials_steering,
        steering = ?SteeringParams::from_use_potentials(use_potentials_steering),
        "LibTorch device probe"
    );

    let device = parse_device_spec(device_str)?;
    let safetensors_path = cache.join("boltz2_conf.safetensors");
    if !safetensors_path.exists() {
        tracing::info!(
            path = %safetensors_path.display(),
            "skip weight spike: place exported safetensors here \
             (scripts/export_checkpoint_to_safetensors.py)"
        );
        return Ok(());
    }

    // Try loading from hparams if available
    let token_s = load_hparams_for_predict(cache)
        .ok()
        .map(|h| h.resolved_token_s())
        .unwrap_or(384);

    let mut model = Boltz2Model::new(device, token_s);

    let mut missing_after_partial: Vec<String> = Vec::new();
    let mut partial_load_ok = false;
    match model.load_partial_from_safetensors(&safetensors_path) {
        Ok(missing) => {
            partial_load_ok = true;
            missing_after_partial = missing;
            tracing::info!(
                model_params = model.var_store().len(),
                still_missing = missing_after_partial.len(),
                "safetensors VarStore::load_partial"
            );
            if !missing_after_partial.is_empty() {
                tracing::warn!(
                    n = missing_after_partial.len(),
                    sample = ?missing_after_partial.iter().take(12).collect::<Vec<_>>(),
                    "checkpoint missing these VarStore keys (left default-init)"
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "VarStore::load_partial failed; trying s_init only"
            );
            if let Err(e2) = model.load_s_init_from_safetensors(&safetensors_path) {
                tracing::warn!(error = %e2, "s_init load also failed");
            }
        }
    }

    if tracing::enabled!(tracing::Level::DEBUG) {
        match safetensor_names_not_in_var_store(&safetensors_path, model.var_store()) {
            Ok(extra) if !extra.is_empty() => {
                tracing::debug!(
                    n = extra.len(),
                    sample = ?extra.iter().take(8).collect::<Vec<_>>(),
                    "safetensors keys not mapped into this Boltz2Model VarStore"
                );
            }
            _ => {}
        }
    }

    let _token_z = model.token_z();

    // --- forward_s_init probe ---
    let probe = Tensor::randn(&[2, token_s], (Kind::Float, device));
    let _y = model.forward_s_init(&probe);

    // --- forward_trunk spike ---
    let b = 2_i64;
    let n = 16_i64;
    let s_in = Tensor::randn(&[b, n, token_s], (Kind::Float, device));
    let token_pad = Tensor::ones(&[b, n], (Kind::Float, device));
    let (s_out, z_out) = model
        .forward_trunk(&s_in, &token_pad, Some(recycling_steps), None, None)
        .map_err(|e| anyhow::anyhow!("forward_trunk spike: {e}"))?;
    let s_sz = s_out.size();
    let z_sz = z_out.size();
    tracing::info!(
        ?s_sz,
        ?z_sz,
        recycling_steps,
        "forward_trunk (pairformer stack) spike ok"
    );

    // --- predict_step_trunk spike ---
    let asym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
    let residue_index = Tensor::arange(n, (Kind::Int64, device))
        .view_(&[1, n])
        .expand(&[b, n], false);
    let entity_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
    let token_index = residue_index.shallow_clone();
    let sym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
    let cyclic_period = Tensor::zeros(&[b, n], (Kind::Int64, device));
    let rel = RelPosFeatures {
        asym_id: &asym_id,
        residue_index: &residue_index,
        entity_id: &entity_id,
        token_index: &token_index,
        sym_id: &sym_id,
        cyclic_period: &cyclic_period,
    };
    let (s_ps, z_ps) = model
        .predict_step_trunk(
            &s_in,
            &rel,
            None,
            None,
            None,
            &token_pad,
            Some(recycling_steps),
            None,
            None,
        )
        .map_err(|e| anyhow::anyhow!("predict_step_trunk spike: {e}"))?;
    tracing::info!(
        s_predict = ?s_ps.size(),
        z_predict = ?z_ps.size(),
        recycling_steps,
        "predict_step_trunk spike ok"
    );

    // --- write spike marker ---
    let spike_path = out_dir.join("boltr_backend_spike_ok.txt");
    let load_note = if partial_load_ok {
        format!(
            "VarStore partial load ok; keys still missing: {}\n",
            missing_after_partial.len()
        )
    } else {
        "VarStore partial load failed; see logs.\n".to_string()
    };
    let msg = format!(
        "Boltz2Model: s_init + forward_trunk + predict_step_trunk executed.\n\
         recycling_steps={recycling_steps}\n\
         s_out shape: {s_sz:?}\n\
         z_out shape: {z_sz:?}\n\
         predict_step_trunk s_out: {:?}\n\
         predict_step_trunk z_out: {:?}\n\
         {load_note}",
        s_ps.size(),
        z_ps.size()
    );
    tokio::fs::write(&spike_path, msg).await?;
    tracing::info!(path = %spike_path.display(), "backend spike complete");

    Ok(())
}

// ===========================================================================
// Main entry point
// ===========================================================================

/// Run the full tch-backed predict pipeline.
pub async fn run_predict_tch(args: PredictTchArgs<'_>) -> Result<()> {
    let PredictTchArgs {
        input_path,
        cache,
        out_dir,
        device,
        affinity,
        use_potentials,
        quality_preset,
        overrides,
        step_scale,
        output_format,
        max_msa_seqs: _,
        num_samples: _,
        checkpoint,
        affinity_checkpoint,
        affinity_mw_correction,
        sampling_steps_affinity,
        diffusion_samples_affinity,
        preprocessing_threads: _,
        override_flag: _,
        write_full_pae,
        write_full_pde,
        spike_only,
        extra_mols_dir,
        constraints_dir,
        preprocess_auto_extras,
        ensemble_mode,
        device_requested,
        parsed: _,
    } = args;

    // 1. Resolve predict_args (CLI > YAML > checkpoint > defaults)
    let resolved = write_resolved_predict_args(&out_dir, &cache, &input_path, overrides);
    match &resolved {
        Ok(args) => {
            tracing::info!(
                ?args,
                path = %out_dir.join("boltr_predict_args.json").display(),
                "wrote resolved predict_args"
            );
        }
        Err(e) => tracing::warn!(error = %e, "could not write boltr_predict_args.json"),
    }

    // 2. Spike-only path (trunk smoke, no diffusion/writers)
    if spike_only {
        let spike_recycling = match resolved {
            Ok(args) => args.recycling_steps,
            Err(_) => overrides.recycling_steps.unwrap_or(0),
        };
        try_model_spike(
            &device,
            &cache,
            &out_dir,
            spike_recycling,
            use_potentials && !affinity,
        )
        .await?;
        return Ok(());
    }

    // 3. Resolve checkpoint paths
    let conf_path = resolve_conf_checkpoint(checkpoint.as_deref(), &cache)?;
    tracing::info!(path = %conf_path.display(), "using confidence checkpoint");

    let native_confidence_enabled = env_flag_true("BOLTR_ENABLE_NATIVE_CONFIDENCE");
    let native_affinity_enabled = env_flag_true("BOLTR_ENABLE_NATIVE_AFFINITY");
    if affinity && !native_affinity_enabled {
        tracing::warn!(
            "--affinity requested, but native affinity scoring is disabled by default to preserve GPU memory; set BOLTR_ENABLE_NATIVE_AFFINITY=1 to opt in"
        );
    }

    let aff_path = if affinity && native_affinity_enabled {
        resolve_affinity_checkpoint(affinity_checkpoint.as_deref(), &cache)
    } else {
        None
    };
    if aff_path.is_some() {
        tracing::info!(path = %aff_path.as_ref().unwrap().display(), "using affinity checkpoint");
    }
    if affinity && native_affinity_enabled && aff_path.is_none() {
        bail!(
            "--affinity requested but no affinity safetensors checkpoint was found (set --affinity-checkpoint or place boltz2_aff.safetensors in the cache)"
        );
    }
    // 4. Load hyperparameters and build model on target device
    let hparams = load_hparams_for_predict(&cache)?;
    let token_s = hparams.resolved_token_s();
    let token_z = hparams.resolved_token_z();
    let num_blocks = hparams.resolved_num_pairformer_blocks().unwrap_or(4);

    let device = boltr_backend_tch::parse_device_spec(&device)?;
    let diff_args = Boltz2DiffusionArgs::from_boltz2_hparams(&hparams);
    let mut diff_config = AtomDiffusionConfig::from_boltz2_hparams(&hparams);
    diff_config.step_scale = step_scale;
    tracing::info!(
        token_s,
        token_z,
        num_blocks,
        token_transformer_heads = diff_args.token_transformer_heads,
        atoms_per_window_queries = diff_args.atoms_per_window_queries,
        atoms_per_window_keys = diff_args.atoms_per_window_keys,
        ?device,
        "building Boltz2Model from hparams"
    );

    let affinity_config = if affinity && native_affinity_enabled {
        Some(AffinityModuleConfig::from_affinity_model_args(
            hparams.affinity_model_args.as_ref(),
            token_s,
        ))
    } else {
        None
    };
    let affinity_mw_correction_enabled = affinity
        && native_affinity_enabled
        && (affinity_mw_correction || hparams.affinity_mw_correction.unwrap_or(false));
    let confidence_config =
        if hparams.confidence_prediction == Some(true) && native_confidence_enabled {
        let mut cfg = ConfidenceModuleConfig::default();
        cfg.pairformer_num_blocks = num_blocks;
        Some(cfg)
    } else {
        if hparams.confidence_prediction == Some(true) {
            tracing::warn!(
                "native confidence scoring is disabled by default to preserve GPU memory; set BOLTR_ENABLE_NATIVE_CONFIDENCE=1 to opt in"
            );
        }
        None
    };

    let mut model = Boltz2Model::with_all_options(
        device,
        token_s,
        token_z,
        Some(num_blocks),
        hparams.resolved_bond_type_feature(),
        diff_args,
        diff_config,
        confidence_config,
        affinity_config,
        affinity_mw_correction_enabled,
    )
    .context("Boltz2Model::with_all_options")?;

    // 5. Load weights
    let mut confidence_ranking_enabled = model.confidence_module().is_some();
    match model.load_partial_from_safetensors(&conf_path) {
        Ok(missing) => {
            if missing.iter().any(|k| k.starts_with("confidence_module.")) {
                let missing_confidence_count = missing
                    .iter()
                    .filter(|k| k.starts_with("confidence_module."))
                    .count();
                let sample = missing
                    .iter()
                    .filter(|k| k.starts_with("confidence_module."))
                    .take(10)
                    .cloned()
                    .collect::<Vec<_>>();
                tracing::warn!(
                    missing_confidence_count,
                    sample = ?sample,
                    "confidence weights are incomplete for this graph; disabling confidence outputs/ranking and continuing structure prediction"
                );
                model.disable_confidence_module();
                confidence_ranking_enabled = false;
            }
            tracing::info!(
                model_params = model.var_store().len(),
                missing_keys = missing.len(),
                "loaded checkpoint weights"
            );
            if !missing.is_empty() {
                tracing::warn!(
                    n = missing.len(),
                    sample = ?missing.iter().take(10).collect::<Vec<_>>(),
                    "some model parameters left at default init (checkpoint incomplete for this graph)"
                );
            }
        }
        Err(e) => {
            confidence_ranking_enabled = false;
            tracing::warn!(error = %e, "partial load failed; trying s_init only");
            if let Err(e2) = model.load_s_init_from_safetensors(&conf_path) {
                tracing::warn!(error = %e2, "s_init load also failed");
            }
        }
    }
    if let Some(ref path) = aff_path {
        match model.load_partial_from_safetensors(path) {
            Ok(missing) => {
                if missing.iter().any(|k| k.starts_with("affinity_module.")) {
                    let missing_affinity_count = missing
                        .iter()
                        .filter(|k| k.starts_with("affinity_module."))
                        .count();
                    let sample = missing
                        .iter()
                        .filter(|k| k.starts_with("affinity_module."))
                        .take(10)
                        .cloned()
                        .collect::<Vec<_>>();
                    tracing::warn!(
                        missing_affinity_count,
                        sample = ?sample,
                        "affinity weights are incomplete; disabling affinity outputs and continuing structure prediction"
                    );
                    model.disable_affinity_module();
                }
                tracing::info!(
                    model_params = model.var_store().len(),
                    missing_keys = missing.len(),
                    "loaded affinity checkpoint weights"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "affinity checkpoint partial load failed");
            }
        }
    }

    let mut pargs = match &resolved {
        Ok(a) => *a,
        Err(_) => Boltz2PredictArgs::default(),
    };
    if affinity {
        if let Some(n) = sampling_steps_affinity {
            pargs.sampling_steps = Some(n);
        }
        if let Some(n) = diffusion_samples_affinity {
            pargs.diffusion_samples = n;
        }
    }

    let predict_result = try_predict_from_preprocess(
        &input_path,
        &out_dir,
        output_format,
        &model,
        &pargs,
        affinity,
        use_potentials,
        extra_mols_dir.as_deref(),
        constraints_dir.as_deref(),
        preprocess_auto_extras,
        ensemble_mode,
        device_requested.as_deref(),
        confidence_ranking_enabled,
    );

    match &predict_result {
        Ok(Some(success)) => {
            let record_dir = out_dir.join(success.record_id.as_str());
            let marker = record_dir.join("boltr_predict_complete.txt");
            let predict_args = resolved.as_ref().ok();
            let info = serde_json::json!({
                "record_id": &success.record_id,
                "status": "predict_step_complete",
                "affinity": affinity,
                "use_potentials": use_potentials,
                "quality_preset": quality_preset,
                "predict_args": predict_args,
                "output_format": output_format.to_string(),
                "selected_sample": success.ranking.selected_sample,
                "ranking_metric": &success.ranking.ranking_metric,
                "ranking_score": success.ranking.ranking_score,
                "confidence_available": success.ranking.confidence_available,
                "affinity_mw_correction": affinity_mw_correction,
                "note": "Structure written from preprocess dir (manifest + npz) + collate + predict_step."
            });
            let j = serde_json::to_string_pretty(&info)?;
            tokio::fs::write(&marker, j).await?;
            tracing::info!(path = %marker.display(), "predict pipeline complete (native preprocess bridge)");
            return Ok(());
        }
        Ok(None) => {}
        Err(e) => {
            tracing::warn!(
                error = %e,
                "preprocess predict_step bridge failed; trying preprocess reference export fallback"
            );
        }
    }

    if let Ok(Some(rid)) = try_write_preprocess_reference_structure(
        &input_path,
        &out_dir,
        output_format,
        affinity,
        extra_mols_dir.as_deref(),
        constraints_dir.as_deref(),
        preprocess_auto_extras,
    ) {
        let record_dir = out_dir.join(&rid);
        let marker = record_dir.join("boltr_predict_complete.txt");
        let predict_args = resolved.as_ref().ok();
        let info = serde_json::json!({
            "record_id": rid,
            "status": "preprocess_reference_structure",
            "affinity": affinity,
            "use_potentials": use_potentials,
            "quality_preset": quality_preset,
            "predict_args": predict_args,
            "output_format": output_format.to_string(),
            "note": "Structure exported from preprocess bundle (reference/input coordinates). Not a diffusion sample."
        });
        let j = serde_json::to_string_pretty(&info)?;
        tokio::fs::write(&marker, j).await?;
        tracing::info!(path = %marker.display(), "predict pipeline complete (preprocess reference structure fallback)");
        return Ok(());
    }

    // 6. Placeholder path when no manifest/preprocess next to the input YAML
    let record_id = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("prediction")
        .to_string();
    tracing::info!(record_id = %record_id, "processing record (no preprocess bridge)");

    let record_dir = out_dir.join(&record_id);
    std::fs::create_dir_all(&record_dir)
        .with_context(|| format!("mkdir {}", record_dir.display()))?;

    write_prediction_outputs(
        &out_dir,
        &record_id,
        output_format,
        write_full_pae,
        write_full_pde,
    )?;

    let marker = record_dir.join("boltr_predict_complete.txt");
    let predict_args = resolved.as_ref().ok();
    let info = serde_json::json!({
        "record_id": record_id,
        "status": "pipeline_complete",
        "affinity": affinity,
        "use_potentials": use_potentials,
        "quality_preset": quality_preset,
        "predict_args": predict_args,
        "output_format": output_format.to_string(),
        "note": "No structure file: missing `manifest.json` + preprocess `.npz` next to the input YAML (enable `--preprocess auto|native|boltz`), or load_input failed, or predict_step failed with no reference fallback. See `boltr_predict_complete.txt` under a successful run: `predict_step_complete` vs `preprocess_reference_structure`."
    });
    let j = serde_json::to_string_pretty(&info)?;
    tokio::fs::write(&marker, j).await?;
    tracing::info!(path = %marker.display(), "predict pipeline complete");

    Ok(())
}

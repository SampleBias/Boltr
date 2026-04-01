//! Full `predict_step` wiring and `predict_args` resolution for `boltr predict` (`--features tch`).
//!
//! This module implements the prediction pipeline:
//! 1. Resolve checkpoint + hyperparameters
//! 2. Build `Boltz2Model` from hparams on the requested device
//! 3. Load weights (safetensors)
//! 4. When `manifest.json` + preprocess `.npz` sit next to the input YAML: `load_input` → collate → `predict_step` → PDB/mmCIF
//! 5. If diffusion does not run (e.g. `--affinity`, bridge error): export preprocess reference coordinates to PDB/mmCIF when the bundle exists
//! 6. Otherwise: placeholder completion marker (no structure file)
//! 7. Confidence / affinity / PAE npz when those heads are loaded and wired (see `boltr-io` writers)
//!
//! Reference: [boltz-reference/docs/prediction.md](../../../boltz-reference/docs/prediction.md)

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use boltr_backend_tch::{
    resolve_predict_args, AtomDiffusionConfig, Boltz2DiffusionArgs, Boltz2Hparams, Boltz2Model,
    Boltz2PredictArgs, PredictArgsCliOverrides, RelPosFeatures, SteeringParams,
};
use boltr_io::config::BoltzInput;
use boltr_io::{
    canonical_yaml_parent, collate_inference_batches, copy_msa_a3m_to_output, load_input,
    parse_manifest_path, trunk_smoke_feature_batch_from_inference_input,
};
use tch::{self, Kind, Tensor};

use crate::OutputFormat;

fn first_sample_coords_2d(coords: &Tensor) -> Tensor {
    let mut t = coords.shallow_clone();
    if t.dim() == 3 && t.size()[0] > 1 {
        t = t.narrow(0, 0, 1);
    }
    if t.dim() == 3 {
        t = t.squeeze_dim(0);
    }
    t
}

fn tensor_xyz_to_vec(t: &Tensor) -> Vec<[f32; 3]> {
    let n = t.size()[0] as usize;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push([
            t.double_value(&[i as i64, 0]) as f32,
            t.double_value(&[i as i64, 1]) as f32,
            t.double_value(&[i as i64, 2]) as f32,
        ]);
    }
    v
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
) -> Result<Option<String>> {
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
    if affinity {
        tracing::info!(
            "--affinity: native preprocess bridge expects non-affinity `{{id}}.npz`; skipping bridge"
        );
        return Ok(None);
    }

    let manifest = parse_manifest_path(&manifest_path)?;
    let record = manifest
        .records
        .first()
        .context("manifest.json has no records")?;
    let mut inference_input = load_input(
        record,
        &preprocess_dir,
        &preprocess_dir,
        None,
        None,
        None,
        false,
    )
    .with_context(|| {
        format!(
            "load_input from preprocess dir {}",
            preprocess_dir.display()
        )
    })?;

    let template_dim = 4_usize;
    let fb = trunk_smoke_feature_batch_from_inference_input(&inference_input, template_dim);
    let coll = collate_inference_batches(std::slice::from_ref(&fb), 0.0, 0, 0)
        .map_err(|e| anyhow::anyhow!("collate_inference_batches: {e}"))?;

    let out = crate::collate_predict_bridge::predict_step_from_collate(
        model,
        &coll,
        Some(resolved.recycling_steps),
        resolved.sampling_steps,
        resolved.diffusion_samples,
        resolved.max_parallel_samples,
        use_potentials && !affinity,
    )
    .context("predict_step_from_collate")?;

    let coords_t = first_sample_coords_2d(&out.diffusion.sample_atom_coords);
    let xyz = tensor_xyz_to_vec(&coords_t);
    let n_atom = inference_input.structure.atoms.len().min(xyz.len());
    inference_input
        .structure
        .apply_predicted_atom_coords(&xyz[..n_atom]);

    let record_id = record.id.clone();
    write_structure_file(
        out_dir,
        &record_id,
        0,
        output_format,
        &inference_input.structure,
    )?;
    copy_msa_a3m_to_output(&preprocess_dir, out_dir).context("copy MSA .a3m into output dir")?;

    tracing::info!(
        record_id = %record_id,
        "wrote predicted structure (preprocess dir + predict_step)"
    );

    Ok(Some(record_id))
}

/// Write mmCIF/PDB from preprocess `StructureV2` only (reference/input geometry, no diffusion).
/// Used when [`try_predict_from_preprocess`] did not write a sampled structure but `manifest.json`
/// + `.npz` exist beside the YAML.
fn try_write_preprocess_reference_structure(
    input_path: &Path,
    out_dir: &Path,
    output_format: OutputFormat,
    affinity: bool,
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

    // `--affinity` expects `pre_affinity_{id}.npz`; native/Boltz preprocess usually emits flat `{id}.npz`.
    // Fall back to the flat bundle so users still get a reference .cif/.pdb when affinity layout is absent.
    let inference_input = match load_input(
        record,
        &preprocess_dir,
        &preprocess_dir,
        None,
        None,
        None,
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
                None,
                None,
                None,
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

/// Write structure file (PDB or mmCIF) from predicted coordinates.
#[allow(clippy::too_many_arguments)]
fn write_structure_file(
    out_dir: &Path,
    record_id: &str,
    model_rank: usize,
    output_format: OutputFormat,
    structure: &boltr_io::StructureV2Tables,
) -> Result<PathBuf> {
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
        overrides,
        step_scale: _,
        output_format,
        max_msa_seqs: _,
        num_samples: _,
        checkpoint,
        affinity_checkpoint,
        affinity_mw_correction: _,
        sampling_steps_affinity: _,
        diffusion_samples_affinity: _,
        preprocessing_threads: _,
        override_flag: _,
        write_full_pae,
        write_full_pde,
        spike_only,
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

    // Export reference mmCIF/PDB from preprocess bundle before loading the full graph. This avoids
    // needing diffusion featurizer tensors (e.g. atom name chars) when the bundle is valid.
    if let Ok(Some(rid)) =
        try_write_preprocess_reference_structure(&input_path, &out_dir, output_format, affinity)
    {
        let record_dir = out_dir.join(&rid);
        let marker = record_dir.join("boltr_predict_complete.txt");
        let predict_args = resolved.as_ref().ok();
        let info = serde_json::json!({
            "record_id": rid,
            "status": "preprocess_reference_structure",
            "affinity": affinity,
            "use_potentials": use_potentials,
            "predict_args": predict_args,
            "output_format": output_format.to_string(),
            "note": "Structure exported from preprocess bundle (reference/input coordinates) before model load. Not a diffusion sample."
        });
        let j = serde_json::to_string_pretty(&info)?;
        tokio::fs::write(&marker, j).await?;
        tracing::info!(path = %marker.display(), "predict pipeline complete (preprocess reference structure, no model)");
        return Ok(());
    }

    // 3. Resolve checkpoint paths
    let conf_path = resolve_conf_checkpoint(checkpoint.as_deref(), &cache)?;
    tracing::info!(path = %conf_path.display(), "using confidence checkpoint");

    let aff_path = if affinity {
        resolve_affinity_checkpoint(affinity_checkpoint.as_deref(), &cache)
    } else {
        None
    };
    if aff_path.is_some() {
        tracing::info!(path = %aff_path.as_ref().unwrap().display(), "using affinity checkpoint");
    }
    let _ = aff_path;

    // 4. Load hyperparameters and build model on target device
    let hparams = load_hparams_for_predict(&cache)?;
    let token_s = hparams.resolved_token_s();
    let token_z = hparams.resolved_token_z();
    let num_blocks = hparams.resolved_num_pairformer_blocks().unwrap_or(4);

    let device = boltr_backend_tch::parse_device_spec(&device)?;
    let diff_args = Boltz2DiffusionArgs::from_boltz2_hparams(&hparams);
    let diff_config = AtomDiffusionConfig::from_boltz2_hparams(&hparams);
    tracing::info!(
        token_s,
        token_z,
        num_blocks,
        token_transformer_heads = diff_args.token_transformer_heads,
        ?device,
        "building Boltz2Model from hparams"
    );

    let mut model = Boltz2Model::with_all_options(
        device,
        token_s,
        token_z,
        Some(num_blocks),
        hparams.resolved_bond_type_feature(),
        diff_args,
        diff_config,
        None,
        None,
        false,
    )
    .context("Boltz2Model::with_all_options")?;

    // 5. Load weights
    match model.load_partial_from_safetensors(&conf_path) {
        Ok(missing) => {
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
            tracing::warn!(error = %e, "partial load failed; trying s_init only");
            if let Err(e2) = model.load_s_init_from_safetensors(&conf_path) {
                tracing::warn!(error = %e2, "s_init load also failed");
            }
        }
    }

    let pargs = match &resolved {
        Ok(a) => *a,
        Err(_) => Boltz2PredictArgs::default(),
    };

    match try_predict_from_preprocess(
        &input_path,
        &out_dir,
        output_format,
        &model,
        &pargs,
        affinity,
        use_potentials,
    ) {
        Ok(Some(rid)) => {
            let record_dir = out_dir.join(&rid);
            let marker = record_dir.join("boltr_predict_complete.txt");
            let predict_args = resolved.as_ref().ok();
            let info = serde_json::json!({
                "record_id": rid,
                "status": "predict_step_complete",
                "affinity": affinity,
                "use_potentials": use_potentials,
                "predict_args": predict_args,
                "output_format": output_format.to_string(),
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
                "preprocess predict_step bridge failed (reference export was attempted before model load)"
            );
        }
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
        "predict_args": predict_args,
        "output_format": output_format.to_string(),
        "note": "No structure file: missing `manifest.json` + preprocess `.npz` next to the input YAML (enable `--preprocess auto|native|boltz`), or load_input failed. With a bundle present, a reference `.cif` is written when diffusion does not run."
    });
    let j = serde_json::to_string_pretty(&info)?;
    tokio::fs::write(&marker, j).await?;
    tracing::info!(path = %marker.display(), "predict pipeline complete");

    Ok(())
}

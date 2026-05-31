//! Full `predict_step` wiring and `predict_args` resolution for `boltr predict` (`--features tch`).
//!
//! This module implements the prediction pipeline:
//! 1. Resolve checkpoint + hyperparameters
//! 2. Build `Boltz2Model` from hparams on the requested device
//! 3. Load weights (safetensors)
//! 4. When `manifest.json` + preprocess `.npz` sit next to the input YAML: `load_input` → collate → `predict_step` → PDB/mmCIF
//! 5. If step 4 did not write (e.g. `--affinity` without flat `{id}.npz`, no manifest, or `predict_step` failed): try preprocess reference export from the same bundle
//! 6. Otherwise: placeholder completion marker (no structure file)
//! 7. Affinity JSON and, with `--write-full-pae`, `pae_{id}_model_0.npz` via `boltr-io` writers
//!
//! Reference: [boltz-reference/docs/prediction.md](../../../boltz-reference/docs/prediction.md)

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use boltr_backend_tch::{
    resolve_predict_args, AffinityModuleConfig, AtomDiffusionConfig, Boltz2DiffusionArgs,
    Boltz2Hparams, Boltz2Model, Boltz2PredictArgs, ConfidenceModuleConfig, ConfidenceOutput,
    PredictArgsCliOverrides, PredictStepOutput, RelPosFeatures, SteeringParams,
};
use boltr_io::config::BoltzInput;
use boltr_io::{
    canonical_yaml_parent, collate_inference_batches, copy_msa_a3m_to_output,
    featurized_atom_token_sum, load_input, parse_manifest_path, resolve_preprocess_load_dirs,
    trunk_smoke_feature_batch_from_inference_input_with_ensemble, AffinitySummary,
    InferenceEnsembleMode,
};
use ndarray::Array2;
use tch::{self, Device, Kind, Tensor};

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

fn diffusion_sample_count(coord_tensor: &Tensor) -> i64 {
    if coord_tensor.dim() == 3 {
        coord_tensor.size().first().copied().unwrap_or(1).max(1)
    } else {
        1
    }
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
    source_note: String,
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

/// Expected-value PAE matrix `[N, N]` (Å) for one diffusion sample index.
fn confidence_pae_matrix_for_sample(
    conf: &ConfidenceOutput,
    sample_idx: i64,
) -> Result<Array2<f32>> {
    let pae = conf
        .pae
        .to_kind(Kind::Float)
        .to_device(Device::Cpu)
        .contiguous();
    let matrix = match pae.dim() {
        2 => {
            if sample_idx != 0 {
                bail!(
                    "PAE tensor is 2D but sample_idx={sample_idx} (expected 0 for single-sample output)"
                );
            }
            pae
        }
        3 => {
            let batch = pae.size()[0];
            if sample_idx < 0 || sample_idx >= batch {
                bail!("PAE sample index {sample_idx} out of range for batch size {batch}");
            }
            pae.narrow(0, sample_idx, 1).squeeze_dim(0)
        }
        d => bail!("unexpected PAE tensor rank {d} (expected 2 or 3)"),
    };
    let shape = matrix.size();
    if shape.len() != 2 {
        bail!("PAE matrix must be rank 2 after sample selection, got {:?}", shape);
    }
    let n0 = shape[0] as usize;
    let n1 = shape[1] as usize;
    let mut data = vec![0f32; n0 * n1];
    matrix.copy_data(&mut data, n0 * n1);
    Array2::from_shape_vec((n0, n1), data).map_err(|e| anyhow!("PAE matrix shape: {e}"))
}

/// Write `pae_{record_id}_model_{rank}.npz` when `--write-full-pae` is set.
fn maybe_write_full_pae_npz(
    out_dir: &Path,
    record_id: &str,
    out: &PredictStepOutput,
    ranking: &SampleRanking,
    write_full_pae: bool,
) -> Result<()> {
    if !write_full_pae {
        return Ok(());
    }
    let conf = out.confidence.as_ref().ok_or_else(|| {
        anyhow!(
            "--write-full-pae requested but confidence output is missing (set BOLTR_ENABLE_NATIVE_CONFIDENCE=1 and ensure confidence checkpoint weights load)"
        )
    })?;
    let pae = confidence_pae_matrix_for_sample(conf, ranking.selected_sample)
        .context("extract PAE matrix from confidence head")?;
    let path = boltr_io::write_pae_npz_path(out_dir, record_id, 0, pae.view())
        .context("write PAE npz")?;
    tracing::info!(
        path = %path.display(),
        tokens = pae.nrows(),
        sample = ranking.selected_sample,
        "wrote full PAE matrix"
    );
    Ok(())
}

fn select_best_sample(
    out: &boltr_backend_tch::PredictStepOutput,
    confidence_ranking_enabled: bool,
) -> SampleRanking {
    sample_rankings(out, confidence_ranking_enabled)
        .into_iter()
        .next()
        .unwrap_or(SampleRanking {
            selected_sample: 0,
            ranking_metric: "sample0_no_confidence".to_string(),
            ranking_score: None,
            confidence_available: false,
        })
}

fn sample_rankings(
    out: &boltr_backend_tch::PredictStepOutput,
    confidence_ranking_enabled: bool,
) -> Vec<SampleRanking> {
    let sample_count = diffusion_sample_count(&out.diffusion.sample_atom_coords);
    let Some(conf) = out
        .confidence
        .as_ref()
        .filter(|_| confidence_ranking_enabled)
    else {
        return (0..sample_count)
            .map(|i| SampleRanking {
                selected_sample: i,
                ranking_metric: if i == 0 {
                    "sample0_no_confidence".to_string()
                } else {
                    "sample_no_confidence".to_string()
                },
                ranking_score: None,
                confidence_available: false,
            })
            .collect();
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
        return (0..sample_count)
            .map(|i| SampleRanking {
                selected_sample: i,
                ranking_metric: if i == 0 {
                    "sample0_confidence_unrankable".to_string()
                } else {
                    "confidence_unrankable".to_string()
                },
                ranking_score: None,
                confidence_available: true,
            })
            .collect();
    };
    let mut ranked: Vec<_> = (0..sample_count)
        .map(|i| (i, scores.double_value(&[i])))
        .collect();
    ranked.sort_by(|(a_idx, a), (b_idx, b)| {
        b.partial_cmp(a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a_idx.cmp(b_idx))
    });
    ranked
        .into_iter()
        .map(|(i, score)| SampleRanking {
            selected_sample: i,
            ranking_metric: metric.to_string(),
            ranking_score: score.is_finite().then_some(score),
            confidence_available: true,
        })
        .collect()
}

fn is_likely_cuda_oom(err_msg: &str) -> bool {
    let l = err_msg.to_lowercase();
    l.contains("out of memory")
        || l.contains("cuda out of memory")
        || (l.contains("cuda") && l.contains("alloc") && l.contains("fail"))
}

fn crop_map_from_feature_batch(fb: &boltr_io::FeatureBatch) -> Result<Vec<i64>> {
    let arr = fb
        .get_i64("crop_to_all_atom_map")
        .context("feature batch missing crop_to_all_atom_map")?;
    Ok(arr.iter().copied().collect())
}

fn crop_to_structure_atom_indices(
    structure: &boltr_io::StructureV2Tables,
    crop_to_all_atom_map: &[i64],
) -> Result<Vec<usize>> {
    let mut all_to_structure = Vec::new();
    for chain in &structure.chains {
        let start = usize::try_from(chain.atom_idx)
            .with_context(|| format!("negative chain atom_idx {}", chain.atom_idx))?;
        let count = usize::try_from(chain.atom_num)
            .with_context(|| format!("negative chain atom_num {}", chain.atom_num))?;
        for i in 0..count {
            let atom_idx = start + i;
            if atom_idx < structure.atoms.len() {
                all_to_structure.push(atom_idx);
            }
        }
    }
    if all_to_structure.is_empty() && !structure.atoms.is_empty() {
        all_to_structure.extend(0..structure.atoms.len());
    }

    let mut mapped = Vec::with_capacity(crop_to_all_atom_map.len());
    for &all_idx in crop_to_all_atom_map {
        let u = usize::try_from(all_idx)
            .with_context(|| format!("negative crop_to_all_atom_map entry {all_idx}"))?;
        let atom_idx = all_to_structure.get(u).copied().with_context(|| {
            format!(
                "crop_to_all_atom_map entry {u} outside all-atom map length {}",
                all_to_structure.len()
            )
        })?;
        mapped.push(atom_idx);
    }
    Ok(mapped)
}

fn apply_mapped_predicted_coords(
    structure: &mut boltr_io::StructureV2Tables,
    xyz: &[[f32; 3]],
    crop_atom_indices: &[usize],
) -> Result<usize> {
    if xyz.len() < crop_atom_indices.len() {
        bail!(
            "diffusion returned {} coordinates, but crop mapping needs {} real atom coordinates",
            xyz.len(),
            crop_atom_indices.len()
        );
    }
    structure.apply_predicted_atom_coords_by_atom_indices(
        &xyz[..crop_atom_indices.len()],
        crop_atom_indices,
    );
    Ok(crop_atom_indices.len())
}

fn coord_stats_json(xyz: &[[f32; 3]]) -> serde_json::Value {
    let mut finite = 0_usize;
    let mut non_finite = 0_usize;
    let mut near_zero_atoms = 0_usize;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for c in xyz {
        if c.iter().all(|v| v.abs() <= 1e-12) {
            near_zero_atoms += 1;
        }
        for axis in 0..3 {
            let v = c[axis];
            if v.is_finite() {
                finite += 1;
                min[axis] = min[axis].min(v);
                max[axis] = max[axis].max(v);
            } else {
                non_finite += 1;
            }
        }
    }
    serde_json::json!({
        "atom_count": xyz.len(),
        "finite_scalar_count": finite,
        "non_finite_scalar_count": non_finite,
        "near_zero_atom_count": near_zero_atoms,
        "min_xyz": if finite == 0 { serde_json::Value::Null } else { serde_json::json!(min) },
        "max_xyz": if finite == 0 { serde_json::Value::Null } else { serde_json::json!(max) },
    })
}

fn qc_dry_run_report(
    structure: &boltr_io::StructureV2Tables,
    model_filename: String,
) -> boltr_io::QcReport {
    let thresholds = boltr_io::QcThresholds::default();
    let initial = boltr_io::validate_structure_qc(
        structure,
        model_filename.clone(),
        thresholds,
        false,
        false,
    );
    if initial.passed {
        return initial;
    }
    let mut relaxed = structure.clone();
    boltr_io::relax_structure(&mut relaxed, thresholds);
    let mut relaxed_report =
        boltr_io::validate_structure_qc(&relaxed, model_filename, thresholds, true, false);
    if relaxed_report.passed {
        relaxed_report.relaxation_fixed = true;
    }
    relaxed_report
}

fn first_failed_geometry_json(report: &boltr_io::QcReport) -> serde_json::Value {
    serde_json::json!({
        "backbone_bond_distances": report.backbone_bond_distances
            .iter()
            .filter(|m| !m.passed)
            .take(8)
            .collect::<Vec<_>>(),
        "peptide_bond_distances": report.peptide_bond_distances
            .iter()
            .filter(|m| !m.passed)
            .take(8)
            .collect::<Vec<_>>(),
        "ca_ca_distances": report.ca_ca_distances
            .iter()
            .filter(|m| !m.passed)
            .take(8)
            .collect::<Vec<_>>(),
        "omega_torsions": report.omega_torsions
            .iter()
            .filter(|m| !m.passed)
            .take(8)
            .collect::<Vec<_>>(),
    })
}

fn promote_upstream_boltz_prediction(
    preprocess_dir: &Path,
    out_dir: &Path,
    record_id: &str,
) -> Result<Option<PathBuf>> {
    let upstream_dir = preprocess_dir.join(".boltr_upstream_predictions");
    if !upstream_dir.is_dir() {
        return Ok(None);
    }
    let record_dir = out_dir.join(record_id);
    std::fs::create_dir_all(&record_dir)
        .with_context(|| format!("mkdir {}", record_dir.display()))?;

    let mut promoted_cif = None;
    for ext in ["cif", "pdb"] {
        let src = upstream_dir.join(format!("{record_id}_model_0.{ext}"));
        if !src.is_file() {
            continue;
        }
        let dst = record_dir.join(format!("{record_id}_model_0.{ext}"));
        std::fs::copy(&src, &dst)
            .with_context(|| format!("copy upstream Boltz prediction {}", src.display()))?;
        if ext == "cif" {
            promoted_cif = Some(dst);
        }
    }
    if promoted_cif.is_some() {
        let note = record_dir.join(format!("{record_id}_model_0.upstream_boltz.txt"));
        std::fs::write(
            &note,
            "Final structure promoted from upstream Boltz high-fidelity prediction because native Rust diffusion did not pass QC.\n",
        )
        .with_context(|| format!("write {}", note.display()))?;
    }
    Ok(promoted_cif)
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
    write_full_pae: bool,
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

    // Structure diffusion always reads the flat `{record_id}.npz` bundle. Treat a
    // manifest without that file as an incomplete preprocess result instead of
    // falling through to zero-coordinate reference export.
    let flat_npz = preprocess_dir.join(format!("{}.npz", record.id));
    if !flat_npz.is_file() {
        bail!(
            "preprocess bundle incomplete: found {} for record {}, but missing flat structure npz {}. Re-run with preprocess auto/native/boltz and verify upstream Boltz copied both manifest.json and the .npz bundle.",
            manifest_path.display(),
            record.id,
            flat_npz.display()
        );
    }
    if affinity {
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
    let inference_input = load_input(
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

    let rankings = sample_rankings(&out, confidence_ranking_enabled);
    let raw_shape = out.diffusion.sample_atom_coords.size();
    let crop_to_all_atom_map = crop_map_from_feature_batch(&fb)?;
    let crop_atom_indices =
        crop_to_structure_atom_indices(&inference_input.structure, &crop_to_all_atom_map)?;
    let n_featurized = featurized_atom_token_sum(&inference_input);
    let n_atoms_struct = inference_input.structure.atoms.len();
    if n_featurized != crop_atom_indices.len() {
        tracing::warn!(
            n_featurized,
            n_crop_map = crop_atom_indices.len(),
            "featurized atom count differs from crop_to_all_atom_map length"
        );
    }

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

    let record_id = record.id.clone();
    let model_filename = structure_primary_path(out_dir, &record_id, 0, output_format)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("prediction_model_0.cif")
        .to_string();
    let mut selected: Option<(SampleRanking, boltr_io::StructureV2Tables)> = None;
    let mut sample_summaries = Vec::new();

    for ranking in rankings {
        let xyz = diffusion_sample_coords_to_xyz_vec(
            &out.diffusion.sample_atom_coords,
            ranking.selected_sample,
        )
        .map_err(|e| anyhow!("diffusion coords → xyz: {e}"))?;
        let n_from_model = xyz.len();
        if n_from_model < crop_atom_indices.len() {
            tracing::warn!(
                n_from_model,
                n_crop_map = crop_atom_indices.len(),
                raw_shape = ?raw_shape,
                sample = ranking.selected_sample,
                "diffusion returned fewer coords than crop map real atoms"
            );
        }
        if n_from_model > crop_atom_indices.len() {
            tracing::debug!(
                n_from_model,
                n_crop_map = crop_atom_indices.len(),
                sample = ranking.selected_sample,
                "diffusion atom dim exceeds crop map real atoms (expected padding)"
            );
        }

        let mut candidate = inference_input.structure.clone();
        let applied = apply_mapped_predicted_coords(&mut candidate, &xyz, &crop_atom_indices)?;
        let qc_report = qc_dry_run_report(&candidate, model_filename.clone());
        let passed_qc = qc_report.passed;
        sample_summaries.push(serde_json::json!({
            "sample": ranking.selected_sample,
            "ranking_metric": &ranking.ranking_metric,
            "ranking_score": ranking.ranking_score,
            "confidence_available": ranking.confidence_available,
            "applied_atom_count": applied,
            "coord_stats": coord_stats_json(&xyz),
            "qc_passed_after_relax": passed_qc,
            "fail_reasons": &qc_report.fail_reasons,
            "steric_clashes": qc_report.steric_clashes.len(),
            "first_failed_geometry": first_failed_geometry_json(&qc_report),
        }));
        if passed_qc && selected.is_none() {
            selected = Some((ranking, candidate));
        }
    }

    let debug_dir = out_dir.join(&record_id);
    std::fs::create_dir_all(&debug_dir)
        .with_context(|| format!("mkdir {}", debug_dir.display()))?;
    let selected_sample = selected.as_ref().map(|(r, _)| r.selected_sample);
    let debug_json = serde_json::json!({
        "record_id": &record_id,
        "raw_sample_atom_coords_shape": raw_shape,
        "selected_sample_after_qc": selected_sample,
        "n_featurized": n_featurized,
        "n_atoms_struct": n_atoms_struct,
        "crop_to_all_atom_map_len": crop_to_all_atom_map.len(),
        "mapped_atom_count": crop_atom_indices.len(),
        "samples": sample_summaries,
    });
    let debug_path = debug_dir.join(format!("{record_id}_native_predict_debug.json"));
    std::fs::write(&debug_path, serde_json::to_vec_pretty(&debug_json)?)
        .with_context(|| format!("write {}", debug_path.display()))?;

    let (ranking, selected_structure, source_note) = if let Some(selected) = selected {
        (
            selected.0,
            selected.1,
            "Structure written from native Rust diffusion after QC passed.".to_string(),
        )
    } else {
        if promote_upstream_boltz_prediction(&preprocess_dir, out_dir, &record_id)?.is_some() {
            let ranking = select_best_sample(&out, confidence_ranking_enabled);
            maybe_write_full_pae_npz(out_dir, &record_id, &out, &ranking, write_full_pae)?;
            copy_msa_a3m_to_output(&preprocess_dir, out_dir)
                .context("copy MSA .a3m into output dir")?;
            tracing::warn!(
                record_id = %record_id,
                "native Rust diffusion failed QC; promoted upstream Boltz high-fidelity prediction"
            );
            return Ok(Some(PredictBridgeSuccess {
                record_id,
                ranking,
                source_note: "Structure promoted from upstream Boltz high-fidelity prediction because native Rust diffusion did not pass QC.".to_string(),
            }));
        }
        let fallback = select_best_sample(&out, confidence_ranking_enabled);
        let xyz = diffusion_sample_coords_to_xyz_vec(
            &out.diffusion.sample_atom_coords,
            fallback.selected_sample,
        )
        .map_err(|e| anyhow!("diffusion coords → xyz for failed fallback sample: {e}"))?;
        let mut candidate = inference_input.structure.clone();
        apply_mapped_predicted_coords(&mut candidate, &xyz, &crop_atom_indices)?;
        (
            fallback,
            candidate,
            "Native Rust diffusion did not pass dry-run QC; failed artifacts are diagnostics."
                .to_string(),
        )
    };
    write_structure_file(out_dir, &record_id, 0, output_format, &selected_structure)?;
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
    maybe_write_full_pae_npz(out_dir, &record_id, &out, &ranking, write_full_pae)?;
    copy_msa_a3m_to_output(&preprocess_dir, out_dir).context("copy MSA .a3m into output dir")?;

    tracing::info!(
        record_id = %record_id,
        selected_sample = ranking.selected_sample,
        ranking_metric = %ranking.ranking_metric,
        "wrote predicted structure (preprocess dir + predict_step)"
    );

    Ok(Some(PredictBridgeSuccess {
        record_id,
        ranking,
        source_note,
    }))
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

fn structure_primary_path(
    out_dir: &Path,
    record_id: &str,
    model_rank: usize,
    output_format: OutputFormat,
) -> PathBuf {
    let record_dir = out_dir.join(record_id);
    match output_format {
        OutputFormat::Pdb => record_dir.join(format!("{record_id}_model_{model_rank}.pdb")),
        OutputFormat::Mmcif | OutputFormat::Both => {
            record_dir.join(format!("{record_id}_model_{model_rank}.cif"))
        }
    }
}

fn write_structure_pair(
    record_dir: &Path,
    base: &str,
    suffix: Option<&str>,
    structure: &boltr_io::StructureV2Tables,
) -> Result<(PathBuf, PathBuf)> {
    let stem = match suffix {
        Some(s) => format!("{base}.{s}"),
        None => base.to_string(),
    };
    let pdb = record_dir.join(format!("{stem}.pdb"));
    let cif = record_dir.join(format!("{stem}.cif"));
    let pdb_bytes = boltr_io::structure_v2_to_pdb(structure, None);
    let cif_bytes = boltr_io::structure_v2_to_mmcif(structure);
    std::fs::write(&pdb, &pdb_bytes).with_context(|| format!("write {}", pdb.display()))?;
    std::fs::write(&cif, &cif_bytes).with_context(|| format!("write {}", cif.display()))?;
    Ok((pdb, cif))
}

fn write_qc_reports(record_dir: &Path, base: &str, report: &boltr_io::QcReport) -> Result<()> {
    let json = record_dir.join(format!("{base}.qc.json"));
    let txt = record_dir.join(format!("{base}.qc.txt"));
    let bytes = serde_json::to_vec_pretty(report).context("serialize QC report JSON")?;
    std::fs::write(&json, bytes).with_context(|| format!("write {}", json.display()))?;
    std::fs::write(&txt, boltr_io::render_qc_text(report))
        .with_context(|| format!("write {}", txt.display()))?;
    Ok(())
}

/// Write structure files only after universal QC passes. Failed structures are emitted with
/// `.failed.*` suffixes and never become the canonical final `.cif`/`.pdb`.
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

    let base = format!("{record_id}_model_{model_rank}");
    if structure.present_atoms_all_coords_near_zero(1e-12) {
        warn_if_all_present_coords_zero(structure, "write_structure_file");
        bail!(
            "structure export aborted for {record_id} model {model_rank}: all present atom coordinates are zero. This is a placeholder/reference bundle, not a usable predicted structure; check preprocess bundle completeness and predict_step logs before QC."
        );
    }
    let primary_path = structure_primary_path(out_dir, record_id, model_rank, output_format);
    let primary_name = primary_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("prediction_model_0.cif")
        .to_string();
    let thresholds = boltr_io::QcThresholds::default();
    let initial_report =
        boltr_io::validate_structure_qc(structure, primary_name.clone(), thresholds, false, false);

    if initial_report.passed {
        let (_pdb, cif) = write_structure_pair(&record_dir, &base, None, structure)?;
        write_qc_reports(&record_dir, &base, &initial_report)?;
        tracing::debug!(path = %cif.display(), "wrote QC-passed structure files");
        return Ok(primary_path);
    }

    let mut relaxed = structure.clone();
    let relax_outcome = boltr_io::relax_structure(&mut relaxed, thresholds);
    let relaxed_report =
        boltr_io::validate_structure_qc(&relaxed, primary_name, thresholds, true, false);
    if relaxed_report.passed {
        write_structure_pair(&record_dir, &base, Some("relaxed"), &relaxed)?;
        write_structure_pair(&record_dir, &base, None, &relaxed)?;
        let mut final_report = relaxed_report;
        final_report.relaxation_fixed = true;
        write_qc_reports(&record_dir, &base, &final_report)?;
        tracing::info!(
            max_displacement = relax_outcome.max_displacement,
            iterations = relax_outcome.iterations,
            "QC relaxation fixed structure geometry"
        );
        return Ok(primary_path);
    }

    write_structure_pair(&record_dir, &base, Some("failed"), structure)?;
    write_qc_reports(&record_dir, &base, &relaxed_report)?;
    let severe_geometry = relaxed_report.steric_clashes.len() >= 1000
        || relaxed_report.fail_reasons.iter().any(|r| {
            let r = r.to_ascii_lowercase();
            r.contains("hard steric overlap") || r.contains("ca-ca distance")
        });
    if severe_geometry {
        tracing::warn!(
            record_id,
            model_rank,
            steric_clashes = relaxed_report.steric_clashes.len(),
            reasons = %relaxed_report.fail_reasons.join("; "),
            "structure QC detected severe geometry collapse; failed structure files are debug artifacts"
        );
    }
    let guidance = if severe_geometry {
        " Severe geometry collapse detected; `.failed.cif` / `.failed.pdb` are debug artifacts, not usable structures. If high-fidelity/Boltz preprocess already ran, inspect the native prediction debug JSON, coordinate mapping, inputs/templates/constraints, and upstream Boltz comparison."
    } else {
        " See the `.qc.txt` / `.qc.json` report and `.failed.*` files for diagnostics."
    };
    bail!(
        "structure QC failed after relaxation for {record_id} model {model_rank}: {}. Steric clashes: {}.{guidance}",
        relaxed_report.fail_reasons.join("; "),
        relaxed_report.steric_clashes.len()
    )
}

#[cfg(test)]
mod qc_export_tests {
    use super::*;
    use boltr_io::boltz_const::{chain_type_id, token_id};
    use boltr_io::{AtomV2Row, ChainRow, EnsembleRow, ResidueRow, StructureV2Tables};

    fn two_ala_structure() -> StructureV2Tables {
        let protein = chain_type_id("PROTEIN").expect("PROTEIN") as i8;
        let ala = token_id("ALA").expect("ALA") as i8;
        let names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB"];
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.46, 0.0, 0.0],
            [2.36, 1.24, 0.0],
            [2.20, 2.46, 0.0],
            [1.46, -1.50, 0.0],
            [3.66, 1.52, 0.0],
            [3.85, 2.95, 0.20],
            [4.75, 4.19, 0.20],
            [4.59, 5.41, 0.20],
            [3.85, 2.95, 1.70],
        ];
        let atoms = names
            .iter()
            .zip(coords.iter())
            .map(|(&name, &coords)| AtomV2Row {
                name: name.to_string(),
                coords,
                is_present: true,
                bfactor: 0.0,
                plddt: 0.0,
            })
            .collect();
        StructureV2Tables {
            atoms,
            residues: vec![
                ResidueRow {
                    name: "ALA".to_string(),
                    res_type: ala,
                    res_idx: 0,
                    atom_idx: 0,
                    atom_num: 5,
                    atom_center: 1,
                    atom_disto: 4,
                    is_standard: true,
                    is_present: true,
                },
                ResidueRow {
                    name: "ALA".to_string(),
                    res_type: ala,
                    res_idx: 1,
                    atom_idx: 5,
                    atom_num: 5,
                    atom_center: 6,
                    atom_disto: 9,
                    is_standard: true,
                    is_present: true,
                },
            ],
            chains: vec![ChainRow {
                name: "A".to_string(),
                mol_type: protein,
                sym_id: 0,
                asym_id: 0,
                entity_id: 0,
                atom_idx: 0,
                atom_num: 10,
                res_idx: 0,
                res_num: 2,
                cyclic_period: 0,
            }],
            chain_mask: vec![true],
            coords: coords.clone(),
            ensemble: vec![EnsembleRow {
                atom_coord_idx: 0,
                atom_num: 10,
            }],
            ensemble_atom_coord_idx: 0,
            bonds: vec![],
        }
    }

    #[test]
    fn qc_export_writes_final_structures_and_reports() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let p = write_structure_file(
            tmp.path(),
            "prediction",
            0,
            OutputFormat::Mmcif,
            &two_ala_structure(),
        )
        .expect("QC-passed export");
        let dir = tmp.path().join("prediction");
        assert_eq!(p, dir.join("prediction_model_0.cif"));
        assert!(dir.join("prediction_model_0.cif").is_file());
        assert!(dir.join("prediction_model_0.pdb").is_file());
        assert!(dir.join("prediction_model_0.qc.json").is_file());
        assert!(dir.join("prediction_model_0.qc.txt").is_file());
        assert!(!dir.join("prediction_model_0.failed.cif").exists());
    }

    #[test]
    fn qc_export_fails_closed_for_missing_backbone() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut s = two_ala_structure();
        s.atoms[1].is_present = false;
        let err = write_structure_file(tmp.path(), "prediction", 0, OutputFormat::Both, &s)
            .expect_err("missing CA must fail");
        assert!(err.to_string().contains("structure QC failed"));
        let dir = tmp.path().join("prediction");
        assert!(!dir.join("prediction_model_0.cif").exists());
        assert!(!dir.join("prediction_model_0.pdb").exists());
        assert!(dir.join("prediction_model_0.failed.cif").is_file());
        assert!(dir.join("prediction_model_0.failed.pdb").is_file());
        assert!(dir.join("prediction_model_0.qc.json").is_file());
        assert!(dir.join("prediction_model_0.qc.txt").is_file());
    }

    #[test]
    fn qc_export_rejects_all_zero_placeholder_without_failed_artifacts() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut s = two_ala_structure();
        for atom in &mut s.atoms {
            atom.coords = [0.0, 0.0, 0.0];
        }
        let err = write_structure_file(tmp.path(), "prediction", 0, OutputFormat::Both, &s)
            .expect_err("all-zero placeholder must fail before file export");
        assert!(err
            .to_string()
            .contains("all present atom coordinates are zero"));
        let dir = tmp.path().join("prediction");
        assert!(!dir.join("prediction_model_0.cif").exists());
        assert!(!dir.join("prediction_model_0.pdb").exists());
        assert!(!dir.join("prediction_model_0.failed.cif").exists());
        assert!(!dir.join("prediction_model_0.failed.pdb").exists());
        assert!(!dir.join("prediction_model_0.qc.json").exists());
    }

    #[test]
    fn diffusion_coords_extracts_supported_shapes() {
        let coords = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 3]);
        assert_eq!(diffusion_sample_count(&coords), 1);
        let xyz = diffusion_sample_coords_to_xyz_vec(&coords, 0).expect("2D coords");
        assert_eq!(xyz, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let transposed = Tensor::from_slice(&[1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]).view([3, 2]);
        let xyz = diffusion_sample_coords_to_xyz_vec(&transposed, 0).expect("transposed coords");
        assert_eq!(xyz, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let batched = Tensor::from_slice(&[
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ])
        .view([2, 2, 3]);
        assert_eq!(diffusion_sample_count(&batched), 2);
        let xyz = diffusion_sample_coords_to_xyz_vec(&batched, 1).expect("batched coords");
        assert_eq!(xyz, vec![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);
    }

    #[test]
    fn crop_map_applies_coords_to_explicit_structure_atoms() {
        let mut s = two_ala_structure();
        s.chains[0].atom_idx = 5;
        s.chains[0].atom_num = 5;
        s.chains.push(ChainRow {
            name: "B".to_string(),
            mol_type: s.chains[0].mol_type,
            sym_id: 0,
            asym_id: 1,
            entity_id: 1,
            atom_idx: 0,
            atom_num: 5,
            res_idx: 0,
            res_num: 1,
            cyclic_period: 0,
        });
        let mapped = crop_to_structure_atom_indices(&s, &[0, 4, 5, 9]).expect("crop map");
        assert_eq!(mapped, vec![5, 9, 0, 4]);

        let xyz = vec![
            [50.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [0.0, 50.0, 0.0],
            [0.0, 90.0, 0.0],
        ];
        apply_mapped_predicted_coords(&mut s, &xyz, &mapped).expect("apply mapped coords");
        assert_eq!(s.atoms[5].coords, [50.0, 0.0, 0.0]);
        assert_eq!(s.atoms[9].coords, [90.0, 0.0, 0.0]);
        assert_eq!(s.atoms[0].coords, [0.0, 50.0, 0.0]);
        assert_eq!(s.atoms[4].coords, [0.0, 90.0, 0.0]);
        assert_eq!(s.coords[5], [50.0, 0.0, 0.0]);
    }

    #[test]
    fn qc_dry_run_identifies_passing_sample_after_failed_sample() {
        let crop_atoms: Vec<_> = (0..two_ala_structure().atoms.len()).collect();
        let mut failed = two_ala_structure();
        let collapsed = vec![[0.0_f32, 0.0, 0.0]; crop_atoms.len()];
        apply_mapped_predicted_coords(&mut failed, &collapsed, &crop_atoms)
            .expect("apply collapsed sample");
        let failed_report = qc_dry_run_report(&failed, "prediction_model_0.cif".to_string());
        assert!(!failed_report.passed);

        let mut passing = two_ala_structure();
        let coords: Vec<_> = two_ala_structure().atoms.iter().map(|a| a.coords).collect();
        apply_mapped_predicted_coords(&mut passing, &coords, &crop_atoms)
            .expect("apply passing sample");
        let passing_report = qc_dry_run_report(&passing, "prediction_model_0.cif".to_string());
        assert!(passing_report.passed);
    }
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

    let native_confidence_enabled = env_flag_true("BOLTR_ENABLE_NATIVE_CONFIDENCE")
        || write_full_pae
        || write_full_pde;
    if (write_full_pae || write_full_pde) && !env_flag_true("BOLTR_ENABLE_NATIVE_CONFIDENCE") {
        tracing::info!(
            write_full_pae,
            write_full_pde,
            "enabling native confidence module for full PAE/PDE export"
        );
    }
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
    let confidence_config = if hparams.confidence_prediction == Some(true)
        && native_confidence_enabled
    {
        let token_level = hparams
            .other
            .get("token_level_confidence")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        Some(ConfidenceModuleConfig::from_confidence_model_args(
            hparams.confidence_model_args.as_ref(),
            token_level,
        ))
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

    if write_full_pae && model.confidence_module().is_none() {
        bail!(
            "--write-full-pae requires the confidence module (checkpoint must include confidence_module weights; set BOLTR_ENABLE_NATIVE_CONFIDENCE=1 if disabled)"
        );
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
        write_full_pae,
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
                "note": &success.source_note
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
            if e.to_string().contains("preprocess bundle incomplete") {
                bail!("preprocess predict_step bridge requires a complete flat bundle: {e}");
            }
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

# Boltr — §6 `boltr-cli`: User-Facing Commands

## Context
From TODO.md §6: Build the complete `boltr-cli` user-facing commands with precision and accuracy.
The CLI must match the upstream Boltz `boltz predict` command interface and produce identical output layouts.

## Status Key
| Mark | Meaning |
|------|---------|
| `[x]` | **Done** |
| `[~]` | **Partial / stub** |
| `[ ]` | **Open** |

---

## Tasks

### 1. Update CLI flags for full Boltz parity
- [x] 1.1 Add `--output-format` flag (`pdb` | `mmcif`, default `mmcif`) matching Boltz `--output_format`
- [x] 1.2 Add `--checkpoint` flag for custom checkpoint path (Boltz `--checkpoint`)
- [x] 1.3 Add `--affinity-checkpoint` flag for affinity model checkpoint
- [x] 1.4 Add `--step-scale` flag (Boltz diffusion step scale, default 1.638)
- [x] 1.5 Add `--max-msa-seqs` flag (Boltz `--max_msa_seqs`, default 8192)
- [x] 1.6 Add `--override` flag to re-run predictions (Boltz `--override`)
- [x] 1.7 Add `--write-full-pae` / `--write-full-pde` flags
- [x] 1.8 Add `--affinity-mw-correction` flag
- [x] 1.9 Add `--sampling-steps-affinity` and `--diffusion-samples-affinity` flags
- [x] 1.10 Add `--preprocessing-threads` flag
- [x] 1.11 Wire all new flags into predict_flow

### 2. Implement full predict pipeline (`predict_tch.rs`)
- [x] 2.1 Create `PredictContext` struct to hold resolved CLI args + cache paths + device
- [x] 2.2 Implement `resolve_checkpoint_path` — CLI path > cache dir > auto-download
- [x] 2.3 Implement `load_or_download_hparams` — hparams.json next to checkpoint or from env
- [x] 2.4 Implement `build_model_from_hparams` — construct `Boltz2Model` from hparams JSON
- [x] 2.5 Implement `build_feature_batch_from_input` — full featurizer pipeline (YAML → preprocess → tokens → features → collate)
- [x] 2.6 Implement `run_predict_step` — trunk → diffusion → distogram → confidence → affinity
- [x] 2.7 Implement `write_prediction_outputs` — structure files + confidence JSON + affinity JSON + PAE/PDE/pLDDT npz
- [x] 2.8 Implement full `predict_flow_tch` orchestrating 2.1–2.7
- [x] 2.9 Handle `--spike-only` path (trunk-only smoke, no diffusion/writers)
- [x] 2.10 Handle `--affinity` path (affinity crop, affinity model, affinity writer)
- [x] 2.11 Handle `--use-potentials` path (steering params, potential feats)
- [x] 2.12 Proper `BOLTR_CACHE` / `--cache` directory resolution matching Boltz `~/.boltz`

### 3. MSA integration in predict flow
- [x] 3.1 Implement `--use_msa_server` path: fetch MSAs → write `.a3m` → feed into featurizer
- [x] 3.2 Implement local MSA path resolution (YAML `msa:` field → `.a3m` → `.npz`)
- [x] 3.3 Wire MSA into `msa_features_from_inference_input`

### 4. Output directory layout (matches Boltz `predictions/`)
- [x] 4.1 Record directory creation: `{output_dir}/{record_id}/`
- [x] 4.2 Structure output: `{record_id}_model_{rank}.{pdb|cif}` sorted by confidence
- [x] 4.3 Confidence JSON: `confidence_{record_id}_model_{rank}.json`
- [x] 4.4 Affinity JSON: `affinity_{record_id}.json`
- [x] 4.5 PAE/PDE/pLDDT npz files per model rank
- [x] 4.6 `boltr_run_summary.json` at output dir root
- [x] 4.7 `boltr_predict_args.json` at output dir root

### 5. Download command improvements
- [x] 5.1 Support `BOLTZ_CACHE` env var for cache directory
- [x] 5.2 Download safetensors + hparams JSON for boltz2
- [x] 5.3 Progress logging for downloads

### 6. Eval command
- [x] 6.1 Improve eval stub with clearer documentation

### 7. Testing
- [x] 7.1 Add CLI integration test for `predict` with minimal YAML + `--spike-only`
- [x] 7.2 Add CLI integration test for `download` command
- [x] 7.3 Add unit tests for flag parsing / resolution
- [x] 7.4 Verify `cargo test -p boltr-cli` passes (no tch feature required)
- [x] 7.5 Verify `cargo test -p boltr-cli --features tch` passes when LibTorch available

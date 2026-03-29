# Boltz2 tensor contract (Python reference)

This document tracks the data path Rust must mirror for parity with `boltz-reference` when `use_kernels=False`.

## Inference pipeline

1. **YAML** → `boltz.data.parse.yaml.parse_yaml` → `Target` / `Record` (schema in `boltz.data.parse.schema`).
2. **Preprocess** (Boltz CLI) builds `processed/` targets, MSAs (`.npz`), constraints, templates.
3. **`Boltz2InferenceDataModule`** ([`inferencev2.py`](../boltz-reference/src/boltz/data/module/inferencev2.py)):
   - `load_input` → `Input` (structure, MSAs, record, templates, …).
   - `Boltz2Tokenizer.tokenize` → `Tokenized`.
   - `Boltz2Featurizer.process` → `dict[str, Tensor]` batch features.
4. **Collate** stacks / pads batch dimensions (see `collate()` in the same module).
5. **`Boltz2.forward` / `predict_step`** consumes those tensors on the configured device (Boltz2 uses mixed bf16 on GPU with many `autocast("cuda", enabled=False)` islands; match dtype policy when porting).

## `predict_args` (inference steps)

Checkpoint `hyper_parameters["predict_args"]` is a JSON object (optional). Rust uses [`Boltz2PredictArgs`](../../boltr-backend-tch/src/predict_args.rs) with keys:

| Key | Role |
|-----|------|
| `recycling_steps` | Trunk loop count (`0..=recycling_steps` in Rust). |
| `num_sampling_steps` or `sampling_steps` | Diffusion sampler steps; `None` defers to [`AtomDiffusionConfig::num_sampling_steps`](../../boltr-backend-tch/src/boltz2/diffusion.rs). |
| `diffusion_samples` | Passed as `multiplicity` to the atom diffusion sampler. |
| `max_parallel_samples` | Optional cap for diffusion (steering / extended sampler). |

**Precedence:** CLI flags (`boltr predict`) override YAML top-level `predict_args:` (if present), which overrides checkpoint JSON, which fills gaps from defaults (`recycling_steps=0`, `num_sampling_steps=None`, `diffusion_samples=1`, `max_parallel_samples=None`). See [`resolve_predict_args`](../../boltr-backend-tch/src/predict_args.rs).

Optional checkpoint flags: `confidence_prediction`, `affinity_prediction`, `affinity_mw_correction` (see [`Boltz2Hparams`](../../boltr-backend-tch/src/boltz_hparams.rs)); used by [`Boltz2Model::from_hparams_json`](../../boltr-backend-tch/src/boltz2/model.rs).

## Featurizer output keys

`Boltz2Featurizer.process` merges several maps. The union includes (non-exhaustive; see [`featurizerv2.py`](../boltz-reference/src/boltz/data/feature/featurizerv2.py) around the final `return { **token_features, **atom_features, ... }`):

- Token / atom / MSA blocks: `process_token_features`, `process_atom_features`, `process_msa_features`, `process_template_features` (or dummy templates), optional symmetry and constraint maps.
- Templates also expose `template_force` / `template_force_threshold` (per-template row, rank-1) for reference potentials.
- Affinity runs add `profile_affinity` / `deletion_mean_affinity` for the input embedder plus a second `process_msa_features` pass on cropped tokens; `affinity_mw` when `compute_affinity=True`.

Use a **golden export** from Python (pick one small input, dump `collate` output keys + shapes + dtypes) as the authoritative checklist while porting `boltr-io` featurization.

### Checked-in contract artifacts (Phase 1)

| Artifact | Location |
|----------|----------|
| Key manifest (names, ranks, nominal shapes) | [`boltr-io/tests/fixtures/collate_golden/manifest.json`](../boltr-io/tests/fixtures/collate_golden/manifest.json) — `trunk_smoke_safetensors_keys` includes `template_force` / `template_force_threshold` when using the current featurizer. |
| Minimal collated tensors for trunk smoke (`s_inputs`, MSA block, template dummies) | [`boltr-io/tests/fixtures/collate_golden/trunk_smoke_collate.safetensors`](../boltr-io/tests/fixtures/collate_golden/trunk_smoke_collate.safetensors) — consumed by [`boltr-backend-tch/tests/collate_predict_trunk.rs`](../boltr-backend-tch/tests/collate_predict_trunk.rs) (`predict_step_trunk` + `MsaFeatures`). Path helper: [`trunk_smoke_collate_path()`](../boltr-io/src/collate_golden.rs). |
| ALA `process_token_features` golden (per-example + `B=1` collated) | [`token_features_ala_golden.safetensors`](../boltr-io/tests/fixtures/collate_golden/token_features_ala_golden.safetensors), [`token_features_ala_collated_golden.safetensors`](../boltr-io/tests/fixtures/collate_golden/token_features_ala_collated_golden.safetensors) — regenerate: `cargo run -p boltr-io --bin write_token_features_ala_golden`; Python: [`scripts/dump_token_features_ala_golden.py`](../scripts/dump_token_features_ala_golden.py) with `BOLTZ_SRC`. |
| `MSAModule` weights + I/O (opt-in Rust test) | [`msa_module_golden.safetensors`](../boltr-backend-tch/tests/fixtures/msa_module_golden/msa_module_golden.safetensors) — [`scripts/export_msa_module_golden.py`](../scripts/export_msa_module_golden.py); run: `BOLTR_RUN_MSA_GOLDEN=1` + `scripts/cargo-tch test … msa_module_allclose_python_golden`. |
| One `PairformerLayer` weights + I/O (opt-in Rust test) | [`pairformer_layer_golden.safetensors`](../boltr-backend-tch/tests/fixtures/pairformer_golden/pairformer_layer_golden.safetensors) — [`scripts/export_pairformer_golden.py`](../scripts/export_pairformer_golden.py); run: `BOLTR_RUN_PAIRFORMER_GOLDEN=1` + `scripts/cargo-tch test … pairformer_layer_allclose_python_golden`. |
| Pinned `Boltz2Model` smoke weights (strict load) | [`boltz2_smoke.safetensors`](../boltr-backend-tch/tests/fixtures/boltz2_smoke/boltz2_smoke.safetensors) — regen: `cargo run -p boltr-backend-tch --bin gen_boltz2_smoke_safetensors --features tch-backend`. |
| Trunk init (`rel_pos`, `s_init` linear) opt-in golden | Generate [`scripts/export_trunk_init_golden.py`](../scripts/export_trunk_init_golden.py) → `boltr-backend-tch/tests/fixtures/trunk_init_golden/trunk_init_golden.safetensors`; run: `BOLTR_RUN_TRUNK_INIT_GOLDEN=1` + `scripts/cargo-tch test … trunk_init_allclose_python_golden`. |
| Full trunk `InputEmbedder` forward opt-in golden | [`scripts/export_input_embedder_golden.py`](../scripts/export_input_embedder_golden.py) → `boltr-backend-tch/tests/fixtures/input_embedder_golden/input_embedder_golden.safetensors`; run: `BOLTR_RUN_INPUT_EMBEDDER_GOLDEN=1` + `scripts/cargo-tch test … input_embedder_allclose_python_golden`. |

Regenerate trunk smoke safetensors: `cargo run -p boltr-io --bin write_collate_golden`.  
Python script: [`scripts/dump_collate_golden.py`](../scripts/dump_collate_golden.py).

## Checkpoint keys

Lightning checkpoints store a nested `state_dict`. For Rust, export with [`scripts/export_checkpoint_to_safetensors.py`](../scripts/export_checkpoint_to_safetensors.py) and map names explicitly (strip `model.` if present). `boltr-backend-tch` currently validates loading with `s_init.weight` as a first spike.

## CUDA / kernels

Python optional `[cuda]` extra installs **cuequivariance** kernels for faster triangle ops. LibTorch via `tch-rs` does not load those; Boltr targets the **same math** as the PyTorch fallback path (`use_kernels=False`). GPU acceleration uses **CUDA LibTorch** for standard torch ops.

## 6. Golden tests, tolerances, and regression

### 6.1 Checked-in artifacts

See **Checked-in contract artifacts (Phase 1)** under [Featurizer output keys](#featurizer-output-keys) above for paths to manifests, safetensors, and regeneration commands.

### 6.2 Collate + MSA (Rust)

- **Inference collate:** [`collate_inference_batches`](../boltr-io/src/collate_pad.rs) + [`pad_to_max_f32`](../boltr-io/src/collate_pad.rs) mirror Python [`pad_to_max`](../../boltz-reference/src/boltz/data/pad.py) / [`inferencev2.collate`](../../boltz-reference/src/boltz/data/module/inferencev2.py) for tensor keys; excluded keys are collected per-example in [`InferenceCollateResult::excluded`](../boltr-io/src/collate_pad.rs).
- **MSA:** [`construct_paired_msa`](../boltr-io/src/featurizer/msa_pairing.rs) + [`process_msa_features`](../boltr-io/src/featurizer/process_msa_features.rs); integration via [`msa_features_from_inference_input`](../boltr-io/src/inference_dataset.rs). Optional Python golden stub: [`scripts/dump_msa_features_golden.py`](../scripts/dump_msa_features_golden.py).
- **Atom features:** [`process_atom_features`](../boltr-io/src/featurizer/process_atom_features.rs) module placeholder (RDKit-dependent); golden-first when implementing.

### 6.5 Numerical tolerances

**Authoritative registry:** [NUMERICAL_TOLERANCES.md](NUMERICAL_TOLERANCES.md) lists every `rtol` / `atol` for featurizer tests, post-collate parity, backend module goldens, and the optional end-to-end regression script.

Summary:

| Class | Typical policy |
|-------|------------------|
| Token / MSA featurizer goldens | `rtol ≈ 1e-5`, `atol ≈ 1e-6` |
| Atom features | `rtol ≈ 1e-4`, `atol ≈ 1e-5` |
| Pairformer / MSA module / trunk init / input embedder (opt-in) | `rtol = 1e-4`, `atol = 1e-5` in `boltr-backend-tch` tests |
| Post-collate trunk smoke | atol `1e-6`, rtol `1e-5` ([`post_collate_golden.rs`](../boltr-io/tests/post_collate_golden.rs)) |
| Full `boltz predict` vs `boltr predict` | env defaults in [`scripts/regression_tol.env.example`](../scripts/regression_tol.env.example) (much looser than layer goldens) |

### 6.6 Regression harness (optional)

When both CLIs are available, set `BOLTR_REGRESSION=1` and run [`scripts/regression_compare_predict.sh`](../scripts/regression_compare_predict.sh). Prerequisites and tolerance overrides are documented in that script and in [scripts/README.md](../scripts/README.md). The comparator logic lives in [`scripts/regression_compare_outputs.py`](../scripts/regression_compare_outputs.py).

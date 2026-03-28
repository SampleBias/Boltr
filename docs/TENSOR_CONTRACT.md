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

## Featurizer output keys

`Boltz2Featurizer.process` merges several maps. The union includes (non-exhaustive; see [`featurizerv2.py`](../boltz-reference/src/boltz/data/feature/featurizerv2.py) around the final `return { **token_features, **atom_features, ... }`):

- Token / atom / MSA blocks: `process_token_features`, `process_atom_features`, `process_msa_features`, `process_template_features` (or dummy templates), optional symmetry and constraint maps.
- Affinity runs add a second `process_msa_features` pass and `affinity_mw` when `compute_affinity=True`.

Use a **golden export** from Python (pick one small input, dump `collate` output keys + shapes + dtypes) as the authoritative checklist while porting `boltr-io` featurization.

### Checked-in contract artifacts (Phase 1)

| Artifact | Location |
|----------|----------|
| Key manifest (names, ranks, nominal shapes) | [`boltr-io/tests/fixtures/collate_golden/manifest.json`](../boltr-io/tests/fixtures/collate_golden/manifest.json) |
| Minimal collated tensors for trunk smoke (`s_inputs`, MSA block, template dummies) | [`boltr-io/tests/fixtures/collate_golden/trunk_smoke_collate.safetensors`](../boltr-io/tests/fixtures/collate_golden/trunk_smoke_collate.safetensors) — consumed by [`boltr-backend-tch/tests/collate_predict_trunk.rs`](../boltr-backend-tch/tests/collate_predict_trunk.rs) (`predict_step_trunk` + `MsaFeatures`). Path helper: [`trunk_smoke_collate_path()`](../boltr-io/src/collate_golden.rs). |
| ALA `process_token_features` golden (per-example + `B=1` collated) | [`token_features_ala_golden.safetensors`](../boltr-io/tests/fixtures/collate_golden/token_features_ala_golden.safetensors), [`token_features_ala_collated_golden.safetensors`](../boltr-io/tests/fixtures/collate_golden/token_features_ala_collated_golden.safetensors) — regenerate: `cargo run -p boltr-io --bin write_token_features_ala_golden`; Python: [`scripts/dump_token_features_ala_golden.py`](../scripts/dump_token_features_ala_golden.py) with `BOLTZ_SRC`. |
| `MSAModule` weights + I/O (opt-in Rust test) | [`msa_module_golden.safetensors`](../boltr-backend-tch/tests/fixtures/msa_module_golden/msa_module_golden.safetensors) — [`scripts/export_msa_module_golden.py`](../scripts/export_msa_module_golden.py); run: `BOLTR_RUN_MSA_GOLDEN=1` + `scripts/cargo-tch test … msa_module_allclose_python_golden`. |
| One `PairformerLayer` weights + I/O (opt-in Rust test) | [`pairformer_layer_golden.safetensors`](../boltr-backend-tch/tests/fixtures/pairformer_golden/pairformer_layer_golden.safetensors) — [`scripts/export_pairformer_golden.py`](../scripts/export_pairformer_golden.py); run: `BOLTR_RUN_PAIRFORMER_GOLDEN=1` + `scripts/cargo-tch test … pairformer_layer_allclose_python_golden`. |
| Pinned `Boltz2Model` smoke weights (strict load) | [`boltz2_smoke.safetensors`](../boltr-backend-tch/tests/fixtures/boltz2_smoke/boltz2_smoke.safetensors) — regen: `cargo run -p boltr-backend-tch --bin gen_boltz2_smoke_safetensors --features tch-backend`. |

Regenerate trunk smoke safetensors: `cargo run -p boltr-io --bin write_collate_golden`.  
Python script: [`scripts/dump_collate_golden.py`](../scripts/dump_collate_golden.py).

## Checkpoint keys

Lightning checkpoints store a nested `state_dict`. For Rust, export with [`scripts/export_checkpoint_to_safetensors.py`](../scripts/export_checkpoint_to_safetensors.py) and map names explicitly (strip `model.` if present). `boltr-backend-tch` currently validates loading with `s_init.weight` as a first spike.

## CUDA / kernels

Python optional `[cuda]` extra installs **cuequivariance** kernels for faster triangle ops. LibTorch via `tch-rs` does not load those; Boltr targets the **same math** as the PyTorch fallback path (`use_kernels=False`). GPU acceleration uses **CUDA LibTorch** for standard torch ops.

## Numerical tolerances (Phase 7)

Document per-test `rtol` / `atol` as goldens land ([TODO.md](../TODO.md) §6.5 Testing):

| Class | Typical policy |
|-------|------------------|
| Embeddings / linear outputs | `rtol ≈ 1e-5`, `atol ≈ 1e-6` (see `token_features_golden.rs`) |
| Pairformer / MSA module (opt-in goldens) | match export script + same dtype (F32); tighten if CPU/CUDA diverge |
| Sampling / diffusion | looser; compare distributions or fixed-seed step parity |
| Collated batch dict | per-key in `#[test]` once Rust `collate` matches Python |

Add a **regression harness** (`boltz predict` vs `boltr predict`) when the CLI pipeline is complete (placeholder: [`scripts/regression_compare_predict.sh`](../scripts/regression_compare_predict.sh)).

### Collate + MSA (Rust)

- **Inference collate:** [`collate_inference_batches`](../boltr-io/src/collate_pad.rs) + [`pad_to_max_f32`](../boltr-io/src/collate_pad.rs) mirror Python [`pad_to_max`](../../boltz-reference/src/boltz/data/pad.py) / [`inferencev2.collate`](../../boltz-reference/src/boltz/data/module/inferencev2.py) for tensor keys; excluded keys are collected per-example in [`InferenceCollateResult::excluded`](../boltr-io/src/collate_pad.rs).
- **MSA:** [`construct_paired_msa`](../boltr-io/src/featurizer/msa_pairing.rs) + [`process_msa_features`](../boltr-io/src/featurizer/process_msa_features.rs); integration via [`msa_features_from_inference_input`](../boltr-io/src/inference_dataset.rs). Optional Python golden stub: [`scripts/dump_msa_features_golden.py`](../scripts/dump_msa_features_golden.py).
- **Atom features:** [`process_atom_features`](../boltr-io/src/featurizer/process_atom_features.rs) module placeholder (RDKit-dependent); golden-first when implementing.

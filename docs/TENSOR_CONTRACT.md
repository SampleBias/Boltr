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

## Checkpoint keys

Lightning checkpoints store a nested `state_dict`. For Rust, export with [`scripts/export_checkpoint_to_safetensors.py`](../scripts/export_checkpoint_to_safetensors.py) and map names explicitly (strip `model.` if present). `boltr-backend-tch` currently validates loading with `s_init.weight` as a first spike.

## CUDA / kernels

Python optional `[cuda]` extra installs **cuequivariance** kernels for faster triangle ops. LibTorch via `tch-rs` does not load those; Boltr targets the **same math** as the PyTorch fallback path (`use_kernels=False`). GPU acceleration uses **CUDA LibTorch** for standard torch ops.

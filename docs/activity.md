# Boltr — activity log

Chronological notes for **what shipped** and **when**. For the live backlog, use **[TODO.md](../TODO.md)**. For rolling featurizer notes, see **[tasks/todo.md](../tasks/todo.md)**.

---

## How to read this file

| Section | Contents |
|--------|----------|
| [Milestones (by theme)](#milestones-by-theme) | Backend, I/O, featurizer — major deliverables |
| [Timeline (chronological)](#timeline-chronological) | Dated entries, shortest path through history |
| [Current snapshot](#current-snapshot-march-2026) | Where the repo stands for the next phase |

---

## Milestones (by theme)

### A. Backend (`boltr-backend-tch`) — trunk and layers

- **Pairformer stack (§5.5):** `AttentionPairBiasV2`, triangular mult/attn (fallback), `Transition`, `OuterProductMean`, `PairformerLayer`, `PairformerModule`. Doc: [PAIRFORMER_IMPLEMENTATION.md](PAIRFORMER_IMPLEMENTATION.md).
- **TrunkV2:** Owns pairformer, recycling projections/norms, `forward_pairformer`, recycling loop. [boltz2/trunk.rs](../boltr-backend-tch/src/boltz2/trunk.rs).
- **Boltz2Model:** Single `VarStore`, `forward_trunk`, `predict_step_trunk` (trunk-only; no full `predict_step` yet). [boltz2/model.rs](../boltr-backend-tch/src/boltz2/model.rs).
- **Embeddings / init:** `RelativePositionEncoder`, `token_bonds` (+ optional type), `ContactConditioning`, partial `InputEmbedder` (res_type + msa_profile + external atom repr `a`).
- **MSAModule:** Real stack (`PairWeightedAveraging`, `OuterProductMeanMsa`, `PairformerNoSeqLayer`); golden export + opt-in Rust test (`BOLTR_RUN_MSA_GOLDEN=1`).
- **TemplateModule:** Stub (no template bias); still TBD for parity.
- **Integration:** [collate_predict_trunk.rs](../boltr-backend-tch/tests/collate_predict_trunk.rs) loads `trunk_smoke_collate.safetensors` → `predict_step_trunk` + `MsaFeatures`.
- **Tooling:** `scripts/cargo-tch`, `scripts/with_dev_venv.sh` — `LD_LIBRARY_PATH` for LibTorch when using PyTorch’s `torch/lib`.

### B. I/O (`boltr-io`) — preprocess-shaped data

- **Structures:** `StructureV2` tables, [structure_v2_npz.rs](../boltr-io/src/structure_v2_npz.rs) read/write; ALA fixtures; token batch `.npz` ([token_npz.rs](../boltr-io/src/token_npz.rs)).
- **MSA:** A3M/CSV parse, MSA `.npz`, `boltr msa-to-npz`, golden verification workflow.
- **Tokenizer:** `tokenize_structure`, `TokenData` / bonds on `StructureV2Tables` (partial vs full Python `Tokenized` / template loop).
- **Featurizer:** `process_token_features`, `process_msa_features`, `process_atom_features` (canonical AA + nucleic paths, `AtomRefDataProvider`), dummy templates; inference helpers on `Boltz2InferenceInput`.
- **Collate:** `FeatureBatch`, `pad_to_max_f32`, `collate_inference_batches`, manifest + goldens under `tests/fixtures/collate_golden/`.
- **Inference:** `load_input` + manifest JSON; `trunk_smoke_feature_batch_from_inference_input` merges token + MSA + atoms + dummy templates (no `s_inputs` — model-side).

### C. CLI & repo hygiene

- **`boltr download`**, partial **`predict`**, YAML parsing, `minimal_protein.yaml` fixture.
- **Makefile / scripts:** checkpoint export, hparams export, safetensors verify, regression script placeholders.

---

## Timeline (chronological)

| Period | Focus |
|--------|--------|
| **2025-03-22** | Pairformer stack + TrunkV2 wiring; project scaffold. |
| **2026-03-23–25** | Relative position, bonds, contact conditioning, partial input embedder, MSAModule, Boltz2Model APIs, collate → `predict_step_trunk` test, LibTorch env fixes. |
| **2026-03-23** | `boltr-io` expansion: constants, ref_atoms, MSA npz, structure npz, inference_dataset, collate goldens, tooling. |
| **2026-03-24** | Pairformer layer Python golden + mask fix; attention pairwise mask broadcast. |
| **2026-03-27** | Inference collate + MSA + merged `FeatureBatch`; atom golden fixtures; two-example MSA collate golden; `featurizer/mod.rs` repair + type fixes; **`atom_features_from_inference_input`** + merge + partial atom allclose vs Python safetensors (`ATOM_GOLDEN_SKIP_*` for RDKit/geometry keys); manifest `atom_features_ala_golden_keys`. |

---

## Current snapshot (March 2026)

**In good shape**

- Trunk forward path through pairformer + optional MSA; numerical goldens for pairformer layer, MSA module (opt-in), pairformer mask behavior documented.
- End-to-end **data path** from preprocess-shaped inputs to a **single merged `FeatureBatch`** (token, MSA, atom, dummy templates) for smoke tests; `load_input` + ALA smoke fixtures.

**Not done (see [TODO.md](../TODO.md))**

- Full **`predict_step`** (diffusion, confidence, affinity).
- Real **template** featurizer + **TemplateModule** (backend stub).
- **Writers** (mmcif/pdb, prediction layout).
- Full **schema/CCD** in Rust, full **collate dict** allclose vs Python, full **atom** allclose on identical NPZ + mols.

---

*This file is the narrative checkpoint; [TODO.md](../TODO.md) is the actionable checklist.*


## 2026-03-27 21:18 - Session Started
- Project structure files verified
- Resumed work on existing project
- Todo.md updated with new session section
- PROJECT_README.md context checked
- Ready for continued development



## 2026-03-28 07:00 - Session Started
- Project structure files verified
- Resumed work on existing project
- Todo.md updated with new session section
- PROJECT_README.md context checked
- Ready for continued development



## 2026-03-28 08:49 - Session Started
- Project structure files verified
- Resumed work on existing project
- Todo.md updated with new session section
- PROJECT_README.md context checked
- Ready for continued development


## 2026-03-28 09:30 - Pairformer Stack Dropout/Mask Audit Completed

**Context:** Section 5.5 of TODO.md identified incomplete dropout/mask audit.

**Issues Found:**
1. Rust `PairformerLayer` did not respect training mode (always applied dropout)
2. Dropout mask used full tensor instead of slice-based subsample
3. Used `>` comparison instead of `>=` (Python uses `>=`)
4. Missing `set_training` API on PairformerModule and TrunkV2

**Changes Made:**

**1. boltr-backend-tch/src/layers/pairformer.rs:**
- Added `training: bool` parameter to `PairformerLayer::forward()`
- Fixed `create_dropout_mask()` to use slice-based approach `z[:, :, 0:1, 0:1]`
- Fixed `create_dropout_mask_columnwise()` to use slice-based approach `z[:, 0:1, :, 0:1]`
- Changed comparison from `gt_tensor()` to `ge_tensor()` to match Python
- Updated all dropout mask applications to respect `training` flag
- Added `training` field to `PairformerModule` struct
- Added `set_training()` method to `PairformerModule`
- Fixed chunking logic: training mode uses `chunk_size=None`, eval uses threshold-based

**2. boltr-backend-tch/src/layers/training_tests.rs:** (NEW FILE)
- Added `test_pairformer_layer_training_mode()` - verifies dropout application
- Added `test_pairformer_layer_eval_mode_no_dropout()` - verifies determinism
- Added `test_pairformer_module_training_mode()` - tests mode switching
- Added `test_pairformer_module_chunk_size_training()` - tests chunking logic
- Added `test_dropout_mask_shape_broadcast()` - verifies mask shapes

**3. boltr-backend-tch/src/boltz2/trunk.rs:**
- Added `training: bool` field to `TrunkV2` struct
- Added `set_training()` method to cascade training flag to pairformer
- Enables training mode control at trunk level

**4. boltr-backend-tch/tests/pairformer_golden.rs:**
- Updated golden test call to include `training=false` parameter

**Test Results:**
- ✅ All 8 pairformer tests pass
- ✅ Golden test passes (BOLTR_RUN_PAIRFORMER_GOLDEN=1)
- ✅ All 36 backend tests pass (no regressions)
- ✅ Build succeeds with only warnings (unused fields)

**Documentation Created:**
- docs/PAIRFORMER_DROPOUT_FIX.md - Comprehensive fix documentation

**Status:**
- ✅ **Section 5.5 Pairformer stack** - DROPOUT/MASK AUDIT **COMPLETED**
- All implementation now matches Python reference behavior
- Ready for training/inference mode switching

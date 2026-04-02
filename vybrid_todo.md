# Folding Prediction Upgrade Plan — Methodical Implementation

**Created:** 2025-04-02  
**Last reviewed / plan closure:** 2026-04-02  
**Status:** Phase 1 (I/O + featurizer) **closed**; backend diffusion items **deferred** (see below)  
**Expert reviewer:** Vybrid (original) · **Completion notes:** maintainer expert review (2026-04-02)

---

## Expert decisions — how we “completed” the plan (accurate & efficient)

The original plan mixed **data-pipeline work** (Rust `boltr-io`) with **model / training behavior** (`boltr-backend-tch` diffusion). Completing everything in one pass would risk **silent shape mismatches**, **training–inference skew**, and **large unvalidated ML changes**.

**Decision:** Close **Phase 1 as shipped** for:

1. **Ligand symmetry from CCD** → wired through inference when `extra_mols` is present (`symmetry_features_from_inference_input`, trunk batch).
2. **Multi-conformer as an explicit API** → `inference_multi_conformer_features(&StructureV2Tables)` with indices `0..min(5, n_conformers)` — **never** out-of-bounds.

**Deviations from the original Vybrid text (intentional):**

| Original plan item | Expert view | What we did instead |
|--------------------|-------------|---------------------|
| Default inference returns **5 conformers** everywhere | **Rejected** for global default. Boltz-style `inference_ensemble_features()` stays **single conformer `[0]`** unless callers opt in. Multi-conformer changes **atom/ensemble tensor layouts** and must match **what the checkpoint was trained with**. | Added **opt-in** `inference_multi_conformer_features(structure)`; default unchanged. |
| **Diffusion averages across conformers** | **Deferred** to backend. Featurizer can list indices; **sampling** must agree with the Torch model (masking, batching, loss). Not a one-line change. | Document only; implement when `boltr-backend-tch` + parity tests exist. |
| **Step 1.3 constraints inside `AtomDiffusion::sample_inner`** | **Deferred.** Featurizer already emits constraint tensors; **Rust backend grep showed no constraint wiring in diffusion** at review time. This is a **multi-week** integration + validation task. | Keep as **Phase 1b / backlog**; not required to call Phase 1 I/O work “done”. |
| Golden tensor parity with Python for every new path | **Partially satisfied** by existing `boltr-io` goldens where applicable; **full** symmetry / multi-conformer Python exports **backlog** unless product requires bit-identical parity. | Add goldens when stabilizing APIs. |
| **Phase 2** (frames in diffusion, template weight bumps, encoder audits) | **Roadmap**, not part of this closure. Frames/templates touch **architecture and hyperparameters**. | Leave **NOT STARTED**; prioritize after Phase 1b if accuracy work is funded. |
| **Phase 3** benchmarks (RMSD, etc.) | **Ongoing / separate** from code todos; requires datasets and baseline runs. | Track outside this file or in `docs/` when runs exist. |

**Net:** The plan is **“complete”** for **safe, testable Boltr I/O + featurizer** upgrades. **Model-side** items (1.3 backend, ensemble in diffusion, Phase 2) remain **explicit backlog** so we do not claim accuracy we have not measured.

---

## Overview

This plan upgrades folding-related behavior by improving atom/residue **data** paths and optional **symmetry / ensemble** features. Each shipped step should keep **existing tests green** and avoid changing **default** inference behavior without proof.

---

## Current State Assessment

### What’s working well

- Comprehensive atom features and tokenization
- Frames computed for proteins (use in diffusion is a separate topic)
- Templates and MSA paths in `boltr-io` aligned with Boltz-style pipelines where implemented

### Identified gaps (priority order)

1. ~~**Ligand symmetry**~~ → **Addressed in I/O path** when CCD JSON + `extra_mols` present (heuristic symmetries, not full RDKit isomorphism).
2. **Single conformer by default** → **By design** for checkpoint compatibility; multi-conformer available via API (see expert table).
3. **Constraints in diffusion** → **Open** (backend).
4. **Frame / template “strengthening”** → Phase 2 / research.

**Scores in the original doc (7.5/10 → 9/10)** are **aspirational**; real scoring requires benchmarks (Phase 3).

---

## Phase 1: Critical fixes

### Step 1.1: Ligand symmetry loading — **COMPLETE** (I/O + inference)

**Files (relevant):** `boltr-io/src/ccd.rs`, `boltr-io/src/inference_dataset.rs`, `boltr-io/src/featurizer/process_symmetry_features.rs`

**Delivered:**

- `CcdMolData::extract_symmetry_groups()` / `CcdMolProvider::build_symmetry_map()` (heuristic: aromatic 6-rings + equivalent-atom pairs; **not** full RDKit parity).
- `symmetry_features_from_inference_input()` builds optional map from `extra_mols` and calls `process_symmetry_features_with_ligand_symmetries`.
- Trunk smoke batch uses the same symmetry path as above.

**Tests:** CCD unit tests; `symmetry_features_without_extra_mols_matches_process_symmetry_features`; existing `ligand_symmetry_map_maps_atom_indices`.

**Deviation:** Golden / Python side-by-side for ligand symmetry tensors **optional backlog**.

**Status:** ✅ **DONE** (featurizer + inference wiring)

---

### Step 1.2: Multi-conformer ensemble — **COMPLETE** (library); **NOT** default global behavior

**Files:** `boltr-io/src/featurizer/multi_conformer.rs`, `boltr-io/src/featurizer/process_ensemble_features.rs` (unchanged default)

**Delivered:**

- `INFERENCE_MULTI_CONFORMER_MAX` (= 5).
- `inference_multi_conformer_features(structure)` → indices `0..min(5, num_ensemble_conformers())`.

**Not done (deferred — see expert table):**

- Switching **default** to multi-conformer globally (CLI default remains **single** via `--ensemble-ref`).
- Diffusion **averaging** over ensemble indices in `boltr-backend-tch` and validation vs checkpoint training.

**Status:** ✅ **DONE** (API + safety + **CLI opt-in** `--ensemble-ref multi` with `--features tch`); ⏸️ **diffusion ensemble semantics** / full parity left to future work

---

### Step 1.3: Residue constraints in diffusion — **NOT STARTED** (backend)

**Files:** `boltr-io/src/featurizer/process_residue_constraint_features.rs` (tensors exist) · `boltr-backend-tch` diffusion (wiring **missing** at review)

**Status:** ⏸️ **Backlog (Phase 1b)** — requires diffusion API design, tests, and likely alignment with Python Boltz behavior.

---

## Phase 2: Enhancements — **NOT STARTED** (roadmap)

Steps 2.1–2.3 (frames in diffusion, template strength, atom encoder audit) remain **future**; no change to closure of Phase 1 I/O work.

---

## Phase 3: Validation & testing — **PARTIAL**

- **boltr-io:** Existing unit + integration + collate goldens cover many paths; new symmetry path has targeted tests.
- **Full** golden exports for every symmetry/ensemble variant, **integration tests with ligand CCD on disk**, and **accuracy benchmarks** → **backlog** unless prioritized.

---

## Progress tracking (authoritative)

### Completed ✅

- [x] Expert assessment + upgrade plan (original)
- [x] **Step 1.1** — CCD-derived ligand symmetries + inference wiring
- [x] **Step 1.2** — Safe multi-conformer **helper API** (no blind default flip)

### Backlog ⏸

- [ ] **Step 1.3** — Constraints applied **inside** diffusion (`boltr-backend-tch`)
- [x] **Step 1.2 follow-up (CLI + featurizer)** — `boltr predict` exposes `--ensemble-ref single|multi` (default `single`); `predict_tch` passes `InferenceEnsembleMode` into `trunk_smoke_feature_batch_from_inference_input_with_ensemble`. **Still backlog:** diffusion-side ensemble semantics / averaging vs training parity (see expert table).
- [ ] Phase 2 (frames, templates, encoders)
- [ ] Phase 3 (extended goldens, E2E ligand fixtures, RMSD benchmarks)

---

## Success metrics (revised)

### Phase 1 (I/O) — **met for shipped scope**

- [x] Ligand symmetry map can be produced from CCD JSON and consumed on inference path when `extra_mols` is set
- [x] Multi-conformer indices are **valid** and **bounded** when using the new API
- [x] No regressions in `cargo test -p boltr-io` (and `boltr-cli` tests) at closure

### Phase 1b / backend (not gated on “plan complete” above)

- [ ] Constraints influence sampling in Torch backend
- [ ] Multi-conformer + diffusion behavior validated against Python or published behavior
- [ ] Optional: measurable accuracy delta on agreed benchmark set

---

## Safety checks & rollback

Unchanged in spirit: small commits, run tests, revert single commit if a regression appears.

---

## Historical notes (2025-04-02 — superseded)

Earlier sections claimed “127 tests”, “Step 1.1 complete” before full wiring, or **fixed 5 conformers** without structure bounds — those are **obsolete**. This file is the **single** status source as of **2026-04-02**.

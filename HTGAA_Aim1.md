# HTGAA-26 — Aim 1: Boltr comparison study (basic design)

This document is **Aim 1** for HTGAA-26: a small, reproducible study design to compare **Boltr** (Rust-native Boltz2 inference in this repository) against **upstream Boltz2**, optionally **Boltz-1**, and an **external AlphaFold** workflow. It emphasizes **prediction accuracy** (structure vs reference coordinates), **affinity accuracy** where experimental data exist, and **what Rust-backed tooling is good for** in this project (reproducible CLI, parity discipline, deployment, operational logging).

**Related implementation docs in this repo**

- Prediction parity baseline and fixture names: [docs/PREDICTION_PARITY_BASELINE.md](docs/PREDICTION_PARITY_BASELINE.md)
- `boltr predict` flags and output layout: [docs/PROJECT_README.md](docs/PROJECT_README.md)
- Boltz YAML schema (comments and examples): [boltr-io/src/config.rs](boltr-io/src/config.rs)
- Example YAML with template mmCIF: [boltr-io/tests/fixtures/yaml/template_cif.yaml](boltr-io/tests/fixtures/yaml/template_cif.yaml)
- Example YAML with protein–ligand affinity: [boltr-io/tests/fixtures/yaml/properties_affinity.yaml](boltr-io/tests/fixtures/yaml/properties_affinity.yaml)
- Upstream evaluation context (OpenStructure / benchmarks): [boltz-reference/docs/evaluation.md](boltz-reference/docs/evaluation.md)

---

## 1. Aim 1 statement (narrative)

Boltr implements the Boltz2 predict path in Rust (`load_input` → collate → `Boltz2Model::predict_step`) with a **Boltz-compatible CLI** and serialized outputs (structures, confidence JSON, optional affinity JSON). Aim 1 establishes whether, under **controlled inputs** (same YAML, checkpoint family, preprocess artifacts, and inference hyperparameters), Boltr **matches** upstream Boltz2 on selected outputs; extends comparison to **reference PDB/mmCIF** for global and interface structure quality; and records **operational** outcomes (wall time, success rate, environment pins) that motivate a Rust-native stack alongside scientific metrics.

**Important scope notes**

- Boltr does **not** accept a bare `.pdb` / `.cif` as the sole predict input. Jobs are driven by **Boltz YAML**; PDB/mmCIF appear as **templates** (`templates:` → `cif:` / `pdb:`) and/or as **external reference** structures for scoring after prediction.
- `**boltr eval`** is a **stub**; structural scores (lDDT, DockQ, interface RMSD, and so on) are computed with an **agreed external tool** (see [boltz-reference/docs/evaluation.md](boltz-reference/docs/evaluation.md) and [docs/PROJECT_README.md](docs/PROJECT_README.md)).
- **AlphaFold** is **not** implemented inside Boltr. Any AlphaFold arm uses a separate pipeline (e.g. ColabFold AF2 / AF2-Multimer, or AF3 if licensed). It is a strong comparator for **protein / assembly backbone** quality; it is **not** a drop-in replacement for Boltz2 **small-molecule affinity** prediction. The affinity comparison arm stays **Boltz-family (Boltr vs upstream Boltz2)** plus **experimental** affinities where available.

**Known Rust-bridge limitations** (from [docs/PREDICTION_PARITY_BASELINE.md](docs/PREDICTION_PARITY_BASELINE.md)): template tensors and potentials/steering are not fully wired through the current Rust predict bridge; paired/ensemble affinity parity is an open decision. **Primary endpoints** for Aim 1 should use **Panel A** and **Panel B** below; **Panel C** is exploratory only.

---

## 2. Research questions


| ID      | Question                                                                                                                                                                                                 |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RQ1** | Under fixed YAML, checkpoints, preprocess mode, and resolved predict arguments, does Boltr match upstream Boltz2 on selected **confidence** and **affinity** scalars within predefined tolerances?       |
| **RQ2** | Against a deposited **reference** PDB/mmCIF, how do **Boltr** and **AlphaFold** predictions compare on agreed **global** and/or **interface** metrics (same reference, same alignment protocol)?         |
| **RQ3** | What are **operational** differences (wall-clock time, peak memory if measured, completion vs failure) between Boltr and comparators on the same hardware class, holding inputs as equal as practicable? |


---

## 3. Comparator definitions


| Arm                   | Definition                                                                                              | Role in Aim 1                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Boltr**             | `boltr predict` built with `--features tch`; flags per [docs/PROJECT_README.md](docs/PROJECT_README.md) | **Primary** system under test                                                                                                                           |
| **Boltz2 (upstream)** | Python `boltz-reference` `boltz predict` with matched YAML, checkpoints, preprocess                     | **Parity / numerical** reference                                                                                                                        |
| **Boltz-1**           | Legacy Boltz-1 stack if available in the lab                                                            | **Optional**; do not mix with Boltz2 metrics without clear labeling                                                                                     |
| **AlphaFold**         | External AF2 / ColabFold or AF3 (if licensed)                                                           | **Structure** comparator for protein-centric cases; affinity table **N/A** unless you define a separate non-Boltz affinity (out of default Aim 1 scope) |


---

## 4. Materials — benchmark panels (YAML + PDB/mmCIF)

Align case IDs with the parity fixture names in [docs/PREDICTION_PARITY_BASELINE.md](docs/PREDICTION_PARITY_BASELINE.md) where possible.

### Panel A — `protein_minimal` (primary structure + confidence)

- **Input**: YAML with at least one protein chain (`sequences`). Optionally add `templates:` pointing to a small reference `**.cif`** or `**.pdb`** (pattern: [boltr-io/tests/fixtures/yaml/template_cif.yaml](boltr-io/tests/fixtures/yaml/template_cif.yaml)).
- **Preprocess**: run both `native` and `boltz` if the study compares preprocess behavior; otherwise pick one and hold it fixed across arms.
- **Required Boltr outputs** (per parity doc): structure, `boltr_predict_complete.txt` (or equivalent completion marker), resolved predict args artifact (`boltr_predict_args.json` where applicable).

### Panel B — `protein_ligand_affinity` (primary affinity)

- **Input**: YAML with protein + ligand (`ccd` or `smiles`) and `properties: - affinity: binder: …` (pattern: [boltr-io/tests/fixtures/yaml/properties_affinity.yaml](boltr-io/tests/fixtures/yaml/properties_affinity.yaml)).
- **Preprocess**: parity doc prefers `**boltz`** for this fixture.
- **Required outputs**: structure, **affinity JSON** (`affinity_pred_value`, `affinity_probability_binary`, selected-sample metadata when present), MW correction flag if used (`--affinity-mw-correction` / hparams).
- **Ground truth**: experimental pKd / pKi / IC50 (with citation) when available; otherwise record predictions only and use **binary** concordance vs `affinity_probability_binary` as a secondary descriptor.

### Panel C — exploratory only (`template_complex`, `constrained_complex`)

- Use only if explicitly in scope. Document **unsupported** or partial behavior per [docs/PREDICTION_PARITY_BASELINE.md](docs/PREDICTION_PARITY_BASELINE.md) (templates/steering in Rust bridge).

### Reference structures for scoring

- For each case, record a **reference PDB ID** (or path to an in-repo **reference `.cif` / `.pdb`**) used **only** for post-hoc superposition and metrics, not necessarily as a Boltz template.

---

## 5. Methods (executable outline)

### 5.1 Software bill of materials (record per run)

- Boltr **git commit** hash; `boltr --version` (if available).
- `**boltr doctor`** output (JSON recommended): LibTorch / CUDA resolution — see `boltr doctor` in [boltr-cli/src/doctor.rs](boltr-cli/src/doctor.rs).
- Checkpoint paths: `--checkpoint`, `--affinity-checkpoint`; cache directory `--cache-dir` or `BOLTZ_CACHE` / `BOLTR_CACHE` per [docs/PROJECT_README.md](docs/PROJECT_README.md).
- Upstream Boltz2: Python and package versions; exact `boltz predict` command line.
- AlphaFold: pipeline name and version (e.g. ColabFold commit / Docker image).

### 5.2 Hold constant across Boltr and Boltz2

Mirror [docs/PREDICTION_PARITY_BASELINE.md](docs/PREDICTION_PARITY_BASELINE.md) **Comparison Checklist**:

- `recycling_steps`, `sampling_steps`, `diffusion_samples`, `max_parallel_samples`
- Preprocess mode and whether upstream produced the same preprocess bundle
- Selected structure sample and ranking metric (e.g. IPTM-based selection)
- Confidence availability and key scores (`confidence_*_model_*.json`: e.g. `confidence_score`, `iptm`, `ligand_iptm`, `pair_chains_iptm`)
- Affinity value, binary probability, MW correction, selected sample
- Completion status (`predict_step_complete` vs fallback)

**Boltr CLI flags** (non-exhaustive; full list: [docs/PROJECT_README.md](docs/PROJECT_README.md)): `--device`, `--cache-dir`, `--use-msa-server`, `--msa-server-url`, `--affinity`, `--affinity-mw-correction`, `--recycling-steps`, `--sampling-steps`, `--diffusion-samples`, `--max-parallel-samples`, `--output-format` (`mmcif` | `pdb`), `--write-full-pae`, `--write-full-pde`, `--override`.

### 5.3 MSA policy (Boltr vs AlphaFold)

State explicitly whether **the same** `.a3m` (or server) seeds both Boltr and AF2, or document divergence as a **limitation** for RQ2.

### 5.4 Structural accuracy (external)

1. Align predicted model (Boltr / Boltz2 / AlphaFold) to the **reference** PDB/mmCIF using one agreed method (e.g. OpenStructure, DockQ pipeline, PyMOL superpose — pick one per study and record tool + version).
2. Record **global** and, if applicable, **interface** metrics (examples: lDDT, TM-score, DockQ, interface RMSD).

### 5.5 Affinity accuracy

- If experimental affinity exists: report error per compound, and optionally **Spearman** correlation across Panel B after log-transform where appropriate.
- If not: tabulate `affinity_pred_value` and `affinity_probability_binary` for Boltr vs Boltz2 only.

### 5.6 Opt-in parity gate (optional automation)

From [docs/PREDICTION_PARITY_BASELINE.md](docs/PREDICTION_PARITY_BASELINE.md):

```bash
BOLTR_RUN_PREDICT_PARITY=1 scripts/cargo-tch test -p boltr-cli --features tch predict_parity_fixture_matrix_is_documented
```

Full numerical parity remains **environment- and GPU-dependent**.

---

## 6. Endpoints and pass / fail criteria (fill thresholds after pilot)

Use checkboxes during analysis.

### Parity (Boltr vs Boltz2)

- Confidence scalars within agreed tolerance (define rtol/atol per field after pilot).
- Affinity `affinity_pred_value` within agreed tolerance when affinity module enabled.
- Same ranking of diffusion samples when using the same ranking metric.

### Structure vs reference

- Median global lDDT ≥ **___** (set X after pilot) for N/M cases on Panel A.
- Interface metric (e.g. DockQ / interface RMSD) meets **___** for selected complexes.

### Operations (Rust / deployment narrative)

- Clean completion on Panel A and Panel B: **___** % of runs.
- Median wall time Boltr vs Boltz2 ≤ **___** × (hypothesis: often similar GPU kernel path; document I/O and preprocess differences).

---

## 7. Recording forms

Copy tables into a lab notebook or spreadsheet; keep one row per **run** (not only per case) if hyperparameters vary.

### 7.1 Run log


| run_id | date | machine_id | gpu_model | boltr_commit | boltr_cmd | boltz2_cmd | af_cmd | preprocess_mode | wall_s_boltr | wall_s_boltz2 | wall_s_af | peak_ram_gb | status | notes |
| ------ | ---- | ---------- | --------- | ------------ | --------- | ---------- | ------ | --------------- | ------------ | ------------- | --------- | ----------- | ------ | ----- |
|        |      |            |           |              |           |            |        |                 |              |               |           |             |        |       |


### 7.2 Structure metrics (vs reference)


| case_id | reference_pdb_id | ref_file_path | metric_tool_ver | metric_name | boltr_value | boltz2_value | af_value | pass_criterion | notes |
| ------- | ---------------- | ------------- | --------------- | ----------- | ----------- | ------------ | -------- | -------------- | ----- |
|         |                  |               |                 |             |             |              |          |                |       |


### 7.3 Affinity (Panel B)


| case_id | exp_pkd_or_pki | exp_citation | boltr_affinity_pred | boltr_prob_binary | boltz2_affinity_pred | boltz2_prob_binary | delta_pred | rank_concordant_y_n | notes |
| ------- | -------------- | ------------ | ------------------- | ----------------- | -------------------- | ------------------ | ---------- | ------------------- | ----- |
|         |                |              |                     |                   |                      |                    |            |                     |       |


### 7.4 Parity checklist (per fixture / run pair)


| field                              | boltr_value | boltz2_value | match_y_n |
| ---------------------------------- | ----------- | ------------ | --------- |
| recycling_steps                    |             |              |           |
| sampling_steps                     |             |              |           |
| diffusion_samples                  |             |              |           |
| max_parallel_samples               |             |              |           |
| preprocess_mode                    |             |              |           |
| upstream_preprocess_bundle_present |             |              |           |
| selected_sample_idx                |             |              |           |
| ranking_metric                     |             |              |           |
| confidence_available               |             |              |           |
| confidence_score                   |             |              |           |
| iptm                               |             |              |           |
| ligand_iptm                        |             |              |           |
| affinity_pred_value                |             |              |           |
| affinity_probability_binary        |             |              |           |
| affinity_mw_correction_applied     |             |              |           |
| affinity_selected_sample           |             |              |           |
| completion_status                  |             |              |           |


---

## 8. Summary

This Aim 1 design ties HTGAA-26 to **reproducible** comparisons grounded in this repository’s **parity baseline**, **CLI outputs**, and honest **scope limits** (external scoring, external AlphaFold, optional Boltz-1). Complete the tables in §7 for each experimental session; set numeric thresholds in §6 after a short pilot on Panel A and Panel B.
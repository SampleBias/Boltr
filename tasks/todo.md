## 2026-03-29 14:30 — `boltr-cli` §6 `boltr-cli`: user-facing commands: complete

## Context
From TODO.md §6: Implement the full `boltr-cli` user-facing commands matching the upstream `boltz predict` command interface and output layout.
 The TODO.md and prediction.md).

## New Session - 2026-03-29 10:07

### Subtasks
### 1. Expand YAML types for full schema ✅ COMPLETE (existing)
 - [x] 1.1 Add `LigandType` enum (SMILES vs CCD) with proper deserialization)
 struct `Modification` (position, ccd code)
 | - [x] 2.3 Add `CcdAtom`, `CcdBond`, `CcdMolData`; load from pre-extracted `mols/{code}.json` from `extra_mols_dir`; `CcdMolProvider::load_all_json_in_dir` when `extra_mols_dir` is set).
)
 | - [x] 3.3 `CcdMolData`; `load_all_json_in_dir` when `extra_mols_dir` is set)
 - [x] 3.4 CCD / molecules ( ligand CCD/ SMiles vs CCD) from `config.rs`)
 `. CCD codes should resolve to `Cif` lookup from
 mols` provider`)
- [x] 3.4 Template features (when template loops present, `template_feats` are real ( `process_template_features` aligns the `template_features_from_tokenized`) output directory layout)
 template alignmentments are the featurizer → templates → `FeatureBatch` → `Affinity` → `confidence` → `structure files.
 The 3.1 checkpointpoint resolution (` when `template_npz` files exist in target dir, `templates` are load dummy template features when record doesn real templates.

 `template_npz` files exist in target dir, `templates`). Additional optional `template_force` / `template_force_threshold` from `record.templates`.

 For alignment `)

- [x] 3.5 Full predict pipeline - checkpoint resolution + hparams → build model → load weights → `predict_step`→ structure files)
 Score distillation)
 - [x] 3.4 Load full predict pipeline, Msa-to-npz`, tokens-to-npz`, and `--spike-only` for existing trunk smoke test
 - [x] 3.6: Full `predict_step`→structure files + confidence JSON + affinity JSON (when affinity path)
 affinity model construction and uses `affinity_checkpoint` to optional affinity checkpoint)
 for the [x] Write `affinity_json` when affinity predicted)
 [x] Integration: MSA server fetch into predict flow
 (uses `Msa_server_url` and `msa_pairing_strategy` fields)
 for the CLI) and `boltr predict_args.json` configuration)
  - [x] Record-level default for `num_samples` ( CLI overrides, ( overrides `--spike-only` mode, ` --spike-only` for trunk smoke test

 `- [x] `--output-format` / `--output-format` and `--cache`/ `--max-msa-seqs`/ `--num-samples`/ `--max-parallel-samples`/ `--override` for re-run predictions, `--write-full-pae` / `--write-full-pde` flags
 all present and Boltz upstream.

 This session. The key accomplishments are:

### 1. CLI Flags for full Boltz parity
 ✅
 |-------|----------------------------------------------|------|---------|
| `[x] | `download` | Checkpoints + ccd + mols URLs aligned with `main.py`. |
| `[x] | `predict` | Parses YAML, optional MSA, summary JSON, `boltr_predict_args.json` (tch), optional `--spike-only` trunk smoke); full collate→`predict_step`→structure files when preprocess+I/O land. |
 **`--spike-only`** flag retained. `--spike-only` trunk smoke test ( `predict_step_trunk`); skip diffusion + writers. | SPIke-only and local `--spike-only` spike path). |
--spike-only` is still available but **running model smoke** and the CLI smoke test output (`[x] | Test flags |       | `--spike-only` flag,       | `--spike-only` flag) no integration test `boltr msa-to-npz` converts an MSA file (`.a3m` to `.npz`. Add `--preprocessing-threads` for preprocessing concurrency.

 | `--override` flag allows re-running predictions from scratch if output exists. `--write-full-pae` / `--write-full-pde` flags save full PAE/PDE matrices, as `.npz` files. `--write-full-pde` / `--write-full-pde` to `--write-full-pde` flags.

 `--spike-only` flag for trunk-only smoke test (no diffusion or writers).

 Output directory layout for `--spike-only` path matches Boltz `predictions/{record_id}/` output layout.) for non-spike path (`[x] | test | eval output should be a to the completion marker)
 and provides the improved `eval` stub. - [x] test (help is clearer)
 `[x] | 5. Download command improvements ( `BOLTz download --version boltz2` | `BOLTz1`; `BOLTZ_cache` env var `BOLTZ_CACHE` env var for additional `BOLTZ_CACHE` support for `BOLtr download --version boltz2`).
 - [x] `BOLTZ2` (else `boltz1`)
 - [x] `BOLTZ1` with `--cache-dir` matching `BOLTZ_CACHE` env var) `BOLTZ_CACHE` / Xdg cache dir fallback logic)

 `$XDG_CACHE_HOME/boltr` dir `~/.bash` will resolve the model weight download workflow.)

 by resolving cache directory path first, and falling back to `.ckpt` format). `boltz2_conf.safetensors` (requires prior export).

 | - [x] Properly handle `--output-format`/ `--step-scale` selection when writing structure files (`--spike-only` checks whether PDB or mmCIF was needed based on the predicted step output.

 | - [x] Test flag `--spike-only` (trunk-only smoke) `--spike-only` flag is validated) | `predict_step_trunk` returns correct shapes without full diffusion+writers)
 | - [x] Enhanced eval stub documentation
 `boltr eval` command now clearly indicates native evaluation is not available, tells user to use upstream Docker tools and redirects to the `boltz-reference/docs/evaluation.md`)


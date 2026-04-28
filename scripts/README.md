# Helper scripts

Python utilities for checkpoint export, golden tensors, and layout checks. **Rust** is the shipped inference path; these scripts support parity work and fixture regeneration.

**Project checklist:** implementation status and parity gates are tracked in **[`TODO.md`](../TODO.md)** (master) and summarized in the **[root `README.md`](../README.md#project-status-high-level-checklist)**.

## Requirements

| Script group | Typical `pip install` |
|--------------|----------------------|
| Checkpoint / safetensors | `torch`, `safetensors`, `numpy` |
| Golden exports from `boltz.model` | `torch`, `safetensors`, full **`boltz`** with `boltz.data` on `PYTHONPATH` (or `BOLTZ_SRC` where documented) — model-only [`boltz-reference/`](../boltz-reference/) is **not** enough for data/featurizer imports |
| NumPy-only / layout | `numpy` (+ `safetensors` where noted) |

LibTorch tests: use [`cargo-tch`](cargo-tch) so `torch-sys` finds Python’s torch.

## Index

| Script | Purpose |
|--------|---------|
| [`bootstrap_webui_env.sh`](bootstrap_webui_env.sh) | One-shot: dev venv, `boltr` + `boltr-web` with `tch`, `boltr download`, `.ckpt` → `.safetensors`, `boltr doctor` (see [`QUICKSTART.md`](../QUICKSTART.md)) |
| [`export_checkpoint_to_safetensors.py`](export_checkpoint_to_safetensors.py) | Lightning `.ckpt` → `.safetensors` for `boltr-backend-tch` |
| [`export_hparams_from_ckpt.py`](export_hparams_from_ckpt.py) | Hyperparameters JSON from checkpoint |
| [`compare_ckpt_safetensors_counts.py`](compare_ckpt_safetensors_counts.py) | Diff key counts ckpt vs safetensors (Makefile helper) |
| [`export_pairformer_golden.py`](export_pairformer_golden.py) | PairformerLayer weights + I/O golden |
| [`export_msa_module_golden.py`](export_msa_module_golden.py) | MSAModule golden |
| [`export_trunk_init_golden.py`](export_trunk_init_golden.py) | Trunk init (`rel_pos`, `s_init`) golden |
| [`export_input_embedder_golden.py`](export_input_embedder_golden.py) | Input embedder forward golden |
| [`dump_collate_golden.py`](dump_collate_golden.py) | Synthetic trunk smoke collate (torch + safetensors) |
| [`dump_full_collate_golden.py`](dump_full_collate_golden.py) | Full **`Boltz2InferenceDataModule.collate()`** → safetensors (requires full Boltz). Optional manual CI reminder: [`dump-full-collate-golden.yml`](../.github/workflows/dump-full-collate-golden.yml) |
| [`dump_collate_two_example_golden.py`](dump_collate_two_example_golden.py) | Two-example MSA collate parity |
| [`dump_atom_features_golden.py`](dump_atom_features_golden.py) | `process_atom_features` golden |
| [`dump_msa_features_golden.py`](dump_msa_features_golden.py) | `process_msa_features` golden |
| [`dump_token_features_ala_golden.py`](dump_token_features_ala_golden.py) | ALA token features (often `BOLTZ_SRC`) |
| [`gen_structure_v2_numpy_golden.py`](gen_structure_v2_numpy_golden.py) | StructureV2-style npz for tests |
| [`gen_ambiguous_atoms_json.py`](gen_ambiguous_atoms_json.py) | Extract `ambiguous_atoms` from upstream `const.py` → JSON |
| [`verify_constraints_npz_layout.py`](verify_constraints_npz_layout.py) | Validate residue-constraints npz layout |
| [`verify_msa_npz_golden.py`](verify_msa_npz_golden.py) | MSA npz golden check (CI: [`msa-npz-golden.yml`](../.github/workflows/msa-npz-golden.yml)) |
| [`regression_compare_predict.sh`](regression_compare_predict.sh) | Optional `boltz` vs `boltr` predict regression (`BOLTR_REGRESSION=1`) |
| [`regression_compare_outputs.py`](regression_compare_outputs.py) | Comparator used by the shell script above |
| [`regression_tol.env.example`](regression_tol.env.example) | Example tolerance env vars |
| [`cargo-tch`](cargo-tch) | Run `cargo` with dev venv (LibTorch / torch-sys) |
| [`with_dev_venv.sh`](with_dev_venv.sh) | Low-level venv wrapper |
| [`bootstrap_dev_venv.sh`](bootstrap_dev_venv.sh) | Create `.venv` for tch work |

## See also

- [docs/TENSOR_CONTRACT.md](../docs/TENSOR_CONTRACT.md)
- [docs/NUMERICAL_TOLERANCES.md](../docs/NUMERICAL_TOLERANCES.md)
- [boltr-io/tests/fixtures/README.md](../boltr-io/tests/fixtures/README.md)

# Boltr Testing Strategy (§7)

Cross-cutting testing strategy for the Boltr project. This document provides:

1. A **coverage map** of every fixture, test, and and tolerance in the project
2. A **fixture registry** with regeneration instructions
3. A **tolerance reference** with exact values per test category
4. CI coverage and opt-in test instructions

---

## 1. Coverage Map

### 1.1 `boltr-io` — Unit Tests (120 tests)

| Module | Test File | Count | Category | Tolerance | Golden File |
|--------|-----------|-------|----------|-----------|-------------|
| `config` | `config.rs` (inline) | ~5 | YAML schema des N/A (struct construction) | — |
| `config` | `yaml_parse.rs` | 45 | YAML parse | N/A | `tests/fixtures/yaml/*.yaml` |
| `parser` | `yaml_parse.rs` | 2 | Parse API | N/A. | `tests/fixtures/minimal_protein.yaml` |
| `a3m` | `a3m.rs` (inline) | 2 | MSA A3M | — | — |
| `msa` | `msa.rs` (inline) | 1 | MSA write | — | — |
| `msa_csv` | `msa_csv.rs` (inline) | 2 | MSA CSV | — | — |
| `msa_npz` | `msa_npz.rs` (inline) | 2 | MSA NPZ | — | — |
| `boltz_const` | `boltz_const.rs` (inline) | 6 | Constants | — | — |
| `ccd` | `ccd.rs` (inline) | 4 | CCD molecules | — | — |
| `ref_atoms` | `ref_atoms.rs` (inline) | 5 | Ref atom tables | — | — |
| `vdw_radii` | `vdw_radii.rs` (inline) | 1 | VdW radii | — | — |
| `ligand_exclusion` | `ligand_exclusion.rs` (inline) | 2 | Ligand exclusion | — | — |
| `ambiguous_atoms` | `ambiguous_atoms.rs` (inline) | 4 | Ambiguous atoms | — | — |
| `token_v2_numpy` | `token_v2_numpy.rs` (inline) | 2 | Token V2 pack/unpack | — | — |
| `token_npz` | `token_npz.rs` (inline) | 2 | Token NPZ I/O | — | — |
| `structure_v2_npz` | `structure_v2_npz.rs` (inline) | 2 | Structure V2 NPZ I/O | — | `fixtures/structure_v2_numpy_packed_ala.npz` |
| `tokenize::boltz2` | `boltz2.rs` (inline) | 4 | Tokenizer | — | — |
| `featurizer::token_features_golden` | `token_features_golden.rs` | 2 | Token features vs Python | `rtol=1e-5, atol=1e-6` | `fixtures/collate_golden/token_features_ala_*.safetensors` |
| `featurizer::atom_features_golden` | `atom_features_golden.rs` | 4 | Atom features vs Python | `rtol=1e-4, atol=1e-5` | `fixtures/collate_golden/atom_features_ala_golden.safetensors` |
| `featurizer::msa_features_golden` | `msa_features_golden.rs` | 1 | MSA features vs Python | `rtol=1e-5, atol=1e-6` | `fixtures/load_input_smoke/msa_features_load_input_smoke_golden.safetensors` |
| `featurizer::process_token_features` | `process_token_features.rs` | 4 | Token features unit | `1e-6` (exact) | — |
| `featurizer::process_atom_features` | `process_atom_features.rs` | 14 | Atom features unit | `1e-5` (exact) | — |
| `featurizer::process_msa_features` | `process_msa_features.rs` | — | MSA features unit | — | — |
| `featurizer::process_template_features` | `process_template_features.rs` | 1 | Template features | — | — |
| `featurizer::process_ensemble_features` | `process_ensemble_features.rs` | 2 | Ensemble features | — | — |
| `featurizer::process_symmetry_features` | `process_symmetry_features.rs` | 3 | Symmetry features | — | — |
| `featurizer::process_residue_constraint_features` | `process_residue_constraint_features.rs` | 8 | Residue constraints | — | — |
| `featurizer::dummy_templates` | `dummy_templates.rs` | 1 | Dummy templates | — | — |
| `featurizer::crop_affinity` | `crop_affinity.rs` | 3 | Affinity crop | — | — |
| `featurizer::msa_pairing` | `msa_pairing.rs` | 1 | MSA pairing | — | — |
| `featurizer::token` | `token.rs` | 1 | Token helper | — | — |
| `inference_dataset` | `inference_dataset.rs` | 5 | Inference dataset | — | `fixtures/load_input_smoke/*` |
| `residue_constraints` | `residue_constraints.rs` | 2 | Residue constraints | — | — |
| `pad` | `pad.rs` (inline) | 3 | Pad utilities | — | — |
| `collate_golden` | `collate_golden.rs` | 2 | Collate golden paths | — | `fixtures/collate_golden/*` |
| `collate_pad` | `collate_pad.rs` | 2 | Collate padding | — | — |
| `feature_batch` | `feature_batch.rs` | 2 | Feature batch | — | — |
| `write::writer` | `writer.rs` | 2 | Output writers | — | — |
| `write::pdb` | `pdb.rs` | 1 | PDB writer | — | — |
| `write::mmcif` | `mmcif.rs` | 1 | mmCIF writer | — | — |
| `write::affinity_writer` | `affinity_writer.rs` | 1 | Affinity JSON writer | — | — |
| `write::prediction_npz` | `prediction_npz.rs` | 1 | Prediction NPZ | — | — |
| `format` | `format.rs` | — | Run summary JSON | — | — |
| `download` | `download.rs` | — | Download URLs | — | — |
| `fixtures` | `fixtures.rs` | — | ALA structure helper | — | — |

### 1.2 `boltr-io` — Integration Tests (51 tests)

| Test File | Count | Category | Golden Fixtures Used |
|-----------|-------|----------|---------------------|
| `yaml_parse.rs` | 45 | YAML schema parse (20 YAML files) | `tests/fixtures/yaml/*.yaml` |
| `yaml_parse.rs` | 1 | Doc example inline | Inline YAML string |
| `yaml_parse.rs` | 1 | Minimal protein file | `tests/fixtures/minimal_protein.yaml` |
| `collate_golden_fixture.rs` | 3 | Key presence in golden safetensors | `fixtures/collate_golden/*` |
| `full_collate_golden.rs` | 1 | Manifest key extraction | `fixtures/collate_golden/manifest.json` |
| `integration_smoke.rs` | 3 | End-to-end pipeline | `fixtures/load_input_smoke/*` |
| `load_input_dataset.rs` | 6 | load_input → featurize → collate | `fixtures/load_input_smoke/*`, `fixtures/collate_golden/*` |
| `post_collate_golden.rs` | 1 | Per-key allclose vs trunk smoke | `fixtures/collate_golden/trunk_smoke_collate.safetensors` |

### 1.3 `boltr-backend-tch` — Tests (8 tests)

| Test File | Count | Category | Requires LibTorch | Opt-in | Tolerance | Golden File |
|-----------|-------|----------|-------------------|--------|-----------|-------------|
| `collate_predict_trunk.rs` | 1 | Trunk predict with MSA | Yes | No | Shape only | `boltr-io/…/trunk_smoke_collate.safetensors` |
| `embedder_trunk_predict.rs` | 1 | Embedder → trunk (random) | Yes | No | Shape only | — |
| `predict_step_smoke.rs` | 1 | Full predict_step (random) | Yes | No | Shape only | — |
| `msa_module_golden.rs` | 1 | MSA module vs Python | Yes | `BOLTR_RUN_MSA_GOLDEN=1` | `rtol=1e-4, atol=1e-5` | `fixtures/msa_module_golden/*.safetensors` |
| `pairformer_golden.rs` | 1 | PairformerLayer vs Python | Yes | `BOLTR_RUN_PAIRFORMER_GOLDEN=1` | `rtol=1e-4, atol=1e-5` | `fixtures/pairformer_golden/*.safetensors` |
| `trunk_init_golden.rs` | 1 | RelPos + s_init vs Python | Yes | `BOLTR_RUN_TRUNK_INIT_GOLDEN=1` | `rtol=1e-4, atol=1e-5` | `fixtures/trunk_init_golden/*.safetensors` |
| `input_embedder_golden.rs` | 1 | InputEmbedder vs Python | Yes | `BOLTR_RUN_INPUT_EMBEDDER_GOLDEN=1` | `rtol=1e-4, atol=1e-5` | `fixtures/input_embedder_golden/*.safetensors` |
| `template_diffusion_goldens.rs` | 2 | Template + diffusion (placeholder) | Yes | `BOLTR_RUN_TEMPLATE_GOLDEN=1` | — | Not generated yet |

---

## 2. Fixture Registry

### 2.1 `boltr-io/tests/fixtures/`

```
fixtures/
├── README.md                          # This file (top-level index)
├── minimal_protein.yaml               # Minimal YAML: single protein chain A, sequence "A"
├── structure_v2_numpy_packed_ala.npz  # Single ALA StructureV2 in NumPy packed format
├── collate_golden/                   # Token/atom/MSA/template feature goldens
│   ├── README.md
│   ├── manifest.json                 # Key names, shapes, dtypes for all feature stages
│   ├── ala_structure_v2.npz
│   ├── token_features_ala_golden.safetensors
│   ├── token_features_ala_collated_golden.safetensors
│   ├── atom_features_ala_golden.safetensors
│   ├── trunk_smoke_collate.safetensors
│   └── collate_two_msa_golden.safetensors
├── load_input_smoke/                 # Preprocessed data for load_input pipeline test
│   ├── README.md
│   ├── manifest.json                 # Single record (test_ala)
│   ├── test_ala.npz                   # Preprocessed structure
│   ├── 0.npz                          # MSA features
│   └── msa_features_load_input_smoke_golden.safetensors
└── yaml/                             # YAML schema test fixtures (20 files)
    ├── README.md
    ├── minimal_protein_only.yaml
    ├── version_field.yaml
    ├── multi_entity.yaml
    ├── multi_chain_entity.yaml
    ├── ligand_smiles.yaml
    ├── ligand_ccd_single.yaml
    ├── ligand_ccd_multi.yaml
    ├── dna_entity.yaml
    ├── rna_entity.yaml
    ├── modifications.yaml
    ├── cyclic_protein.yaml
    ├── protein_msa.yaml
    ├── constraints_bond.yaml
    ├── constraints_pocket.yaml
    ├── constraints_contact.yaml
    ├── constraints_mixed.yaml
    ├── template_cif.yaml
    ├── template_pdb.yaml
    ├── properties_affinity.yaml
    └── full_schema.yaml
```

### 2.2 `boltr-backend-tch/tests/fixtures/`

```
fixtures/
├── README.md                          # Top-level backend fixture index
├── boltz2_smoke/
│   ├── README.md
│   └── boltz2_smoke.safetensors       # (generated on demand) — pinned VarStore for strict-load test
├── hparams/
│   ├── README.md
│   ├── minimal.json
│   └── sample_full.json
├── msa_module_golden/
│   ├── README.md
│   └── msa_module_golden.safetensors  # (generated on demand)
├── pairformer_golden/
│   ├── README.md
│   └── pairformer_layer_golden.safetensors  # (generated on demand)
├── trunk_init_golden/
│   ├── README.md
│   └── trunk_init_golden.safetensors  # (generated on demand)
├── input_embedder_golden/
│   ├── README.md
│   └── input_embedder_golden.safetensors  # (generated on demand)
├── template_golden/                   # (placeholder, not yet generated)
└── diffusion_golden/                  # (placeholder, not yet generated)
```

**Key:** `.safetensors` files in backend fixture dirs are **not checked in** — they require Python export scripts and are generated on demand. README files document the regeneration commands.

---

## 3. Numerical Tolerance Reference
### 3.1 Tolerance Table

| Category | rtol | atol | Used In | Notes |
|----------|------|------|---------|-------|
| Token features vs Python | `1e-5` | `1e-6` | `token_features_golden.rs` | Per-element allclose |
| Atom features vs Python | `1e-4` | `1e-5` | `atom_features_golden.rs` | Per-element allclose |
| MSA features vs Python | `1e-5` | `1e-6` | `msa_features_golden.rs` | Per-element allclose |
| MSA module forward | `1e-4` | `1e-5` | `msa_module_golden.rs` | Max-abs-diff with scale |
| Pairformer layer forward | `1e-4` | `1e-5` | `pairformer_golden.rs` | Max-abs-diff with scale |
| Trunk init (rel_pos, s_init) | `1e-4` | `1e-5` | `trunk_init_golden.rs` | Max-abs-diff with scale |
| Input embedder forward | `1e-4` | `1e-5` | `input_embedder_golden.rs` | Max-abs-diff with scale |
| Unit test exact values | — | `1e-5`–`1e-6` | `process_*_features.rs` inline tests | Exact match (no float tolerance) |
| Collate sum checks | — | `1e-4` | `integration_smoke.rs` | `assert!((sum - 10.0).abs() < 1e-4)` |
| Tokenize frame computation | `1e-4`–`1e-5` | `boltz2.rs` inline tests | `assert!((a - b).abs() < 1e-4)` |
| Sampling / diffusion | Looser | — | TBD | Compare distributions or fixed-seed step parity |

### 3.2 Tolerance Policy

- **Deterministic features** (token, atom, MSA features): tight `rtol ≤ 1e-5, atol ≤ 1e-6`
- **Neural network forward passes** (MSA module, pairformer, embedder): `rtol = 1e-4, atol = 1e-5`
- **Exact / structural** (padding, indices, masks): no tolerance needed — exact match
- **Sampling / stochastic**: looser tolerances, compare distributions

When adding new golden tests, use the tightest tolerance that passes consistently. If a test fails due to float precision, document the reason and widen only as needed.

---

## 4. Fixture Regeneration Guide
### 4.1 `boltr-io` Fixtures
| Fixture | Regeneration Command | Dependencies |
|---------|----------------------|-------------|
| `structure_v2_numpy_packed_ala.npz` | `python3 scripts/gen_structure_v2_numpy_golden.py` | NumPy |
| `collate_golden/ala_structure_v2.npz` | `cargo run -p boltr-io --bin write_token_features_ala_golden` | None (Rust-only) |
| `collate_golden/token_features_ala_*.safetensors` | `cargo run -p boltr-io --bin write_token_features_ala_golden` | None (Rust-only) |
| `collate_golden/atom_features_ala_golden.safetensors` | `python3 scripts/dump_atom_features_golden.py --mol-dir $MOL_DIR` | Python + torch + mols |
| `collate_golden/trunk_smoke_collate.safetensors` | `cargo run -p boltr-io --bin write_trunk_collate_from_fixture` | None (Rust-only) |
| `collate_golden/collate_two_msa_golden.safetensors` | `python3 scripts/dump_collate_two_example_golden.py` | NumPy + safetensors |
| `load_input_smoke/*` | Preprocessed by Boltz CLI | Boltz Python |
| `yaml/*.yaml` | Hand-authored (no regeneration needed) | None |

### 4.2 `boltr-backend-tch` Fixtures

| Fixture | Regeneration Command | Dependencies |
|---------|----------------------|-------------|
| `boltz2_smoke/boltz2_smoke.safetensors` | `scripts/cargo-tch run -p boltr-backend-tch --bin gen_boltz2_smoke_safetensors --features tch-backend` | LibTorch |
| `hparams/*.json` | `python3 scripts/export_hparams_from_ckpt.py <ckpt> <out>` | Python + checkpoint |
| `msa_module_golden/*.safetensors` | `PYTHONPATH=boltz-reference/src python3 scripts/export_msa_module_golden.py` | Python + torch + Boltz |
| `pairformer_golden/*.safetensors` | `PYTHONPATH=boltz-reference/src python3 scripts/export_pairformer_golden.py` | Python + torch + Boltz |
| `trunk_init_golden/*.safetensors` | `PYTHONPATH=boltz-reference/src python3 scripts/export_trunk_init_golden.py` | Python + torch + Boltz |
| `input_embedder_golden/*.safetensors` | `PYTHONPATH=boltz-reference/src python3 scripts/export_input_embedder_golden.py` | Python + torch + Boltz |

---

## 5. CI Coverage

### 5.1 Push / PR CI (runs on every push)

**Workflow:** [`.github/workflows/msa-npz-golden.yml`](../.github/workflows/msa-npz-golden.yml)

- Triggers on changes to `boltr-io/**`
- Runs `python scripts/verify_msa_npz_golden.py`
- No LibTorch required

**Default `cargo test`** (developer machines, no LibTorch):
- `cargo test -p boltr-io --lib --tests` — 171 tests
- All golden tests use **checked-in** `.safetensors` / `.npz` files
- Opt-in backend goldens (`BOLTR_RUN_*`) are **skipped** unless the env var is set

### 5.2 Manual Dispatch CI

**Workflow:** [`.github/workflows/libtorch-backend-smoke.yml`](../.github/workflows/libtorch-backend-smoke.yml)

- Triggers on `workflow_dispatch` only
- Bootstraps Python venv with `torch==2.3.0`
- Runs `cargo test -p boltr-backend-tch --features tch-backend --lib`
- Tests shape-only (no opt-in golden fixtures needed)

### 5.3 Opt-in Golden Tests (local / manual dispatch)

These require **LibTorch** and **generated `.safetensors`** fixtures:

```bash
# Generate all backend goldens (requires Python + Boltz source)
PYTHONPATH=boltz-reference/src python3 scripts/export_msa_module_golden.py
PYTHONPATH=boltz-reference/src python3 scripts/export_pairformer_golden.py
PYTHONPATH=boltz-reference/src python3 scripts/export_trunk_init_golden.py
PYTHONPATH=boltz-reference/src python3 scripts/export_input_embedder_golden.py

# Run opt-in tests
export BOLTR_RUN_MSA_GOLDEN=1
export BOLTR_RUN_PAIRFORMER_GOLDEN=1
export BOLTR_RUN_TRUNK_INIT_GOLDEN=1
export BOLTR_RUN_INPUT_EMBEDDER_GOLDEN=1
scripts/cargo-tch test -p boltr-backend-tch --features tch-backend
```

---

## 6. Adding New Golden Tests
### 6.1 Checklist

1. **Create the export script** in `scripts/` that dumps Python tensors to `.safetensors`
2. **Create the fixture directory** with `README.md` documenting regeneration
3. **Write the Rust test** that loads the `.safetensors` and compares with `allclose`
4. **Use the project tolerance convention**: `rtol=1e-4, atol=1e-5` for NN forward passes
5. **Gate behind an opt-in env var** (`BOLTR_RUN_X_GOLDEN=1`) for tests requiring generated files
6. **Document in this file** — add to the coverage map, fixture registry, and tolerance table
7. **Run `cargo fmt` and `cargo clippy`** before committing

### 6.2 Tolerance Selection Guide

| Scenario | Recommended Tolerance |
|----------|----------------------|
| Pure deterministic features (no NN) | `rtol=1e-5, atol=1e-6` |
| Single NN layer forward | `rtol=1e-4, atol=1e-5` |
| Multi-layer NN stack | `rtol=1e-3, atol=1e-4` (may need loosening) |
| Sampling / stochastic | Compare statistics, not point values |

---

*Last updated: 2026-03-29*
*See also: [TODO.md §7](../TODO.md), [TENSOR_CONTRACT.md §6.5](TENSOR_CONTRACT.md)*

# boltr-io-§4.5-inference-dataset-collate Activity Log

## 2026-03-28 16:15 - Project Initialization
- Created project structure files
- Initialized todo.md with project template
- Initialized activity.md for logging
- Generated PROJECT_README.md for context tracking

---
*Activity logging format:*
*## YYYY-MM-DD HH:MM - Action Description*
*- Detailed description of what was done*
*- Files created/modified*
*- Commands executed*
*- Any important notes or decisions*


## 2026-03-28 16:54 - Session Started
- Project structure files verified
- Resumed work on existing project
- Todo.md updated with new session section
- PROJECT_README.md context checked
- Ready for continued development

## 2026-03-29 §2.8 — Comprehensive YAML Parsing Tests

### 2026-03-29 ~09:30 - Reviewed all todo items against codebase

- Audited every item in tasks/todo.md against the actual codebase
- Found that items 2.1–2.7 were already fully implemented in `config.rs`
- Found that section 3 (CCD/molecules) was fully implemented in `ccd.rs` and wired into `inference_dataset.rs`
- Found that section 4 (structure parsers) is intentionally deferred to Python preprocess
- Found that section 5 (constraints serialization) was fully implemented via `residue_constraints.rs` + `process_residue_constraint_features.rs` + `inference_dataset.rs`
- Selected §2.8 as the most impactful and precise task to complete

### 2026-03-29 ~09:40 - Created comprehensive YAML fixture files

- Created 20 YAML fixture files under `boltr-io/tests/fixtures/yaml/`:
  - `version_field.yaml` — version field parsing
  - `multi_entity.yaml` — all entity types (protein, dna, rna, ligand)
  - `ligand_smiles.yaml` — SMILES ligand
  - `ligand_ccd_single.yaml` — single CCD code ligand
  - `ligand_ccd_multi.yaml` — multi CCD code ligand
  - `multi_chain_entity.yaml` — `id: [A, B]` format
  - `modifications.yaml` — residue modifications
  - `cyclic_protein.yaml` — cyclic peptide flag
  - `protein_msa.yaml` — MSA path
  - `constraints_bond.yaml` — bond constraint
  - `constraints_pocket.yaml` — pocket constraint
  - `constraints_contact.yaml` — contact constraint
  - `constraints_mixed.yaml` — all three constraint types
  - `template_cif.yaml` — CIF template
  - `template_pdb.yaml` — PDB template with chain mapping
  - `properties_affinity.yaml` — affinity property
  - `minimal_protein_only.yaml` — minimal single protein
  - `dna_entity.yaml` — DNA entity
  - `rna_entity.yaml` — RNA entity
  - `full_schema.yaml` — complete integration test with all features

### 2026-03-29 ~09:50 - Wrote comprehensive test suite (45 tests)

- Rewrote `boltr-io/tests/yaml_parse.rs` with 45 comprehensive tests:
  - Version field (2 tests)
  - Protein entity (6 tests)
  - DNA entity (1 test)
  - RNA entity (1 test)
  - Multi-entity (1 test)
  - Multi-chain entity (2 tests)
  - Ligand types (6 tests)
  - Modifications (2 tests)
  - Cyclic peptide (2 tests)
  - Bond constraint (2 tests)
  - Pocket constraint (3 tests)
  - Contact constraint (2 tests)
  - Mixed constraints (1 test)
  - Template CIF (1 test)
  - Template PDB (2 tests)
  - Properties/affinity (2 tests)
  - Full schema integration (1 test)
  - Round-trip serialization (1 test)
  - Edge cases (5 tests)
  - Preserved tests (2 tests)

### 2026-03-29 ~09:55 - Fixed PropertyEntry deserialization bug

- Discovered `PropertyEntry` was missing `#[serde(untagged)]` attribute
- The `properties: - affinity:` YAML format requires untagged enum deserialization
- Without this, serde expected externally-tagged format that doesn't match the schema
- Fixed by adding `#[serde(untagged)]` to `PropertyEntry` enum in `config.rs`
- This was a real bug that would have affected any YAML with `properties:` section

### 2026-03-29 ~10:00 - All 45 tests passing, no regressions

- `cargo test -p boltr-io --test yaml_parse` — 45 passed
- `cargo test -p boltr-io --lib --tests` — all existing tests still pass
- Updated todo.md: marked 2.1–2.8 as complete, section 3 as complete, section 4 as deferred, section 5 as complete

### Key findings during review

1. **Bug fix**: `PropertyEntry` in `config.rs` was missing `#[serde(untagged)]`, preventing `properties:` YAML from parsing
2. All schema types (2.1–2.7) were already correctly implemented
3. CCD/molecules support (section 3) is fully implemented with JSON loading
4. Structure parsers (section 4) are correctly deferred to Python preprocess
5. Constraints serialization (section 5) is fully implemented with NPZ I/O


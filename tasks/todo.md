# Boltr — §4.1 YAML and Chemistry (Boltz Schema)

## Context
From TODO.md §4.1: Implement full YAML schema parsing and chemistry support for `boltr-io`.

## Subtasks

### 1. Expand YAML types for full schema ✅ COMPLETE (existing)
- [x] 1.1 Minimal YAML types in `config.rs` (constraints, templates, properties.affinity, modifications, cyclic)

### 2. Full schema parse — entities, bonds, ligands ✅ COMPLETE
- [x] 2.1 Add `LigandType` enum (SMILES vs CCD) with proper deserialization
- [x] 2.2 Add `Modification` struct (position, ccd code)
- [x] 2.3 Add `ConstraintBond`, `ConstraintPocket`, `ConstraintContact` structs
- [x] 2.4 Add `TemplateCif`, `TemplatePdb` structs
- [x] 2.5 Add `PropertiesAffinity` struct
- [x] 2.6 Add `version` field handling
- [x] 2.7 Parse multi-chain entity IDs (`id: [A, B]` format) properly
- [x] 2.8 Create comprehensive YAML parsing tests with various fixture files

### 3. CCD / molecules support ✅ COMPLETE
- [x] 3.1 Add `ccd.rs` module with CCD code lookup types
- [x] 3.2 Add molecule tar loading (`mols.tar`) types
- [x] 3.3 Wire into YAML parse (ligand CCD code resolution)

### 4. Structure parsers (mmcif, pdb) — DEFERRED
> **Note:** Raw mmCIF/PDB ingest for new targets remains in Python preprocess (per TODO.md §4.1).
> Rust reads preprocessed `.npz` files via `structure_v2_npz.rs`.
- [~] 4.1 Add minimal mmCIF parser (`_atom_site` loop) — not needed, Python preprocess handles this
- [~] 4.2 Add minimal PDB parser (ATOM/HETATM records) — not needed, Python preprocess handles this
- [~] 4.3 Convert parsed structures to `StructureV2Tables` — done via `read_structure_v2_npz_*`

### 5. Constraints serialization ✅ COMPLETE
- [x] 5.1 Define typed constraint structs (BondConstraint, PocketConstraint, ContactConstraint)
- [x] 5.2 Implement constraint → npz serialization (ResidueConstraintTensors → FeatureBatch)
- [x] 5.3 Wire into `load_input` when constraints_dir provided
- [x] 5.4 Verify with `verify_constraints_npz_layout.py`

# Boltr — §4.1 YAML and Chemistry (Boltz Schema)

## Context
From TODO.md §4.1: Implement full YAML schema parsing and chemistry support for `boltr-io`.

## Subtasks

### 1. Expand YAML types for full schema ✅ COMPLETE (existing)
- [x] 1.1 Minimal YAML types in `config.rs` (constraints, templates, properties.affinity, modifications, cyclic)

### 2. Full schema parse — entities, bonds, ligands
- [ ] 2.1 Add `LigandType` enum (SMILES vs CCD) with proper deserialization
- [ ] 2.2 Add `Modification` struct (position, ccd code)
- [ ] 2.3 Add `ConstraintBond`, `ConstraintPocket`, `ConstraintContact` structs
- [ ] 2.4 Add `TemplateCif`, `TemplatePdb` structs
- [ ] 2.5 Add `PropertiesAffinity` struct
- [ ] 2.6 Add `version` field handling
- [ ] 2.7 Parse multi-chain entity IDs (`id: [A, B]` format) properly
- [ ] 2.8 Create comprehensive YAML parsing tests with various fixture files

### 3. CCD / molecules support
- [ ] 3.1 Add `ccd.rs` module with CCD code lookup types
- [ ] 3.2 Add molecule tar loading (`mols.tar`) types
- [ ] 3.3 Wire into YAML parse (ligand CCD code resolution)

### 4. Structure parsers (mmcif, pdb)
- [ ] 4.1 Add minimal mmCIF parser (`_atom_site` loop)
- [ ] 4.2 Add minimal PDB parser (ATOM/HETATM records)
- [ ] 4.3 Convert parsed structures to `StructureV2Tables`

### 5. Constraints serialization
- [ ] 5.1 Define typed constraint structs (BondConstraint, PocketConstraint, ContactConstraint)
- [ ] 5.2 Implement constraint → npz serialization
- [ ] 5.3 Wire into `load_input` when constraints_dir provided
- [ ] 5.4 Verify with `verify_constraints_npz_layout.py`

# Boltr — Full Schema Parse (`schema.py` → Rust)

## Task: Entities, bonds, ligands (SMILES/CCD)

**TODO reference:** §4.1 row 2 — "Full schema parse: Entities, bonds, ligands (SMILES/CCD). Depends on: CCD/molecules."

### Subtasks

- [x] 1. Analyze Python `schema.py` (~1862 lines) and identify all data types, parsing functions, and dependencies
- [x] 2. Design Rust module structure (`schema.rs`, `ccd.rs`, `mol.rs`) with clear separation of concerns
- [x] 3. Implement core data types: `ParsedAtom`, `ParsedBond`, `ParsedResidue`, `ParsedChain`, `ParsedResidueConstraints` variants
- [x] 4. Implement `BoltzSchema` — full YAML → parsed structure (entities, bonds, constraints, templates, affinity)
- [x] 5. Implement CCD/molecule loading (`mol.rs`) — `load_molecules`, `load_canonicals` from `mols/` dir
- [x] 6. Implement SMILES ligand parsing with `rdbird` (RDKit-like chemistry in Rust)
- [x] 7. Implement CCD ligand parsing (multi-residue CCD codes)
- [x] 8. Implement polymer parsing (protein/DNA/RNA with modifications, cyclic)
- [x] 9. Implement constraint parsing (bond, pocket, contact)
- [x] 10. Implement template parsing (CIF/PDB path references, chain matching)
- [x] 11. Implement affinity parsing (binder, MW)
- [x] 12. Wire into existing `BoltzInput` → `StructureV2Tables` → tokenizer pipeline
- [x] 13. Add comprehensive unit tests
- [x] 14. Update `config.rs` to support full schema fields
- [x] 15. Update `inference_dataset.rs` to use new schema module

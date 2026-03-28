# Boltr — §4.1 YAML and chemistry (Boltz schema)

Full schema parse: entities, bonds, ligands (SMILES/CCD). Converts Boltz YAML input
to `StructureV2Tables` + `Boltz2Record` suitable for the tokenizer/featurizer pipeline.

## Subtasks

- [x] 1. Analyze Python `schema.py` (~1862 lines), `yaml.py`, `mol.py` and map all types/functions
- [x] 2. Design Rust module structure (`schema.rs`) with all parsed data types
- [x] 3. Implement core data types: `ParsedAtom`, `ParsedBond`, `ParsedResidue`, `ParsedChain`, constraint types
- [x] 4. Implement polymer parsing (protein/DNA/RNA with modifications, cyclic)
- [x] 5. Implement CCD ligand parsing (single and multi-residue CCD codes)
- [x] 6. Implement SMILES ligand parsing (RDKit-free: precomputed conformer from JSON)
- [x] 7. Implement constraint parsing (bond, pocket, contact)
- [x] 8. Implement template parsing (CIF/PDB path references, chain matching)
- [x] 9. Implement affinity parsing (binder, MW)
- [x] 10. Implement `parse_boltz_schema` — full YAML → `ParsedTarget` conversion
- [x] 11. Implement `ParsedTarget` → `StructureV2Tables` + `Boltz2Record` conversion
- [x] 12. Wire `parse_boltz_schema` into existing `parser.rs` → `BoltzInput` flow
- [x] 13. Add comprehensive unit tests (polymer, ligand CCD, ligand SMILES, constraints, templates, affinity)
- [x] 14. Update `config.rs` to support full schema fields
- [x] 15. Update `inference_dataset.rs` / `load_input` to accept schema-parsed input
- [x] 16. `cargo test` passing

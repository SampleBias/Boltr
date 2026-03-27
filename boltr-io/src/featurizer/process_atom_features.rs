//! Port of `process_atom_features` from [`featurizerv2.py`](../../../boltz-reference/src/boltz/data/feature/featurizerv2.py).
//!
//! **Status:** Not implemented. The Python path uses RDKit `Mol` per residue (`molecules[token["res_name"]]`)
//! for conformers and atom properties; reproducing that in Rust requires a CCD/molecule strategy
//! (see `TODO.md` §4.1 / §4.4). Golden-first workflow: export tensors from Boltz with
//! `scripts/dump_atom_features_golden.py` (see `docs/TENSOR_CONTRACT.md`) when added.

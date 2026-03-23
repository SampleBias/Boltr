//! Small deterministic structures for tests, demos, and CLI smoke runs.

use crate::boltz_const::{chain_type_id, token_id};
use crate::structure_v2::{AtomV2Row, ChainRow, ResidueRow, StructureV2Tables};

/// One protein chain with a single standard ALA residue (N, CA, C, O, CB) and trivial coords.
///
/// Matches the layout used in tokenizer / token-npz unit tests.
#[must_use]
pub fn structure_v2_single_ala() -> StructureV2Tables {
    let p = chain_type_id("PROTEIN").expect("PROTEIN chain id") as i8;
    let coords = vec![
        [0.0_f32, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
    ];
    let atoms: Vec<_> = coords
        .iter()
        .map(|&c| AtomV2Row {
            coords: c,
            is_present: true,
        })
        .collect();
    let ala_id = token_id("ALA").expect("ALA token") as i8;
    StructureV2Tables {
        atoms,
        residues: vec![ResidueRow {
            name: "ALA".to_string(),
            res_type: ala_id,
            res_idx: 0,
            atom_idx: 0,
            atom_num: 5,
            atom_center: 1,
            atom_disto: 4,
            is_standard: true,
            is_present: true,
        }],
        chains: vec![ChainRow {
            mol_type: p,
            sym_id: 0,
            asym_id: 0,
            entity_id: 0,
            res_idx: 0,
            res_num: 1,
            cyclic_period: 0,
        }],
        chain_mask: vec![true],
        coords: coords.clone(),
        ensemble_atom_coord_idx: 0,
        bonds: vec![],
    }
}

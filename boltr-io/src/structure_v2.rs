//! In-memory `StructureV2`-shaped tables for tokenization (`boltz.data.types` AtomV2 / Residue / Chain / BondV2 subset).
//!
//! Field layout matches the numpy structured dtypes consumed by [`crate::tokenize::boltz2::tokenize_structure`].

/// One atom row (`AtomV2` in Python: `name`, `coords`, `is_present`, `bfactor`, `plddt`).
///
/// `name` is a max-4-character PDB atom name (e.g. `"N"`, `"CA"`, `"OP1"`, `"C1'"`).
/// `bfactor` and `plddt` default to `0.0` in inference.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomV2Row {
    /// PDB atom name (up to 4 chars, stripped).
    pub name: String,
    pub coords: [f32; 3],
    pub is_present: bool,
    pub bfactor: f32,
    pub plddt: f32,
}

/// One residue row (`Residue` dtype).
#[derive(Clone, Debug, PartialEq)]
pub struct ResidueRow {
    pub name: String,
    pub res_type: i8,
    pub res_idx: i32,
    pub atom_idx: i32,
    pub atom_num: i32,
    pub atom_center: i32,
    pub atom_disto: i32,
    pub is_standard: bool,
    pub is_present: bool,
}

/// One chain row (`Chain` dtype).
#[derive(Clone, Debug, PartialEq)]
pub struct ChainRow {
    /// PDB chain name (e.g. `"A"`), used by template alignment (`TemplateInfo.query_chain`).
    pub name: String,
    pub mol_type: i8,
    pub sym_id: i32,
    pub asym_id: i32,
    pub entity_id: i32,
    pub atom_idx: i32,
    pub atom_num: i32,
    pub res_idx: i32,
    pub res_num: i32,
    pub cyclic_period: i32,
}

/// Bond row subset used by tokenizer (`BondV2`: global atom indices + bond type).
#[derive(Clone, Debug, PartialEq)]
pub struct BondV2AtomRow {
    pub atom_1: i32,
    pub atom_2: i32,
    /// Raw `type` from structure (tokenizer stores `type + 1` in `TokenBondV2`).
    pub bond_type: i8,
}

/// Minimal structure matching Boltz `StructureV2` fields needed for `tokenize_structure`.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureV2Tables {
    pub atoms: Vec<AtomV2Row>,
    pub residues: Vec<ResidueRow>,
    pub chains: Vec<ChainRow>,
    /// Per-chain include flag (`struct.mask` in Python).
    pub chain_mask: Vec<bool>,
    /// Flattened coords table (`Coords`: one `[f32;3]` per atom slot in the ensemble slice).
    pub coords: Vec<[f32; 3]>,
    /// `ensemble[0]["atom_coord_idx"]` — base offset into `coords` for the first conformer.
    pub ensemble_atom_coord_idx: i32,
    pub bonds: Vec<BondV2AtomRow>,
}

impl StructureV2Tables {
    /// Coords row for `ensemble[0].atom_coord_idx + atom_index` (Boltz coord table).
    #[inline]
    pub(crate) fn ensemble_coords(&self, atom_index: i32) -> Option<[f32; 3]> {
        let o = self.ensemble_atom_coord_idx as i64 + i64::from(atom_index);
        let u = usize::try_from(o).ok()?;
        self.coords.get(u).copied()
    }
}

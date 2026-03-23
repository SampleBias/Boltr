// Boltr - Rust Native Boltz Implementation
// I/O Operations
//
// This crate handles all input/output operations including:
// - YAML/JSON config parsing
// - File I/O for sequences, structures, MSA data
// - MSA server communication
// - Output formatting

pub mod a3m;
pub mod ambiguous_atoms;
pub mod boltz_const;
pub mod structure_v2;
pub mod tokenize;
pub mod ref_atoms;
pub mod config;
pub mod download;
pub mod feature_batch;
pub mod format;
pub mod ligand_exclusion;
pub mod msa;
pub mod msa_csv;
pub mod msa_npz;
pub mod pad;
pub mod parser;
pub mod token_npz;
pub mod vdw_radii;

pub use a3m::{parse_a3m_path, parse_a3m_str, A3mMsa, A3mSequenceMeta};
pub use ambiguous_atoms::{
    pdb_atom_key, resolve_ambiguous_element, AMBIGUOUS_ATOMS_TOP_LEVEL_COUNT,
};
pub use structure_v2::{
    AtomV2Row, BondV2AtomRow, ChainRow, ResidueRow, StructureV2Tables,
};
pub use tokenize::boltz2::{
    compute_frame, tokenize_structure, TokenBondV2, TokenData,
};
pub use token_npz::{
    read_token_batch_npz_bytes, read_token_batch_npz_path, write_token_batch_npz_compressed,
    write_token_batch_npz_to_vec,
};
pub use boltz_const::{
    bond_type_id, chain_type_id, chirality_type_id, contact_conditioning_id, dna_letter_to_token_id,
    hybridization_type_id, method_type_id, ph_bin_id, pocket_contact_id, prot_letter_to_token_id,
    rna_letter_to_token_id, temperature_bin_id, token_id, token_name, unk_token_id, BOND_TYPES,
    CHAIN_TYPES, CHIRALITY_TYPES, CHUNK_SIZE_THRESHOLD, HYBRIDIZATION_MAP, INTERFACE_CUTOFF,
    MAX_MSA_SEQS, MAX_PAIRED_SEQS, MIN_COVERAGE_FRACTION, MIN_COVERAGE_RESIDUES, NUM_ELEMENTS,
    NUM_METHOD_TYPES, NUM_PH_BINS, NUM_TEMP_BINS,
    NUM_TOKENS, TOKENS, ATOM_INTERFACE_CUTOFF, UNK_BOND_TYPE, UNK_CHIRALITY_TYPE,
    UNK_HYBRIDIZATION_TYPE,
};
pub use ref_atoms::{
    center_atom_index, disto_atom_index, nucleic_backbone_atom_index, protein_backbone_atom_index,
    ref_atom_names, ref_atom_names_for_token, ref_atoms_key_from_token, ref_symmetry_groups,
    ref_symmetry_groups_for_token, NUCLEIC_BACKBONE_ATOM_NAMES, PROTEIN_BACKBONE_ATOM_NAMES,
};
pub use msa_csv::{parse_csv_path, parse_csv_str};
pub use msa_npz::{read_msa_npz_bytes, read_msa_npz_path, write_msa_npz_compressed};
pub use pad::{
    pad_1d, pad_ragged_rows, row_pad_mask_from_lengths, stack_tokens_2d, token_pad_mask, PadError,
};
pub use config::BoltzInput;
/// Backward-compatible name for parsed YAML root.
pub type BoltzConfig = BoltzInput;
pub use ligand_exclusion::{is_ligand_excluded, LIGAND_EXCLUSION_CODES, LIGAND_EXCLUSION_COUNT};
pub use feature_batch::{
    collate_feature_batches, stack_f32_views, CollateError, FeatureBatch, FeatureTensor,
};
pub use download::download_model_assets;
pub use format::PredictionRunSummary;
pub use msa::{write_a3m, MsaProcessor};
pub use parser::parse_input_path as parse_input;
pub use parser::{parse_input_path, parse_input_str};
pub use vdw_radii::{vdw_radius, VDW_RADII, VDW_RADII_LEN};

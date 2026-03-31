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
pub mod ccd;
pub mod collate_golden;
pub mod collate_pad;
pub mod config;
pub mod download;
pub mod feature_batch;
pub mod featurizer;
pub mod fixtures;
pub mod format;
pub mod inference_collate_serialize;
pub mod inference_dataset;
pub mod ligand_exclusion;
pub mod msa;
pub mod msa_csv;
pub mod msa_npz;
pub mod pad;
pub mod preprocess;
pub mod parser;
pub mod ref_atoms;
pub mod residue_constraints;
pub mod structure_v2;
pub mod structure_v2_npz;
pub mod token_npz;
pub mod token_v2_numpy;
pub mod tokenize;
pub mod vdw_radii;
pub mod write;

pub use a3m::{parse_a3m_path, parse_a3m_str, A3mMsa, A3mSequenceMeta};
pub use ambiguous_atoms::{
    pdb_atom_key, resolve_ambiguous_element, AMBIGUOUS_ATOMS_TOP_LEVEL_COUNT,
};
pub use boltz_const::{
    bond_type_id, chain_type_id, chain_type_to_out_single_type, chirality_type_id,
    clash_type_for_chain_pair, contact_conditioning_id, dna_letter_to_token_id,
    dna_token_id_to_letter, hybridization_type_id, is_canonical_token, method_type_id,
    out_type_weight, out_type_weight_af3, ph_bin_id, pocket_contact_id, prot_letter_to_token_id,
    prot_token_id_to_letter, rna_letter_to_token_id, rna_token_id_to_letter, temperature_bin_id,
    token_id, token_name, unk_token_id, ATOM_INTERFACE_CUTOFF, BOND_TYPES, CANONICAL_TOKENS,
    CHAIN_TYPES, CHIRALITY_TYPES, CHUNK_SIZE_THRESHOLD, CLASH_TYPES, HYBRIDIZATION_MAP,
    INTERFACE_CUTOFF, MAX_MSA_SEQS, MAX_PAIRED_SEQS, MIN_COVERAGE_FRACTION, MIN_COVERAGE_RESIDUES,
    NUM_ELEMENTS, NUM_METHOD_TYPES, NUM_PH_BINS, NUM_TEMP_BINS, NUM_TOKENS, OUT_SINGLE_TYPES,
    OUT_TYPES, TOKENS, UNK_BOND_TYPE, UNK_CHIRALITY_TYPE, UNK_HYBRIDIZATION_TYPE,
};
pub use collate_golden::{
    trunk_smoke_collate_path, trunk_smoke_collate_shapes, trunk_smoke_collate_shapes_from_path,
};
pub use ccd::{serialize_ccd_mol_json, CcdAtom, CcdBond, CcdMolData, CcdMolProvider};
pub use config::BoltzInput;
pub use featurizer::{
    ala_tokenized_smoke, atom_ref_data_from_ccd_mol, dummy_templates_as_feature_batch,
    inference_ensemble_features, inference_residue_constraint_features, load_dummy_templates_features,
    pad_template_tdim, process_atom_features, process_ensemble_features, process_msa_features,
    process_symmetry_features, process_symmetry_features_with_ligand_symmetries,
    process_template_features, process_token_features, get_ligand_symmetries_for_tokens,
    stack_template_feature_rows, token_feature_key_names, AffinityCropper, AffinityTokenized,
    AtomFeatureConfig, AtomFeatureTensors, AtomRefDataProvider, ChainSwap, DummyTemplateTensors,
    InferenceAtomRefProvider, MsaFeatureTensors, ResidueConstraintTensors, StandardAminoAcidRefData,
    SymmetryFeatures, TemplateAlignment, TokenFeatureTensors, ALA_STANDARD_HEAVY_ATOM_COUNT,
    ATOM_FEATURE_KEYS_ALA, CONTACT_CONDITIONING_NUM_CLASSES,
};
pub use msa_csv::{parse_csv_path, parse_csv_str};
pub use msa_npz::{read_msa_npz_bytes, read_msa_npz_path, write_msa_npz_compressed};
pub use pad::{
    pad_1d, pad_ragged_rows, row_pad_mask_from_lengths, stack_tokens_2d, token_pad_mask, PadError,
};
pub use ref_atoms::{
    center_atom_index, disto_atom_index, nucleic_backbone_atom_index, protein_backbone_atom_index,
    ref_atom_names, ref_atom_names_for_token, ref_atoms_key_from_token, ref_symmetry_groups,
    ref_symmetry_groups_for_token, NUCLEIC_BACKBONE_ATOM_NAMES, PROTEIN_BACKBONE_ATOM_NAMES,
};
pub use residue_constraints::ResidueConstraints;
pub use structure_v2::{
    AtomV2Row, BondV2AtomRow, ChainRow, EnsembleRow, ResidueRow, StructureV2Tables,
};
pub use structure_v2_npz::{
    read_structure_v2_npz_bytes, read_structure_v2_npz_path, write_structure_v2_npz_compressed,
    write_structure_v2_npz_to_vec,
};
pub use token_npz::{
    read_token_batch_npz_bytes, read_token_batch_npz_path, write_token_batch_npz_compressed,
    write_token_batch_npz_to_vec,
};
pub use token_v2_numpy::{
    decode_res_name_unicode_u8, encode_res_name_unicode_u8, pack_token_v2_row, unpack_token_v2_row,
    TOKEN_V2_NUMPY_ITEMSIZE,
};
pub use tokenize::boltz2::{compute_frame, tokenize_structure, TokenBondV2, TokenData};
/// Backward-compatible name for parsed YAML root.
pub type BoltzConfig = BoltzInput;
pub use collate_pad::{
    collate_inference_batches, pad_to_max_f32, InferenceCollateError, InferenceCollateResult,
    PadToMaxResult,
};
pub use download::download_model_assets;
pub use feature_batch::{
    collate_feature_batches, stack_f32_views, CollateError, FeatureBatch, FeatureTensor,
    INFERENCE_COLLATE_EXCLUDED_KEYS,
};
pub use fixtures::structure_v2_single_ala;
pub use format::PredictionRunSummary;
pub use inference_collate_serialize::{
    compare_inference_collate_to_safetensors, inference_collate_to_golden_tensors,
    write_inference_collate_golden, TRUNK_COLLATE_S_INPUT_LAST,
};
pub use inference_dataset::{
    affinity_asym_id_from_record, atom_features_from_inference_input, load_input,
    msa_features_from_inference_input, parse_manifest_json, parse_manifest_path,
    template_features_from_tokenized, token_features_from_inference_input,
    tokenize_boltz2_inference, trunk_smoke_feature_batch_from_inference_input, Boltz2ChainInfo,
    Boltz2InferenceInput, Boltz2InterfaceInfo, Boltz2Manifest, Boltz2Record, Boltz2Tokenized,
    Boltz2Tokenizer, StructureInfo, TemplateInfo, TokenizeBoltz2Input,
};
pub use ligand_exclusion::{is_ligand_excluded, LIGAND_EXCLUSION_CODES, LIGAND_EXCLUSION_COUNT};
pub use msa::{write_a3m, MsaProcessor};
pub use parser::parse_input_path as parse_input;
pub use parser::{parse_input_path, parse_input_str};
pub use preprocess::{
    copy_flat_preprocess_bundle, find_boltz_manifest_path, validate_native_eligible,
    write_native_preprocess_bundle, NativePreprocessError,
};
pub use vdw_radii::{vdw_radius, VDW_RADII, VDW_RADII_LEN};
pub use write::{
    confidence_json_filename, pae_npz_filename, pde_npz_filename, plddt_npz_filename,
    structure_v2_to_mmcif, structure_v2_to_pdb, write_affinity_json, write_confidence_json,
    write_pae_npz_path, write_pde_npz_path, write_plddt_npz_path, AffinitySummary, ConfidenceSummary,
    PredictionFileNames,
};

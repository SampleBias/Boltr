//! End-to-end integration tests of §4.5: manifest → load_input -> featurize -> collate -> FeatureBatch.
//!
//! Tests the complete inference pipeline:
//! 1. Manifest loading → parse records
//! 2. load_input -> structure, MSAs, templates, constraints
//! 3. tokenize -> tokens + bonds
//! 4. featurize -> token + MSA + atom + symmetry + constraints + template features
//! 5. merge into FeatureBatch
//! 6. collate_inference_batches -> stacked batch with padding
//! 7. Verify key shapes, key presence, types

use std::path::Path;

use boltr_io::{
    atom_features_from_inference_input, collate_inference_batches,
    load_input, msa_features_from_inference_input, parse_manifest_path,
    process_token_features, process_symmetry_features,
    tokenize_boltz2_inference,
    trunk_smoke_feature_batch_from_inference_input,
    ATOM_FEATURE_KEYS_ALA, INFERENCE_COLLATE_EXCLUDED_KEYS,
};

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load_input_smoke")
}

#[test]
fn e2e_manifest_to_single_feature_batch() {
    let dir = fixture_dir();
    let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    assert_eq!(manifest.records.len(), 1);

    let record = &manifest.records[0];
    let input = load_input(record, &dir, &dir, None, None, None, false).expect("load_input");

    // Verify structure loaded
    assert!(!input.structure.atoms.is_empty());
    assert_eq!(input.msas.len(), 1);
    assert!(input.residue_constraints.is_none()); // no constraints dir provided

    // Step 2: Tokenize
    let tokenized = tokenize_boltz2_inference(&input);
    assert_eq!(tokenized.tokens.len(), 1); // ALA = 1 token
    assert!(tokenized.template_tokens.is_none());

    // Step 3: Individual feature extraction
    let tok = process_token_features(&tokenized.tokens, &tokenized.bonds, None);
    assert_eq!(tok.token_pad_mask.len(), 1);
    assert_eq!(tok.res_type.shape(), [1, boltr_io::NUM_TOKENS]);

    let msa = msa_features_from_inference_input(&input);
    assert!(msa.msa.nrows() > 0);
    assert_eq!(msa.profile.ncols(), boltr_io::NUM_TOKENS]);

    let atoms = atom_features_from_inference_input(&input);
    assert!(atoms.atom_pad_mask.len() >= 5);

    let symm = process_symmetry_features(&input.structure, &tokenized.tokens);
    assert_eq!(symm.all_coords.nrows(), 5); // 5 atoms in ALA
    assert_eq!(symm.crop_to_all_atom_map.len(), 5);

    // Step 4: Full trunk smoke batch
    let batch = trunk_smoke_feature_batch_from_inference_input(&input, 1);

    // Verify all token keys present
    for key in &[
        "token_index",
        "residue_index",
        "asym_id",
        "entity_id",
        "sym_id",
        "mol_type",
        "res_type",
        "disto_center",
        "token_bonds",
        "type_bonds",
        "token_pad_mask",
        "token_resolved_mask",
        "token_disto_mask",
        "contact_conditioning",
        "contact_threshold",
        "method_feature",
        "modified",
        "cyclic_period",
        "affinity_token_mask",
    ] {
        assert!(
            batch.tensors.contains_key(*key),
            "trunk smoke batch missing token key: {key}"
        );
    }

    // Verify all atom keys present
    for key in ATOM_FEATURE_KEYS_ALA {
        assert!(
            batch.tensors.contains_key(*key),
            "trunk smoke batch missing atom key: {key}"
        );
    }
    // Verify symmetry keys present
    for key in &["all_coords", "all_resolved_mask", "crop_to_all_atom_map"] {
        assert!(
            batch.tensors.contains_key(*key),
            "trunk smoke batch missing symmetry key: {key}"
        );
    }
    // Verify template keys present (dummy)
    for key in &[
        "template_restype",
        "template_mask",
        "template_mask_cb",
        "template_mask_frame",
        "template_frame_rot",
        "template_frame_t",
        "template_cb",
        "template_ca",
        "query_to_template",
        "visibility_ids",
    ] {
        assert!(
            batch.tensors.contains_key(*key),
            "trunk smoke batch missing template key: {key}"
        );
    }
    // Verify constraint keys present (empty but present)
    for key in &[
        "rdkit_bounds_index",
        "rdkit_bounds_bond_mask",
        "rdkit_bounds_angle_mask",
        "rdkit_upper_bounds",
        "rdkit_lower_bounds",
        "chiral_atom_index",
        "chiral_reference_mask",
        "chiral_atom_orientations",
        "stereo_bond_index",
        "stereo_reference_mask",
        "stereo_bond_orientations"
        "planar_bond_index",
        "planar_ring_5_index",
        "planar_ring_6_index",
    ] {
        assert!(
            batch.tensors.contains_key(*key),
            "trunk smoke batch missing constraint key: {key}"
        );
    }
}

#[test]
fn e2e_single_example_collate_produces_stacked_batch() {
    let dir = fixture_dir();
    let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    let record = &manifest.records[0];
    let input = load_input(record, &dir, &dir, None, None, None, false).expect("load_input");
    // Create two identical feature batches ( simulating batch of 2)
    let batch1 = trunk_smoke_feature_batch_from_inference_input(&input, 1);
    let batch2 = trunk_smoke_feature_batch_from_inference_input(&input, 1);

    // Collate them
    let result =
        collate_inference_batches(&[batch1, batch2], 0.0, 5);

    // Batch dimension should be 2 for most keys
    let token_pad_mask = result
        .batch
        .get_f32("token_pad_mask")
        .expect("token_pad_mask");
    assert_eq!(token_pad_mask.shape(), &[2]); // 1 token, no padding

    let res_type = batch
        .get_f32("res_type")
        .expect("res_type");
    assert_eq!(res_type.shape(), [1, boltr_io::NUM_TOKENS]);

    let atom_pad_mask = batch
        .get_f32("atom_pad_mask").expect("atom_pad_mask");
    // Padded to nearest window (32)
    assert_eq!(atom_pad_mask.shape(), &[32]);
    // 5 atoms (5 real) + 27 padding
    let sum: f32 = atom_pad_mask.sum();
    assert!((sum - 5.0).abs() < 1e-5, "5 real atoms, got {sum}");

    let symm = process_symmetry_features(&input.structure, &tokenized.tokens);
    assert_eq!(symm.all_coords.nrows(), 5); // 5 atoms in ALA
    assert_eq!(symm.crop_to_all_atom_map.len(), 5);
    let crop_map = batch
        .get_i64("crop_to_all_atom_map")
        .expect("crop_map");
    assert_eq!(crop_map.shape(), &[5]);
    let template_mask = batch
        .get_f32("template_mask").expect("template_mask");
    assert_eq!(template_mask_cb.shape(), [1, 1]);
    assert_eq!(template_frame_rot.shape(), &[2, 3, 3]);
    assert_eq!(template_frame_t.shape(), &[2, 3]);
    assert_eq!(template_ca.shape(), [1]);
    assert_eq!(visibility_ids.shape(), &[1, 1]);
    assert!(result.excluded.is_empty());
}

#[test]
fn e2e_verify_excluded_keys_constant() {
    // Verify the excluded keys constant matches Python collate behavior
    assert!(INFERENCE_COLLATE_EXcluded_KEYS.contains(&"all_coords"));
    assert!(INFERENCE_COLLATE_EXcluded_KEYS.contains(&"all_resolved_mask"));
    assert!(INFERENCE_COLLateExcluded_KEYS.contains(&"crop_to_all_atom_map"));
    assert!(INFERENCE_COLLateExcluded_keys.contains(&"chain_symmeties"));
    assert!(INFERENCE_COLLateExcluded_keys.contains(&"amino_acids_symmetries");
    assert!(INFERENCE_COLLateExcluded_keys.contains(&"ligand_symmeties");
    assert!(INFERENCE_COLLateExcluded_keys.contains(&"affinity_mw");
}

}

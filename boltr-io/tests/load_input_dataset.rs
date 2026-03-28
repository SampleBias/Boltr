//! Integration tests for [`boltr_io::inference_dataset`] (`load_input` / manifest parsing).

use std::path::{Path, PathBuf};

use boltr_io::{
    load_input, msa_features_from_inference_input, parse_manifest_path, process_token_features,
    structure_v2_single_ala, token_features_from_inference_input,
    tokenize::boltz2::tokenize_structure, trunk_smoke_feature_batch_from_inference_input,
    ATOM_FEATURE_KEYS_ALA, A3mMsa, A3mSequenceMeta,
};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load_input_smoke")
}

#[test]
fn parse_manifest_fixture() {
    let dir = fixture_dir();
    let m = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    assert_eq!(m.records.len(), 1);
    assert_eq!(m.records[0].id, "test_ala");
}

#[test]
fn load_input_smoke_fixture() {
    let dir = fixture_dir();
    let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    let record = &manifest.records[0];

    let input = load_input(
        record,
        &dir,
        &dir,
        None,
        None,
        None,
        false,
    )
    .expect("load_input");

    assert!(!input.structure.atoms.is_empty());
    assert_eq!(input.msas.len(), 1);
    let msa = input.msas.get(&0).expect("chain 0 msa");
    assert_eq!(
        *msa,
        A3mMsa {
            residues: vec![2, 3, 4],
            deletions: vec![(1, 2)],
            sequences: vec![A3mSequenceMeta {
                seq_idx: 0,
                taxonomy_id: 9606,
                res_start: 0,
                res_end: 3,
                del_start: 0,
                del_end: 1,
            }],
        }
    );
    assert!(input.templates.is_none());
}

#[test]
fn load_input_skips_msa_id_minus_one() {
    let dir = fixture_dir();
    let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    let mut record = manifest.records[0].clone();
    record.chains[0].msa_id = serde_json::json!(-1);

    let input = load_input(&record, &dir, &dir, None, None, None, false).expect("load");
    assert!(input.msas.is_empty());
}

/// `test_ala.npz` is the packed ALA golden; same tables as `structure_v2_single_ala`, so token
/// features match `token_features_ala_golden.safetensors` (see `featurizer::token_features_golden`).
#[test]
fn load_input_msa_features_runs() {
    let dir = fixture_dir();
    let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    let input = load_input(&manifest.records[0], &dir, &dir, None, None, None, false)
        .expect("load_input");
    let m = msa_features_from_inference_input(&input);
    assert_eq!(m.msa.ncols(), m.profile.nrows());
    assert_eq!(m.profile.ncols(), boltr_io::NUM_TOKENS);
}

/// Featurizer keys expected by [`boltr_io::collate_golden::trunk_smoke_collate_path`] minus `s_inputs`
/// (computed inside the model, not the featurizer), plus atom keys from [`ATOM_FEATURE_KEYS_ALA`].
#[test]
fn trunk_smoke_feature_batch_covers_collate_manifest_keys() {
    let dir = fixture_dir();
    let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    let input = load_input(&manifest.records[0], &dir, &dir, None, None, None, false)
        .expect("load_input");
    let batch = trunk_smoke_feature_batch_from_inference_input(&input, 1);
    for key in [
        "token_pad_mask",
        "msa",
        "msa_paired",
        "msa_mask",
        "has_deletion",
        "deletion_value",
        "deletion_mean",
        "profile",
        "template_restype",
        "template_mask",
    ] {
        assert!(
            batch.tensors.contains_key(key),
            "trunk smoke batch missing {key}"
        );
    }
    for key in ATOM_FEATURE_KEYS_ALA {
        assert!(
            batch.tensors.contains_key(*key),
            "trunk smoke batch missing atom key {key}"
        );
    }
}

#[test]
fn load_input_token_features_match_canonical_ala_golden_path() {
    let dir = fixture_dir();
    let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
    let input = load_input(&manifest.records[0], &dir, &dir, None, None, None, false)
        .expect("load_input");
    let got = token_features_from_inference_input(&input);

    let (tokens, bonds) = tokenize_structure(&structure_v2_single_ala(), None);
    let expected = process_token_features(&tokens, &bonds, None);
    assert_eq!(got, expected);
}

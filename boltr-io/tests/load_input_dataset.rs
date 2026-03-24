//! Integration tests for [`boltr_io::inference_dataset`] (`load_input` / manifest parsing).

use std::path::{Path, PathBuf};

use boltr_io::{load_input, parse_manifest_path, A3mMsa, A3mSequenceMeta};

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

//! Tests for preprocess bundle discovery and native writer.

use std::fs;
use boltr_io::{
    copy_flat_preprocess_bundle, find_boltz_manifest_path, parse_input_path, parse_manifest_path,
    validate_native_eligible, write_native_preprocess_bundle,
};

#[test]
fn find_manifest_prefers_boltz_results_stem() {
    let tmp = tempfile::tempdir().unwrap();
    let stem = "case1";
    let processed = tmp
        .path()
        .join(format!("boltz_results_{stem}"))
        .join("processed");
    fs::create_dir_all(processed.join("structures")).unwrap();
    let manifest_path = processed.join("manifest.json");
    let npz = processed.join("structures").join("rec1.npz");
    fs::write(
        &manifest_path,
        r#"{"records":[{"id":"rec1","structure":{},"chains":[{"chain_id":0,"chain_name":"A","mol_type":0,"cluster_id":0,"msa_id":0,"num_residues":1,"valid":true}],"interfaces":[]}]}"#,
    )
    .unwrap();
    fs::write(&npz, b"x").unwrap();
    let found = find_boltz_manifest_path(tmp.path(), stem).unwrap();
    assert_eq!(found, manifest_path);
}

#[test]
fn find_manifest_prefers_processed_stem() {
    let tmp = tempfile::tempdir().unwrap();
    let stem = "case1";
    let processed = tmp.path().join("processed").join(stem);
    fs::create_dir_all(&processed).unwrap();
    let manifest_path = processed.join("manifest.json");
    let npz = processed.join("rec1.npz");
    fs::write(
        &manifest_path,
        r#"{"records":[{"id":"rec1","structure":{},"chains":[{"chain_id":0,"chain_name":"A","mol_type":0,"cluster_id":0,"msa_id":0,"num_residues":1,"valid":true}],"interfaces":[]}]}"#,
    )
    .unwrap();
    fs::write(&npz, b"x").unwrap();
    let found = find_boltz_manifest_path(tmp.path(), stem).unwrap();
    assert_eq!(found, manifest_path);
}

#[test]
fn copy_flat_bundle_copies_npz() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("manifest.json"),
        r#"{"records":[{"id":"r1","structure":{},"chains":[{"chain_id":0,"chain_name":"A","mol_type":0,"cluster_id":0,"msa_id":0,"num_residues":1,"valid":true}],"interfaces":[]}]}"#,
    )
    .unwrap();
    fs::write(src.join("r1.npz"), b"npz").unwrap();
    fs::write(src.join("0.npz"), b"msa").unwrap();
    let dst = tmp.path().join("dst");
    copy_flat_preprocess_bundle(&src.join("manifest.json"), &dst, false).unwrap();
    assert!(dst.join("manifest.json").is_file());
    assert_eq!(fs::read(dst.join("r1.npz")).unwrap(), b"npz");
    assert_eq!(fs::read(dst.join("0.npz")).unwrap(), b"msa");
}

#[test]
fn copy_flat_bundle_copies_from_structures_and_msa_dirs() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(src.join("structures")).unwrap();
    fs::create_dir_all(src.join("msa")).unwrap();
    fs::write(
        src.join("manifest.json"),
        r#"{"records":[{"id":"r1","structure":{},"chains":[{"chain_id":0,"chain_name":"A","mol_type":0,"cluster_id":0,"msa_id":0,"num_residues":1,"valid":true}],"interfaces":[]}]}"#,
    )
    .unwrap();
    fs::write(src.join("structures").join("r1.npz"), b"npz").unwrap();
    fs::write(src.join("msa").join("0.npz"), b"msa").unwrap();
    let dst = tmp.path().join("dst");
    copy_flat_preprocess_bundle(&src.join("manifest.json"), &dst, false).unwrap();
    assert_eq!(fs::read(dst.join("r1.npz")).unwrap(), b"npz");
    assert_eq!(fs::read(dst.join("0.npz")).unwrap(), b"msa");
}

#[test]
fn validate_native_eligible_accepts_empty_templates_and_constraints() {
    let tmp = tempfile::tempdir().unwrap();
    let yaml = tmp.path().join("x.yaml");
    fs::write(
        &yaml,
        r#"sequences:
  - protein:
      id: A
      sequence: "A"
templates: []
constraints: []
"#,
    )
    .unwrap();
    let parsed = parse_input_path(&yaml).unwrap();
    assert!(validate_native_eligible(&parsed).is_ok());
}

#[test]
fn native_bundle_writes_manifest_and_structure() {
    let tmp = tempfile::tempdir().unwrap();
    let yaml = tmp.path().join("t.yaml");
    fs::write(
        &yaml,
        r#"sequences:
  - protein:
      id: A
      sequence: "A"
      msa: empty
"#,
    )
    .unwrap();
    write_native_preprocess_bundle(&yaml, tmp.path(), Some("rec_x"), Some(16), None).unwrap();
    let m = parse_manifest_path(&tmp.path().join("manifest.json")).unwrap();
    assert_eq!(m.records[0].id, "rec_x");
    assert!(tmp.path().join("rec_x.npz").is_file());
    assert!(tmp.path().join("0.npz").is_file());
}

mod common;

use std::path::PathBuf;
use std::process::Command;

fn boltr_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_boltr") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return pb;
        }
    }
    let mut p = std::env::current_exe().expect("current_exe");
    p.pop();
    p.pop();
    p.push("boltr");
    assert!(
        p.is_file(),
        "boltr binary missing at {} (rebuild with cargo test -p boltr-cli)",
        p.display()
    );
    p
}

#[test]
fn tokens_to_npz_from_structure_npz_roundtrip() {
    let dir = tempfile::tempdir_in(common::test_temp_root()).unwrap();
    let s = boltr_io::structure_v2_single_ala();
    let str_npz = dir.path().join("structure.npz");
    boltr_io::write_structure_v2_npz_compressed(&str_npz, &s).unwrap();
    let tok_npz = dir.path().join("tokens.npz");
    let status = Command::new(boltr_bin())
        .args(["tokens-to-npz", "-i"])
        .arg(&str_npz)
        .arg("-o")
        .arg(&tok_npz)
        .status()
        .unwrap();
    assert!(status.success());
    let (tokens, bonds) = boltr_io::read_token_batch_npz_path(&tok_npz).unwrap();
    assert_eq!(tokens.len(), 1);
    assert!(bonds.is_empty());
}

#[test]
fn tokens_to_npz_ala_demo_writes_file() {
    let dir = tempfile::tempdir_in(common::test_temp_root()).unwrap();
    let npz = dir.path().join("tok.npz");

    let status = Command::new(boltr_bin())
        .args(["tokens-to-npz", "ala", "-o"])
        .arg(&npz)
        .status()
        .unwrap();
    assert!(status.success());
    assert!(npz.is_file());

    let (tokens, bonds) = boltr_io::read_token_batch_npz_path(&npz).unwrap();
    assert_eq!(tokens.len(), 1);
    assert!(bonds.is_empty());
    assert_eq!(tokens[0].res_name, "ALA");
}

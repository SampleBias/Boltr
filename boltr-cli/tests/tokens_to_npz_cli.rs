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
fn tokens_to_npz_ala_demo_writes_file() {
    let dir = tempfile::tempdir().unwrap();
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

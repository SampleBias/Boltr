use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn boltr_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_boltr") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return pb;
        }
    }
    // Same `target/*/debug/` as this integration test (`deps/…` → `boltr`).
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
fn msa_to_npz_writes_file() {
    let dir = tempfile::tempdir().unwrap();
    let a3m = dir.path().join("tiny.a3m");
    fs::write(&a3m, ">q\nACDEF\n").unwrap();
    let npz = dir.path().join("tiny.npz");

    let status = Command::new(boltr_bin())
        .args(["msa-to-npz"])
        .arg("-o")
        .arg(&npz)
        .arg(&a3m)
        .status()
        .unwrap();
    assert!(status.success());
    assert!(npz.is_file());
    let back = boltr_io::read_msa_npz_path(&npz).unwrap();
    assert_eq!(back.sequences.len(), 1);
}

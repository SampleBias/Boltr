mod common;

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

fn minimal_yaml_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../boltr-io/tests/fixtures/minimal_protein.yaml");
    p
}

#[test]
fn predict_summary_json_no_tch() {
    let dir = tempfile::tempdir_in(common::test_temp_root()).unwrap();
    let out = dir.path().join("output");

    let status = Command::new(boltr_bin())
        .args([
            "predict",
            minimal_yaml_path().to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "boltr predict should succeed without tch");

    // Check boltr_run_summary.json was written
    let summary_path = out.join("boltr_run_summary.json");
    assert!(summary_path.is_file(), "missing {}", summary_path.display());

    let text = fs::read_to_string(&summary_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["use_msa_server"], false);
    assert_eq!(v["device"], "cpu");
    assert_eq!(v["affinity"], false);
}

#[test]
fn predict_with_affinity_flag() {
    let dir = tempfile::tempdir_in(common::test_temp_root()).unwrap();
    let out = dir.path().join("out_aff");

    let status = Command::new(boltr_bin())
        .args([
            "predict",
            minimal_yaml_path().to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--affinity",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let summary_path = out.join("boltr_run_summary.json");
    assert!(summary_path.is_file());
    let text = fs::read_to_string(&summary_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["affinity"], true);
}

#[test]
fn predict_help_shows_all_flags() {
    let output = Command::new(boltr_bin())
        .args(["predict", "--help"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Check Boltz-compatible flags appear
    assert!(stdout.contains("--recycling-steps"));
    assert!(stdout.contains("--sampling-steps"));
    assert!(stdout.contains("--diffusion-samples"));
    assert!(stdout.contains("--max-parallel-samples"));
    assert!(stdout.contains("--step-scale"));
    assert!(stdout.contains("--output-format"));
    assert!(stdout.contains("--max-msa-seqs"));
    assert!(stdout.contains("--checkpoint"));
    assert!(stdout.contains("--affinity-checkpoint"));
    assert!(stdout.contains("--affinity-mw-correction"));
    assert!(stdout.contains("--use-msa-server"));
    assert!(stdout.contains("--msa-server-url"));
    assert!(stdout.contains("--use-potentials"));
    assert!(stdout.contains("--spike-only"));
    assert!(stdout.contains("--write-full-pae"));
    assert!(stdout.contains("--write-full-pde"));
    assert!(stdout.contains("--override"));
    assert!(stdout.contains("--num-samples"));
}

#[test]
fn download_help() {
    let output = Command::new(boltr_bin())
        .args(["download", "--help"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cache-dir") || stdout.contains("cache"));
    assert!(stdout.contains("boltz2"));
}

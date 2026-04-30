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
    let dev = v["device"].as_str().expect("device string");
    assert!(
        dev == "cpu" || dev == "cuda" || dev.starts_with("cuda:"),
        "unexpected resolved device: {dev}"
    );
    assert_eq!(v["device_requested"], "auto");
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
    assert!(stdout.contains("--quality-preset"));
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
    assert!(stdout.contains("--extra-mols-dir"));
    assert!(stdout.contains("--constraints-dir"));
    assert!(stdout.contains("--preprocess-auto-extras"));
    assert!(stdout.contains("--ensemble-ref"));
    assert!(stdout.contains("--preprocess-cuda-visible-devices"));
    assert!(stdout.contains("--preprocess-boltz-cpu"));
    assert!(stdout.contains("--preprocess-auto-boltz-gpu"));
    assert!(stdout.contains("--preprocess-post-boltz-empty-cache"));
    assert!(stdout.contains("--device"));
    assert!(stdout.contains("auto"));
    assert!(stdout.contains("BOLTR_AUTO_MIN_FREE_VRAM_MB"));
    assert!(stdout.contains("BOLTR_AUTO_PREPROCESS_BOLTZ_CPU"));
}

#[test]
fn prediction_parity_baseline_documented() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../docs/PREDICTION_PARITY_BASELINE.md");
    let text = fs::read_to_string(&path).expect("read parity baseline doc");
    for fixture in [
        "protein_minimal",
        "protein_ligand_affinity",
        "template_complex",
        "constrained_complex",
    ] {
        assert!(text.contains(fixture), "missing fixture {fixture}");
    }
    assert!(text.contains("BOLTR_RUN_PREDICT_PARITY"));
}

#[test]
fn predict_parity_fixture_matrix_is_documented() {
    if std::env::var("BOLTR_RUN_PREDICT_PARITY").ok().as_deref() != Some("1") {
        return;
    }
    let boltz = Command::new("boltz").arg("--help").status();
    assert!(
        boltz.map(|s| s.success()).unwrap_or(false),
        "BOLTR_RUN_PREDICT_PARITY=1 requires upstream `boltz` on PATH"
    );
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../docs/PREDICTION_PARITY_BASELINE.md");
    let text = fs::read_to_string(&path).expect("read parity baseline doc");
    assert!(text.contains("protein_ligand_affinity"));
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

#[test]
fn doctor_json_reports_tch_feature() {
    let output = Command::new(boltr_bin())
        .args(["doctor", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    #[cfg(feature = "tch")]
    assert_eq!(v["tch_feature"], true);
    #[cfg(not(feature = "tch"))]
    assert_eq!(v["tch_feature"], false);
    #[cfg(not(feature = "tch"))]
    assert!(v["libtorch_runtime_ok"].is_null());
}

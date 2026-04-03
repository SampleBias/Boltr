//! Regression: YAML parse + summary JSON (no LibTorch required).

use std::path::PathBuf;

use boltr_io::{parse_input_path, PredictionRunSummary};

#[test]
fn prediction_summary_roundtrip() {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../boltr-io/tests/fixtures/minimal_protein.yaml");
    let input = parse_input_path(&p).expect("parse");
    let s = PredictionRunSummary::from_input(
        p.display().to_string(),
        &input,
        false,
        "cpu",
        None,
        1,
        "test",
        false,
        false,
        false,
        None,
    );
    assert_eq!(s.chain_ids, vec!["A".to_string()]);
    assert!(!s.affinity);
    assert!(!s.use_potentials);
    assert!(!s.spike_only);
    assert!(s.boltr_predict_args_path.is_none());
}

#[test]
fn summary_json_roundtrip_to_file() {
    let dir = tempfile::tempdir_in(test_temp_root()).unwrap();
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../boltr-io/tests/fixtures/minimal_protein.yaml");
    let input = parse_input_path(&p).expect("parse");
    let s = PredictionRunSummary::from_input(
        "test.yaml",
        &input,
        true,
        "cuda:0",
        None,
        5,
        "tch",
        true,
        true,
        false,
        Some("boltr_predict_args.json".to_string()),
    );
    let path = dir.path().join("summary.json");
    s.write_json(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["use_msa_server"], true);
    assert_eq!(v["device"], "cuda:0");
    assert_eq!(v["num_samples"], 5);
    assert_eq!(v["affinity"], true);
    assert_eq!(v["use_potentials"], true);
    assert_eq!(v["boltr_predict_args_path"], "boltr_predict_args.json");
}

/// Shared temp root under `target/` for CLI tests.
fn test_temp_root() -> PathBuf {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("boltr-cli-test-tmp");
    std::fs::create_dir_all(&root).expect("create test temp root under target/");
    root
}

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

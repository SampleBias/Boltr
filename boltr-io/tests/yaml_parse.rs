use std::path::PathBuf;

use boltr_io::parse_input_path;

#[test]
fn parse_minimal_protein_fixture() {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/minimal_protein.yaml");
    let input = parse_input_path(&p).expect("parse fixture");
    assert_eq!(input.summary_chain_ids(), vec!["A".to_string()]);
}

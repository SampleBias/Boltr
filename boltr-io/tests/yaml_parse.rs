use std::path::PathBuf;

use boltr_io::{parse_input_path, parse_input_str};

/// Same assertion as the module doc example on `BoltzInput` (doctest disabled on `[lib]` — see Cargo.toml).
#[test]
fn boltz_input_deserializes_doc_example_yaml() {
    let yaml = r#"
sequences:
  - protein:
      id: A
      sequence: "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSQ"
      msa: ./msa.a3m
  - ligand:
      id: B
      ccd: HEM
constraints:
  - bond:
      atom1: [A, 1, N]
      atom2: [B, 1, C1]
templates:
  - cif: template.cif
    chain_id: A
"#;
    let input = parse_input_str(yaml).expect("parse doc-example YAML");
    assert_eq!(input.summary_chain_ids(), vec!["A", "B"]);
}

#[test]
fn parse_minimal_protein_fixture() {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/minimal_protein.yaml");
    let input = parse_input_path(&p).expect("parse fixture");
    assert_eq!(input.summary_chain_ids(), vec!["A".to_string()]);
}

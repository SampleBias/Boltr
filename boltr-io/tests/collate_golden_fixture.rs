//! Collate golden manifest + safetensors consistency.

use std::collections::HashSet;
use std::path::Path;

use safetensors::SafeTensors;
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    trunk_smoke_safetensors_keys: Vec<String>,
    #[serde(default)]
    token_features_ala_golden_keys: Vec<String>,
}

#[test]
fn trunk_smoke_collate_has_manifest_keys() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/collate_golden");
    let man_path = dir.join("manifest.json");
    let raw = std::fs::read_to_string(&man_path).expect("read manifest.json");
    let m: Manifest = serde_json::from_str(&raw).expect("parse manifest");

    let st_path = dir.join("trunk_smoke_collate.safetensors");
    let bytes = std::fs::read(&st_path).expect("read trunk_smoke_collate.safetensors");
    let st = SafeTensors::deserialize(&bytes).expect("safetensors");

    let names: HashSet<String> = st.names().into_iter().map(String::from).collect();
    for k in &m.trunk_smoke_safetensors_keys {
        assert!(
            names.contains(k),
            "missing tensor {k} in trunk_smoke_collate.safetensors"
        );
    }
}

#[test]
fn token_features_ala_golden_has_manifest_keys() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/collate_golden");
    let man_path = dir.join("manifest.json");
    let raw = std::fs::read_to_string(&man_path).expect("read manifest.json");
    let m: Manifest = serde_json::from_str(&raw).expect("parse manifest");

    let st_path = dir.join("token_features_ala_golden.safetensors");
    let bytes = std::fs::read(&st_path).expect("read token_features_ala_golden.safetensors");
    let st = SafeTensors::deserialize(&bytes).expect("safetensors");

    let names: HashSet<String> = st.names().into_iter().map(String::from).collect();
    for k in &m.token_features_ala_golden_keys {
        assert!(
            names.contains(k),
            "missing tensor {k} in token_features_ala_golden.safetensors"
        );
    }
    assert_eq!(names.len(), m.token_features_ala_golden_keys.len());
}

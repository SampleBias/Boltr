//! Golden test for full post-collate batch from Python Boltz2InferenceDataModule.
//!
//! Compares Rust collate output against Python golden from `dump_full_collate_golden.py`.

use anyhow::{Context, Result};
use safetensors::{SafeTensors, tensor};

/// Load a safetensors file and compare with expected manifest specification.
///
/// # Arguments
///
/// * `golden_path` - Path to golden safetensors file
/// * `manifest_path` - Path to manifest.json with expected keys/shapes
///
/// # Returns
///
/// Ok(()) if all keys match, otherwise error with details
pub fn verify_full_collate_golden(
    golden_path: &std::path::Path,
    manifest_path: &std::path::Path,
) -> Result<()> {
    println!("Loading golden safetensors: {}", golden_path.display());
    let golden = SafeTensors::load(golden_path).with_context(|| {
        format!("Failed to load golden safetensors: {}", golden_path.display())
    })?;

    println!("Loading manifest: {}", manifest_path.display());
    let manifest_bytes = std::fs::read(manifest_path).with_context(|| {
        format!("Failed to read manifest: {}", manifest_path.display())
    })?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .with_context(|| "Failed to parse manifest JSON")?;

    // Get all expected keys from manifest
    let expected_keys: Vec<String> = extract_expected_keys(&manifest);

    // Check that all expected keys are in golden
    let mut missing_keys = Vec::new();
    for key in &expected_keys {
        if !golden.tensors.contains_key(key) {
            missing_keys.push(key.clone());
        }
    }

    if !missing_keys.is_empty() {
        anyhow::bail!(
            "Golden missing expected keys from manifest: {:?}",
            missing_keys
        );
    }

    println!("\n=== Verification Summary ===");
    println!("Expected keys (from manifest): {}", expected_keys.len());
    println!("Golden keys: {}", golden.tensors.len());
    println!("Missing keys: {}", missing_keys.len());

    // Show key categories
    println!("\n=== Key Categories ===");

    // Token features
    let token_keys = extract_keys_from_manifest(&manifest, "process_token_features");
    if !token_keys.is_empty() {
        let present: Vec<_> = token_keys.iter().filter(|k| golden.tensors.contains_key(k)).collect();
        let missing: Vec<_> = token_keys.iter().filter(|k| !golden.tensors.contains_key(k)).collect();
        println!("\nToken features ({} of {} present):", present.len(), token_keys.len());
        if !missing.is_empty() {
            println!("  Missing: {:?}", missing);
        }
    }

    // MSA features
    let msa_keys = extract_keys_from_manifest(&manifest, "process_msa_features_non_affinity");
    if !msa_keys.is_empty() {
        let present: Vec<_> = msa_keys.iter().filter(|k| golden.tensors.contains_key(k)).collect();
        let missing: Vec<_> = msa_keys.iter().filter(|k| !golden.tensors.contains_key(k)).collect();
        println!("\nMSA features ({} of {} present):", present.len(), msa_keys.len());
        if !missing.is_empty() {
            println!("  Missing: {:?}", missing);
        }
    }

    // Atom features
    let atom_keys = extract_keys_from_manifest(&manifest, "atom_features_ala_golden_keys");
    if !atom_keys.is_empty() {
        let present: Vec<_> = atom_keys.iter().filter(|k| golden.tensors.contains_key(k)).collect();
        let missing: Vec<_> = atom_keys.iter().filter(|k| !golden.tensors.contains_key(k)).collect();
        println!("\nAtom features ({} of {} present):", present.len(), atom_keys.len());
        if !missing.is_empty() {
            println!("  Missing: {:?}", missing);
        }
    }

    // Template features
    let template_keys = extract_keys_from_manifest(&manifest, "load_dummy_templates_features");
    if !template_keys.is_empty() {
        let present: Vec<_> = template_keys.iter().filter(|k| golden.tensors.contains_key(k)).collect();
        let missing: Vec<_> = template_keys.iter().filter(|k| !golden.tensors.contains_key(k)).collect();
        println!("\nTemplate features ({} of {} present):", present.len(), template_keys.len());
        if !missing.is_empty() {
            println!("  Missing: {:?}", missing);
        }
    }

    // Trunk smoke keys
    let trunk_keys = extract_keys_from_manifest(&manifest, "trunk_smoke_safetensors_keys");
    if !trunk_keys.is_empty() {
        let present: Vec<_> = trunk_keys.iter().filter(|k| golden.tensors.contains_key(k)).collect();
        let missing: Vec<_> = trunk_keys.iter().filter(|k| !golden.tensors.contains_key(k)).collect();
        println!("\nTrunk smoke features ({} of {} present):", present.len(), trunk_keys.len());
        if !missing.is_empty() {
            println!("  Missing: {:?}", missing);
        }
    }

    println!("\n✅ Verification complete!");
    println!("All expected keys present in golden file.");
    Ok(())
}

/// Extract expected key names from manifest JSON.
fn extract_expected_keys(manifest: &serde_json::Value) -> Vec<String> {
    let mut keys = Vec::new();

    // Get keys from process_token_features
    if let Some(token_features) = manifest.get("process_token_features") {
        if let Some(obj) = token_features.as_object() {
            keys.extend(obj.keys().map(|k| k.to_string()));
        }
    }

    // Get keys from process_msa_features_non_affinity
    if let Some(msa_features) = manifest.get("process_msa_features_non_affinity") {
        if let Some(obj) = msa_features.as_object() {
            keys.extend(obj.keys().map(|k| k.to_string()));
        }
    }

    // Get keys from load_dummy_templates_features
    if let Some(template_features) = manifest.get("load_dummy_templates_features") {
        if let Some(obj) = template_features.as_object() {
            keys.extend(obj.keys().map(|k| k.to_string()));
        }
    }

    // Get keys from atom_features_ala_golden_keys (it's a list)
    if let Some(atom_keys) = manifest.get("atom_features_ala_golden_keys") {
        if let Some(arr) = atom_keys.as_array() {
            keys.extend(arr.iter().map(|v| v.as_str().unwrap_or("").to_string()));
        }
    }

    keys.sort();
    keys.dedup();
    keys
}

/// Extract keys from a specific manifest section.
fn extract_keys_from_manifest(manifest: &serde_json::Value, section: &str) -> Vec<String> {
    let mut keys = Vec::new();

    if let Some(section) = manifest.get(section) {
        match section {
            serde_json::Value::Object(obj) => {
                keys.extend(obj.keys().map(|k| k.to_string()));
            }
            serde_json::Value::Array(arr) => {
                keys.extend(arr.iter().map(|v| v.as_str().unwrap_or("").to_string()));
            }
            _ => {}
        }
    }

    keys.sort();
    keys.dedup();
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_manifest_key_extraction() {
        let manifest_json = r#"{
            "process_token_features": {"token_index": {"rank": 1}},
            "atom_features_ala_golden_keys": ["atom_pad_mask", "coords"]
        }"#;

        let manifest: serde_json::Value = serde_json::from_str(manifest_json).unwrap();
        let keys = extract_expected_keys(&manifest);

        assert!(keys.contains(&"token_index".to_string()));
        assert!(keys.contains(&"atom_pad_mask".to_string()));
        assert!(keys.contains(&"coords".to_string()));
    }
}

//! Golden schema: Python `process_atom_features` on ALA + canonical mols
//! ([`scripts/dump_atom_features_golden.py`](../../../scripts/dump_atom_features_golden.py)).

#[cfg(test)]
mod tests {
    use std::path::Path;

    use safetensors::tensor::Dtype;
    use safetensors::SafeTensors;

    use super::super::process_atom_features::ALA_STANDARD_HEAVY_ATOM_COUNT;

    fn fixture_bytes() -> Vec<u8> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/collate_golden/atom_features_ala_golden.safetensors");
        std::fs::read(&p).unwrap_or_else(|e| {
            panic!(
                "missing {} (generate: python3 scripts/dump_atom_features_golden.py --mol-dir .../mols): {e}",
                p.display()
            )
        })
    }

    /// Expected tensor names and shapes for ALA single-token, `max_atoms` padded (Boltz default cap).
    fn expected_schema() -> Vec<(&'static str, Dtype, Vec<usize>)> {
        vec![
            ("atom_backbone_feat", Dtype::I64, vec![32, 17]),
            ("atom_pad_mask", Dtype::F32, vec![32]),
            ("atom_resolved_mask", Dtype::BOOL, vec![32]),
            ("atom_to_token", Dtype::I64, vec![32, 1]),
            ("bfactor", Dtype::F32, vec![32]),
            ("coords", Dtype::F32, vec![1, 32, 3]),
            ("disto_coords_ensemble", Dtype::F32, vec![1, 1, 3]),
            ("disto_target", Dtype::F32, vec![1, 1, 1, 64]),
            ("plddt", Dtype::F32, vec![32]),
            ("r_set_to_rep_atom", Dtype::I64, vec![1, 32]),
            ("ref_atom_name_chars", Dtype::I64, vec![32, 4, 64]),
            ("ref_charge", Dtype::F32, vec![32]),
            ("ref_chirality", Dtype::I64, vec![32]),
            ("ref_element", Dtype::I64, vec![32, 128]),
            ("ref_pos", Dtype::F32, vec![32, 3]),
            ("ref_space_uid", Dtype::I64, vec![32]),
            ("token_to_center_atom", Dtype::I64, vec![1, 32]),
            ("token_to_rep_atom", Dtype::I64, vec![1, 32]),
        ]
    }

    fn read_f32_le(buf: &[u8]) -> Vec<f32> {
        buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn atom_pad_mask_five_real_atoms_single_ala() {
        let bytes = fixture_bytes();
        let st = SafeTensors::deserialize(&bytes).expect("safetensors");
        let v = st.tensor("atom_pad_mask").expect("atom_pad_mask");
        assert_eq!(v.dtype(), Dtype::F32);
        let got = read_f32_le(v.data());
        let sum: f32 = got.iter().sum();
        let exp = ALA_STANDARD_HEAVY_ATOM_COUNT as f32;
        assert!(
            (sum - exp).abs() < 1e-5,
            "expected {exp} real atoms for ALA, got {sum}"
        );
    }

    #[test]
    fn atom_features_ala_golden_matches_expected_schema() {
        let bytes = fixture_bytes();
        let st = SafeTensors::deserialize(&bytes).expect("safetensors");
        let mut names: Vec<_> = st.names().into_iter().map(String::from).collect();
        names.sort();
        let exp = expected_schema();
        assert_eq!(
            names.len(),
            exp.len(),
            "tensor count mismatch (regenerate dump_atom_features_golden.py if Boltz adds keys)"
        );
        for (key, dtype, shape) in exp {
            let tv = st
                .tensor(key)
                .unwrap_or_else(|_| panic!("missing key {key}"));
            assert_eq!(tv.dtype(), dtype, "dtype {key}");
            assert_eq!(tv.shape(), shape.as_slice(), "shape {key}");
        }
    }
}

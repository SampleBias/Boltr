//! Golden schema: Python `process_atom_features` on ALA + canonical mols
//! ([`scripts/dump_atom_features_golden.py`](../../../scripts/dump_atom_features_golden.py)).
//!
//! [`tests::atom_features_ala_rust_matches_python_golden_allclose`] compares live Rust tensors
//! (fixture [`structure_v2_single_ala`](crate::fixtures::structure_v2_single_ala) + canonical
//! ref data) to the checked-in safetensors. The Python dump uses
//! `structure_v2_numpy_packed_ala.npz` + RDKit mols; small numeric drift on `ref_pos` / `coords` is
//! possible if the structure NPZ differs from the in-code fixture.

#[cfg(test)]
mod tests {
    use std::path::Path;

    use safetensors::tensor::Dtype;
    use safetensors::SafeTensors;

    use super::super::process_atom_features::{
        inference_ensemble_features, process_atom_features, AtomFeatureConfig, AtomFeatureTensors,
        StandardAminoAcidRefData, ALA_STANDARD_HEAVY_ATOM_COUNT, ATOM_FEATURE_KEYS_ALA,
    };
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::tokenize_structure;

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

    const RTOL: f32 = 1e-4;
    const ATOL: f32 = 1e-5;

    fn allclose(a: f32, b: f32) -> bool {
        (a - b).abs() <= ATOL + RTOL * b.abs()
    }

    fn read_i64_le(buf: &[u8]) -> Vec<i64> {
        buf.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn read_bool_le(buf: &[u8]) -> Vec<f32> {
        buf.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect()
    }

    fn ala_atom_features_rust() -> AtomFeatureTensors {
        let s = structure_v2_single_ala();
        let (tokens, _bonds) = tokenize_structure(&s, None);
        let provider = StandardAminoAcidRefData::new();
        let config = AtomFeatureConfig::default();
        process_atom_features(
            &tokens,
            &s,
            &inference_ensemble_features(),
            &provider,
            &config,
        )
    }

    fn flatten_row_major_f32(a: &ndarray::ArrayD<f32>) -> Vec<f32> {
        a.iter().cloned().collect()
    }

    /// Keys excluded from Rust-vs-Python allclose: golden is built from `structure_v2_numpy_packed_ala.npz`
    /// + RDKit `mols/*.pkl`; this test uses [`structure_v2_single_ala`] + idealized conformers.
    const ATOM_GOLDEN_SKIP_ALCLOSE: &[&str] = &[
        "ref_chirality", // RDKit per-atom tags vs `CHI_OTHER` in `StandardAminoAcidRefData`
        "coords",        // input Cartesian vs packed NPZ
        "ref_pos",       // RDKit conformer vs idealized ALA
        "disto_coords_ensemble",
        "disto_target", // depends on structure geometry
    ];

    /// Compare Rust `process_atom_features` (ALA in-code fixture) to Python golden safetensors.
    /// Golden may use I64/BOOL where Rust uses F32; compare numerically after casting.
    #[test]
    fn atom_features_ala_rust_matches_python_golden_allclose() {
        let feat = ala_atom_features_rust();
        let bytes = fixture_bytes();
        let st = SafeTensors::deserialize(&bytes).expect("safetensors");

        for key in ATOM_FEATURE_KEYS_ALA {
            if ATOM_GOLDEN_SKIP_ALCLOSE.contains(key) {
                continue;
            }
            let view = st
                .tensor(key)
                .unwrap_or_else(|e| panic!("golden missing {key}: {e:?}"));
            match *key {
                "atom_backbone_feat" => {
                    let got = flatten_row_major_f32(&feat.atom_backbone_feat.clone().into_dyn());
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "atom_pad_mask" => {
                    let got = feat.atom_pad_mask.as_slice().unwrap().to_vec();
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "atom_resolved_mask" => {
                    let got = feat.atom_resolved_mask.as_slice().unwrap().to_vec();
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "atom_to_token" => {
                    let got = flatten_row_major_f32(&feat.atom_to_token.clone().into_dyn());
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "bfactor" => {
                    let got = feat.bfactor.as_slice().unwrap().to_vec();
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "plddt" => {
                    let got = feat.plddt.as_slice().unwrap().to_vec();
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "r_set_to_rep_atom" => {
                    let got = flatten_row_major_f32(&feat.r_set_to_rep_atom.clone().into_dyn());
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "ref_atom_name_chars" => {
                    let got = flatten_row_major_f32(&feat.ref_atom_name_chars.clone().into_dyn());
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "ref_charge" => {
                    let got = feat.ref_charge.as_slice().unwrap().to_vec();
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "ref_element" => {
                    let got = flatten_row_major_f32(&feat.ref_element.clone().into_dyn());
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "ref_space_uid" => {
                    let got = feat.ref_space_uid.as_slice().unwrap().to_vec();
                    assert_golden_int_exact(&view, &got, key);
                }
                "token_to_center_atom" => {
                    let got = flatten_row_major_f32(&feat.token_to_center_atom.clone().into_dyn());
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                "token_to_rep_atom" => {
                    let got = flatten_row_major_f32(&feat.token_to_rep_atom.clone().into_dyn());
                    assert_golden_numeric_allclose(&view, &got, key);
                }
                _ => panic!("unhandled key {key} in ATOM_FEATURE_KEYS_ALA"),
            }
        }
    }

    fn golden_as_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
        match view.dtype() {
            Dtype::F32 => read_f32_le(view.data()),
            Dtype::I64 => read_i64_le(view.data())
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            Dtype::BOOL => read_bool_le(view.data()),
            d => panic!("unexpected dtype {d:?} for float-like compare"),
        }
    }

    fn assert_golden_numeric_allclose(
        view: &safetensors::tensor::TensorView<'_>,
        expected: &[f32],
        key: &str,
    ) {
        let got = golden_as_f32(view);
        assert_eq!(
            got.len(),
            expected.len(),
            "{key}: len mismatch golden {} vs rust {}",
            got.len(),
            expected.len()
        );
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                allclose(*g, *e),
                "{key}[{i}]: golden {g} vs rust {e} (rtol={RTOL}, atol={ATOL})",
            );
        }
    }

    fn assert_golden_int_exact(
        view: &safetensors::tensor::TensorView<'_>,
        expected: &[i64],
        key: &str,
    ) {
        let got = match view.dtype() {
            Dtype::I64 => read_i64_le(view.data()),
            Dtype::F32 => read_f32_le(view.data())
                .into_iter()
                .map(|v| v as i64)
                .collect(),
            d => panic!("{key}: unexpected dtype {d:?} for int compare"),
        };
        assert_eq!(got.as_slice(), expected, "{key}: i64 mismatch");
    }
}

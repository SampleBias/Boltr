//! Golden parity: checked-in safetensors vs live [`process_token_features`].
//!
//! Generate fixtures: `cargo run -p boltr-io --bin write_token_features_ala_golden`.  
//! Regenerate from upstream Boltz: `scripts/dump_token_features_ala_golden.py` (see repo root).

#[cfg(test)]
mod tests {
    use std::path::Path;

    use safetensors::tensor::Dtype;
    use safetensors::SafeTensors;

    use crate::featurizer::process_token_features;
    use crate::featurizer::token::token_feature_key_names;
    use crate::featurizer::TokenFeatureTensors;
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::tokenize_structure;

    const RTOL: f32 = 1e-5;
    const ATOL: f32 = 1e-6;

    fn allclose(a: f32, b: f32) -> bool {
        (a - b).abs() <= ATOL + RTOL * b.abs()
    }

    fn read_f32_le(buf: &[u8]) -> Vec<f32> {
        buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn read_i64_le(buf: &[u8]) -> Vec<i64> {
        buf.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn expect_shape_collated(collated: bool, shape: &[usize], ranks: &[usize]) {
        if collated {
            assert_eq!(shape.first().copied(), Some(1), "collated must lead with B=1");
            assert_eq!(&shape[1..], ranks);
        } else {
            assert_eq!(shape, ranks);
        }
    }

    fn compare_1d_i64(view: safetensors::tensor::TensorView<'_>, expected: &[i64], collated: bool) {
        assert_eq!(view.dtype(), Dtype::I64);
        let sh = view.shape();
        let n = expected.len();
        if collated {
            expect_shape_collated(true, sh, &[n]);
        } else {
            expect_shape_collated(false, sh, &[n]);
        }
        let got = read_i64_le(view.data());
        assert_eq!(got.as_slice(), expected, "i64 1d mismatch");
    }

    fn compare_1d_f32(view: safetensors::tensor::TensorView<'_>, expected: &[f32], collated: bool) {
        assert_eq!(view.dtype(), Dtype::F32);
        let sh = view.shape();
        let n = expected.len();
        if collated {
            expect_shape_collated(true, sh, &[n]);
        } else {
            expect_shape_collated(false, sh, &[n]);
        }
        let got = read_f32_le(view.data());
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                allclose(*g, *e),
                "f32 mismatch: got {g} expected {e} (rtol={RTOL}, atol={ATOL})"
            );
        }
    }

    fn compare_2d_f32(
        view: safetensors::tensor::TensorView<'_>,
        expected: &[f32],
        exp_rows: usize,
        exp_cols: usize,
        collated: bool,
    ) {
        assert_eq!(view.dtype(), Dtype::F32);
        let sh = view.shape();
        if collated {
            assert_eq!(sh, &[1, exp_rows, exp_cols]);
        } else {
            assert_eq!(sh, &[exp_rows, exp_cols]);
        }
        let got = read_f32_le(view.data());
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(allclose(*g, *e), "f32 2d mismatch: got {g} expected {e}");
        }
    }

    fn compare_2d_i64(
        view: safetensors::tensor::TensorView<'_>,
        expected: &[i64],
        exp_rows: usize,
        exp_cols: usize,
        collated: bool,
    ) {
        assert_eq!(view.dtype(), Dtype::I64);
        let sh = view.shape();
        if collated {
            assert_eq!(sh, &[1, exp_rows, exp_cols]);
        } else {
            assert_eq!(sh, &[exp_rows, exp_cols]);
        }
        let got = read_i64_le(view.data());
        assert_eq!(got.as_slice(), expected);
    }

    fn compare_3d_f32(
        view: safetensors::tensor::TensorView<'_>,
        expected: &[f32],
        d0: usize,
        d1: usize,
        d2: usize,
        collated: bool,
    ) {
        assert_eq!(view.dtype(), Dtype::F32);
        let sh = view.shape();
        if collated {
            assert_eq!(sh, &[1, d0, d1, d2]);
        } else {
            assert_eq!(sh, &[d0, d1, d2]);
        }
        let got = read_f32_le(view.data());
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(allclose(*g, *e), "f32 3d mismatch: got {g} expected {e}");
        }
    }

    fn assert_features_match_golden(feat: &TokenFeatureTensors, bytes: &[u8], collated: bool) {
        let st = SafeTensors::deserialize(bytes).expect("safetensors");
        let names: std::collections::HashSet<String> =
            st.names().into_iter().map(String::from).collect();
        for key in token_feature_key_names() {
            assert!(names.contains(*key), "golden missing key {key}");
        }
        assert_eq!(names.len(), token_feature_key_names().len());

        compare_1d_i64(
            st.tensor("token_index").unwrap(),
            feat.token_index.as_slice().unwrap(),
            collated,
        );
        compare_1d_i64(
            st.tensor("residue_index").unwrap(),
            feat.residue_index.as_slice().unwrap(),
            collated,
        );
        compare_1d_i64(
            st.tensor("asym_id").unwrap(),
            feat.asym_id.as_slice().unwrap(),
            collated,
        );
        compare_1d_i64(
            st.tensor("entity_id").unwrap(),
            feat.entity_id.as_slice().unwrap(),
            collated,
        );
        compare_1d_i64(
            st.tensor("sym_id").unwrap(),
            feat.sym_id.as_slice().unwrap(),
            collated,
        );
        compare_1d_i64(
            st.tensor("mol_type").unwrap(),
            feat.mol_type.as_slice().unwrap(),
            collated,
        );
        let (nr, nc) = feat.res_type.dim();
        compare_2d_f32(
            st.tensor("res_type").unwrap(),
            feat.res_type.as_slice().unwrap(),
            nr,
            nc,
            collated,
        );
        let (nr, nc) = feat.disto_center.dim();
        compare_2d_f32(
            st.tensor("disto_center").unwrap(),
            feat.disto_center.as_slice().unwrap(),
            nr,
            nc,
            collated,
        );
        let (a, b, c) = feat.token_bonds.dim();
        compare_3d_f32(
            st.tensor("token_bonds").unwrap(),
            feat.token_bonds.as_slice().unwrap(),
            a,
            b,
            c,
            collated,
        );
        let (nr, nc) = feat.type_bonds.dim();
        compare_2d_i64(
            st.tensor("type_bonds").unwrap(),
            feat.type_bonds.as_slice().unwrap(),
            nr,
            nc,
            collated,
        );
        compare_1d_f32(
            st.tensor("token_pad_mask").unwrap(),
            feat.token_pad_mask.as_slice().unwrap(),
            collated,
        );
        compare_1d_f32(
            st.tensor("token_resolved_mask").unwrap(),
            feat.token_resolved_mask.as_slice().unwrap(),
            collated,
        );
        compare_1d_f32(
            st.tensor("token_disto_mask").unwrap(),
            feat.token_disto_mask.as_slice().unwrap(),
            collated,
        );
        let (a, b, c) = feat.contact_conditioning.dim();
        compare_3d_f32(
            st.tensor("contact_conditioning").unwrap(),
            feat.contact_conditioning.as_slice().unwrap(),
            a,
            b,
            c,
            collated,
        );
        let (nr, nc) = feat.contact_threshold.dim();
        compare_2d_f32(
            st.tensor("contact_threshold").unwrap(),
            feat.contact_threshold.as_slice().unwrap(),
            nr,
            nc,
            collated,
        );
        compare_1d_i64(
            st.tensor("method_feature").unwrap(),
            feat.method_feature.as_slice().unwrap(),
            collated,
        );
        compare_1d_i64(
            st.tensor("modified").unwrap(),
            feat.modified.as_slice().unwrap(),
            collated,
        );
        compare_1d_f32(
            st.tensor("cyclic_period").unwrap(),
            feat.cyclic_period.as_slice().unwrap(),
            collated,
        );
        compare_1d_f32(
            st.tensor("affinity_token_mask").unwrap(),
            feat.affinity_token_mask.as_slice().unwrap(),
            collated,
        );
    }

    fn live_features() -> TokenFeatureTensors {
        let s = structure_v2_single_ala();
        let (tokens, bonds) = tokenize_structure(&s, None);
        process_token_features(&tokens, &bonds, None)
    }

    #[test]
    fn token_features_ala_matches_per_example_golden() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/collate_golden");
        let path = dir.join("token_features_ala_golden.safetensors");
        let bytes = std::fs::read(&path).expect("read token_features_ala_golden.safetensors");
        let feat = live_features();
        assert_features_match_golden(&feat, &bytes, false);
    }

    #[test]
    fn token_features_ala_matches_collated_golden() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/collate_golden");
        let path = dir.join("token_features_ala_collated_golden.safetensors");
        let bytes = std::fs::read(&path).expect("read collated golden");
        let feat = live_features();
        assert_features_match_golden(&feat, &bytes, true);
    }
}

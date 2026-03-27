//! Golden parity: [`process_msa_features`] / [`crate::inference_dataset::msa_features_from_inference_input`]
//! vs `msa_features_load_input_smoke_golden.safetensors` from
//! [`scripts/dump_msa_features_golden.py`](../../../scripts/dump_msa_features_golden.py).

#[cfg(test)]
mod tests {
    use std::path::Path;

    use safetensors::tensor::Dtype;
    use safetensors::SafeTensors;

    use crate::inference_dataset::{load_input, msa_features_from_inference_input};
    use crate::parse_manifest_path;

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

    fn read_bool_le(buf: &[u8]) -> Vec<bool> {
        buf.iter().map(|&b| b != 0).collect()
    }

    fn fixture_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load_input_smoke")
    }

    #[test]
    fn msa_features_load_input_smoke_match_python_golden() {
        let dir = fixture_dir();
        let golden_path = dir.join("msa_features_load_input_smoke_golden.safetensors");
        let bytes = std::fs::read(&golden_path).unwrap_or_else(|e| {
            panic!(
                "missing {golden_path:?} (generate with scripts/dump_msa_features_golden.py): {e}"
            )
        });

        let manifest = parse_manifest_path(&dir.join("manifest.json")).expect("manifest");
        let input = load_input(
            &manifest.records[0],
            &dir,
            &dir,
            None,
            None,
            None,
            false,
        )
        .expect("load_input");
        let got = msa_features_from_inference_input(&input);

        let st = SafeTensors::deserialize(&bytes).expect("safetensors");
        for key in [
            "msa",
            "msa_paired",
            "deletion_value",
            "has_deletion",
            "deletion_mean",
            "profile",
            "msa_mask",
        ] {
            assert!(st.tensor(key).is_ok(), "golden missing key {key}");
        }

        let msa = st.tensor("msa").expect("msa");
        assert_eq!(msa.dtype(), Dtype::I64);
        assert_eq!(msa.shape(), got.msa.shape());
        let exp_msa = read_i64_le(msa.data());
        assert_eq!(exp_msa.as_slice(), got.msa.as_slice().unwrap());

        let msa_paired = st.tensor("msa_paired").expect("msa_paired");
        assert_eq!(msa_paired.dtype(), Dtype::F32);
        assert_eq!(msa_paired.shape(), got.msa_paired.shape());
        let exp_paired = read_f32_le(msa_paired.data());
        let got_paired: Vec<f32> = got.msa_paired.iter().map(|&x| x as f32).collect();
        assert_eq!(exp_paired.len(), got_paired.len());
        for (a, b) in exp_paired.iter().zip(got_paired.iter()) {
            assert!(allclose(*a, *b), "msa_paired mismatch {a} {b}");
        }

        let del_v = st.tensor("deletion_value").expect("deletion_value");
        assert_eq!(del_v.dtype(), Dtype::F32);
        assert_eq!(del_v.shape(), got.deletion_value.shape());
        let exp = read_f32_le(del_v.data());
        assert_eq!(exp.len(), got.deletion_value.len());
        for (a, b) in exp.iter().zip(got.deletion_value.iter()) {
            assert!(allclose(*a, *b), "deletion_value mismatch {a} {b}");
        }

        let has_d = st.tensor("has_deletion").expect("has_deletion");
        assert_eq!(has_d.dtype(), Dtype::BOOL);
        assert_eq!(has_d.shape(), got.has_deletion.shape());
        let exp_b = read_bool_le(has_d.data());
        let got_h: Vec<bool> = got.has_deletion.iter().map(|&x| x != 0).collect();
        assert_eq!(exp_b, got_h);

        let dm = st.tensor("deletion_mean").expect("deletion_mean");
        assert_eq!(dm.dtype(), Dtype::F32);
        assert_eq!(dm.shape(), got.deletion_mean.shape());
        let exp_dm = read_f32_le(dm.data());
        assert_eq!(exp_dm.len(), got.deletion_mean.len());
        for (a, b) in exp_dm.iter().zip(got.deletion_mean.iter()) {
            assert!(allclose(*a, *b), "deletion_mean mismatch {a} {b}");
        }

        let prof = st.tensor("profile").expect("profile");
        assert_eq!(prof.dtype(), Dtype::F32);
        assert_eq!(prof.shape(), got.profile.shape());
        let exp_p = read_f32_le(prof.data());
        assert_eq!(exp_p.len(), got.profile.len());
        for (a, b) in exp_p.iter().zip(got.profile.iter()) {
            assert!(allclose(*a, *b), "profile mismatch {a} {b}");
        }

        let mask = st.tensor("msa_mask").expect("msa_mask");
        assert_eq!(mask.dtype(), Dtype::I64);
        assert_eq!(mask.shape(), got.msa_mask.shape());
        let exp_mask = read_i64_le(mask.data());
        assert_eq!(exp_mask.as_slice(), got.msa_mask.as_slice().unwrap());
    }
}

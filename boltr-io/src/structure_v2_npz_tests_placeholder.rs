        f.write_all(&blob)?;
    }
    zw.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::tokenize_structure;

    #[test]
    fn roundtrip_structure_npz_matches_fixture() {
        let s = structure_v2_single_ala();
        let bytes = write_structure_v2_npz_to_vec(&s).unwrap();
        let back = read_structure_v2_npz_bytes(&bytes).unwrap();
        assert_eq!(back, s);
        let (t1, _) = tokenize_structure(&s, None);
        let (t2, _) = tokenize_structure(&back, None);
        assert_eq!(t1, t2);
    }

    /// NumPy **packed** structured dtypes (Boltz `types.py` lists, default `np.dtype` layout),
    /// optional `interfaces`, and **multi-row** `ensemble`. Regenerate:
    /// `python3 scripts/gen_structure_v2_numpy_golden.py`.
    #[test]
    fn golden_numpy_packed_structure_v2_matches_ala_fixture() {
        const BYTES: &[u8] = include_bytes!("../tests/fixtures/structure_v2_numpy_packed_ala.npz");
        let s = structure_v2_single_ala();
        let got = read_structure_v2_npz_bytes(BYTES).expect("golden npz");
        assert_eq!(got, s);
        let (t1, _) = tokenize_structure(&s, None);
        let (t2, _) = tokenize_structure(&got, None);
        assert_eq!(t1, t2);
    }
}

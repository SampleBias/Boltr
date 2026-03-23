//! Van der Waals radii (Å) from `boltz.data.const.vdw_radii` — 118 values (Python list length).

/// Same order and values as `const.py` `vdw_radii` (indices `0..118` match Python `vdw_radii[i]`).
pub const VDW_RADII: [f64; 118] = [
    1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54, 2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88,
    2.8, 2.4, 2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.05, 1.9, 1.9,
    2.02, 2.9, 2.55, 2.4, 2.3, 2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25, 2.2, 2.1,
    2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45, 2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3,
    2.28, 2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05, 2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.4, 2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
];

pub const VDW_RADII_LEN: usize = VDW_RADII.len();

#[inline]
#[must_use]
pub fn vdw_radius(element_index: usize) -> Option<f64> {
    VDW_RADII.get(element_index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_and_spot_values_match_python() {
        assert_eq!(VDW_RADII_LEN, 118);
        assert!((VDW_RADII[0] - 1.2).abs() < 1e-9);
        assert!((VDW_RADII[5] - 1.7).abs() < 1e-9);
        assert!((vdw_radius(117).unwrap() - 2.0).abs() < 1e-9);
        assert!(vdw_radius(118).is_none());
    }
}

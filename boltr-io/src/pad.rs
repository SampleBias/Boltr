//! Batch padding helpers aligned with typical Boltz collate (`pad_to_max` / `pad_sequence`-style).
//!
//! Pads **at the end** of the sequence axis (post-padding). Use returned **lengths** for masks.

use ndarray::{s, Array2, Array3, ArrayView2};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PadError {
    #[error("empty batch")]
    EmptyBatch,
    #[error("inconsistent feature dim: expected {expected}, got {got}")]
    InconsistentFeatureDim { expected: usize, got: usize },
}

/// Right-pad a 1-D sequence to `target_len` with `fill` (truncate if longer).
#[must_use]
pub fn pad_1d<T: Clone>(seq: &[T], target_len: usize, fill: T) -> Vec<T> {
    let mut v = seq.to_vec();
    if v.len() > target_len {
        v.truncate(target_len);
    } else {
        v.resize(target_len, fill);
    }
    v
}

/// Ragged rows → dense `[n_rows, max_width]`; `lengths[i]` is the original row length before padding.
pub fn pad_ragged_rows<T: Clone>(rows: &[Vec<T>], fill: T) -> (Array2<T>, Vec<usize>) {
    let n = rows.len();
    let w = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut out = Array2::from_elem((n, w), fill.clone());
    let mut lengths = Vec::with_capacity(n);
    for (i, row) in rows.iter().enumerate() {
        lengths.push(row.len());
        for (j, v) in row.iter().enumerate().take(w) {
            out[[i, j]] = v.clone();
        }
    }
    (out, lengths)
}

/// Stack `[n_i, C]` matrices into `[batch, max_n, C]` with post-padding on the token axis (dim 0 of each item).
pub fn stack_tokens_2d<T: Clone>(
    batch: &[ArrayView2<T>],
    fill: T,
) -> Result<(Array3<T>, Vec<usize>), PadError> {
    let b = batch.len();
    if b == 0 {
        return Err(PadError::EmptyBatch);
    }
    let c = batch[0].shape()[1];
    for m in batch.iter() {
        if m.shape()[1] != c {
            return Err(PadError::InconsistentFeatureDim {
                expected: c,
                got: m.shape()[1],
            });
        }
    }
    let max_n = batch.iter().map(|m| m.shape()[0]).max().unwrap_or(0);
    let mut out = Array3::from_elem((b, max_n, c), fill.clone());
    let mut lengths = Vec::with_capacity(b);
    for (i, m) in batch.iter().enumerate() {
        let n = m.shape()[0];
        lengths.push(n);
        out.slice_mut(s![i, 0..n, ..]).assign(m);
    }
    Ok((out, lengths))
}

/// `1` for valid positions `j < lengths[i]`, `0` for padding (f32 mask `[batch, max_n]`).
#[must_use]
pub fn token_pad_mask(lengths: &[usize], max_n: usize) -> Array2<f32> {
    let b = lengths.len();
    let mut m = Array2::zeros((b, max_n));
    for (i, &len) in lengths.iter().enumerate() {
        let end = len.min(max_n);
        for j in 0..end {
            m[[i, j]] = 1.0;
        }
    }
    m
}

/// Convenience: mask from `pad_ragged_rows` lengths.
#[inline]
pub fn row_pad_mask_from_lengths(lengths: &[usize], max_w: usize) -> Array2<f32> {
    token_pad_mask(lengths, max_w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pad_1d_truncates_and_fills() {
        assert_eq!(pad_1d(&[1, 2, 3], 5, 0), vec![1, 2, 3, 0, 0]);
        assert_eq!(pad_1d(&[1, 2, 3, 4, 5], 3, 0), vec![1, 2, 3]);
    }

    #[test]
    fn pad_ragged_and_mask() {
        let rows = vec![vec![1.0, 2.0], vec![3.0], vec![4.0, 5.0, 6.0]];
        let (a, lens) = pad_ragged_rows(&rows, 0.0);
        assert_eq!(a.shape(), &[3, 3]);
        assert_eq!(lens, vec![2, 1, 3]);
        assert_eq!(a[[0, 2]], 0.0);
        let m = row_pad_mask_from_lengths(&lens, 3);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 2]], 0.0);
        assert_eq!(m[[1, 0]], 1.0);
        assert_eq!(m[[1, 1]], 0.0);
    }

    #[test]
    fn stack_tokens_2d_matches_expected_shape() {
        let a0 = array![[1.0_f32, 10.0], [2.0, 20.0]];
        let a1 = array![[3.0_f32, 30.0]];
        let views = [a0.view(), a1.view()];
        let (stacked, lens) = stack_tokens_2d(&views, 0.0).unwrap();
        assert_eq!(stacked.shape(), &[2, 2, 2]);
        assert_eq!(lens, vec![2, 1]);
        assert_eq!(stacked[[1, 1, 0]], 0.0);
        let m = token_pad_mask(&lens, 2);
        assert_eq!(m[[1, 1]], 0.0);
    }
}

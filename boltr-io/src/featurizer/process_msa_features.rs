//! Port of `process_msa_features` from [`featurizerv2.py`](../../../boltz-reference/src/boltz/data/feature/featurizerv2.py).

use std::collections::HashMap;

use ndarray::{s, Array1, Array2, Axis};

use crate::boltz_const::NUM_TOKENS;
use crate::featurizer::msa_pairing::construct_paired_msa;
use crate::structure_v2::StructureV2Tables;
use crate::tokenize::boltz2::TokenData;
use crate::a3m::A3mMsa;

use rand::Rng;

fn pad_dim_i64(a: &Array2<i64>, dim: usize, pad_len: usize, value: i64) -> Array2<i64> {
    if pad_len == 0 {
        return a.clone();
    }
    let (r, c) = (a.nrows(), a.ncols());
    if dim == 0 {
        let mut out = Array2::from_elem((r + pad_len, c), value);
        out.slice_mut(s![0..r, ..]).assign(a);
        out
    } else {
        let mut out = Array2::from_elem((r, c + pad_len), value);
        out.slice_mut(s![.., 0..c]).assign(a);
        out
    }
}

fn pad_dim_f32(a: &Array2<f32>, dim: usize, pad_len: usize, value: f32) -> Array2<f32> {
    if pad_len == 0 {
        return a.clone();
    }
    let (r, c) = (a.nrows(), a.ncols());
    if dim == 0 {
        let mut out = Array2::from_elem((r + pad_len, c), value);
        out.slice_mut(s![0..r, ..]).assign(a);
        out
    } else {
        let mut out = Array2::from_elem((r, c + pad_len), value);
        out.slice_mut(s![.., 0..c]).assign(a);
        out
    }
}

fn pad_dim_i64_zeros(a: &Array2<i64>, dim: usize, pad_len: usize) -> Array2<i64> {
    pad_dim_i64(a, dim, pad_len, 0)
}

fn pad_dim1_f32(a: &Array1<f32>, pad_len: usize, value: f32) -> Array1<f32> {
    if pad_len == 0 {
        return a.clone();
    }
    let n = a.len();
    let mut out = Array1::from_elem(n + pad_len, value);
    out.slice_mut(s![0..n]).assign(a);
    out
}

/// Non-affinity MSA feature tensors (names align with [`crate::collate_golden`] manifest).
#[derive(Clone, Debug, PartialEq)]
pub struct MsaFeatureTensors {
    pub msa: Array2<i64>,
    pub msa_paired: Array2<i64>,
    pub deletion_value: Array2<f32>,
    pub has_deletion: Array2<i64>,
    pub deletion_mean: Array1<f32>,
    pub profile: Array2<f32>,
    pub msa_mask: Array2<i64>,
}

/// Inference MSA path: paired construction + transforms + optional padding.
#[must_use]
pub fn process_msa_features(
    tokens: &[TokenData],
    structure: &StructureV2Tables,
    msas: &HashMap<i32, A3mMsa>,
    rng: &mut impl Rng,
    max_seqs_batch: usize,
    max_seqs: usize,
    max_tokens: Option<usize>,
    pad_to_max_seqs: bool,
    msa_sampling: bool,
) -> MsaFeatureTensors {
    let (msa_nt, deletion_nt, paired_nt) = construct_paired_msa(
        tokens,
        structure,
        msas,
        rng,
        max_seqs_batch,
        8192,
        16384,
        msa_sampling,
    );

    let mut msa = msa_nt.t().to_owned();
    let deletion_i = deletion_nt.mapv(|x| x as f32).t().to_owned();
    let mut paired = paired_nt.mapv(|x| x as f32).t().to_owned();

    let num_classes = NUM_TOKENS as i64;
    assert!(msa.iter().all(|&v| v >= 0 && v < num_classes));

    let s_msa = msa.nrows();
    let n = msa.ncols();
    let mut profile = Array2::<f32>::zeros((n, NUM_TOKENS));
    for i in 0..s_msa {
        for j in 0..n {
            let t = msa[[i, j]] as usize;
            if t < NUM_TOKENS {
                profile[[j, t]] += 1.0;
            }
        }
    }
    profile /= s_msa as f32;

    let mut msa_mask = Array2::<i64>::ones(msa.raw_dim());
    let mut has_deletion = deletion_i.mapv(|x| if x > 0.0 { 1i64 } else { 0 });
    let mut deletion = deletion_i.mapv(|d| std::f32::consts::FRAC_PI_2 * (d / 3.0).atan());
    let mut deletion_mean: Array1<f32> = deletion
        .mean_axis(Axis(0))
        .expect("mean_axis")
        .into_dimensionality()
        .expect("deletion_mean 1d");

    if pad_to_max_seqs {
        let pad_len = max_seqs.saturating_sub(msa.nrows());
        if pad_len > 0 {
            let gap = i64::from(crate::boltz_const::token_id("-").expect("gap"));
            msa = pad_dim_i64(&msa, 0, pad_len, gap);
            paired = pad_dim_f32(&paired, 0, pad_len, 0.0);
            msa_mask = pad_dim_i64_zeros(&msa_mask, 0, pad_len);
            has_deletion = pad_dim_i64_zeros(&has_deletion, 0, pad_len);
            deletion = pad_dim_f32(&deletion, 0, pad_len, 0.0);
        }
    }

    if let Some(max_t) = max_tokens {
        let pad_len = max_t.saturating_sub(msa.ncols());
        if pad_len > 0 {
            let gap = i64::from(crate::boltz_const::token_id("-").expect("gap"));
            msa = pad_dim_i64(&msa, 1, pad_len, gap);
            paired = pad_dim_f32(&paired, 1, pad_len, 0.0);
            msa_mask = pad_dim_i64_zeros(&msa_mask, 1, pad_len);
            has_deletion = pad_dim_i64_zeros(&has_deletion, 1, pad_len);
            deletion = pad_dim_f32(&deletion, 1, pad_len, 0.0);
            profile = pad_dim_f32(&profile, 0, pad_len, 0.0);
            deletion_mean = pad_dim1_f32(&deletion_mean, pad_len, 0.0);
        }
    }

    MsaFeatureTensors {
        msa,
        msa_paired: paired.mapv(|x| x as i64),
        deletion_value: deletion,
        has_deletion,
        deletion_mean,
        profile,
        msa_mask,
    }
}

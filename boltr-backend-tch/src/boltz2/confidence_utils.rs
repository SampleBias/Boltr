//! Helpers aligned with `boltz-reference/src/boltz/model/layers/confidence_utils.py`.
//!
//! `compute_frame_pred` in Python adjusts frames for **ligand** chains; until that logic is ported,
//! [`compute_frame_pred_stub`] repeats `frames_idx` and uses valid-token masks so `compute_ptms`
//! runs end-to-end (scores match Python when ligand frame logic is inactive).

use std::collections::BTreeMap;
use std::ops::Mul;

use tch::{Device, Kind, Tensor};

/// Boltz `chain_type_ids["PROTEIN"]` / `["NONPOLYMER"]` (`boltz/data/const.py` order).
pub const CHAIN_TYPE_PROTEIN: i64 = 0;
pub const CHAIN_TYPE_NONPOLYMER: i64 = 3;

/// `compute_aggregated_metric(logits, end)` — expected value over softmax bins.
pub fn compute_aggregated_metric(logits: &Tensor, end: f64) -> Tensor {
    let num_bins = *logits.size().last().expect("logits rank");
    let bin_width = end / num_bins as f64;
    let device = logits.device();
    let bounds = Tensor::arange(num_bins, (Kind::Float, device))
        .g_mul_scalar(bin_width)
        .g_add_scalar(0.5 * bin_width);
    let probs = logits.softmax(-1, Kind::Float);
    let n = probs.dim() as usize;
    let mut view_shape = vec![1i64; n];
    view_shape[n - 1] = num_bins;
    let bounds = bounds.view(view_shape.as_slice());
    probs
        .mul(&bounds)
        .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
}

fn tm_function(d: &Tensor, n_res: &Tensor) -> Tensor {
    let d0 = n_res
        .clamp_min(19.0)
        .g_sub_scalar(15.0)
        .pow_tensor_scalar(1.0 / 3.0)
        .g_mul_scalar(1.24)
        .g_sub_scalar(1.8);
    let ratio = d / d0;
    Tensor::ones_like(&ratio) / (Tensor::ones_like(&ratio) + ratio.pow_tensor_scalar(2.0))
}

/// Repeat `frames_idx` with multiplicity and build a collinear-valid mask from `token_pad_mask`.
/// Full ligand `compute_frame_pred` from Python is TODO; this matches the common protein-only path.
pub fn compute_frame_pred_stub(
    frames_idx_true: &Tensor,
    token_pad_mask: &Tensor,
    multiplicity: i64,
) -> (Tensor, Tensor) {
    let b_total = frames_idx_true.size()[0];
    let n_atom = frames_idx_true.size()[1];
    let b0 = b_total / multiplicity;
    let frames_idx_pred = frames_idx_true
        .repeat_interleave_self_int(multiplicity, Some(0), None)
        .view([b0, multiplicity, n_atom, 3]);
    let n_tok = token_pad_mask.size()[1];
    let mask_collinear = token_pad_mask
        .view([b0, 1, n_tok])
        .expand([b0, multiplicity, n_tok], true)
        .to_kind(Kind::Float);
    (frames_idx_pred, mask_collinear)
}

/// `compute_ptms` — `boltz.model.layers.confidence_utils`.
pub fn compute_ptms(
    pae_logits: &Tensor,
    _x_preds: &Tensor,
    frames_idx: &Tensor,
    asym_id: &Tensor,
    mol_type: &Tensor,
    token_pad_mask: &Tensor,
    multiplicity: i64,
) -> (
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    BTreeMap<i64, BTreeMap<i64, Tensor>>,
) {
    let (frames_idx_pred, mask_collinear_pred) =
        compute_frame_pred_stub(frames_idx, token_pad_mask, multiplicity);
    let _ = frames_idx_pred;

    let mask_pad = token_pad_mask.repeat_interleave_self_int(multiplicity, Some(0), None);
    let n = mask_collinear_pred.size()[2];
    let maski = mask_collinear_pred.reshape(&[-1, n]);

    let pad_pair = mask_pad.unsqueeze(1) * mask_pad.unsqueeze(2);
    let pair_mask_ptm = maski.unsqueeze(2) * &pad_pair;

    let asym_r = asym_id.repeat_interleave_self_int(multiplicity, Some(0), None);
    let ne = asym_r.unsqueeze(2).ne_tensor(&asym_r.unsqueeze(1));
    let pair_mask_iptm = maski.unsqueeze(2) * ne.to_kind(Kind::Float) * &pad_pair;

    let num_bins = *pae_logits.size().last().expect("pae bins");
    let bin_width = 32.0 / num_bins as f64;
    let device = pae_logits.device();
    let pae_value = Tensor::arange(num_bins, (Kind::Float, device))
        .g_mul_scalar(bin_width)
        .g_add_scalar(0.5 * bin_width)
        .unsqueeze(0);
    let n_res = mask_pad.sum_dim_intlist(&[-1i64][..], true, Kind::Float);
    let tm_w = tm_function(&pae_value, &n_res);
    let tm_value = tm_w.unsqueeze(1).unsqueeze(2);
    let probs = pae_logits.softmax(-1, Kind::Float);
    let tm_expected_value = probs
        .mul(&tm_value)
        .sum_dim_intlist(&[-1i64][..], false, Kind::Float);

    let ptm = (tm_expected_value * &pair_mask_ptm).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    ) / (pair_mask_ptm.sum_dim_intlist(&[-1i64][..], false, Kind::Float) + 1e-5);
    let ptm = ptm.max_dim(1, false).0;

    let iptm = (tm_expected_value * &pair_mask_iptm).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    ) / (pair_mask_iptm.sum_dim_intlist(&[-1i64][..], false, Kind::Float) + 1e-5);
    let iptm = iptm.max_dim(1, false).0;

    let token_type = mol_type
        .repeat_interleave_self_int(multiplicity, Some(0), None)
        .to_kind(Kind::Float);
    let is_ligand = token_type.eq_tensor(
        &Tensor::from(CHAIN_TYPE_NONPOLYMER as f64).to_device(token_type.device()),
    );
    let is_protein = token_type.eq_tensor(
        &Tensor::from(CHAIN_TYPE_PROTEIN as f64).to_device(token_type.device()),
    );

    let ligand_iptm_mask = maski.unsqueeze(2)
        * ne.to_kind(Kind::Float)
        * &pad_pair
        * ((is_ligand.unsqueeze(2) * is_protein.unsqueeze(1))
            + (is_protein.unsqueeze(2) * is_ligand.unsqueeze(1)));
    let protein_iptm_mask = maski.unsqueeze(2)
        * ne.to_kind(Kind::Float)
        * &pad_pair
        * (is_protein.unsqueeze(2) * is_protein.unsqueeze(1));

    let ligand_iptm = ((tm_expected_value * &ligand_iptm_mask).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    ) / (ligand_iptm_mask.sum_dim_intlist(&[-1i64][..], false, Kind::Float) + 1e-5))
        .max_dim(1, false)
        .0;
    let protein_iptm = ((tm_expected_value * &protein_iptm_mask).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    ) / (protein_iptm_mask.sum_dim_intlist(&[-1i64][..], false, Kind::Float) + 1e-5))
        .max_dim(1, false)
        .0;

    let mut pair_chains: BTreeMap<i64, BTreeMap<i64, Tensor>> = BTreeMap::new();
    let asym_cpu = asym_r.to_device(Device::Cpu);
    let flat = asym_cpu.reshape(&[-1]);
    let nuniq = flat.numel();
    let mut uniq_ids = Vec::new();
    for u in 0..nuniq {
        uniq_ids.push(flat.get(u as i64).int64_value(&[]));
    }
    uniq_ids.sort_unstable();
    uniq_ids.dedup();

    for &idx1 in &uniq_ids {
        let mut inner = BTreeMap::new();
        for &idx2 in &uniq_ids {
            let i1 = Tensor::from(idx1).to_device(asym_r.device());
            let i2 = Tensor::from(idx2).to_device(asym_r.device());
            let mask_pair_chain = maski.unsqueeze(2)
                * asym_r.unsqueeze(2).eq_tensor(&i1).to_kind(Kind::Float)
                * asym_r.unsqueeze(1).eq_tensor(&i2).to_kind(Kind::Float)
                * &pad_pair;
            let v = ((tm_expected_value * &mask_pair_chain).sum_dim_intlist(
                &[-1i64][..],
                false,
                Kind::Float,
            ) / (mask_pair_chain.sum_dim_intlist(&[-1i64][..], false, Kind::Float) + 1e-5))
                .max_dim(1, false)
                .0;
            inner.insert(idx2, v);
        }
        pair_chains.insert(idx1, inner);
    }

    (ptm, iptm, ligand_iptm, protein_iptm, pair_chains)
}

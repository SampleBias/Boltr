//! Windowed key indexing shared by [`super::encoders::AtomEncoder`] and [`super::transformers::AtomTransformer`].
//!
//! Reference: `boltz-reference/src/boltz/model/modules/encodersv2.py` (`single_to_keys`, `get_indexing_matrix`).

use tch::{Device, Kind, Tensor};

/// Build the static indexing matrix for windowed attention keys.
/// Equivalent to Python `get_indexing_matrix(K, W, H, device)`.
pub fn get_indexing_matrix(k: i64, w: i64, h: i64, device: Device) -> Tensor {
    assert!(w % 2 == 0, "W must be even");
    let half_w = w / 2;
    assert!(h % half_w == 0, "H must be divisible by W/2");
    let h_ratio = h / half_w;
    assert!(h_ratio % 2 == 0, "h ratio must be even");

    let arange = Tensor::arange(2 * k, (Kind::Int64, device));
    let diff = arange.unsqueeze(0) - arange.unsqueeze(1); // [2K, 2K]
    let index = (diff + h_ratio / 2).clamp(0, h_ratio + 1);
    let index = index.reshape(&[k, 2, 2 * k]).select(1, 0); // [K, 2K]
    let onehot = index.one_hot(h_ratio + 2);
    let onehot = onehot.slice(2, 1, h_ratio + 1, 1);
    let onehot = onehot.transpose(0, 1);
    onehot.reshape(&[2 * k, h_ratio * k]).to_kind(Kind::Float)
}

/// Map single representation from query windows to key windows.
/// `single [B, N, D]` → `[B, K, H, D]`.
pub fn single_to_keys(single: &Tensor, indexing_matrix: &Tensor, w: i64, h: i64) -> Tensor {
    let size = single.size();
    let (b, n, d) = (size[0], size[1], size[2]);
    let k = n / w;
    let single_r = single.reshape(&[b, 2 * k, w / 2, d]);
    let out = Tensor::einsum("bjid,jk->bkid", &[&single_r, indexing_matrix], None::<i64>);
    out.reshape(&[b, k, h, d])
}

/// Boltz `AtomTransformer.to_keys_new`: map `[B*NW, W, D]` → `[B*NW, H, D]`.
pub fn windowed_to_keys(
    x: &Tensor,
    batch: i64,
    n_atoms: i64,
    w: i64,
    h: i64,
    indexing_matrix: &Tensor,
) -> Tensor {
    let size = x.size();
    let d = size[2];
    let nw = n_atoms / w;
    assert_eq!(size[0], batch * nw, "windowed_to_keys: batch*NW mismatch");
    assert_eq!(size[1], w, "windowed_to_keys: W mismatch");
    let x_flat = x.view([batch, n_atoms, d]);
    let sk = single_to_keys(&x_flat, indexing_matrix, w, h);
    sk.reshape([batch * nw, h, d])
}

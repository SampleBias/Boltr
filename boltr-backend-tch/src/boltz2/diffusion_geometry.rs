//! Random rigid augmentation and weighted rigid alignment for diffusion sampling.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/utils.py` (quaternions / rotations),
//! `boltz-reference/src/boltz/model/loss/diffusionv2.py` (`weighted_rigid_align`).

use tch::{Device, Kind, Tensor};

/// Random rotation matrices `(multiplicity, 3, 3)` and translation `(multiplicity, 1, 3)`.
pub fn compute_random_augmentation(
    multiplicity: i64,
    device: Device,
    kind: Kind,
) -> (Tensor, Tensor) {
    let quat = random_quaternions(multiplicity, device, kind);
    let r = quaternion_to_matrix(&quat);
    let random_tr = Tensor::randn(&[multiplicity, 1, 3], (kind, device));
    (r, random_tr)
}

fn random_quaternions(n: i64, device: Device, kind: Kind) -> Tensor {
    let o = Tensor::randn(&[n, 4], (kind, device));
    let s = o.multiply(&o).sum_dim_intlist(&[1i64][..], false, kind);
    let sgn = o.select(1, 0).sign();
    let sqrt_s = s.sqrt();
    o.multiply(&(sgn / sqrt_s).unsqueeze(-1))
}

fn quaternion_to_matrix(quaternions: &Tensor) -> Tensor {
    let r = quaternions.select(1, 0);
    let i = quaternions.select(1, 1);
    let j = quaternions.select(1, 2);
    let k = quaternions.select(1, 3);
    let two_s = 2.0
        / quaternions
            .multiply(quaternions)
            .sum_dim_intlist(&[1i64][..], false, quaternions.kind());
    let one = Tensor::ones_like(&two_s);

    let row0 = Tensor::stack(
        &[
            &(&one - &(&two_s * &(j.multiply(&j) + k.multiply(&k)))),
            &(&two_s * &(i.multiply(&j) - k.multiply(&r))),
            &(&two_s * &(i.multiply(&k) + j.multiply(&r))),
        ],
        -1,
    );
    let row1 = Tensor::stack(
        &[
            &(&two_s * &(i.multiply(&j) + k.multiply(&r))),
            &(&one - &(&two_s * &(i.multiply(&i) + k.multiply(&k)))),
            &(&two_s * &(j.multiply(&k) - i.multiply(&r))),
        ],
        -1,
    );
    let row2 = Tensor::stack(
        &[
            &(&two_s * &(i.multiply(&k) - j.multiply(&r))),
            &(&two_s * &(j.multiply(&k) + i.multiply(&r))),
            &(&one - &(&two_s * &(i.multiply(&i) + j.multiply(&j)))),
        ],
        -1,
    );
    Tensor::stack(&[row0, row1, row2], -2)
}

/// Algorithm 28 (`weighted_rigid_align`): align `true_coords` to `pred_coords` with weights.
pub fn weighted_rigid_align(
    true_coords: &Tensor,
    pred_coords: &Tensor,
    weights: &Tensor,
    mask: &Tensor,
) -> Tensor {
    let device = true_coords.device();
    let weights = mask * weights;
    let weights = weights.unsqueeze(-1);

    let wsum = weights.sum_dim_intlist(&[-2i64][..], true, Kind::Float);
    let true_centroid =
        (true_coords * &weights).sum_dim_intlist(&[-2i64][..], true, Kind::Float) / &wsum;
    let pred_centroid =
        (pred_coords * &weights).sum_dim_intlist(&[-2i64][..], true, Kind::Float) / &wsum;

    let true_centered = true_coords - &true_centroid;
    let pred_centered = pred_coords - &pred_centroid;

    let cov_matrix = Tensor::einsum(
        "bni,bnj->bij",
        &[&(weights * &pred_centered), &true_centered],
        None::<Vec<i64>>,
    );

    let original_dtype = cov_matrix.kind();
    let cov_32 = cov_matrix.to_kind(Kind::Float);

    let (u, _s, vh) = Tensor::linalg_svd(&cov_32, false, "");

    let rot_matrix = u.matmul(&vh).to_kind(Kind::Float);

    let dim = 3i64;
    let sz = u.size();
    let batch_rank = sz.len().saturating_sub(2);
    let mut batch_prefix: Vec<i64> = Vec::new();
    for i in 0..batch_rank {
        batch_prefix.push(sz[i]);
    }

    let mut f = Tensor::eye(dim, (Kind::Float, device));
    for _ in 0..batch_prefix.len() {
        f = f.unsqueeze(0);
    }
    if !batch_prefix.is_empty() {
        let mut expand_dims = batch_prefix.clone();
        expand_dims.push(dim);
        expand_dims.push(dim);
        f = f.expand(&expand_dims, true);
    }

    let det = rot_matrix.det();
    // F = I with F[..., -1, -1] = det(rot_matrix).
    let corner_one = Tensor::from_slice(&[0_f32, 0., 0., 0., 0., 0., 0., 0., 1.])
        .view([1, dim, dim])
        .to_device(device);
    let mut corner = corner_one;
    for _ in 0..batch_prefix.len() {
        corner = corner.unsqueeze(0);
    }
    if !batch_prefix.is_empty() {
        let mut expand_dims = batch_prefix.clone();
        expand_dims.push(dim);
        expand_dims.push(dim);
        corner = corner.expand(&expand_dims, true);
    }
    let f = &f + (&det.unsqueeze(-1).unsqueeze(-1) - 1.0) * &corner;

    let rot_matrix = u.matmul(&f).matmul(&vh).to_kind(original_dtype);

    let aligned = true_centered.matmul(&rot_matrix.transpose(-2, -1)) + pred_centroid;
    aligned.detach()
}

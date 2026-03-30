//! Boltz2 `TemplateV2Module` — full port from `modules/trunkv2.py`.
//!
//! Processes template structural information (coordinates, frames, residue types)
//! through a distogram + unit-vector featurizer and a pairformer stack to produce
//! a template-derived pairwise bias that is added to `z`.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/trunkv2.py`

use crate::layers::PairformerNoSeqModule;
use crate::tch_compat::{layer_norm_1d, linear_no_bias};
use tch::nn::{Module, Path};
use tch::{Device, Kind, Tensor};

use super::input_embedder::BOLTZ_NUM_TOKENS;

/// Template features for forward pass.
///
/// Contains all necessary template information from the featurizer.
pub struct TemplateFeatures<'a> {
    /// Template residue types (one-hot float): `[B, T, N, num_tokens]`
    pub template_restype: &'a Tensor,
    /// Template frame rotation matrices: `[B, T, N, 3, 3]`
    pub template_frame_rot: &'a Tensor,
    /// Template frame translation vectors: `[B, T, N, 3]`
    pub template_frame_t: &'a Tensor,
    /// Template frame mask: `[B, T, N]`
    pub template_mask_frame: &'a Tensor,
    /// Template CB coordinates: `[B, T, N, 3]`
    pub template_cb: &'a Tensor,
    /// Template CA coordinates: `[B, T, N, 3]`
    pub template_ca: &'a Tensor,
    /// Template CB mask: `[B, T, N]`
    pub template_mask_cb: &'a Tensor,
    /// Visibility IDs for per-template chain pairing: `[B, T, N]`
    pub visibility_ids: &'a Tensor,
    /// Overall template mask: `[B, T, N]` (`.any(dim=2)` is applied internally)
    pub template_mask: &'a Tensor,
}

/// Backward-compatible no-op stub (returns `z` unchanged).
#[derive(Debug, Default, Clone, Copy)]
pub struct TemplateModule;

impl TemplateModule {
    #[must_use]
    pub fn forward_trunk_step(&self, z: &Tensor) -> Tensor {
        z.shallow_clone()
    }
}

/// Full `TemplateV2Module` — processes template structural info into a pairwise bias.
///
/// Architecture:
/// 1. Compute distogram from CB–CB pairwise distances → one-hot
/// 2. Compute unit vectors from frame rotations + CA coordinates
/// 3. Concatenate `[distogram, cb_mask, unit_vector, frame_mask, res_type_i, res_type_j]`
/// 4. Project through `a_proj`; add projected z (`z_proj(z_norm(z))`)
/// 5. Process through a `PairformerNoSeqModule` stack
/// 6. Aggregate over templates (weighted mean)
/// 7. ReLU + final `u_proj` → pairwise bias `[B, N, N, token_z]`
pub struct TemplateV2Module {
    min_dist: f64,
    max_dist: f64,
    num_bins: i64,
    token_z: i64,
    template_dim: i64,

    z_norm: tch::nn::LayerNorm,
    v_norm: tch::nn::LayerNorm,
    z_proj: tch::nn::Linear,
    a_proj: tch::nn::Linear,
    u_proj: tch::nn::Linear,
    pairformer: PairformerNoSeqModule,

    device: Device,
}

impl TemplateV2Module {
    /// Construct under the given `VarStore` `path` (e.g. `root.sub("template_module")`).
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        path: Path<'a>,
        token_z: i64,
        template_dim: i64,
        template_blocks: i64,
        dropout: Option<f64>,
        pairwise_head_width: Option<i64>,
        pairwise_num_heads: Option<i64>,
        post_layer_norm: Option<bool>,
        activation_checkpointing: Option<bool>,
        min_dist: Option<f64>,
        max_dist: Option<f64>,
        num_bins: Option<i64>,
        device: Device,
    ) -> Self {
        let min_dist = min_dist.unwrap_or(3.25);
        let max_dist = max_dist.unwrap_or(50.75);
        let num_bins = num_bins.unwrap_or(38);

        let z_norm = layer_norm_1d(path.sub("z_norm"), token_z);
        let v_norm = layer_norm_1d(path.sub("v_norm"), template_dim);
        let z_proj = linear_no_bias(path.sub("z_proj"), token_z, template_dim);

        let a_in_dim = BOLTZ_NUM_TOKENS * 2 + num_bins + 5;
        let a_proj = linear_no_bias(path.sub("a_proj"), a_in_dim, template_dim);
        let u_proj = linear_no_bias(path.sub("u_proj"), template_dim, token_z);

        let pairformer = PairformerNoSeqModule::new(
            path.sub("pairformer"),
            template_dim,
            template_blocks,
            dropout,
            pairwise_head_width,
            pairwise_num_heads,
            post_layer_norm,
            activation_checkpointing,
            device,
        );

        Self {
            min_dist,
            max_dist,
            num_bins,
            token_z,
            template_dim,
            z_norm,
            v_norm,
            z_proj,
            a_proj,
            u_proj,
            pairformer,
            device,
        }
    }

    /// Forward pass — returns template bias `u` of shape `[B, N, N, token_z]`.
    ///
    /// Caller adds to `z`: `z = z + template_module.forward(z, feats, pair_mask, false)`.
    pub fn forward(
        &self,
        z: &Tensor,
        feats: &TemplateFeatures,
        pair_mask: &Tensor,
        use_kernels: bool,
    ) -> Tensor {
        let res_type = feats.template_restype;
        let frame_rot = feats.template_frame_rot;
        let frame_t = feats.template_frame_t;
        let frame_mask = feats.template_mask_frame;
        let cb_coords = feats.template_cb;
        let ca_coords = feats.template_ca;
        let cb_mask = feats.template_mask_cb;
        let visibility_ids = feats.visibility_ids;

        let bt_size = res_type.size();
        let b = bt_size[0];
        let t = bt_size[1];
        let n = bt_size[2];

        // template_mask: [B, T, N] → any over N → [B, T]
        let template_mask = feats
            .template_mask
            .sum_dim_intlist(&[2i64][..], false, Kind::Float)
            .gt(0.0)
            .to_kind(Kind::Float);
        let num_templates = template_mask
            .sum_dim_intlist(&[1i64][..], false, Kind::Float)
            .clamp_min(1.0); // [B] — avoid `f64::MAX` upper bound (tch Scalar overflow on clamp)

        // Pairwise masks: outer product along token dimension, then unsqueeze for feature concat
        // b_cb_mask:    [B, T, N, 1] * [B, T, 1, N] → [B, T, N, N] → [B, T, N, N, 1]
        // b_frame_mask: same pattern
        let b_cb_mask = (cb_mask.unsqueeze(-1) * cb_mask.unsqueeze(-2)).unsqueeze(-1);
        let b_frame_mask = (frame_mask.unsqueeze(-1) * frame_mask.unsqueeze(-2)).unsqueeze(-1);

        // V2 asym mask: same visibility_id ↔ same chain
        let tmlp_pair_mask = visibility_ids
            .unsqueeze(-1)
            .eq_tensor(&visibility_ids.unsqueeze(-2))
            .to_kind(Kind::Float); // [B, T, N, N]

        // ── Feature computation (no autocast) ──────────────────────────────

        // Distogram from CB–CB distances
        let distogram = self.compute_distogram(cb_coords); // [B, T, N, N, num_bins]

        // Unit vectors from frames
        let unit_vector = Self::compute_unit_vectors(ca_coords, frame_rot, frame_t);
        // [B, T, N, N, 3]

        // Concatenate geometric features
        let a_tij = Tensor::cat(
            &[
                &distogram.to_kind(Kind::Float),
                &b_cb_mask,
                &unit_vector,
                &b_frame_mask,
            ],
            -1,
        ); // [B, T, N, N, num_bins + 5]

        // Apply visibility mask
        let a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1);

        // Residue type pairwise features
        let res_type_i = res_type
            .unsqueeze(3)
            .expand(&[b, t, n, n, BOLTZ_NUM_TOKENS], false);
        let res_type_j = res_type
            .unsqueeze(2)
            .expand(&[b, t, n, n, BOLTZ_NUM_TOKENS], false);
        let a_tij = Tensor::cat(&[&a_tij, &res_type_i, &res_type_j], -1);
        // [B, T, N, N, num_bins + 5 + 2*num_tokens]
        let a_tij = self.a_proj.forward(&a_tij); // [B, T, N, N, template_dim]

        // ── Pairformer processing ──────────────────────────────────────────

        // Expand pair_mask for each template: [B, N, N] → [B*T, N, N]
        let pair_mask_t = pair_mask
            .unsqueeze(1)
            .expand(&[b, t, n, n], false)
            .reshape(&[b * t, n, n]);

        // v = z_proj(z_norm(z[:, None])) + a_tij
        let z_proj = self.z_proj.forward(&self.z_norm.forward(&z.unsqueeze(1)));
        // z_proj: [B, 1, N, N, template_dim], broadcasts over T
        let v = z_proj + a_tij; // [B, T, N, N, template_dim]
        let v = v.reshape(&[b * t, n, n, self.template_dim]);

        let v = &v + self.pairformer.forward(&v, &pair_mask_t, use_kernels);
        let v = self.v_norm.forward(&v);
        let v = v.reshape(&[b, t, n, n, self.template_dim]);

        // ── Aggregate templates ────────────────────────────────────────────

        // template_mask: [B, T] → [B, T, 1, 1, 1]
        let tmask = template_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);
        // num_templates: [B] → [B, 1, 1, 1]
        let ntmpl = num_templates.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);

        let u = (&v * &tmask).sum_dim_intlist(&[1i64][..], false, Kind::Float) / &ntmpl; // [B, N, N, template_dim]

        // Output projection
        self.u_proj.forward(&u.relu())
    }

    /// CB–CB pairwise distances → binned one-hot distogram.
    ///
    /// Returns `[B, T, N, N, num_bins]` float.
    fn compute_distogram(&self, cb_coords: &Tensor) -> Tensor {
        // Pairwise L2 distances via ||a-b||² = ||a||² + ||b||² - 2·a·b
        let sq = cb_coords
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float); // [B, T, N, 1]
        let inner = cb_coords.matmul(&cb_coords.transpose(-1, -2)); // [B, T, N, N]
        let dists = (&sq + &sq.transpose(-1, -2) - inner * 2.0)
            .clamp_min(0.0)
            .sqrt(); // [B, T, N, N]

        // Bin into histogram
        let boundaries = Tensor::linspace(
            self.min_dist,
            self.max_dist,
            self.num_bins - 1,
            (Kind::Float, self.device),
        ); // [num_bins - 1]
        let bin_idx = dists
            .unsqueeze(-1)
            .gt_tensor(&boundaries)
            .to_kind(Kind::Float)
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
            .to_kind(Kind::Int64); // [B, T, N, N]
        bin_idx.one_hot(self.num_bins).to_kind(Kind::Float)
    }

    /// Compute unit vectors from frame rotations and CA coordinates.
    ///
    /// Applies `R^T @ (ca_j - t_i)` for every (i,j) pair, then normalizes.
    /// Faithfully ports the Python: `torch.norm(vector, dim=-1, keepdim=True)`.
    ///
    /// Returns `[B, T, N, N, 3]` float.
    fn compute_unit_vectors(ca_coords: &Tensor, frame_rot: &Tensor, frame_t: &Tensor) -> Tensor {
        // frame_rot: [B, T, N, 3, 3] → [B, T, 1, N, 3, 3] (transposed = R^T)
        let rot_t = frame_rot.unsqueeze(2).transpose(-1, -2);
        // frame_t: [B, T, N, 3] → [B, T, 1, N, 3, 1]
        let t_exp = frame_t.unsqueeze(2).unsqueeze(-1);
        // ca_coords: [B, T, N, 3] → [B, T, N, 1, 3, 1]
        let ca_exp = ca_coords.unsqueeze(3).unsqueeze(-1);

        // R^T @ (ca_j - t_i): [B, T, N, N, 3, 1]
        let vector = rot_t.matmul(&(&ca_exp - &t_exp));

        // norm along last dim (size 1) — matches Python exactly
        let norm = vector
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
            .sqrt();
        let zero = Tensor::zeros_like(&vector);
        let unit = (&vector / &norm).where_self(&norm.gt(0.0), &zero);
        unit.squeeze_dim(-1) // [B, T, N, N, 3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;
    use tch::IndexOp;

    #[test]
    fn template_v2_forward_shapes() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_z: i64 = 64;
        let template_dim: i64 = 32;
        let template_blocks: i64 = 1;
        let b: i64 = 2;
        let t: i64 = 3;
        let n: i64 = 8;

        let vs = VarStore::new(device);
        let root = vs.root();
        let tmpl = TemplateV2Module::new(
            root.sub("template_module"),
            token_z,
            template_dim,
            template_blocks,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            device,
        );

        let z = Tensor::randn(&[b, n, n, token_z], (Kind::Float, device));
        let pair_mask = Tensor::ones(&[b, n, n], (Kind::Float, device));

        let feats = TemplateFeatures {
            template_restype: &Tensor::zeros(&[b, t, n, BOLTZ_NUM_TOKENS], (Kind::Float, device)),
            template_frame_rot: &Tensor::eye(3, (Kind::Float, device))
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(&[b, t, n, 3, 3], false),
            template_frame_t: &Tensor::zeros(&[b, t, n, 3], (Kind::Float, device)),
            template_mask_frame: &Tensor::ones(&[b, t, n], (Kind::Float, device)),
            template_cb: &Tensor::randn(&[b, t, n, 3], (Kind::Float, device)),
            template_ca: &Tensor::randn(&[b, t, n, 3], (Kind::Float, device)),
            template_mask_cb: &Tensor::ones(&[b, t, n], (Kind::Float, device)),
            visibility_ids: &Tensor::zeros(&[b, t, n], (Kind::Int64, device)),
            template_mask: &Tensor::ones(&[b, t, n], (Kind::Float, device)),
        };

        let u = tmpl.forward(&z, &feats, &pair_mask, false);
        assert_eq!(u.size(), vec![b, n, n, token_z]);
    }

    #[test]
    fn distogram_bin_range() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let root = vs.root();
        let tmpl = TemplateV2Module::new(
            root.sub("t"),
            64,
            32,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            device,
        );
        let coords = Tensor::randn(&[1, 1, 4, 3], (Kind::Float, device)) * 10.0;
        let dg = tmpl.compute_distogram(&coords);
        assert_eq!(dg.size(), vec![1, 1, 4, 4, 38]);
        let row_sums = dg.sum_dim_intlist(&[-1i64][..], false, Kind::Float);
        let ones = Tensor::ones_like(&row_sums);
        assert!(
            (&row_sums - &ones).abs().max().double_value(&[]) < 1e-5,
            "each distogram row must sum to 1 (one-hot)"
        );
    }

    #[test]
    fn unit_vectors_identity_frame() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let b = 1_i64;
        let t = 1_i64;
        let n = 3_i64;

        let frame_rot = Tensor::eye(3, (Kind::Float, device))
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(&[b, t, n, 3, 3], false);
        let frame_t = Tensor::zeros(&[b, t, n, 3], (Kind::Float, device));
        // Set ca[...,1] = [3,0,0] so the unit vector from residue 0 to 1 should be [1,0,0]
        let row1 = Tensor::from_slice(&[3.0_f32, 0.0, 0.0])
            .view([1, 1, 1, 3])
            .expand(&[b, t, 1, 3], false)
            .to_device(device);
        let ca_data = Tensor::cat(
            &[
                Tensor::zeros(&[b, t, 1, 3], (Kind::Float, device)),
                row1,
                Tensor::zeros(&[b, t, 1, 3], (Kind::Float, device)),
            ],
            2,
        );

        let uv = TemplateV2Module::compute_unit_vectors(&ca_data, &frame_rot, &frame_t);
        assert_eq!(uv.size(), vec![b, t, n, n, 3]);

        // Python layout: diff[i,j] = ca_i - t_j. With (i,j)=(1,0): ca_1 - t_0 = [3,0,0] → unit [1,0,0].
        let v10 = uv.i((0, 0, 1, 0));
        let expected = Tensor::from_slice(&[1.0_f32, 0.0, 0.0]);
        let diff = (&v10 - &expected).abs().max().double_value(&[]);
        assert!(diff < 1e-5, "unit vector mismatch: diff = {diff}");
    }
}

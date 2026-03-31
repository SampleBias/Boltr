//! Relative position encoding for pairwise token features (`z`), Algorithm 3 in AlphaFold3 lineage.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/encodersv2.py` (`RelativePositionEncoder`).
//! Python parity for `rel_pos` + `s_init` weights: opt-in `BOLTR_RUN_TRUNK_INIT_GOLDEN=1` +
//! [`tests/trunk_init_golden.rs`](../../tests/trunk_init_golden.rs).

use tch::nn::{linear, LinearConfig, Module, Path};
use tch::{Device, Kind, Tensor};

/// Index features required for [`RelativePositionEncoder::forward`], matching Boltz featurizer keys.
pub struct RelPosFeatures<'a> {
    pub asym_id: &'a Tensor,
    pub residue_index: &'a Tensor,
    pub entity_id: &'a Tensor,
    pub token_index: &'a Tensor,
    pub sym_id: &'a Tensor,
    /// Use zeros when `cyclic_pos_enc` is false on the encoder.
    pub cyclic_period: &'a Tensor,
}

/// `rel_pos` submodule on the Boltz2 root; `linear_layer.weight` matches Lightning.
pub struct RelativePositionEncoder {
    linear_layer: tch::nn::Linear,
    r_max: i64,
    s_max: i64,
    fix_sym_check: bool,
    cyclic_pos_enc: bool,
    device: Device,
}

impl RelativePositionEncoder {
    pub fn new<'a>(
        path: Path<'a>,
        token_z: i64,
        r_max: Option<i64>,
        s_max: Option<i64>,
        fix_sym_check: bool,
        cyclic_pos_enc: bool,
        device: Device,
    ) -> Self {
        let r_max = r_max.unwrap_or(32);
        let s_max = s_max.unwrap_or(2);
        let in_dim = 4 * (r_max + 1) + 2 * (s_max + 1) + 1;
        let linear_layer = linear(
            path.sub("linear_layer"),
            in_dim,
            token_z,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        Self {
            linear_layer,
            r_max,
            s_max,
            fix_sym_check,
            cyclic_pos_enc,
            device,
        }
    }

    /// Pairwise bias `[B, N, N, token_z]` added to `z_init` in Python `Boltz2.forward`.
    pub fn forward(&self, rel: &RelPosFeatures<'_>) -> Tensor {
        let asym_id = rel.asym_id;
        let residue_index = rel.residue_index;
        let entity_id = rel.entity_id;
        let token_index = rel.token_index;
        let sym_id = rel.sym_id;

        let b_same_chain = asym_id.unsqueeze(2).eq_tensor(&asym_id.unsqueeze(1));
        let b_same_residue = residue_index
            .unsqueeze(2)
            .eq_tensor(&residue_index.unsqueeze(1));
        let b_same_entity = entity_id.unsqueeze(2).eq_tensor(&entity_id.unsqueeze(1));

        let mut d_residue = residue_index.unsqueeze(2) - residue_index.unsqueeze(1);

        if self.cyclic_pos_enc {
            let z0 = Tensor::zeros_like(rel.cyclic_period);
            let pos = rel.cyclic_period.gt_tensor(&z0);
            if pos.any().is_nonzero() {
                let ten_k = Tensor::full_like(rel.cyclic_period, 10_000i64);
                let period = rel.cyclic_period.where_self(&pos, &ten_k);
                let d_f = d_residue.to_kind(Kind::Float);
                let p_f = period.unsqueeze(2).to_kind(Kind::Float);
                let adj = d_f.shallow_clone() / p_f.shallow_clone();
                d_residue = (d_f - p_f * adj.round()).to_kind(Kind::Int64);
            }
        }

        let r2 = 2 * self.r_max;
        let mut d_residue = (d_residue + self.r_max).clamp(0, r2);
        let off_chain = Tensor::full_like(&d_residue, r2 + 1);
        d_residue = d_residue.where_self(&b_same_chain, &off_chain);
        let a_rel_pos = d_residue.one_hot(r2 + 2);

        let mut d_token =
            (token_index.unsqueeze(2) - token_index.unsqueeze(1) + self.r_max).clamp(0, r2);
        let same_res = b_same_chain.bitwise_and_tensor(&b_same_residue);
        let off_tok = Tensor::full_like(&d_token, r2 + 1);
        d_token = d_token.where_self(&same_res, &off_tok);
        let a_rel_token = d_token.one_hot(r2 + 2);

        let mut d_chain =
            (sym_id.unsqueeze(2) - sym_id.unsqueeze(1) + self.s_max).clamp(0, 2 * self.s_max);
        let s2 = 2 * self.s_max;
        let cond = if self.fix_sym_check {
            b_same_entity.logical_not()
        } else {
            b_same_chain.shallow_clone()
        };
        let off_ch = Tensor::full_like(&d_chain, s2 + 1);
        // Match Python `torch.where(cond, off, d_chain)`: `off.where_self(cond, d_chain)`.
        d_chain = off_ch.where_self(&cond, &d_chain);
        let a_rel_chain = d_chain.one_hot(s2 + 2);

        let b_ent_f = b_same_entity.unsqueeze(-1).to_kind(Kind::Float);
        let pieces = vec![
            a_rel_pos.to_kind(Kind::Float),
            a_rel_token.to_kind(Kind::Float),
            b_ent_f,
            a_rel_chain.to_kind(Kind::Float),
        ];
        let cat = Tensor::cat(&pieces, -1);
        self.linear_layer.forward(&cat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn forward_matches_expected_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let b = 2_i64;
        let n = 11_i64;
        let token_z = 64_i64;

        let vs = VarStore::new(device);
        let enc =
            RelativePositionEncoder::new(vs.root(), token_z, None, None, false, false, device);

        let asym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
        let residue_index = Tensor::arange(n, (Kind::Int64, device))
            .view_(&[1, n])
            .expand(&[b, n], false);
        let entity_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
        let token_index = Tensor::arange(n, (Kind::Int64, device))
            .view_(&[1, n])
            .expand(&[b, n], false);
        let sym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
        let cyclic_period = Tensor::zeros(&[b, n], (Kind::Int64, device));

        let rel = RelPosFeatures {
            asym_id: &asym_id,
            residue_index: &residue_index,
            entity_id: &entity_id,
            token_index: &token_index,
            sym_id: &sym_id,
            cyclic_period: &cyclic_period,
        };
        let z = enc.forward(&rel);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }

    /// `cyclic_pos_enc=true` with a non-zero period exercises the wrap branch (Python parity).
    #[test]
    fn forward_cyclic_period_runs() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let b = 1_i64;
        let n = 9_i64;
        let token_z = 48_i64;

        let vs = VarStore::new(device);
        let enc = RelativePositionEncoder::new(vs.root(), token_z, None, None, false, true, device);

        let asym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
        let residue_index = Tensor::arange(n, (Kind::Int64, device))
            .view_(&[1, n])
            .expand(&[b, n], false);
        let entity_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
        let token_index = residue_index.shallow_clone();
        let sym_id = Tensor::zeros(&[b, n], (Kind::Int64, device));
        let mut cyc_row = vec![0_i64; n as usize];
        cyc_row[3] = 5;
        let cyclic_period = Tensor::from_slice(&cyc_row).view([1, n]).to_device(device);

        let rel = RelPosFeatures {
            asym_id: &asym_id,
            residue_index: &residue_index,
            entity_id: &entity_id,
            token_index: &token_index,
            sym_id: &sym_id,
            cyclic_period: &cyclic_period,
        };
        let z = enc.forward(&rel);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }
}

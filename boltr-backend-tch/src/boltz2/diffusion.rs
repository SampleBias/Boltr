//! Structure module: `DiffusionModule` (score network) and `AtomDiffusion` (sampler).
//!
//! Reference: `boltz-reference/src/boltz/model/modules/diffusionv2.py`
//!
//! The score model (`DiffusionModule`) receives noisy coordinates and conditioning
//! from `DiffusionConditioning`, processes through an atom attention encoder,
//! a full-sequence diffusion transformer, and an atom attention decoder to produce
//! position updates.
//!
//! `AtomDiffusion` wraps the score model with EDM-style preconditioning and
//! Karras et al. ODE sampling.

use tch::nn::{Module, Path};
use tch::{Device, Kind, Tensor};

use crate::tch_compat::{layer_norm_1d, linear_no_bias};

use super::diffusion_conditioning::DiffusionConditioningOutput;
use super::encoders::{AtomAttentionDecoder, AtomAttentionEncoder, SingleConditioning};
use super::transformers::DiffusionTransformer;

// ---------------------------------------------------------------------------
// DiffusionModule  (score network)
// ---------------------------------------------------------------------------

/// The score model: atom encoder → token transformer → atom decoder.
pub struct DiffusionModule {
    sigma_data: f64,
    single_conditioner: SingleConditioning,
    atom_attention_encoder: AtomAttentionEncoder,
    s_to_a_linear_norm: tch::nn::LayerNorm,
    s_to_a_linear_linear: tch::nn::Linear,
    token_transformer: DiffusionTransformer,
    a_norm: tch::nn::LayerNorm,
    atom_attention_decoder: AtomAttentionDecoder,
    token_s: i64,
}

impl DiffusionModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: Path<'_>,
        token_s: i64,
        atom_s: i64,
        atoms_per_window_queries: i64,
        atoms_per_window_keys: i64,
        sigma_data: f64,
        dim_fourier: i64,
        atom_encoder_depth: i64,
        atom_encoder_heads: i64,
        token_transformer_depth: i64,
        token_transformer_heads: i64,
        atom_decoder_depth: i64,
        atom_decoder_heads: i64,
        conditioning_transition_layers: i64,
        device: Device,
    ) -> Self {
        let two_ts = 2 * token_s;

        let single_conditioner = SingleConditioning::new(
            path.sub("single_conditioner"),
            sigma_data,
            token_s,
            dim_fourier,
            conditioning_transition_layers,
            2,
            device,
        );

        let atom_attention_encoder = AtomAttentionEncoder::new(
            path.sub("atom_attention_encoder"),
            atom_s,
            token_s,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_encoder_depth,
            atom_encoder_heads,
            true,
            device,
        );

        let s_to_a_linear_norm = layer_norm_1d(path.sub("s_to_a_linear").sub("0"), two_ts);
        let s_to_a_linear_linear =
            linear_no_bias(path.sub("s_to_a_linear").sub("1"), two_ts, two_ts);

        let token_transformer = DiffusionTransformer::new(
            path.sub("token_transformer"),
            token_transformer_depth,
            token_transformer_heads,
            two_ts,
            Some(two_ts),
            true,
            device,
        );

        let a_norm = layer_norm_1d(path.sub("a_norm"), two_ts);

        let atom_attention_decoder = AtomAttentionDecoder::new(
            path.sub("atom_attention_decoder"),
            atom_s,
            token_s,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_decoder_depth,
            atom_decoder_heads,
            device,
        );

        Self {
            sigma_data,
            single_conditioner,
            atom_attention_encoder,
            s_to_a_linear_norm,
            s_to_a_linear_linear,
            token_transformer,
            a_norm,
            atom_attention_decoder,
            token_s,
        }
    }

    /// Score model forward pass.
    ///
    /// * `s_inputs`: `[B, N, token_s]` trunk input embedding
    /// * `s_trunk`: `[B, N, token_s]` trunk output
    /// * `r_noisy`: `[B*mult, M, 3]` scaled noisy coordinates
    /// * `times`: `[B*mult]` scaled noise level
    /// * `cond`: pre-computed `DiffusionConditioningOutput`
    /// * `token_pad_mask`: `[B, N]`
    /// * `atom_pad_mask`: `[B, M]`
    /// * `atom_to_token`: `[B, M, N]`
    /// * `multiplicity`: sampling multiplicity
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        s_inputs: &Tensor,
        s_trunk: &Tensor,
        r_noisy: &Tensor,
        times: &Tensor,
        cond: &DiffusionConditioningOutput,
        token_pad_mask: &Tensor,
        atom_pad_mask: &Tensor,
        atom_to_token: &Tensor,
        multiplicity: i64,
    ) -> Tensor {
        let s_trunk_rep = s_trunk.repeat_interleave_self_int(multiplicity, Some(0), None);
        let s_inputs_rep = s_inputs.repeat_interleave_self_int(multiplicity, Some(0), None);

        let (s, _normed_fourier) =
            self.single_conditioner
                .forward(times, &s_trunk_rep, &s_inputs_rep);

        // Atom attention encoder
        let (a, q_skip, c_skip) = self.atom_attention_encoder.forward(
            &cond.q,
            &cond.c,
            &cond.atom_enc_bias,
            atom_pad_mask,
            atom_to_token,
            r_noisy,
            multiplicity,
            &cond.indexing_matrix,
        );

        // Token-level processing
        let s_to_a = self
            .s_to_a_linear_linear
            .forward(&self.s_to_a_linear_norm.forward(&s));
        let mut a = a + s_to_a;

        let mask = token_pad_mask
            .repeat_interleave_self_int(multiplicity, Some(0), None)
            .to_kind(Kind::Float);

        a = self.token_transformer.forward(
            &a,
            &s,
            Some(&cond.token_trans_bias.to_kind(Kind::Float)),
            &mask,
            multiplicity,
        );
        a = self.a_norm.forward(&a);

        // Atom attention decoder → position update
        self.atom_attention_decoder.forward(
            &a,
            &q_skip,
            &c_skip,
            &cond.atom_dec_bias,
            atom_pad_mask,
            atom_to_token,
            multiplicity,
            &cond.indexing_matrix,
        )
    }
}

// ---------------------------------------------------------------------------
// AtomDiffusion  (EDM sampler + preconditioning)
// ---------------------------------------------------------------------------

/// EDM-style diffusion sampler wrapping the score model.
pub struct AtomDiffusion {
    pub score_model: DiffusionModule,
    sigma_min: f64,
    sigma_max: f64,
    sigma_data: f64,
    rho: f64,
    num_sampling_steps: i64,
    gamma_0: f64,
    gamma_min: f64,
    noise_scale: f64,
    step_scale: f64,
    token_s: i64,
}

/// Configuration for constructing [`AtomDiffusion`].
pub struct AtomDiffusionConfig {
    pub num_sampling_steps: i64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub sigma_data: f64,
    pub rho: f64,
    pub gamma_0: f64,
    pub gamma_min: f64,
    pub noise_scale: f64,
    pub step_scale: f64,
}

impl Default for AtomDiffusionConfig {
    fn default() -> Self {
        Self {
            num_sampling_steps: 5,
            sigma_min: 0.0004,
            sigma_max: 160.0,
            sigma_data: 16.0,
            rho: 7.0,
            gamma_0: 0.8,
            gamma_min: 1.0,
            noise_scale: 1.003,
            step_scale: 1.5,
        }
    }
}

/// Output of the diffusion sampling process.
pub struct DiffusionSampleOutput {
    /// Denoised atom coordinates `[multiplicity, M, 3]`.
    pub sample_atom_coords: Tensor,
}

impl AtomDiffusion {
    /// Build the score model and sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: Path<'_>,
        token_s: i64,
        atom_s: i64,
        atoms_per_window_queries: i64,
        atoms_per_window_keys: i64,
        dim_fourier: i64,
        atom_encoder_depth: i64,
        atom_encoder_heads: i64,
        token_transformer_depth: i64,
        token_transformer_heads: i64,
        atom_decoder_depth: i64,
        atom_decoder_heads: i64,
        conditioning_transition_layers: i64,
        config: AtomDiffusionConfig,
        device: Device,
    ) -> Self {
        let score_model = DiffusionModule::new(
            path.sub("score_model"),
            token_s,
            atom_s,
            atoms_per_window_queries,
            atoms_per_window_keys,
            config.sigma_data,
            dim_fourier,
            atom_encoder_depth,
            atom_encoder_heads,
            token_transformer_depth,
            token_transformer_heads,
            atom_decoder_depth,
            atom_decoder_heads,
            conditioning_transition_layers,
            device,
        );

        Self {
            score_model,
            sigma_min: config.sigma_min,
            sigma_max: config.sigma_max,
            sigma_data: config.sigma_data,
            rho: config.rho,
            num_sampling_steps: config.num_sampling_steps,
            gamma_0: config.gamma_0,
            gamma_min: config.gamma_min,
            noise_scale: config.noise_scale,
            step_scale: config.step_scale,
            token_s,
        }
    }

    // ---- EDM preconditioning scalings ----

    fn c_skip(&self, sigma: &Tensor) -> Tensor {
        let sd2 = self.sigma_data * self.sigma_data;
        sd2 / (sigma.pow_tensor_scalar(2) + sd2)
    }

    fn c_out(&self, sigma: &Tensor) -> Tensor {
        sigma * self.sigma_data
            / (sigma.pow_tensor_scalar(2) + self.sigma_data * self.sigma_data).sqrt()
    }

    fn c_in(&self, sigma: &Tensor) -> Tensor {
        1.0 / (sigma.pow_tensor_scalar(2) + self.sigma_data * self.sigma_data).sqrt()
    }

    fn c_noise(&self, sigma: &Tensor) -> Tensor {
        (sigma / self.sigma_data).clamp_min(1e-20).log() * 0.25
    }

    /// Preconditioned network forward: scale input → run score model → scale output.
    #[allow(clippy::too_many_arguments)]
    fn preconditioned_network_forward(
        &self,
        noised_atom_coords: &Tensor,
        sigma: &Tensor,
        s_inputs: &Tensor,
        s_trunk: &Tensor,
        cond: &DiffusionConditioningOutput,
        token_pad_mask: &Tensor,
        atom_pad_mask: &Tensor,
        atom_to_token: &Tensor,
        multiplicity: i64,
    ) -> Tensor {
        let padded_sigma = sigma.unsqueeze(-1).unsqueeze(-1); // [B, 1, 1]

        let r_noisy = &self.c_in(&padded_sigma) * noised_atom_coords;

        let r_update = self.score_model.forward(
            s_inputs,
            s_trunk,
            &r_noisy,
            &self.c_noise(sigma),
            cond,
            token_pad_mask,
            atom_pad_mask,
            atom_to_token,
            multiplicity,
        );

        &self.c_skip(&padded_sigma) * noised_atom_coords + &self.c_out(&padded_sigma) * r_update
    }

    /// Karras et al. noise schedule: `sigma_i` from `sigma_max` to `sigma_min`.
    fn sample_schedule(&self, num_steps: i64, device: Device) -> Tensor {
        let inv_rho = 1.0 / self.rho;
        let steps = Tensor::arange(num_steps, (Kind::Float, device));
        let sigmas = (self.sigma_max.powf(inv_rho)
            + &steps / ((num_steps - 1) as f64)
                * (self.sigma_min.powf(inv_rho) - self.sigma_max.powf(inv_rho)))
        .pow_tensor_scalar(self.rho);
        let sigmas = sigmas * self.sigma_data;
        Tensor::cat(&[sigmas, Tensor::zeros(&[1], (Kind::Float, device))], 0)
    }

    /// Run the full reverse-diffusion sampling loop.
    ///
    /// Returns denoised atom coordinates `[multiplicity, M, 3]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        s_inputs: &Tensor,
        s_trunk: &Tensor,
        cond: &DiffusionConditioningOutput,
        token_pad_mask: &Tensor,
        atom_pad_mask: &Tensor,
        atom_to_token: &Tensor,
        num_sampling_steps: Option<i64>,
        multiplicity: i64,
    ) -> DiffusionSampleOutput {
        let device = s_trunk.device();
        let num_steps = num_sampling_steps.unwrap_or(self.num_sampling_steps);

        let atom_mask = atom_pad_mask.repeat_interleave_self_int(multiplicity, Some(0), None);
        let shape = [atom_mask.size()[0], atom_mask.size()[1], 3];

        let sigmas = self.sample_schedule(num_steps, device);

        let init_sigma =
            f64::try_from(sigmas.select(0, 0)).unwrap_or(self.sigma_max * self.sigma_data);
        let mut atom_coords = Tensor::randn(&shape, (Kind::Float, device)) * init_sigma;
        let mut atom_coords_denoised: Option<Tensor> = None;

        for step_idx in 0..(num_steps as usize) {
            let sigma_tm = f64::try_from(sigmas.select(0, step_idx as i64)).unwrap();
            let sigma_t = f64::try_from(sigmas.select(0, (step_idx + 1) as i64)).unwrap();
            let gamma = if sigma_t > self.gamma_min {
                self.gamma_0
            } else {
                0.0
            };

            // Center coordinates
            atom_coords = &atom_coords - atom_coords.mean_dim(&[-2i64][..], true, Kind::Float);

            let t_hat = sigma_tm * (1.0 + gamma);
            let noise_var =
                self.noise_scale * self.noise_scale * (t_hat * t_hat - sigma_tm * sigma_tm);
            let eps = Tensor::randn(&shape, (Kind::Float, device)) * noise_var.sqrt();
            let atom_coords_noisy = &atom_coords + &eps;

            let t_hat_tensor =
                Tensor::full(&[atom_coords_noisy.size()[0]], t_hat, (Kind::Float, device));

            let _no_grad = tch::no_grad_guard();
            atom_coords_denoised = Some(self.preconditioned_network_forward(
                &atom_coords_noisy,
                &t_hat_tensor,
                s_inputs,
                s_trunk,
                cond,
                token_pad_mask,
                atom_pad_mask,
                atom_to_token,
                multiplicity,
            ));

            let acd = atom_coords_denoised.as_ref().unwrap();
            let denoised_over_sigma = (&atom_coords_noisy - acd) / t_hat;
            atom_coords =
                &atom_coords_noisy + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma;
        }

        DiffusionSampleOutput {
            sample_atom_coords: atom_coords,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn sample_schedule_shape() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let vs = VarStore::new(device);
        let cfg = AtomDiffusionConfig::default();
        let ad = AtomDiffusion::new(
            vs.root().sub("structure_module"),
            32,
            16,
            4,
            8,
            64,
            1,
            2,
            1,
            2,
            1,
            2,
            2,
            cfg,
            device,
        );
        let sigmas = ad.sample_schedule(5, device);
        assert_eq!(sigmas.size(), vec![6]); // num_steps + 1
        let first = f64::try_from(sigmas.select(0, 0)).unwrap();
        let last = f64::try_from(sigmas.select(0, 5)).unwrap();
        assert!(first > 0.0, "first sigma should be positive");
        assert!((last).abs() < 1e-10, "last sigma should be ~0");
    }
}

//! Boltz2 high-level model: one `VarStore` aligned with Lightning `Boltz2` root names.
//!
//! Python layout reference: `boltz-reference/src/boltz/model/models/boltz2.py`
//! — `s_init`, `z_init_*`, recycling, and `pairformer_module` live on the **model root**
//! (not inside a nested `trunk` submodule).

use std::path::Path;

use anyhow::Result;
use tch::nn::{embedding, linear, EmbeddingConfig, LinearConfig, Module, VarStore};
use tch::{Device, Kind, Tensor};

use super::confidence::{ConfidenceModule, ConfidenceModuleConfig, ConfidenceOutput};
use super::contact_conditioning::{ContactConditioning, ContactFeatures};
use super::diffusion::{AtomDiffusion, AtomDiffusionConfig, DiffusionSampleOutput};
use super::diffusion_conditioning::{DiffusionConditioning, DiffusionConditioningOutput};
use super::distogram::{BFactorModule, DistogramModule};
use super::input_embedder::InputEmbedder;
use super::msa_module::{MsaFeatures, MsaModule};
use super::relative_position::{RelPosFeatures, RelativePositionEncoder};
use super::template_module::TemplateFeatures;
use super::trunk::TrunkV2;
use crate::boltz_hparams::Boltz2Hparams;
use crate::checkpoint::load_tensor_from_safetensors;

/// `len(const.bond_types) + 1` in Boltz (`boltz-reference/.../const.py`).
pub const BOND_TYPE_EMBEDDING_NUM: i64 = 7;

/// Boltz2 inference model: trunk + pairformer + diffusion conditioning + structure module +
/// distogram/bfactor heads on a shared [`VarStore`].
pub struct Boltz2Model {
    device: Device,
    var_store: VarStore,
    trunk: TrunkV2,
    rel_pos: RelativePositionEncoder,
    /// Python `token_bonds`: `Linear(1, token_z, bias=False)`.
    token_bonds: tch::nn::Linear,
    /// Python `token_bonds_type` when `bond_type_feature`; `Embedding(BOND_TYPE_EMBEDDING_NUM, token_z)`.
    token_bonds_type: Option<tch::nn::Embedding>,
    /// Python `contact_conditioning` (cutoffs default 4 Å / 20 Å like `Boltz2`).
    contact_conditioning: ContactConditioning,
    /// Partial `input_embedder`: `res_type` + `msa_profile` linears; pass atom-attn `a` from outside.
    input_embedder: InputEmbedder,
    /// Pre-computes conditioning for the diffusion score model.
    diffusion_conditioning: DiffusionConditioning,
    /// EDM sampler + score network.
    structure_module: AtomDiffusion,
    /// Predicted distogram logits from pair representation.
    distogram_module: DistogramModule,
    /// Optional B-factor head.
    bfactor_module: Option<BFactorModule>,
    /// Optional confidence stack (`confidence_module` in Lightning) — `boltz.model.modules.confidencev2`.
    confidence_module: Option<ConfidenceModule>,
    token_s: i64,
    token_z: i64,
    atom_s: i64,
    atom_z: i64,
    num_bins: i64,
}

/// Boltz2 default score model / diffusion hyper-parameters (matching Python defaults).
#[derive(Debug, Clone)]
pub struct Boltz2DiffusionArgs {
    pub atom_s: i64,
    pub atom_z: i64,
    pub atoms_per_window_queries: i64,
    pub atoms_per_window_keys: i64,
    pub atom_encoder_depth: i64,
    pub atom_encoder_heads: i64,
    pub token_transformer_depth: i64,
    pub token_transformer_heads: i64,
    pub atom_decoder_depth: i64,
    pub atom_decoder_heads: i64,
    pub atom_feature_dim: i64,
    pub conditioning_transition_layers: i64,
    pub dim_fourier: i64,
    pub num_bins: i64,
    pub predict_bfactor: bool,
    /// `Some((template_dim, template_blocks))` to enable templates.
    pub use_templates: Option<(i64, i64)>,
}

/// Tensors required for diffusion sampling + confidence after trunk (collate / featurizer keys).
pub struct PredictStepFeats<'a> {
    pub token_pad_mask: &'a Tensor,
    pub asym_id: &'a Tensor,
    pub mol_type: &'a Tensor,
    pub token_to_rep_atom: &'a Tensor,
    pub frames_idx: &'a Tensor,
    pub ref_pos: &'a Tensor,
    pub ref_charge: &'a Tensor,
    pub ref_element: &'a Tensor,
    pub atom_pad_mask: &'a Tensor,
    pub ref_space_uid: &'a Tensor,
    pub atom_to_token: &'a Tensor,
}

/// Trunk + structure + distogram + optional confidence (Python `Boltz2.forward` predict path).
pub struct PredictStepOutput {
    pub s_trunk: Tensor,
    pub z_trunk: Tensor,
    pub diffusion: DiffusionSampleOutput,
    pub pdistogram: Tensor,
    pub confidence: Option<ConfidenceOutput>,
}

impl Default for Boltz2DiffusionArgs {
    fn default() -> Self {
        Self {
            atom_s: 128,
            atom_z: 16,
            atoms_per_window_queries: 32,
            atoms_per_window_keys: 128,
            atom_encoder_depth: 3,
            atom_encoder_heads: 4,
            token_transformer_depth: 24,
            token_transformer_heads: 8,
            atom_decoder_depth: 3,
            atom_decoder_heads: 4,
            atom_feature_dim: 128,
            conditioning_transition_layers: 2,
            dim_fourier: 256,
            num_bins: 64,
            predict_bfactor: false,
            use_templates: None,
        }
    }
}

impl Boltz2Model {
    /// Build from exported Lightning `hyper_parameters` JSON ([`Boltz2Hparams`]).
    pub fn from_hparams_json(device: Device, json_bytes: &[u8]) -> Result<Self> {
        let h = Boltz2Hparams::from_json_slice(json_bytes)?;
        Self::with_all_options(
            device,
            h.resolved_token_s(),
            h.resolved_token_z(),
            h.resolved_num_pairformer_blocks(),
            h.resolved_bond_type_feature(),
            Boltz2DiffusionArgs::default(),
            AtomDiffusionConfig::default(),
            None,
        )
    }

    /// Build with default pairformer depth (`4` blocks) and `token_z = 128`.
    pub fn new(device: Device, token_s: i64) -> Self {
        Self::with_options(device, token_s, 128, None)
    }

    /// Full constructor: `num_pairformer_blocks` defaults to `4` when `None`; no bond-type embedding.
    pub fn with_options(
        device: Device,
        token_s: i64,
        token_z: i64,
        num_pairformer_blocks: Option<i64>,
    ) -> Self {
        Self::with_options_bonds(device, token_s, token_z, num_pairformer_blocks, false)
    }

    /// Like [`Self::with_options`]; set `bond_type_feature` to match Python `Boltz2(bond_type_feature=…)`.
    pub fn with_options_bonds(
        device: Device,
        token_s: i64,
        token_z: i64,
        num_pairformer_blocks: Option<i64>,
        bond_type_feature: bool,
    ) -> Self {
        Self::with_all_options(
            device,
            token_s,
            token_z,
            num_pairformer_blocks,
            bond_type_feature,
            Boltz2DiffusionArgs::default(),
            AtomDiffusionConfig::default(),
            None,
        )
        .expect("default diffusion args should not fail")
    }

    /// Full constructor with all diffusion / head arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn with_all_options(
        device: Device,
        token_s: i64,
        token_z: i64,
        num_pairformer_blocks: Option<i64>,
        bond_type_feature: bool,
        diff_args: Boltz2DiffusionArgs,
        diff_config: AtomDiffusionConfig,
        confidence: Option<ConfidenceModuleConfig>,
    ) -> Result<Self> {
        let var_store = VarStore::new(device);
        let root = var_store.root();
        let trunk = TrunkV2::new(
            &var_store,
            Some(token_s),
            Some(token_z),
            num_pairformer_blocks,
            diff_args.use_templates,
            device,
        );
        let rel_pos = RelativePositionEncoder::new(
            root.sub("rel_pos"),
            token_z,
            None,
            None,
            false,
            false,
            device,
        );
        let token_bonds = linear(
            root.sub("token_bonds"),
            1,
            token_z,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let token_bonds_type = bond_type_feature.then(|| {
            embedding(
                root.sub("token_bonds_type"),
                BOND_TYPE_EMBEDDING_NUM,
                token_z,
                EmbeddingConfig::default(),
            )
        });
        let contact_conditioning =
            ContactConditioning::new(root.sub("contact_conditioning"), token_z, 4.0, 20.0);
        let input_embedder = InputEmbedder::new(root.sub("input_embedder"), token_s);

        let diffusion_conditioning = DiffusionConditioning::new(
            root.sub("diffusion_conditioning"),
            token_s,
            token_z,
            diff_args.atom_s,
            diff_args.atom_z,
            diff_args.atoms_per_window_queries,
            diff_args.atoms_per_window_keys,
            diff_args.atom_encoder_depth,
            diff_args.atom_encoder_heads,
            diff_args.token_transformer_depth,
            diff_args.token_transformer_heads,
            diff_args.atom_decoder_depth,
            diff_args.atom_decoder_heads,
            diff_args.atom_feature_dim,
            diff_args.conditioning_transition_layers,
            device,
        );

        let structure_module = AtomDiffusion::new(
            root.sub("structure_module"),
            token_s,
            diff_args.atom_s,
            diff_args.atoms_per_window_queries,
            diff_args.atoms_per_window_keys,
            diff_args.dim_fourier,
            diff_args.atom_encoder_depth,
            diff_args.atom_encoder_heads,
            diff_args.token_transformer_depth,
            diff_args.token_transformer_heads,
            diff_args.atom_decoder_depth,
            diff_args.atom_decoder_heads,
            diff_args.conditioning_transition_layers,
            diff_config,
            device,
        );

        let distogram_module = DistogramModule::new(
            root.sub("distogram_module"),
            token_z,
            diff_args.num_bins,
            None,
        );

        let bfactor_module = diff_args.predict_bfactor.then(|| {
            BFactorModule::new(
                root.sub("bfactor_module"),
                token_s,
                diff_args.num_bins,
            )
        });

        let confidence_module = confidence.map(|mut cfg| {
            cfg.pairformer_num_blocks = num_pairformer_blocks.unwrap_or(cfg.pairformer_num_blocks);
            ConfidenceModule::new(
                root.sub("confidence_module"),
                device,
                token_s,
                token_z,
                &cfg,
            )
        });

        Ok(Self {
            device,
            var_store,
            trunk,
            rel_pos,
            token_bonds,
            token_bonds_type,
            contact_conditioning,
            input_embedder,
            diffusion_conditioning,
            structure_module,
            distogram_module,
            bfactor_module,
            confidence_module,
            token_s,
            token_z,
            atom_s: diff_args.atom_s,
            atom_z: diff_args.atom_z,
            num_bins: diff_args.num_bins,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn token_s(&self) -> i64 {
        self.token_s
    }

    pub fn token_z(&self) -> i64 {
        self.token_z
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    pub fn var_store_mut(&mut self) -> &mut VarStore {
        &mut self.var_store
    }

    pub fn trunk(&self) -> &TrunkV2 {
        &self.trunk
    }

    pub fn trunk_mut(&mut self) -> &mut TrunkV2 {
        &mut self.trunk
    }

    /// Lightning `msa_module` subgraph (optional `MsaFeatures` on trunk forwards).
    pub fn msa(&self) -> &MsaModule {
        self.trunk.msa()
    }

    pub fn rel_pos(&self) -> &RelativePositionEncoder {
        &self.rel_pos
    }

    pub fn bond_type_feature(&self) -> bool {
        self.token_bonds_type.is_some()
    }

    pub fn contact_conditioning(&self) -> &ContactConditioning {
        &self.contact_conditioning
    }

    pub fn input_embedder(&self) -> &InputEmbedder {
        &self.input_embedder
    }

    /// Python `input_embedder` tail: `a` + `res_type_encoding` + `msa_profile_encoding`.
    pub fn forward_input_embedder(
        &self,
        atom_attn_out: &Tensor,
        res_type: &Tensor,
        profile: &Tensor,
        deletion_mean: &Tensor,
    ) -> Tensor {
        self.input_embedder
            .forward_with_atom_repr(atom_attn_out, res_type, profile, deletion_mean)
    }

    /// Same as [`Self::forward_input_embedder`] but with `profile_affinity` / `deletion_mean_affinity`
    /// tensors (Boltz `InputEmbedder.forward(..., affinity=True)` in `trunkv2.py`).
    pub fn forward_input_embedder_affinity(
        &self,
        atom_attn_out: &Tensor,
        res_type: &Tensor,
        profile_affinity: &Tensor,
        deletion_mean_affinity: &Tensor,
    ) -> Tensor {
        self.forward_input_embedder(
            atom_attn_out,
            res_type,
            profile_affinity,
            deletion_mean_affinity,
        )
    }

    /// Pairwise contact bias `[B, N, N, token_z]` (Python `contact_conditioning(feats)`).
    pub fn forward_contact_conditioning(&self, feats: &ContactFeatures<'_>) -> Tensor {
        self.contact_conditioning.forward(feats)
    }

    /// `z += token_bonds(feats["token_bonds"])` (+ optional type embedding). See Python `Boltz2.forward`.
    ///
    /// `token_bonds` is `[B, N, N, 1]` float. If `None`, uses zeros (same as a missing bond signal of 0).
    /// `type_bonds` is `[B, N, N]` int64 when bond-type embedding is enabled.
    pub fn forward_token_bonds_bias(
        &self,
        batch: i64,
        num_tokens: i64,
        token_bonds: Option<&Tensor>,
        type_bonds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let tb_in = match token_bonds {
            Some(t) => t.shallow_clone(),
            None => Tensor::zeros(
                &[batch, num_tokens, num_tokens, 1],
                (Kind::Float, self.device),
            ),
        };
        let mut z = self.token_bonds.forward(&tb_in);
        if let Some(emb) = &self.token_bonds_type {
            let idx = type_bonds.ok_or_else(|| {
                anyhow::anyhow!("type_bonds is required when bond_type_feature is enabled")
            })?;
            z = z + emb.forward(idx);
        }
        Ok(z)
    }

    /// Load `s_init.weight` from a safetensors file (exported from Lightning `state_dict`).
    pub fn load_s_init_from_safetensors(&mut self, path: &Path) -> Result<()> {
        let w = load_tensor_from_safetensors(path, "s_init.weight", self.device)?;
        self.trunk.load_s_init_weight(&w);
        Ok(())
    }

    /// Load every tensor that exists in both the safetensors file and this model's [`VarStore`].
    ///
    /// Checkpoint keys must match `VarStore` names (export with
    /// `scripts/export_checkpoint_to_safetensors.py` and `--strip-prefix` if Lightning uses a
    /// `model.` prefix). Returns the names of **model** parameters that were not found in the file
    /// (they remain at default initialization).
    pub fn load_partial_from_safetensors(&mut self, path: &Path) -> Result<Vec<String>> {
        self.var_store
            .load_partial(path)
            .map_err(|e| anyhow::anyhow!("VarStore::load_partial: {e}"))
    }

    /// VarStore keys with no matching tensor name in the file (without loading).
    pub fn var_store_keys_missing_in_safetensors(&self, path: &Path) -> Result<Vec<String>> {
        crate::checkpoint::var_store_keys_missing_in_safetensors(path, &self.var_store)
    }

    /// Safetensors keys not used by this graph (e.g. diffusion heads in a full checkpoint).
    pub fn safetensors_keys_unused_by_model(&self, path: &Path) -> Result<Vec<String>> {
        crate::checkpoint::safetensor_names_not_in_var_store(path, &self.var_store)
    }

    /// Classify all tensor names in a `.safetensors` file into Rust inference–related keys vs
    /// other heads (see [`crate::inference_keys::BOLTZ2_INFERENCE_TOP_LEVEL_KEYS`]).
    pub fn partition_checkpoint_keys_for_inference(
        path: &Path,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let names = crate::checkpoint::list_safetensor_names(path)?;
        Ok(crate::inference_keys::partition_safetensors_keys_for_inference(
            &names,
        ))
    }

    /// [`load_partial_from_safetensors`] then fail if any model parameter was not found in the file.
    pub fn load_from_safetensors_require_all_vars(&mut self, path: &Path) -> Result<()> {
        let missing = self.load_partial_from_safetensors(path)?;
        if !missing.is_empty() {
            anyhow::bail!(
                "safetensors missing {} VarStore keys (export/strip-prefix mismatch?): {:?}",
                missing.len(),
                missing
            );
        }
        Ok(())
    }

    /// Load weights and optionally enforce a **fully consumed** checkpoint (no extra tensors).
    pub fn load_from_safetensors_strict(
        &mut self,
        path: &Path,
        reject_unused_file_keys: bool,
    ) -> Result<()> {
        self.load_from_safetensors_require_all_vars(path)?;
        if reject_unused_file_keys {
            let extra = self.safetensors_keys_unused_by_model(path)?;
            if !extra.is_empty() {
                anyhow::bail!(
                    "safetensors contains {} keys not in this VarStore: {:?}",
                    extra.len(),
                    extra
                );
            }
        }
        Ok(())
    }

    /// Same transform as Python `self.s_init(s_inputs)` on `[B, N, token_s]`.
    pub fn forward_s_init(&self, s: &Tensor) -> Tensor {
        self.trunk.apply_s_init(s)
    }

    /// Trunk + pairformer + recycling, matching Python’s inner loop for a **pre-embedded** batch.
    ///
    /// `s_inputs` is the output of Python `input_embedder(feats)` — shape `[B, N, token_s]`.
    pub fn forward_trunk(
        &self,
        s_inputs: &Tensor,
        recycling_steps: Option<i64>,
        msa_feats: Option<&MsaFeatures<'_>>,
        template_feats: Option<&TemplateFeatures<'_>>,
    ) -> Result<(Tensor, Tensor)> {
        self.trunk
            .forward(s_inputs, recycling_steps, msa_feats, template_feats)
    }

    /// Relative position bias `[B, N, N, token_z]` from tokenizer/featurizer index tensors.
    pub fn forward_rel_pos(&self, rel: &RelPosFeatures<'_>) -> Tensor {
        self.rel_pos.forward(rel)
    }

    /// Trunk forward with Python-aligned `z_init += rel_pos(feats)` before recycling.
    pub fn forward_trunk_with_rel_pos(
        &self,
        s_inputs: &Tensor,
        rel: &RelPosFeatures<'_>,
        recycling_steps: Option<i64>,
        msa_feats: Option<&MsaFeatures<'_>>,
        template_feats: Option<&TemplateFeatures<'_>>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_trunk_with_z_init_terms(
            s_inputs,
            rel,
            None,
            None,
            None,
            recycling_steps,
            msa_feats,
            template_feats,
        )
    }

    /// Full z-init slice matching Python: pair + `rel_pos` + `token_bonds` + `contact_conditioning`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_trunk_with_z_init_terms(
        &self,
        s_inputs: &Tensor,
        rel: &RelPosFeatures<'_>,
        token_bonds: Option<&Tensor>,
        type_bonds: Option<&Tensor>,
        contact: Option<&ContactFeatures<'_>>,
        recycling_steps: Option<i64>,
        msa_feats: Option<&MsaFeatures<'_>>,
        template_feats: Option<&TemplateFeatures<'_>>,
    ) -> Result<(Tensor, Tensor)> {
        let b = s_inputs.size()[0];
        let n = s_inputs.size()[1];
        let (s_init, z_pair) = self.trunk.initialize(s_inputs);
        let z_rel = self.rel_pos.forward(rel);
        let z_bonds = self.forward_token_bonds_bias(b, n, token_bonds, type_bonds)?;
        let z_contact = match contact {
            Some(c) => self.contact_conditioning.forward(c),
            None => Tensor::zeros(&[b, n, n, self.token_z], (Kind::Float, self.device)),
        };
        let z_init = z_pair + z_rel + z_bonds + z_contact;
        self.trunk
            .forward_from_init(&s_init, &z_init, recycling_steps, msa_feats, template_feats)
    }

    pub fn diffusion_conditioning(&self) -> &DiffusionConditioning {
        &self.diffusion_conditioning
    }

    pub fn structure_module(&self) -> &AtomDiffusion {
        &self.structure_module
    }

    pub fn distogram_module(&self) -> &DistogramModule {
        &self.distogram_module
    }

    pub fn bfactor_module(&self) -> Option<&BFactorModule> {
        self.bfactor_module.as_ref()
    }

    pub fn confidence_module(&self) -> Option<&ConfidenceModule> {
        self.confidence_module.as_ref()
    }

    pub fn atom_s(&self) -> i64 {
        self.atom_s
    }

    pub fn atom_z(&self) -> i64 {
        self.atom_z
    }

    pub fn num_bins(&self) -> i64 {
        self.num_bins
    }

    /// Run `DiffusionConditioning` on trunk outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_diffusion_conditioning(
        &self,
        s_trunk: &Tensor,
        z_trunk: &Tensor,
        relative_position_encoding: &Tensor,
        ref_pos: &Tensor,
        ref_charge: &Tensor,
        ref_element: &Tensor,
        atom_pad_mask: &Tensor,
        ref_space_uid: &Tensor,
        atom_to_token: &Tensor,
    ) -> DiffusionConditioningOutput {
        self.diffusion_conditioning.forward(
            s_trunk,
            z_trunk,
            relative_position_encoding,
            ref_pos,
            ref_charge,
            ref_element,
            atom_pad_mask,
            ref_space_uid,
            atom_to_token,
        )
    }

    /// Run the distogram head on pair representation `z`.
    pub fn forward_distogram(&self, z: &Tensor) -> Tensor {
        self.distogram_module.forward(z)
    }

    /// Run the B-factor head on single representation `s` (returns `None` if disabled).
    pub fn forward_bfactor(&self, s: &Tensor) -> Option<Tensor> {
        self.bfactor_module.as_ref().map(|m| m.forward(s))
    }

    /// Run the full reverse-diffusion sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_diffusion_sample(
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
        self.structure_module.sample(
            s_inputs,
            s_trunk,
            cond,
            token_pad_mask,
            atom_pad_mask,
            atom_to_token,
            num_sampling_steps,
            multiplicity,
        )
    }

    /// Full predict path: trunk → diffusion conditioning → sample → distogram → optional confidence.
    /// Aligns with `Boltz2.forward` when `run_trunk_and_structure` and `confidence_prediction` are set.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_step(
        &self,
        s_inputs: &Tensor,
        rel: &RelPosFeatures<'_>,
        token_bonds: Option<&Tensor>,
        type_bonds: Option<&Tensor>,
        contact: Option<&ContactFeatures<'_>>,
        recycling_steps: Option<i64>,
        msa_feats: Option<&MsaFeatures<'_>>,
        template_feats: Option<&TemplateFeatures<'_>>,
        feats: &PredictStepFeats<'_>,
        num_sampling_steps: Option<i64>,
        multiplicity: i64,
    ) -> Result<PredictStepOutput> {
        let (s_trunk, z_trunk) = self.forward_trunk_with_z_init_terms(
            s_inputs,
            rel,
            token_bonds,
            type_bonds,
            contact,
            recycling_steps,
            msa_feats,
            template_feats,
        )?;
        let rel_enc = self.forward_rel_pos(rel);
        let cond = self.forward_diffusion_conditioning(
            &s_trunk,
            &z_trunk,
            &rel_enc,
            feats.ref_pos,
            feats.ref_charge,
            feats.ref_element,
            feats.atom_pad_mask,
            feats.ref_space_uid,
            feats.atom_to_token,
        );
        let diffusion = self.forward_diffusion_sample(
            s_inputs,
            &s_trunk,
            &cond,
            feats.token_pad_mask,
            feats.atom_pad_mask,
            feats.atom_to_token,
            num_sampling_steps,
            multiplicity,
        );
        let pdistogram = self.forward_distogram(&z_trunk);
        let confidence = self.confidence_module.as_ref().map(|cm| {
            cm.forward(
                s_inputs,
                &s_trunk,
                &z_trunk,
                &diffusion.sample_atom_coords,
                feats.token_pad_mask,
                feats.asym_id,
                feats.mol_type,
                feats.token_to_rep_atom,
                feats.frames_idx,
                &pdistogram,
                multiplicity,
            )
        });
        Ok(PredictStepOutput {
            s_trunk,
            z_trunk,
            diffusion,
            pdistogram,
            confidence,
        })
    }

    /// `predict_step`-shaped entry: trunk with recycling + MSA + template + pairformer.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_step_trunk(
        &self,
        s_inputs: &Tensor,
        rel: &RelPosFeatures<'_>,
        token_bonds: Option<&Tensor>,
        type_bonds: Option<&Tensor>,
        contact: Option<&ContactFeatures<'_>>,
        recycling_steps: Option<i64>,
        msa_feats: Option<&MsaFeatures<'_>>,
        template_feats: Option<&TemplateFeatures<'_>>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_trunk_with_z_init_terms(
            s_inputs,
            rel,
            token_bonds,
            type_bonds,
            contact,
            recycling_steps,
            msa_feats,
            template_feats,
        )
    }

    /// Whether templates are enabled (weights created in the trunk).
    pub fn use_templates(&self) -> bool {
        self.trunk.has_template()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boltz2::contact_conditioning::{ContactFeatures, CONTACT_CONDITIONING_CHANNELS};

    #[test]
    fn var_store_keys_missing_in_empty_safetensors() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let m = Boltz2Model::with_options(device, 64, 32, Some(1));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("empty.safetensors");
        // Valid empty safetensors file (matches safetensors crate `test_empty`).
        const EMPTY: &[u8] = &[8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32];
        std::fs::write(&p, EMPTY).unwrap();
        let missing = m.var_store_keys_missing_in_safetensors(&p).unwrap();
        assert!(
            missing.len() > 5,
            "expected many unfilled keys, got {}",
            missing.len()
        );
        let mut m2 = Boltz2Model::with_options(device, 64, 32, Some(1));
        let err = m2
            .load_from_safetensors_require_all_vars(&p)
            .expect_err("strict load should fail");
        let s = format!("{err:#}");
        assert!(
            s.contains("missing") || s.contains("VarStore"),
            "unexpected error: {s}"
        );
    }

    #[test]
    fn s_init_linear_runs() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let ts = 384_i64;
        let m = Boltz2Model::new(device, ts);
        let x = Tensor::randn(&[7, ts], (tch::Kind::Float, device));
        let y = m.forward_s_init(&x);
        assert_eq!(y.size(), vec![7, ts]);
    }

    #[test]
    fn trunk_forward_runs_small() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let token_s = 64_i64;
        let token_z = 32_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = Boltz2Model::with_options(device, token_s, token_z, Some(1));
        let s_in = Tensor::randn(&[b, n, token_s], (tch::Kind::Float, device));
        let (s, z) = m.forward_trunk(&s_in, Some(0), None, None).unwrap();
        assert_eq!(s.size(), vec![b, n, token_s]);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }

    #[test]
    fn trunk_forward_with_rel_pos_runs() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let token_s = 64_i64;
        let token_z = 32_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = Boltz2Model::with_options(device, token_s, token_z, Some(1));
        let s_in = Tensor::randn(&[b, n, token_s], (tch::Kind::Float, device));
        let asym_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let residue_index = Tensor::arange(n, (tch::Kind::Int64, device))
            .view_(&[1, n])
            .expand(&[b, n], false);
        let entity_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let token_index = residue_index.shallow_clone();
        let sym_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let cyclic_period = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let rel = RelPosFeatures {
            asym_id: &asym_id,
            residue_index: &residue_index,
            entity_id: &entity_id,
            token_index: &token_index,
            sym_id: &sym_id,
            cyclic_period: &cyclic_period,
        };
        let (s, z) = m
            .forward_trunk_with_rel_pos(&s_in, &rel, Some(0), None, None)
            .unwrap();
        assert_eq!(s.size(), vec![b, n, token_s]);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }

    #[test]
    fn trunk_forward_with_token_bonds_and_types() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let token_s = 64_i64;
        let token_z = 32_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = Boltz2Model::with_options_bonds(device, token_s, token_z, Some(1), true);
        assert!(m.bond_type_feature());
        let s_in = Tensor::randn(&[b, n, token_s], (tch::Kind::Float, device));
        let asym_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let residue_index = Tensor::arange(n, (tch::Kind::Int64, device))
            .view_(&[1, n])
            .expand(&[b, n], false);
        let entity_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let token_index = residue_index.shallow_clone();
        let sym_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let cyclic_period = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let rel = RelPosFeatures {
            asym_id: &asym_id,
            residue_index: &residue_index,
            entity_id: &entity_id,
            token_index: &token_index,
            sym_id: &sym_id,
            cyclic_period: &cyclic_period,
        };
        let token_bonds = Tensor::randn(&[b, n, n, 1], (tch::Kind::Float, device));
        let type_bonds = Tensor::zeros(&[b, n, n], (tch::Kind::Int64, device));
        let (s, z) = m
            .forward_trunk_with_z_init_terms(
                &s_in,
                &rel,
                Some(&token_bonds),
                Some(&type_bonds),
                None,
                Some(0),
                None,
                None,
            )
            .unwrap();
        assert_eq!(s.size(), vec![b, n, token_s]);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }

    #[test]
    fn input_embedder_tail_then_trunk_runs() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let token_s = 64_i64;
        let token_z = 32_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = Boltz2Model::with_options(device, token_s, token_z, Some(1));
        let a = Tensor::randn(&[b, n, token_s], (tch::Kind::Float, device));
        let res = Tensor::randn(
            &[b, n, crate::boltz2::BOLTZ_NUM_TOKENS],
            (tch::Kind::Float, device),
        );
        let prof = Tensor::randn(
            &[b, n, crate::boltz2::BOLTZ_NUM_TOKENS],
            (tch::Kind::Float, device),
        );
        let del = Tensor::randn(&[b, n], (tch::Kind::Float, device));
        let s_inputs = m.forward_input_embedder(&a, &res, &prof, &del);
        let (s, z) = m.forward_trunk(&s_inputs, Some(0), None, None).unwrap();
        assert_eq!(s.size(), vec![b, n, token_s]);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }

    #[test]
    fn trunk_forward_with_contact_conditioning() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let token_s = 64_i64;
        let token_z = 32_i64;
        let b = 2_i64;
        let n = 8_i64;
        let m = Boltz2Model::with_options(device, token_s, token_z, Some(1));
        let s_in = Tensor::randn(&[b, n, token_s], (tch::Kind::Float, device));
        let asym_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let residue_index = Tensor::arange(n, (tch::Kind::Int64, device))
            .view_(&[1, n])
            .expand(&[b, n], false);
        let entity_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let token_index = residue_index.shallow_clone();
        let sym_id = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let cyclic_period = Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let rel = RelPosFeatures {
            asym_id: &asym_id,
            residue_index: &residue_index,
            entity_id: &entity_id,
            token_index: &token_index,
            sym_id: &sym_id,
            cyclic_period: &cyclic_period,
        };
        let cc = Tensor::zeros(
            &[b, n, n, CONTACT_CONDITIONING_CHANNELS],
            (tch::Kind::Float, device),
        );
        let ct = Tensor::ones(&[b, n, n], (tch::Kind::Float, device)).g_mul_scalar(12.0);
        let contact = ContactFeatures {
            contact_conditioning: &cc,
            contact_threshold: &ct,
        };
        let (s, z) = m
            .forward_trunk_with_z_init_terms(&s_in, &rel, None, None, Some(&contact), Some(0), None, None)
            .unwrap();
        assert_eq!(s.size(), vec![b, n, token_s]);
        assert_eq!(z.size(), vec![b, n, n, token_z]);
    }

    #[test]
    fn safetensors_partial_load_roundtrip_var_store() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let path =
            std::env::temp_dir().join(format!("boltr_vs_rt_{}.safetensors", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let m = Boltz2Model::with_options(device, 64, 32, Some(1));
        m.var_store()
            .save(&path)
            .expect("VarStore::save safetensors");

        let mut m2 = Boltz2Model::with_options(device, 64, 32, Some(1));
        let missing = m2
            .load_partial_from_safetensors(&path)
            .expect("load_partial_from_safetensors");
        let _ = std::fs::remove_file(&path);
        assert!(
            missing.is_empty(),
            "expected full match after self-export, missing: {:?}",
            missing
        );
    }

    /// Pinned smoke weights: `tests/fixtures/boltz2_smoke/boltz2_smoke.safetensors`
    /// (regenerate with `cargo run -p boltr-backend-tch --bin gen_boltz2_smoke_safetensors --features tch-backend`).
    #[test]
    fn pinned_smoke_safetensors_require_all_vars_loads() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/boltz2_smoke/boltz2_smoke.safetensors");
        assert!(
            path.is_file(),
            "missing {}; run: cargo run -p boltr-backend-tch --bin gen_boltz2_smoke_safetensors --features tch-backend",
            path.display()
        );
        let mut m = Boltz2Model::with_options(device, 64, 32, Some(1));
        m.load_from_safetensors_require_all_vars(&path)
            .unwrap_or_else(|e| {
                panic!(
                    "load_from_safetensors_require_all_vars {}: {e:#}",
                    path.display()
                )
            });
    }

    #[test]
    fn confidence_module_optional_on_model() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;
        let m = Boltz2Model::with_all_options(
            device,
            64,
            32,
            Some(1),
            false,
            Boltz2DiffusionArgs::default(),
            AtomDiffusionConfig::default(),
            Some(crate::boltz2::confidence::ConfidenceModuleConfig::default()),
        )
        .expect("with_all_options");
        assert!(m.confidence_module().is_some());
    }
}

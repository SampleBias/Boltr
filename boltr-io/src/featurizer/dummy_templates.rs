//! `load_dummy_templates_features` port (`featurizerv2.py`) — zero dummy template tensors for inference.
//!
//! Real template features: [`crate::featurizer::process_template_features`]. This module matches the
//! Python helper for placeholder slots and padding extra template rows.

use ndarray::{Array2, Array3, Array4};

use crate::boltz_const::NUM_TOKENS;
use crate::feature_batch::FeatureBatch;

/// Dummy template block matching Python `load_dummy_templates_features(tdim, num_tokens)` tensor shapes
/// (before batching). `template_restype` is one-hot float with last dim `NUM_TOKENS`.
#[derive(Debug, Clone, PartialEq)]
pub struct DummyTemplateTensors {
    pub template_restype: Array3<f32>,
    pub template_frame_rot: Array4<f32>,
    pub template_frame_t: Array3<f32>,
    pub template_cb: Array3<f32>,
    pub template_ca: Array3<f32>,
    pub template_mask_cb: Array2<f32>,
    pub template_mask_frame: Array2<f32>,
    pub template_mask: Array2<f32>,
    pub query_to_template: Array2<i64>,
    pub visibility_ids: Array2<f32>,
}

/// Allocate zero-filled dummy template features (`tdim` templates × `num_tokens` tokens).
#[must_use]
pub fn load_dummy_templates_features(tdim: usize, num_tokens: usize) -> DummyTemplateTensors {
    let c = NUM_TOKENS;
    DummyTemplateTensors {
        template_restype: Array3::zeros((tdim, num_tokens, c)),
        template_frame_rot: Array4::zeros((tdim, num_tokens, 3, 3)),
        template_frame_t: Array3::zeros((tdim, num_tokens, 3)),
        template_cb: Array3::zeros((tdim, num_tokens, 3)),
        template_ca: Array3::zeros((tdim, num_tokens, 3)),
        template_mask_cb: Array2::zeros((tdim, num_tokens)),
        template_mask_frame: Array2::zeros((tdim, num_tokens)),
        template_mask: Array2::zeros((tdim, num_tokens)),
        query_to_template: Array2::zeros((tdim, num_tokens)),
        visibility_ids: Array2::zeros((tdim, num_tokens)),
    }
}

impl DummyTemplateTensors {
    /// Pack into a [`FeatureBatch`] (keys match Python `load_dummy_templates_features`).
    #[must_use]
    pub fn into_feature_batch(self) -> FeatureBatch {
        let mut b = FeatureBatch::new();
        b.insert_f32("template_restype", self.template_restype.into_dyn());
        b.insert_f32("template_frame_rot", self.template_frame_rot.into_dyn());
        b.insert_f32("template_frame_t", self.template_frame_t.into_dyn());
        b.insert_f32("template_cb", self.template_cb.into_dyn());
        b.insert_f32("template_ca", self.template_ca.into_dyn());
        b.insert_f32("template_mask_cb", self.template_mask_cb.into_dyn());
        b.insert_f32("template_mask_frame", self.template_mask_frame.into_dyn());
        b.insert_f32("template_mask", self.template_mask.into_dyn());
        b.insert_i64("query_to_template", self.query_to_template.into_dyn());
        b.insert_f32("visibility_ids", self.visibility_ids.into_dyn());
        b
    }
}

/// Pack [`DummyTemplateTensors`] into a [`FeatureBatch`] (keys match Python `load_dummy_templates_features`).
#[must_use]
pub fn dummy_templates_as_feature_batch(tdim: usize, num_tokens: usize) -> FeatureBatch {
    load_dummy_templates_features(tdim, num_tokens).into_feature_batch()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_shapes_match_collate_manifest() {
        let t = load_dummy_templates_features(2, 5);
        assert_eq!(t.template_restype.shape(), &[2, 5, NUM_TOKENS]);
        assert_eq!(t.template_frame_rot.shape(), &[2, 5, 3, 3]);
        assert_eq!(t.query_to_template.shape(), &[2_usize, 5_usize]);
    }
}

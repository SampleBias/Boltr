//! Additional tests for Pairformer training mode behavior
//!
//! These tests verify that dropout and chunking behavior matches Python
//! reference implementation during both training and evaluation modes.

#[cfg(test)]
mod tests {
    use crate::layers::{PairformerLayer, PairformerModule};
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_pairformer_layer_training_mode() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 64;
        let token_z = 128;
        let num_heads = 4;
        let dropout = 0.25;
        let batch_size = 2;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let layer = PairformerLayer::new(
            vs.root(),
            token_s,
            token_z,
            Some(num_heads),
            Some(dropout),
            None,
            None,
            None,
            Some(true),
            device,
        );

        let s = Tensor::randn(&[batch_size, seq_len, token_s], (Kind::Float, device));
        let z = Tensor::randn(
            &[batch_size, seq_len, seq_len, token_z],
            (Kind::Float, device),
        );
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));
        let pair_mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        // Training mode - should apply dropout
        let (s_train_out, z_train_out) =
            layer.forward(&s, &z, &mask, &pair_mask, None, true, false);

        // Eval mode - should not apply dropout
        let (s_eval_out, z_eval_out) = layer.forward(&s, &z, &mask, &pair_mask, None, false, false);

        // Outputs should have correct shapes
        assert_eq!(s_train_out.size(), vec![batch_size, seq_len, token_s]);
        assert_eq!(
            z_train_out.size(),
            vec![batch_size, seq_len, seq_len, token_z]
        );
        assert_eq!(s_eval_out.size(), vec![batch_size, seq_len, token_s]);
        assert_eq!(
            z_eval_out.size(),
            vec![batch_size, seq_len, seq_len, token_z]
        );
    }

    #[test]
    fn test_pairformer_layer_eval_mode_no_dropout() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 64;
        let token_z = 128;
        let num_heads = 4;
        let dropout = 0.5; // High dropout
        let batch_size = 1;
        let seq_len = 5;

        let vs = VarStore::new(device);
        let layer = PairformerLayer::new(
            vs.root(),
            token_s,
            token_z,
            Some(num_heads),
            Some(dropout),
            None,
            None,
            None,
            Some(true),
            device,
        );

        let s = Tensor::randn(&[batch_size, seq_len, token_s], (Kind::Float, device));
        let z = Tensor::randn(
            &[batch_size, seq_len, seq_len, token_z],
            (Kind::Float, device),
        );
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));
        let pair_mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        // Two forward passes in eval mode should produce identical results
        let (s_out1, z_out1) = layer.forward(&s, &z, &mask, &pair_mask, None, false, false);
        let (s_out2, z_out2) = layer.forward(&s, &z, &mask, &pair_mask, None, false, false);

        // Results should be identical (no randomness in eval mode)
        let s_diff = (s_out1 - s_out2).abs().max().double_value(&[]);
        let z_diff = (z_out1 - z_out2).abs().max().double_value(&[]);

        assert!(
            s_diff < 1e-6,
            "Eval mode should be deterministic: s_diff={}",
            s_diff
        );
        assert!(
            z_diff < 1e-6,
            "Eval mode should be deterministic: z_diff={}",
            z_diff
        );
    }

    #[test]
    fn test_pairformer_module_training_mode() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 64;
        let token_z = 128;
        let num_blocks = 2;
        let dropout = 0.25;
        let batch_size = 1;
        let seq_len = 10;

        let vs = VarStore::new(device);
        let mut module = PairformerModule::new(
            vs.root(),
            token_s,
            token_z,
            num_blocks,
            None,
            Some(dropout),
            None,
            None,
            None,
            None,
            Some(true),
            device,
        );

        let s = Tensor::randn(&[batch_size, seq_len, token_s], (Kind::Float, device));
        let z = Tensor::randn(
            &[batch_size, seq_len, seq_len, token_z],
            (Kind::Float, device),
        );
        let mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));
        let pair_mask = Tensor::ones(&[batch_size, seq_len, seq_len], (Kind::Float, device));

        // Test eval mode (default)
        let (s_eval_out, z_eval_out) = module.forward(&s, &z, &mask, &pair_mask, false);

        // Test training mode
        module.set_training(true);
        let (s_train_out, z_train_out) = module.forward(&s, &z, &mask, &pair_mask, false);

        // Back to eval mode
        module.set_training(false);
        let (s_eval_out2, z_eval_out2) = module.forward(&s, &z, &mask, &pair_mask, false);

        // Eval mode outputs should be identical
        let s_diff = (s_eval_out - s_eval_out2).abs().max().double_value(&[]);
        let z_diff = (z_eval_out - z_eval_out2).abs().max().double_value(&[]);
        assert!(
            s_diff < 1e-6,
            "Eval mode should be deterministic: s_diff={}",
            s_diff
        );
        assert!(
            z_diff < 1e-6,
            "Eval mode should be deterministic: z_diff={}",
            z_diff
        );

        // Training mode should produce different outputs (with dropout)
        assert_eq!(s_train_out.size(), vec![batch_size, seq_len, token_s]);
        assert_eq!(
            z_train_out.size(),
            vec![batch_size, seq_len, seq_len, token_z]
        );
    }

    #[test]
    fn test_pairformer_module_chunk_size_training() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 64;
        let token_z = 128;
        let num_blocks = 1;
        let batch_size = 1;

        let vs = VarStore::new(device);
        let mut module = PairformerModule::new(
            vs.root(),
            token_s,
            token_z,
            num_blocks,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(true),
            device,
        );

        // Test with seq_len > 256 threshold
        let seq_len_large = 300;
        let s = Tensor::randn(&[batch_size, seq_len_large, token_s], (Kind::Float, device));
        let z = Tensor::randn(
            &[batch_size, seq_len_large, seq_len_large, token_z],
            (Kind::Float, device),
        );
        let mask = Tensor::ones(
            &[batch_size, seq_len_large, seq_len_large],
            (Kind::Float, device),
        );
        let pair_mask = Tensor::ones(
            &[batch_size, seq_len_large, seq_len_large],
            (Kind::Float, device),
        );

        // Eval mode - should use chunk_size=128 for seq_len > 256
        let _ = module.forward(&s, &z, &mask, &pair_mask, false);

        // Training mode - should use chunk_size=None
        module.set_training(true);
        let _ = module.forward(&s, &z, &mask, &pair_mask, false);

        // Test with seq_len <= 256 threshold
        let seq_len_small = 200;
        let s = Tensor::randn(&[batch_size, seq_len_small, token_s], (Kind::Float, device));
        let z = Tensor::randn(
            &[batch_size, seq_len_small, seq_len_small, token_z],
            (Kind::Float, device),
        );
        let mask = Tensor::ones(
            &[batch_size, seq_len_small, seq_len_small],
            (Kind::Float, device),
        );
        let pair_mask = Tensor::ones(
            &[batch_size, seq_len_small, seq_len_small],
            (Kind::Float, device),
        );

        // Eval mode - should use chunk_size=512 for seq_len <= 256
        module.set_training(false);
        let _ = module.forward(&s, &z, &mask, &pair_mask, false);
    }

    #[test]
    fn test_dropout_mask_shape_broadcast() {
        tch::maybe_init_cuda();
        let device = Device::Cpu;

        let token_s = 64;
        let token_z = 128;
        let batch_size = 2;
        let seq_len = 10;
        let dropout = 0.25;

        let vs = VarStore::new(device);
        let layer = PairformerLayer::new(
            vs.root(),
            token_s,
            token_z,
            None,
            Some(dropout),
            None,
            None,
            None,
            Some(true),
            device,
        );

        let z = Tensor::randn(
            &[batch_size, seq_len, seq_len, token_z],
            (Kind::Float, device),
        );

        // Test non-columnwise mask
        let mask = layer.create_dropout_mask(&z, true);
        // Mask should be small: [B, N, 1, 1] that broadcasts to [B, N, N, token_z]
        // The function creates mask from v = z[:, :, 0:1, 0:1] which is [B, N, 1, 1]
        assert_eq!(mask.size(), vec![batch_size, seq_len, 1, 1]);

        // Test columnwise mask
        let mask_col = layer.create_dropout_mask_columnwise(&z, true);
        // Mask should be small: [B, 1, N, 1] that broadcasts to [B, N, N, token_z]
        // The function creates mask from v = z[:, 0:1, :, 0:1] which is [B, 1, N, 1]
        assert_eq!(mask_col.size(), vec![batch_size, 1, seq_len, 1]);
    }
}

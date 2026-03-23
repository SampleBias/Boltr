//! Numerical golden vs Python `PairformerLayer` (§5.5 / §7).
//!
//! Run with LibTorch: `cargo test -p boltr-backend-tch --features tch-backend pairformer_layer_allclose -- --ignored`
//!
//! Requires a committed `.safetensors` snapshot (see `tests/fixtures/pairformer_golden/README.md`).

#[test]
#[ignore = "add committed pairformer_layer0.safetensors + Python reference dump"]
fn pairformer_layer_allclose_python_golden() {
    // Placeholder: load s/z/mask from fixtures, run `PairformerLayer::forward`, allclose outputs.
}

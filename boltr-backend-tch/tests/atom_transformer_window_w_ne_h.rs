//! Regression: Boltz default `atoms_per_window_queries=32` and `atoms_per_window_keys=128` require
//! `to_keys` in `DiffusionTransformerLayer` (W×H attention + W×H bias). W==H hides the bug.

use boltr_backend_tch::boltz2::atom_window_keys::get_indexing_matrix;
use boltr_backend_tch::boltz2::transformers::AtomTransformer;
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

#[test]
fn atom_transformer_forward_w32_h128() {
    tch::maybe_init_cuda();
    let device = Device::Cpu;
    let w = 32_i64;
    let h = 128_i64;
    let depth = 2_i64;
    let heads = 4_i64;
    let dim = 64_i64;
    let b = 1_i64;
    let n_atoms = 32_i64;
    let nw = n_atoms / w;
    assert_eq!(nw, 1);
    let k = nw;
    let im = get_indexing_matrix(k, w, h, device);

    let vs = VarStore::new(device);
    let at = AtomTransformer::new(vs.root(), w, h, depth, heads, dim, Some(dim), device);

    let q = Tensor::randn(&[b, n_atoms, dim], (Kind::Float, device));
    let c = Tensor::randn(&[b, n_atoms, dim], (Kind::Float, device));
    let d_total = heads * depth;
    let bias = Tensor::randn(&[b, k, w, h, d_total], (Kind::Float, device));
    let mask = Tensor::ones(&[b, n_atoms], (Kind::Float, device));

    let out = at.forward(&q, &c, &bias, &mask, 1, &im);
    assert_eq!(out.size(), vec![b, n_atoms, dim]);
}

# Pairformer numerical golden

Goal: `PairformerLayer` / block output **allclose** vs Python with `use_kernels=False`.

- Stub exporter (docs): [`scripts/export_pairformer_golden.py`](../../../scripts/export_pairformer_golden.py)
- Planned Rust generator (requires LibTorch): `cargo run -p boltr-backend-tch --bin gen_pairformer_golden --features tch-backend`

Until a `.safetensors` snapshot is committed, backend tests may use `#[ignore]` or shape-only checks.

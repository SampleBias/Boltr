# input_embedder_golden.safetensors

Opt-in parity for the full trunk [`InputEmbedder`](../../src/boltz2/input_embedder.rs) vs PyTorch (`scripts/export_input_embedder_golden.py`).

Generate (requires `torch` + Boltz on `PYTHONPATH`):

```bash
PYTHONPATH=boltz-reference/src python3 scripts/export_input_embedder_golden.py
```

Rust: `BOLTR_RUN_INPUT_EMBEDDER_GOLDEN=1 scripts/cargo-tch test -p boltr-backend-tch --features tch-backend input_embedder_allclose_python_golden`

# MSAModule golden (`§5.4`)

Regenerate `msa_module_golden.safetensors` after changing `MSAModule` / layer code:

```bash
# From repo root; needs a Python with torch + safetensors (see DEVELOPMENT.md if pip/torch missing).
PYTHONPATH=boltz-reference/src python3 scripts/export_msa_module_golden.py
```

Rust tests only need LibTorch (CPU zip + `LIBTORCH` is enough — PyTorch in Python is **not** required unless you use `LIBTORCH_USE_PYTORCH=1`). See [DEVELOPMENT.md](../../../../DEVELOPMENT.md) Path A vs B.

```bash
cargo test -p boltr-backend-tch --features tch-backend msa_module_allclose_python_golden -- --ignored
```

The committed `.safetensors` file may be absent until you run the export script; the integration test is `#[ignore]` until the fixture is present in your tree.

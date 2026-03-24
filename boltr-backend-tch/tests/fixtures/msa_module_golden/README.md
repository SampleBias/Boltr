# MSAModule golden (`§5.4`)

Regenerate `msa_module_golden.safetensors` after changing `MSAModule` / layer code:

```bash
# From repo root; needs a Python with torch + safetensors (see DEVELOPMENT.md if pip/torch missing).
PYTHONPATH=boltz-reference/src python3 scripts/export_msa_module_golden.py
```

Rust tests only need LibTorch (CPU zip + `LIBTORCH` is enough — PyTorch in Python is **not** required unless you use `LIBTORCH_USE_PYTORCH=1`). See [DEVELOPMENT.md](../../../../DEVELOPMENT.md) Path A vs B.

```bash
scripts/cargo-tch test -p boltr-backend-tch --features tch-backend msa_module_allclose_python_golden
```

The committed `msa_module_golden.safetensors` is checked in; regenerate it with the export command above after changing Python `MSAModule` or Rust `MsaModule` so numerics stay aligned.

**Python export deps:** Boltz pulls `numpy`, `scipy`, `einops`, etc. A minimal venv can use:

```bash
.venv/bin/pip install numpy scipy einops
```

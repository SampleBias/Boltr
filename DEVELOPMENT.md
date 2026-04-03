# Development Guide

See also the repository **[`README.md`](README.md)** (overview, checklist, doc index).

## Prerequisites

- **Rust** stable toolchain (workspace uses **edition 2021**; see root `Cargo.toml`)
- Git

## Initial Setup

### Build (I/O and CLI only, no LibTorch)

Default workspace crates build **without** linking LibTorch so CI and minimal clones work:

```bash
cargo build --release
cargo test
```

### Build with `tch-rs` (LibTorch)

All tensor work uses the `tch` feature on `boltr-cli` (pulls in `boltr-backend-tch/tch-backend`).

Keep **`LIBTORCH` or `LIBTORCH_USE_PYTORCH`** exported in the **same shell** where you run `cargo` (`torch-sys` reads them at compile time).

**Sanity check (optional):**

```bash
bash scripts/check_tch_prereqs.sh
```

**Arch Linux + PEP 668:** system Python is “externally managed”; `pip install torch` to `/usr` fails with `externally-managed-environment`. Use **Path A** (no Python torch), or a **venv** (Path B below + [`scripts/bootstrap_dev_venv.sh`](scripts/bootstrap_dev_venv.sh)).

#### Path A — Standalone LibTorch (works without PyTorch in Python)

Use this when system `python3` has **no `pip`** or **no `torch`** (common on minimal Arch installs). Rust links against the C++ library only; you do **not** need `import torch` for `cargo build`.

[`tch` 0.16](https://crates.io/crates/tch) matches **LibTorch ~2.3.0** C++ APIs. Avoid **`libtorch-*-latest.zip`** (too new vs bundled `libtch`).

```bash
wget 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.3.0%2Bcpu.zip' -O libtorch-2.3.0-cpu.zip
unzip libtorch-2.3.0-cpu.zip
export LIBTORCH="$(pwd)/libtorch"
unset LIBTORCH_USE_PYTORCH
cargo test -p boltr-backend-tch --features tch-backend
```

For GPU, use a **2.3.x** CUDA LibTorch from [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) (same minor as above), not an unrelated release. After unpacking to `third_party/libtorch`, you can `source scripts/env_libtorch_cuda.sh` (sets `LIBTORCH`, `LD_LIBRARY_PATH`, `LIBTORCH_CXX11_ABI=0`, unsets `LIBTORCH_USE_PYTORCH`).

#### Path B — `LIBTORCH_USE_PYTORCH=1` (reuse PyTorch’s LibTorch)

`torch-sys` runs **`python3` from your `PATH`** and requires `import torch`. That must be an interpreter that actually has PyTorch (venv, conda, etc.) — **not** bare `/usr/bin/python3` on Arch after only `pacman -S python-pip`.

**Recommended on Arch (PEP 668):** repo venv (one-time setup). The script uses **Python 3.12 / 3.11 / 3.10** and **`torch==2.3.0`** so `libtch` compiles (see troubleshooting below if you used Python 3.14 + latest torch).

```bash
bash scripts/bootstrap_dev_venv.sh --force   # --force if an old .venv had wrong Python/torch
scripts/cargo-tch test -p boltr-backend-tch --features tch-backend
```

**Plain `cargo` without activating `.venv`** still uses system `python3` → `ModuleNotFoundError: torch` / `no cxx11 abi returned by python`. From the repo root use the wrapper (same env as `with_dev_venv.sh`):

```bash
scripts/cargo-tch test -p boltr-backend-tch --features tch-backend
```

**Cursor / VS Code (rust-analyzer):** it runs `cargo check` without your shell’s `activate`. Either run **`source .venv/bin/activate`** in the integrated terminal before editing, or add **user** settings (not committed; Path A devs omit `LIBTORCH_*`):

```json
"rust-analyzer.cargo.extraEnv": {
  "PATH": "${workspaceFolder}/.venv/bin:${env:PATH}",
  "LIBTORCH_USE_PYTORCH": "1",
  "LIBTORCH_BYPASS_VERSION_CHECK": "1"
}
```

Or manually (use **python3.12** etc., not 3.13+, and pin torch):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install setuptools wheel 'torch==2.3.0' safetensors
export LIBTORCH_USE_PYTORCH=1
python3 -c "import torch; print('PyTorch', torch.__version__)"
cargo test -p boltr-backend-tch --features tch-backend
```

Other situations:

| Situation | Command |
|-----------|---------|
| No `pip` module | Arch: `sudo pacman -S python-pip`, then use a **venv** row above (not `pip install --user` on Arch). |
| [`uv`](https://github.com/astral-sh/uv) | `uv venv && source .venv/bin/activate && uv pip install torch safetensors` then `export LIBTORCH_USE_PYTORCH=1`. |

Confirm the interpreter `torch-sys` will see (first `python3` on `PATH`):

```bash
which python3
python3 -c "import torch"
```

#### CLI release build (after Path A or B)

```bash
cargo build --release -p boltr-cli --features tch
```

#### Troubleshooting: `Cannot find a libtorch install` (`torch-sys`)

```text
Error: Cannot find a libtorch install ...
```

You did not set `LIBTORCH`, or `LIBTORCH_USE_PYTORCH=1` is set but Python cannot supply LibTorch. **Fix:** use **Path A** (set `LIBTORCH` to an unpacked zip) or fix Python per **Path B**.

#### Troubleshooting: `no cxx11 abi returned by python`

`torch-sys` runs the **first** `python3` on your `PATH` and asks PyTorch for the C++11 ABI flag. That code path imports **`torch.utils.cpp_extension`**, which in turn imports **`setuptools`**. A bare `python -m venv` often has **no setuptools**, so you can see this Rust error even when `torch` is installed.

| Python stderr (inside the Rust error) | Fix |
|---------------------------------------|-----|
| `No module named 'torch'` | Put `.venv` first on `PATH` — `scripts/cargo-tch …` / `scripts/with_dev_venv.sh cargo …`, or `source .venv/bin/activate` before `cargo`. Or **Path A** + `unset LIBTORCH_USE_PYTORCH`. |
| `No module named 'setuptools'` | `.venv/bin/pip install setuptools` or re-run [`scripts/bootstrap_dev_venv.sh`](scripts/bootstrap_dev_venv.sh) (installs setuptools). |

#### Troubleshooting: `externally-managed-environment` (pip on Arch)

System Python refuses `pip install torch` (PEP 668). Use [`scripts/bootstrap_dev_venv.sh`](scripts/bootstrap_dev_venv.sh) or Path A — not `pip install --user` into `/usr`.

#### Golden export scripts (`ModuleNotFoundError: No module named 'torch'`)

Scripts such as [`scripts/export_msa_module_golden.py`](scripts/export_msa_module_golden.py), [`scripts/export_pairformer_golden.py`](scripts/export_pairformer_golden.py), [`scripts/export_trunk_init_golden.py`](scripts/export_trunk_init_golden.py), and [`scripts/export_input_embedder_golden.py`](scripts/export_input_embedder_golden.py) need **`torch`** and **`safetensors`**. With the repo venv:

```bash
source .venv/bin/activate   # after bootstrap_dev_venv.sh
PYTHONPATH=boltz-reference/src python3 scripts/export_msa_module_golden.py
PYTHONPATH=boltz-reference/src python3 scripts/export_pairformer_golden.py
PYTHONPATH=boltz-reference/src python3 scripts/export_trunk_init_golden.py
PYTHONPATH=boltz-reference/src python3 scripts/export_input_embedder_golden.py
```

Opt-in numeric tests (LibTorch): `BOLTR_RUN_MSA_GOLDEN=1` / `BOLTR_RUN_PAIRFORMER_GOLDEN=1` / `BOLTR_RUN_TRUNK_INIT_GOLDEN=1` / `BOLTR_RUN_INPUT_EMBEDDER_GOLDEN=1` with `scripts/cargo-tch test -p boltr-backend-tch --features tch-backend` (see fixture READMEs under `tests/fixtures/`).

#### Troubleshooting: `this tch version expects PyTorch 2.3.0, got …`

[`tch` 0.16](https://crates.io/crates/tch) / `torch-sys` compares the **reported** wheel version. Newer pip `torch` trips this unless you bypass:

```bash
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

[`scripts/with_dev_venv.sh`](scripts/with_dev_venv.sh) sets this by default. Prefer **`torch==2.3.0`** in `.venv` so you can omit bypass (see [`bootstrap_dev_venv.sh`](scripts/bootstrap_dev_venv.sh)).

#### Troubleshooting: `libtch/torch_api.cpp` / `hasORT` / `_scaled_mm` / C++ errors while building `torch-sys`

**Dissection:** `LIBTORCH_BYPASS_VERSION_CHECK` only skips the **Python version string** check. The crate still compiles bundled **`libtch` C++** against **your PyTorch install’s headers**. **Python 3.14 + latest torch** exposes **new ATen C++ APIs** that do not match `tch` 0.16’s generated code → compile errors.

**Fix:** recreate `.venv` with **Python ≤ 3.12** and **`torch==2.3.0`** (matches `tch` 0.16):

```bash
rm -rf .venv
bash scripts/bootstrap_dev_venv.sh
scripts/cargo-tch test -p boltr-backend-tch --features tch-backend
```

If you have no `python3.12` (or 3.11 / 3.10) binary: on **Arch Linux** there is **no** `python312` in the main repos (`pacman -S python312` → *target not found*). Install **3.12** via **AUR** (`yay -S python312` / `paru -S python312`), or **`pyenv`** from `extra` (`sudo pacman -S pyenv`, `pyenv install 3.12`, then `export BOLTR_VENV_PYTHON="$HOME/.pyenv/versions/3.12.x/bin/python"` and run `bootstrap_dev_venv.sh`), or use **Path A** with a **2.3.0** LibTorch zip (not `latest`).

### CUDA vs Python `cuequivariance` wheels

- **GPU in Boltr** comes from a **CUDA build of LibTorch** plus `--device cuda` (or `cuda:N`) on the CLI. Override with env `BOLTR_DEVICE` if needed.
- Upstream Boltz’s optional `pip install boltz[cuda]` adds **cuequivariance** fused kernels. Those are **not** available through `tch-rs`; Boltr targets the same numerics as PyTorch with `use_kernels=False` (the pure PyTorch op path).

### Boltz preprocess vs LibTorch GPU (load balancing)

`boltr predict --preprocess boltz|auto` runs upstream **`boltz predict`** in a subprocess, then LibTorch **`predict_step`** in-process. Both are full inferences; peak VRAM on **one** GPU can OOM the second stage.

| Situation | What to use |
|-----------|-------------|
| **Two GPUs** | `--preprocess-cuda-visible-devices 1` (or env `BOLTR_BOLTZ_CUDA_VISIBLE_DEVICES=1`) so Boltz only sees the second card; LibTorch keeps `--device cuda` / `cuda:0` on the first. |
| **One GPU (default)** | Boltz subprocess gets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (override with `BOLTR_BOLTZ_PYTORCH_CUDA_ALLOC_CONF`; empty string disables). Optional `--preprocess-post-boltz-empty-cache` runs `torch.cuda.empty_cache()` between stages. |
| **OOM after Boltz** | `--preprocess-boltz-cpu` or `BOLTR_PREPROCESS_BOLTZ_CPU=1` forces Boltz to CPU (slow). |
| **Python for empty_cache** | `BOLTR_PYTHON` selects the interpreter (default `python3`). |

**Memory / sampling knobs (upstream Boltz):** repeat `--preprocess-bolt-arg` for flags such as `--max_parallel_samples 1` to cap parallel diffusion work during the Boltz stage. See upstream [boltz-reference/docs/prediction.md](boltz-reference/docs/prediction.md).

**LibTorch / Boltr:** `boltr predict` forwards diffusion and trunk overrides (`--diffusion-samples`, `--sampling-steps`, `--max-parallel-samples`, …) into `boltr_predict_args.json` / checkpoint resolution; large complexes may need lower `diffusion_samples` or steps. See [docs/TENSOR_CONTRACT.md](docs/TENSOR_CONTRACT.md) for `predict_args` keys.

### Checkpoint export for Rust

Lightning `.ckpt` files are not loaded directly in Rust. After `boltr download`, the CLI **attempts** a best-effort export when it can find this repo and a Python with `torch` + `safetensors` (warnings only on failure). Otherwise use:

```bash
python scripts/export_checkpoint_to_safetensors.py ~/.cache/boltr/boltz2_conf.ckpt ~/.cache/boltr/boltz2_conf.safetensors
```

**LibTorch smoke:** `boltr doctor` / `boltr doctor --json` (needs `--features tch` for a real CPU tensor probe; without `tch`, JSON reports `tch_feature: false`).

(Optional: `--strip-prefix model.` if keys are nested.) See [docs/TENSOR_CONTRACT.md](docs/TENSOR_CONTRACT.md).

**Makefile shortcuts** (repo root [Makefile](Makefile)):

```bash
make export-safetensors CKPT=path/to.ckpt OUT=out.safetensors
make verify-safetensors SAFETENSORS=out.safetensors
make export-hparams CKPT=path/to.ckpt HPARAMS_JSON=hparams.json
make compare-ckpt-safetensors-counts CKPT=path/to.ckpt SAFETENSORS=out.safetensors
```

Hyperparameters JSON for [`Boltz2Hparams`](boltr-backend-tch/src/boltz_hparams.rs) / [`Boltz2Model::from_hparams_json`](boltr-backend-tch/src/boltz2/model.rs) can include `bond_type_feature` (bool) when the checkpoint was trained with bond-type embeddings.

### VarStore key alignment vs a real export

[`Boltz2Model::load_from_safetensors_require_all_vars`](boltr-backend-tch/src/boltz2/model.rs) fails when any Rust parameter name is missing from the file (extra diffusion / confidence keys in the file are fine).

1. Export with [`scripts/export_checkpoint_to_safetensors.py`](scripts/export_checkpoint_to_safetensors.py) and the same `--strip-prefix` you use in Lightning (often `model.`).
2. Diff keys (missing / unused):

   ```bash
   scripts/cargo-tch run -p boltr-backend-tch --bin verify_boltz2_safetensors --features tch-backend -- \
     ~/.cache/boltr/boltz2_conf.safetensors
   ```

   Match checkpoint hyperparameters when needed:

   ```bash
   scripts/cargo-tch run -p boltr-backend-tch --bin verify_boltz2_safetensors --features tch-backend -- \
     --token-s 384 --token-z 128 --blocks 4 --bond-type-feature \
     boltz2_export.safetensors
   ```

   Exit code **0** means every VarStore key is present; **1** lists missing names (fix Rust `Path` segments under `boltr-backend-tch/src/boltz2/` / `layers/` or adjust export prefix).

   A **full** Lightning export contains many tensors the Rust trunk model does not load yet (diffusion, confidence, affinity heads, optimiser buffers, etc.). The tool prints **unused file keys** for visibility only; a successful run requires **no missing** VarStore keys, not an empty unused list.

   **Full checkpoint audit checklist (Phase 4):** (1) Export with `--strip-prefix` matching Lightning. (2) Run `verify_boltz2_safetensors` with `--token-s` / `--token-z` / `--blocks` / `--bond-type-feature` as needed. (3) Fix **missing** Rust keys (rename `Path` segments in `boltr-backend-tch`). (4) Record **unused** prefixes for heads not ported yet. (5) Re-run after adding modules.

3. **Collate smoke → trunk:** `boltr-io` fixture [`trunk_smoke_collate.safetensors`](boltr-io/tests/fixtures/collate_golden/trunk_smoke_collate.safetensors) is loaded in [`boltr-backend-tch/tests/collate_predict_trunk.rs`](boltr-backend-tch/tests/collate_predict_trunk.rs), which runs `Boltz2Model::predict_step_trunk` with `MsaFeatures` (no full checkpoint required).

4. **Pinned smoke fixture** (architecture 64 / 32 / 1 pairformer block, no bond-type embedding) used in CI to prove strict load on a committed file:

   ```bash
   scripts/cargo-tch run -p boltr-backend-tch --bin gen_boltz2_smoke_safetensors --features tch-backend
   ```

   Output: [`boltr-backend-tch/tests/fixtures/boltz2_smoke/boltz2_smoke.safetensors`](boltr-backend-tch/tests/fixtures/boltz2_smoke/boltz2_smoke.safetensors). Re-run the generator after adding parameters to [`Boltz2Model`](boltr-backend-tch/src/boltz2/model.rs).

## Project Structure

```
Boltr/
├── boltr-cli/           # Command-line interface
│   ├── src/
│   │   └── main.rs     # CLI entry point
│   └── Cargo.toml
├── boltr-io/            # Input/output handling
│   ├── src/
│   │   ├── config.rs   # YAML types (Boltz2-oriented)
│   │   ├── parser.rs   # Input file parsing
│   │   ├── download.rs # Checkpoints + ccd/mols URLs (aligned with boltz main.py)
│   │   ├── msa.rs      # ColabFold-style MSA server client
│   │   └── format.rs   # Run summary JSON
│   └── Cargo.toml
├── boltr-backend-tch/   # LibTorch backend (`--features tch`)
│   ├── src/
│   │   ├── boltz2/     # Boltz2 module layout (trunk, diffusion, …)
│   │   ├── checkpoint.rs # Safetensors → tch
│   │   ├── device.rs   # cpu / cuda:N
│   │   ├── model.rs    # Re-exports
│   │   ├── layers.rs   # (stubs / future)
│   │   ├── attention.rs
│   │   └── equivariance.rs
│   └── Cargo.toml
├── docs/
│   ├── TENSOR_CONTRACT.md
│   └── PYTHON_REMOVAL.md
├── scripts/
│   └── export_checkpoint_to_safetensors.py
├── boltz-reference/     # Original PyTorch implementation
├── Cargo.toml           # Workspace configuration
├── README.md
└── LICENSE
```

## Development Workflow

### Running the CLI

```bash
# Build debug version
cargo build

# Run from debug build (MSA server optional; default MSA host matches Boltz)
cargo run -p boltr-cli -- predict input.yaml --output ./out

# LibTorch + optional GPU spike (requires LIBTORCH / LIBTORCH_USE_PYTORCH)
cargo run -p boltr-cli --features tch -- predict input.yaml --device cuda --output ./out

cargo run -p boltr-cli -- download --version boltz2
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Skip rustdoc doctests (library + integration tests only) — useful when /tmp is full
cargo test --lib --tests

# Example: run boltr-io tests without doctests
cargo test -p boltr-io --lib --tests
```

### Code Quality

```bash
# Format code
cargo fmt

# Check formatting without applying
cargo fmt -- --check

# Run linter (requires clippy)
cargo clippy
```

## Implementation Roadmap

### Phase 1: Core Infrastructure
- [x] Project structure setup
- [x] Workspace configuration
- [x] Basic CLI framework
- [x] Module stubs
- [ ] Configuration file parsing (YAML)
- [ ] Input file validation
- [ ] Error handling patterns

### Phase 2: I/O Layer (boltr-io)
- [ ] YAML config parser (serde_yaml)
- [ ] Sequence file parsers (FASTA, MSA formats)
- [ ] Structure file parsers (PDB)
- [ ] Output formatters
- [ ] MSA server client
- [ ] Model weight downloader

### Phase 3: Backend Layer (boltr-backend-tch)
- [ ] Model weight loading
- [ ] Layer implementations (Linear, Attention, etc.)
- [ ] Equivariant attention
- [ ] Model architecture (Boltz-1)
- [ ] Model architecture (Boltz-2)
- [ ] Inference pipeline
- [ ] GPU acceleration

### Phase 4: CLI Integration
- [ ] Full prediction workflow
- [ ] Batch processing
- [ ] Model download command
- [ ] Evaluation commands
- [ ] Logging and debugging

### Phase 5: Advanced Features
- [ ] Binding affinity prediction
- [ ] Confidence metrics
- [ ] Multi-sample inference
- [ ] Caching and optimization
- [ ] Alternative backends (e.g., Candle)

## Reference Implementation

The original Boltz implementation is available in `boltz-reference/`. Use this to understand:
- Model architecture details
- Data preprocessing steps
- Inference pipeline
- MSA generation
- Output formats

Key reference files:
- `boltz-reference/src/boltz/` - Core model code
- `boltz-reference/docs/prediction.md` - Input/output formats
- `boltz-reference/examples/` - Example configurations

## Adding New Features

1. Add crate dependency to relevant `Cargo.toml`
2. Run `cargo check` to verify
3. Write tests in `tests/` or `src/`
4. Run `cargo test`
5. Update documentation
6. Commit changes

## Memory and Performance

- Use `--release` for production builds
- Profile with `cargo flamegraph` or `perf`
- Monitor memory usage with `valgrind` or `heaptrack`
- Use `cargo test --release` for performance benchmarks

## Troubleshooting

### LibTorch not found
```bash
# Check if LIBTORCH is set
echo $LIBTORCH

# Use Python PyTorch
export LIBTORCH_USE_PYTORCH=1

# Or download LibTorch manually
```

### CUDA issues
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Compilation errors
```bash
# Clean build artifacts
cargo clean

# Update dependencies
cargo update

# Check Rust toolchain
rustup update
```

### Disk quota / `QuotaExceeded` during `cargo test` (often doctests)

`rustdoc` writes argument files to **`$TMPDIR`** (often `/tmp`). If that volume hits a user or disk quota (Linux error **122**), you may see:

`failed to write arguments to temporary file: QuotaExceeded`

**Fixes:**

1. **Free space** on the volume holding `TMPDIR` and the project `target/` directory (`cargo clean`, remove old `target` trees, clear package caches).
2. **Point temp at a directory with headroom** (same filesystem as your checkout if that is where space exists):

   ```bash
   mkdir -p /path/with/space/tmp
   TMPDIR=/path/with/space/tmp cargo test -p boltr-io
   ```

3. **Skip doctests** while iterating (unit + integration tests still run):

   ```bash
   cargo test -p boltr-io --lib --tests
   ```

4. **`boltr-io` disables library doctests** (`[lib] doctest = false` in its `Cargo.toml`) so `cargo test -p boltr-io` does not invoke `rustdoc` on that crate—avoiding temp-file quota failures. The former `BoltzInput` doc example is covered by `tests/yaml_parse.rs`.

5. **CLI integration tests** (`boltr-cli/tests/*_cli.rs`) use `target/boltr-cli-test-tmp/` (see `tests/common.rs`) instead of `/tmp` so file writes stay on the project volume when `/tmp` is quota-limited.

## Contact & Support

- GitHub Issues: https://github.com/SampleBias/Boltr/issues
- Original Boltz: https://github.com/jwohlwend/boltz

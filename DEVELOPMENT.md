# Development Guide

## Prerequisites

- Rust 1.93 or later
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

First, install LibTorch:

```bash
# Option 1: Use Python environment with PyTorch
pip install torch
export LIBTORCH_USE_PYTORCH=1

# Option 2: Manual LibTorch installation (CPU or CUDA build — use CUDA zip for GPU)
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
export LIBTORCH=$(pwd)/libtorch
```

Then:

```bash
cargo build --release -p boltr-cli --features tch
```

### CUDA vs Python `cuequivariance` wheels

- **GPU in Boltr** comes from a **CUDA build of LibTorch** plus `--device cuda` (or `cuda:N`) on the CLI. Override with env `BOLTR_DEVICE` if needed.
- Upstream Boltz’s optional `pip install boltz[cuda]` adds **cuequivariance** fused kernels. Those are **not** available through `tch-rs`; Boltr targets the same numerics as PyTorch with `use_kernels=False` (the pure PyTorch op path).

### Checkpoint export for Rust

Lightning `.ckpt` files are not loaded directly in Rust. Use:

```bash
python scripts/export_checkpoint_to_safetensors.py ~/.cache/boltr/boltz2_conf.ckpt ~/.cache/boltr/boltz2_conf.safetensors
```

(Optional: `--strip-prefix model.` if keys are nested.) See [docs/TENSOR_CONTRACT.md](docs/TENSOR_CONTRACT.md).

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

## Contact & Support

- GitHub Issues: https://github.com/SampleBias/Boltr/issues
- Original Boltz: https://github.com/jwohlwend/boltz

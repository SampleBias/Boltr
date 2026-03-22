# Development Guide

## Prerequisites

- Rust 1.93 or later
- Git

## Initial Setup

### Build (without PyTorch backend)
```bash
cargo build --release
```

### Build (with PyTorch backend)

First, install LibTorch:

```bash
# Option 1: Use Python environment with PyTorch
pip install torch
export LIBTORCH_USE_PYTORCH=1

# Option 2: Manual LibTorch installation
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
export LIBTORCH=$(pwd)/libtorch
```

Then build with the feature:
```bash
cargo build --release --features tch
```

### Build with GPU support
Requires CUDA-capable GPU and CUDA toolkit:

```bash
# Install CUDA version compatible with your PyTorch installation
# Then build with the tch feature (GPU support is automatic)
cargo build --release --features tch
```

## Project Structure

```
Boltr/
├── boltr-cli/           # Command-line interface
│   ├── src/
│   │   └── main.rs     # CLI entry point
│   └── Cargo.toml
├── boltr-io/            # Input/output handling
│   ├── src/
│   │   ├── config.rs   # Configuration parsing
│   │   ├── parser.rs   # Input file parsing
│   │   ├── msa.rs      # MSA processing
│   │   └── format.rs   # Output formatting
│   └── Cargo.toml
├── boltr-backend-tch/   # PyTorch backend (optional)
│   ├── src/
│   │   ├── model.rs    # Core model
│   │   ├── layers.rs   # Neural network layers
│   │   ├── attention.rs # Attention mechanisms
│   │   └── equivariance.rs # Equivariant operations
│   └── Cargo.toml
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

# Run from debug build
cargo run -- predict input.yaml

# Run with specific features
cargo run --features tch -- predict input.yaml --use_msa_server
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

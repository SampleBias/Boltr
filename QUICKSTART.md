# Quick Start Guide

## Project Status

✅ **Repository Setup Complete**
- Git initialized and connected to `https://github.com/SampleBias/Boltr.git`
- Rust workspace with 3 crates configured
- Original Boltz code available as reference in `boltz-reference/`
- Project builds successfully

## Project Structure

```
Boltr/
├── boltr-cli/              # Command-line interface
│   └── src/main.rs        # Entry point (boltr binary)
├── boltr-io/               # Input/output operations
│   └── src/
│       ├── config.rs      # YAML config parsing
│       ├── parser.rs      # Input file parsing
│       ├── msa.rs         # MSA server communication
│       └── format.rs      # Output formatting
├── boltr-backend-tch/      # PyTorch backend (optional)
│   └── src/
│       ├── model.rs       # Core Boltz model
│       ├── layers.rs      # Neural network layers
│       ├── attention.rs   # Attention mechanisms
│       └── equivariance.rs # Equivariant operations
├── boltz-reference/        # Original PyTorch implementation (reference)
├── Cargo.toml              # Workspace config
├── DEVELOPMENT.md          # Detailed development guide
└── README.md               # Project overview
```

## Getting Started

### 1. Build the Project

```bash
# Build without PyTorch (fastest)
cargo build --release

# Build with PyTorch backend (requires LibTorch)
cargo build --release --features tch
```

The `boltr` binary will be at `target/release/boltr`

### 2. Test the CLI

```bash
# Show help
./target/release/boltr --help

# Test predict command (placeholder)
./target/release/boltr predict test.yaml
```

### 3. Set Up LibTorch (Optional, for full inference)

```bash
# Option 1: Use Python PyTorch
pip install torch
export LIBTORCH_USE_PYTORCH=1

# Option 2: Download LibTorch manually
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
export LIBTORCH=$(pwd)/libtorch

# Rebuild with PyTorch support
cargo build --release --features tch
```

### 4. Development Workflow

```bash
# Check code compiles
cargo check

# Run tests
cargo test

# Format code
cargo fmt

# Lint code (requires clippy)
cargo clippy
```

## Next Steps

### Implementation Priority

1. **I/O Layer (boltr-io)**
   - Implement YAML config parser with serde_yaml
   - Add FASTA parser for sequences
   - Add PDB parser for structures
   - Implement MSA server client

2. **Backend Layer (boltr-backend-tch)**
   - Implement model weight loading
   - Port neural network layers from reference
   - Implement attention mechanisms
   - Add equivariance operations

3. **Model Architecture**
   - Port Boltz-1 architecture from `boltz-reference/src/boltz/model/models/boltz1.py`
   - Port Boltz-2 architecture from `boltz-reference/src/boltz/model/models/boltz2.py`
   - Implement inference pipeline

4. **CLI Integration**
   - Wire up predict command
   - Add download command for model weights
   - Implement evaluation commands
   - Add batch processing support

## Reference Implementation Guide

The original Boltz implementation in `boltz-reference/` contains:

### Key Files to Study

**Model Architecture:**
- `src/boltz/model/models/boltz1.py` - Boltz-1 model
- `src/boltz/model/models/boltz2.py` - Boltz-2 model
- `src/boltz/model/layers/` - Neural network layers
- `src/boltz/model/modules/` - Core modules (attention, diffusion, etc.)

**Data Processing:**
- `src/boltz/main.py` - Main inference pipeline
- `docs/prediction.md` - Input/output formats
- `examples/` - Example configurations

**Documentation:**
- `docs/prediction.md` - How to format inputs
- `docs/training.md` - Model architecture details
- `README.md` - Overview and features

### Architecture Mapping

| Python (Reference) | Rust (Boltr) |
|-------------------|--------------|
| `src/boltz/model/` | `boltr-backend-tch/src/` |
| `src/boltz/main.py` | `boltr-cli/src/main.rs` |
| YAML parsing | `boltr-io/src/config.rs` |
| MSA processing | `boltr-io/src/msa.rs` |

## Common Commands

```bash
# Run the CLI
./target/release/boltr --help
./target/release/boltr predict input.yaml --use_msa_server

# Development
cargo watch -x check          # Watch for changes
cargo watch -x test           # Watch and run tests
cargo expand                  # Expand macros

# Debugging
RUST_BACKTRACE=1 ./target/release/boltr predict test.yaml
cargo build --verbose         # Verbose build

# Release optimization
cargo build --release         # Optimized build
cargo bench                   # Run benchmarks
```

## Troubleshooting

### Build Errors
```bash
# Clean build artifacts
cargo clean
cargo build

# Update dependencies
cargo update
```

### LibTorch Issues
```bash
# Check if LIBTORCH is set
echo $LIBTORCH

# Use Python environment
export LIBTORCH_USE_PYTORCH=1
python3 -c "import torch; print(torch.__version__)"
```

### Git Issues
```bash
# Check remote
git remote -v

# Push to GitHub
git push -u origin master

# Create new branch
git checkout -b feature/my-feature
```

## Resources

- **Original Boltz**: https://github.com/jwohlwend/boltz
- **tch-rs (PyTorch bindings)**: https://github.com/LaurentMazare/tch-rs
- **Rust Book**: https://doc.rust-lang.org/book/
- **Clap (CLI)**: https://docs.rs/clap

## Contributing

1. Make changes in appropriate crate
2. Run `cargo check` and `cargo test`
3. Format with `cargo fmt`
4. Commit with descriptive message
5. Push to GitHub

Example commit:
```bash
git add .
git commit -m "Implement YAML config parser"
git push origin master
```

## License

MIT License - See LICENSE file for details

# Boltr - Rust Native Boltz Implementation

A high-performance Rust reimplementation of the Boltz biomolecular interaction prediction models.

## Overview

Boltr is a native Rust implementation of the Boltz-1 and Boltz-2 models for biomolecular interaction prediction. This project aims to provide:
- **Boltz-1 accuracy** approaching AlphaFold3
- **Boltz-2 capabilities** for joint structure and binding affinity prediction
- **1000x faster** inference than physics-based methods
- **Memory-safe** implementation in Rust
- **Modular architecture** with separate CLI, IO, and backend components

## Architecture

The project is organized as a Rust workspace with three main crates:

- **boltr-cli**: Command-line interface and user interaction layer
- **boltr-io**: Input/output handling, file parsing, and MSA server communication
- **boltr-backend-tch**: Core inference backend using tch-rs (PyTorch bindings)

## Installation

### Prerequisites

- Rust 1.93 or later
- PyTorch with LibTorch (for tch-rs)
- CUDA (optional, for GPU acceleration)

### Build from Source

```bash
git clone https://github.com/SampleBias/Boltr.git
cd Boltr
cargo build --release -p boltr-cli
```

The binary is at `target/release/boltr`. I/O and CLI build **without** LibTorch by default. For `tch-rs` inference, set `LIBTORCH` or `LIBTORCH_USE_PYTORCH=1` and build with:

```bash
cargo build --release -p boltr-cli --features tch
```

Use a **CUDA** LibTorch build for GPU; pass `--device cuda` (or `cuda:N`). See [DEVELOPMENT.md](DEVELOPMENT.md).

## Usage

### Prediction

```bash
boltr predict input.yaml --use_msa_server --output ./results --device cpu
```

Optional: `BOLTR_DEVICE=cuda` instead of `--device`. MSA server base URL defaults to `https://api.colabfold.com` (same as Boltz); override with `--msa-server-url`.

### Download Model Weights

```bash
boltr download --version boltz2
```

### Evaluation

```bash
boltr eval ./test_data
```

## Development Status

Work in progress. See [docs/TENSOR_CONTRACT.md](docs/TENSOR_CONTRACT.md) for the Python tensor path and [docs/PYTHON_REMOVAL.md](docs/PYTHON_REMOVAL.md) for when to shrink `boltz-reference/`.

- [ ] Boltz-1 full model
- [ ] Boltz-2 full forward (in progress: `boltr-backend-tch/src/boltz2/`, safetensors load, `s_init` spike)
- [ ] Attention / pairformer (non-kernel parity with Python)
- [x] MSA server client + YAML parsing (`boltr-io`, ColabFold-compatible protocol)
- [ ] Structure prediction (Rust featurizer + diffusion)
- [ ] Binding affinity head
- [x] GPU path: CUDA LibTorch + CLI `--device cuda` (Python-only cuequivariance kernels not in tch-rs)
- [x] Weight download (`.ckpt`); Rust loads exported `.safetensors` (`scripts/export_checkpoint_to_safetensors.py`)
- [x] CLI: `predict`, `download`, run summary JSON
- [x] YAML configuration types (Boltz2-oriented)

## Reference Implementation

The vendored `boltz-reference/` tree mirrors upstream Boltz for parity testing; removal is gated (see `docs/PYTHON_REMOVAL.md`).

## License

MIT License - See LICENSE file for details

## Citation

If you use this implementation, please cite the original Boltz papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Original Boltz team: https://github.com/jwohlwend/boltz
- tch-rs team for PyTorch bindings

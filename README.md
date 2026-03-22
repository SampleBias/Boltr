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
cargo build --release
```

The binary will be available at `target/release/boltr`.

## Usage

### Prediction

```bash
boltr predict input.yaml --use_msa_server --output ./results
```

### Download Model Weights

```bash
boltr download --version boltz2
```

### Evaluation

```bash
boltr eval ./test_data
```

## Development Status

This is an active work-in-progress project. The implementation follows the original Boltz architecture:

- [ ] Model architecture implementation (Boltz-1)
- [ ] Model architecture implementation (Boltz-2)
- [ ] Attention mechanisms (equivariant and standard)
- [ ] MSA processing pipeline
- [ ] Structure prediction
- [ ] Binding affinity prediction
- [ ] GPU acceleration via CUDA
- [ ] Model weight loading and inference
- [ ] CLI interface
- [ ] Configuration file parsing
- [ ] MSA server integration

## Reference Implementation

This project is based on the original Boltz implementation in PyTorch, which is included in the `boltz-reference/` directory for architectural reference.

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

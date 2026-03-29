# Boltr — Rust-native Boltz2 inference

```text
___
          _____                   _______                   _____        _____                    _____          
         /\    \                 /::\    \                 /\    \      /\    \                  /\    \         
        /::\    \               /::::\    \               /::\____\    /::\    \                /::\    \        
       /::::\    \             /::::::\    \             /:::/    /    \:::\    \              /::::\    \       
      /::::::\    \           /::::::::\    \           /:::/    /      \:::\    \            /::::::\    \      
     /:::/\:::\    \         /:::/~~\:::\    \         /:::/    /        \:::\    \          /:::/\:::\    \     
    /:::/__\:::\    \       /:::/    \:::\    \       /:::/    /          \:::\    \        /:::/__\:::\    \    
   /::::\   \:::\    \     /:::/    / \:::\    \     /:::/    /           /::::\    \      /::::\   \:::\    \   
  /::::::\   \:::\    \   /:::/____/   \:::\____\   /:::/    /           /::::::\    \    /::::::\   \:::\    \  
 /:::/\:::\   \:::\ ___\ |:::|    |     |:::|    | /:::/    /           /:::/\:::\    \  /:::/\:::\   \:::\____\ 
/:::/__\:::\   \:::|    ||:::|____|     |:::|    |/:::/____/           /:::/  \:::\____\/:::/  \:::\   \:::|    |
\:::\   \:::\  /:::|____| \:::\    \   /:::/    / \:::\    \          /:::/    \::/    /\::/   |::::\  /:::|____|
 \:::\   \:::\/:::/    /   \:::\    \ /:::/    /   \:::\    \        /:::/    / \/____/  \/____|:::::\/:::/    / 
  \:::\   \::::::/    /     \:::\    /:::/    /     \:::\    \      /:::/    /                 |:::::::::/    /  
   \:::\   \::::/    /       \:::\__/:::/    /       \:::\    \    /:::/    /                  |::|\::::/    /   
    \:::\  /:::/    /         \::::::::/    /         \:::\    \   \::/    /                   |::| \::/____/    
     \:::\/:::/    /           \::::::/    /           \:::\    \   \/____/                    |::|  ~|          
      \::::::/    /             \::::/    /             \:::\    \                             |::|   |          
       \::::/    /               \::/____/               \:::\____\                            \::|   |          
        \::/____/                 ~~                      \::/    /                             \:|   |          
         ~~                                                \/____/                               \|___|          
                                                                                                                 
___
```

Boltr is a **Rust** reimplementation of **Boltz-2** (and shared Boltz architecture pieces) for biomolecular structure and affinity-related inference. The goal is **numerical parity** with PyTorch Boltz on the **non–cuEquivariance** path (`use_kernels=False`): same tensor contracts, LibTorch-backed modules via **`tch-rs`**, and a small set of **Python golden exporters** for regression.

The original Boltz models are described in the [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) and [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) papers. This repository tracks **Boltr-specific** implementation status in [`TODO.md`](TODO.md).

---

## Workspace layout

| Crate | Role |
|-------|------|
| [**boltr-io**](boltr-io/) | YAML → typed config, StructureV2, tokenizer, **Boltz2 featurizer** (`process_*`), inference **collate**, MSA helpers, **writers** (confidence JSON, PAE/PDE/plddt npz, PDB/mmCIF). Builds **without** LibTorch. |
| [**boltr-backend-tch**](boltr-backend-tch/) | **Boltz2** `VarStore` graph: trunk (input embedder, MSA, templates, pairformer), diffusion + sampling, distogram, confidence, affinity module, potentials / steering. Requires **`--features tch-backend`** and a LibTorch install. |
| [**boltr-cli**](boltr-cli/) | **`boltr`** binary: `download`, `predict` (YAML + optional MSA), `msa-to-npz`, `tokens-to-npz`, device selection. With **`--features tch`**, `predict` can run **preprocess → collate → `predict_step` → PDB/mmCIF** when `manifest.json` and preprocess `.npz` sit **next to the input YAML** ([`predict_tch.rs`](boltr-cli/src/predict_tch.rs)); otherwise it writes the usual summary + placeholder dirs (see checklist). |

Supporting assets:

- **[`boltz-reference/`](boltz-reference/)** — Vendored **model-only** Python tree for reading parity and opt-in golden exports (not a full upstream Boltz clone). See [`boltz-reference/README.md`](boltz-reference/README.md).
- **[`scripts/`](scripts/README.md)** — Checkpoint export, golden generators, regression harness helpers.

---

## Prerequisites

- **Rust**: stable toolchain, **edition 2021** (see workspace `Cargo.toml`).
- **Default build** (`cargo build -p boltr-cli`): **no** LibTorch — I/O and CLI compile for fast iteration.
- **Inference backend** (`boltr-backend-tch`): **LibTorch** matching your `tch` version — set `LIBTORCH`, or `LIBTORCH_USE_PYTORCH=1` with a Python env that has `torch`, or use [`scripts/bootstrap_dev_venv.sh`](scripts/bootstrap_dev_venv.sh) + [`scripts/cargo-tch`](scripts/cargo-tch) (see [`DEVELOPMENT.md`](DEVELOPMENT.md)).
- **CUDA** (optional): CUDA build of LibTorch; CLI `--device cuda` / `BOLTR_DEVICE=cuda`.

---

## Quick start

```bash
git clone https://github.com/SampleBias/Boltr.git
cd Boltr
cargo build --release -p boltr-cli
# Binary: target/release/boltr
```

LibTorch / `tch` backend (from repo root) — pick **one**:

**Path A — standalone LibTorch** (no `pip` / `torch` on system Python; typical for minimal Arch):

1. Unpack **LibTorch 2.3.x** into `third_party/libtorch` ([`DEVELOPMENT.md`](DEVELOPMENT.md): CPU zip, or CUDA cu118/cu121 from [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)). For CUDA, the repo expects `third_party/libtorch/lib/libtorch_cuda.so` (gitignored).
2. In every shell where you run `cargo`:

```bash
source scripts/env_libtorch_cuda.sh
cargo build --release -p boltr-cli --features tch
```

`env_libtorch_cuda.sh` sets `LIBTORCH`, `LD_LIBRARY_PATH`, `LIBTORCH_CXX11_ABI=0` (required for official non–cxx11-abi CUDA zips), and unsets `LIBTORCH_USE_PYTORCH`. Override the tree with `BOLTR_LIBTORCH=/path/to/libtorch`.

**Path B — PyTorch venv** (`torch-sys` uses Python’s LibTorch):

```bash
bash scripts/bootstrap_dev_venv.sh    # once: .venv + torch for torch-sys
bash scripts/cargo-tch build --release -p boltr-cli --features tch
```

### Common commands

| Command | Notes |
|---------|--------|
| `boltr download --version boltz2` | Checkpoints + CCD + mols into cache (URLs aligned with upstream Boltz). |
| `boltr predict input.yaml --output ./out --device cpu` | Build with **`--features tch`**. Parses YAML, optional MSA, summary JSON. **Native structure output:** place Boltz-style **`manifest.json`** + **`{record_id}.npz`** (and MSA `.npz`) in the **same directory as the input YAML**, then run `predict` — see [`TODO.md` §5.10 / §6](TODO.md). Without that layout, outputs are placeholders until preprocess data is present. |
| `cargo test -p boltr-io` | I/O + featurizer tests (CI: [`.github/workflows/boltr-io-test.yml`](.github/workflows/boltr-io-test.yml)). |
| `bash scripts/cargo-tch test -p boltr-backend-tch --features tch-backend --lib` | Backend library tests (manual / dev venv). |

---

## Documentation index

| Document | Contents |
|----------|----------|
| [**TODO.md**](TODO.md) | **Master implementation checklist** — parity rules, `boltr-io` plan, backend graph, CLI, testing, Python removal gates. |
| [**DEVELOPMENT.md**](DEVELOPMENT.md) | LibTorch matrix, `tch` troubleshooting, dev venv. |
| [**docs/TENSOR_CONTRACT.md**](docs/TENSOR_CONTRACT.md) | Python tensor path, featurizer keys, checkpoint naming, **§6** tolerances + regression pointers. |
| [**docs/NUMERICAL_TOLERANCES.md**](docs/NUMERICAL_TOLERANCES.md) | Central `rtol` / `atol` registry for goldens. |
| [**docs/PYTHON_REMOVAL.md**](docs/PYTHON_REMOVAL.md) | When / how to shrink vendored Python. |
| [**scripts/README.md**](scripts/README.md) | All helper scripts and env vars for goldens. |
| [**boltr-io/tests/fixtures/README.md**](boltr-io/tests/fixtures/README.md) | Fixture layout and regeneration. |
| [**boltr-backend-tch/tests/fixtures/README.md**](boltr-backend-tch/tests/fixtures/README.md) | Backend safetensors / opt-in goldens. |
| [**docs/activity.md**](docs/activity.md) | Chronological work log. |

---

## Project status (high-level checklist)

This mirrors [`TODO.md`](TODO.md) at a glance. For authoritative per-row status, use **TODO.md**.

### `boltr-io` (data path)

- [x] **YAML** → `BoltzInput` / entities / constraints / templates / affinity `properties` ([`boltr-io/src/config.rs`](boltr-io/src/config.rs), [`yaml_parse` tests](boltr-io/tests/yaml_parse.rs)); serde **round-trip** coverage for representative fixtures.
- [x] **StructureV2**, NPZ I/O, **CCD** JSON molecules, **MSA** (a3m, CSV, npz, ColabFold client).
- [x] **Boltz2 tokenizer** + **featurizer** (`process_token_features`, `process_atom_features`, `process_msa_features`, templates, symmetry, constraints, ensemble).
- [x] **Inference collate** + **post-collate golden** ([`post_collate_golden.rs`](boltr-io/tests/post_collate_golden.rs), `trunk_smoke_collate.safetensors`).
- [x] **Writers**: confidence JSON, PAE/PDE/plddt npz, PDB/mmCIF from `StructureV2`.
- [x] Optional **larger** Python↔Rust cross-goldens on huge ligand/constraint fixtures remain **optional follow-ups** (not required for core gates); cross-entity schema validation stays in upstream Python where applicable.

### `boltr-backend-tch` (model)

- [x] **Boltz2Model**: `InputEmbedder`, `RelativePositionEncoder`, trunk (MSA, **TemplateV2**, pairformer), **diffusion** + score model, **distogram**, **confidence** v2, **affinity** module, **potentials** / steering hooks.
- [x] **`predict_step`** / **`predict_step_trunk`** (see [`model.rs`](boltr-backend-tch/src/boltz2/model.rs)).
- [x] **Opt-in Python goldens**: MSA module, pairformer layer, trunk init, input embedder (`BOLTR_RUN_*_GOLDEN=1` — see [`boltr-backend-tch/tests/fixtures/README.md`](boltr-backend-tch/tests/fixtures/README.md)).
- [x] **VarStore** / **hparams**: smoke fixtures, [`inference_keys`](boltr-backend-tch/src/inference_keys.rs) taxonomy, [`Boltz2Hparams`](boltr-backend-tch/src/boltz_hparams.rs) + Lightning JSON export; strict pairing via [`verify_boltz2_safetensors --reject-unused-file-keys`](boltr-backend-tch/src/bin/verify_boltz2_safetensors.rs) when you need no-extra-keys checks.
- [x] **Z-init** (`rel_pos`, `s_init`, `token_bonds`, `contact`): [`forward_trunk_with_z_init_terms`](boltr-backend-tch/src/boltz2/model.rs), unit tests + [`trunk_init_golden.rs`](boltr-backend-tch/tests/trunk_init_golden.rs) for `rel_pos`/`s_init`.

### `boltr-cli` (user-facing)

- [x] **`download`**, **`predict`** (YAML, MSA options, summary JSON, `boltr_predict_args.json`, `--spike-only` trunk smoke with `tch`).
- [x] **`predict` (tch):** with preprocess **`manifest.json` + `.npz`** next to the YAML → **`load_input` → collate → `predict_step` → PDB/mmCIF** ([`collate_predict_bridge.rs`](boltr-cli/src/collate_predict_bridge.rs)). Confidence / PAE npz from the CLI still depend on loading the confidence stack and wiring tensors ([`TODO.md` §5.10](TODO.md)).
- [x] **Flags parity** — see [`TODO.md` §6](TODO.md) (`--recycling-steps`, `--sampling-steps`, diffusion samples, potentials, affinity, …).
- [x] **`eval`** — stub with pointer to [upstream evaluation docs](boltz-reference/docs/evaluation.md) (full benchmark tooling not ported).

### Tooling, testing, CI

- [x] **Dev venv** + [`scripts/cargo-tch`](scripts/cargo-tch) for LibTorch tests.
- [x] **Fixture READMEs**, [**NUMERICAL_TOLERANCES.md**](docs/NUMERICAL_TOLERANCES.md), regression scripts ([`regression_compare_predict.sh`](scripts/regression_compare_predict.sh), optional `BOLTR_REGRESSION=1`).
- [x] **CI**: `boltr-io` tests on push/PR ([`boltr-io-test.yml`](.github/workflows/boltr-io-test.yml)); MSA npz golden ([`msa-npz-golden.yml`](.github/workflows/msa-npz-golden.yml)); optional LibTorch smoke ([`libtorch-backend-smoke.yml`](.github/workflows/libtorch-backend-smoke.yml)); optional manual note for full-Python collate export ([`dump-full-collate-golden.yml`](.github/workflows/dump-full-collate-golden.yml)).

### Reference / Python

- [x] **`boltz-reference/`** model slice for parity + golden scripts; removal **gated** ([`docs/PYTHON_REMOVAL.md`](docs/PYTHON_REMOVAL.md)).

---

## Reference implementation

Upstream Boltz: **https://github.com/jwohlwend/boltz**. Boltr’s **vendored** tree is **not** a full substitute — use upstream for `boltz.data`, training, and the full CLI when you need them.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use this implementation, cite the **Boltz** papers (and Boltr as a fork if you publish on this codebase):

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

---

## Contributing

Issues and pull requests are welcome. For large changes, align with **§1 Parity rules** in [`TODO.md`](TODO.md).

## Acknowledgments

- Original Boltz team: https://github.com/jwohlwend/boltz  
- [tch-rs](https://github.com/LaurentMazare/tch-rs) (PyTorch bindings for Rust)

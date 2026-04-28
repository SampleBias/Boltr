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
| [**boltr-cli**](boltr-cli/) | **`boltr`** binary: `download`, `predict` (YAML + optional MSA), **`preprocess`** (Boltz subprocess or **native** protein-only bundle; see [`docs/PREPROCESS_NATIVE.md`](docs/PREPROCESS_NATIVE.md)), `doctor` (LibTorch smoke), `msa-to-npz`, `tokens-to-npz`, device selection. With **`--features tch`**, `predict` can run **preprocess → collate → `predict_step` → PDB/mmCIF** when `manifest.json` and preprocess `.npz` sit **next to the input YAML** ([`predict_tch.rs`](boltr-cli/src/predict_tch.rs)); use **`--preprocess native|boltz|auto`** to generate that bundle first. Otherwise it writes the usual summary + placeholder dirs (see checklist). |
| [**boltr-web**](boltr-web/) | Local **Axum** UI: cache / `boltr doctor`–style status, YAML validation, **`boltr predict`** jobs, log SSE, tarball download, and **`structure_output`** in job status (paths to `.cif`/`.pdb` + explanation when missing). See [`boltr-web/README.md`](boltr-web/README.md) and [`QUICKSTART.md`](QUICKSTART.md). |

Supporting assets:

- **[`boltz-reference/`](boltz-reference/)** — Vendored **model-only** Python tree for reading parity and opt-in golden exports (not a full upstream Boltz clone). See [`boltz-reference/README.md`](boltz-reference/README.md).
- **[`scripts/`](scripts/README.md)** — Checkpoint export, golden generators, regression harness helpers.

---

## Prerequisites

- **Rust**: stable toolchain, **edition 2021** (see workspace [`Cargo.toml`](Cargo.toml)).
- **Default build** (`cargo build -p boltr-cli`): **no** LibTorch — I/O and CLI compile for fast iteration.
- **Inference backend** (`boltr-backend-tch`): **LibTorch** matching your `tch` version — set `LIBTORCH`, or `LIBTORCH_USE_PYTORCH=1` with a Python env that has `torch`, or use [`scripts/bootstrap_dev_venv.sh`](scripts/bootstrap_dev_venv.sh) + [`scripts/cargo-tch`](scripts/cargo-tch) (see [`DEVELOPMENT.md`](DEVELOPMENT.md)).
- **CUDA** (optional): CUDA build of LibTorch; CLI `--device cuda` / `BOLTR_DEVICE=cuda`.

---

## Dependencies (summary)

Everything below is spelled out in [`DEVELOPMENT.md`](DEVELOPMENT.md) and [`QUICKSTART.md`](QUICKSTART.md); this table is the single index.

### Rust workspace

| Item | Notes |
|------|--------|
| **Edition** | **2021** (all crates). |
| **Resolver** | Cargo **workspace** `resolver = "2"` ([`Cargo.toml`](Cargo.toml)). |
| **Crates** | [`boltr-io`](boltr-io/Cargo.toml), [`boltr-backend-tch`](boltr-backend-tch/Cargo.toml), [`boltr-cli`](boltr-cli/Cargo.toml), [`boltr-web`](boltr-web/Cargo.toml). |
| **Key libs** | `tokio`, `serde` / `serde_json` / `serde_yaml`, `clap`, `anyhow`, `tracing`, `ndarray`, `reqwest`, `safetensors`, `flate2`, `tar`, `dirs`, `rand`, `numpy` (crate), `itertools` — versions pinned in **workspace** `[workspace.dependencies]`. |
| **`tch`** | **`0.16`** (optional; enables LibTorch when `--features tch` on `boltr-cli` or `tch-backend` on `boltr-backend-tch`). |
| **`boltr-web`** | `axum` 0.7 (HTTP + multipart), `tokio`, same serde stack as the rest of the workspace. |

### LibTorch / `tch` (native inference)

| Item | Notes |
|------|--------|
| **`tch` crate** | **0.16** — must match **LibTorch C++ API ~2.3.x** (see [`boltr-cli/Cargo.toml`](boltr-cli/Cargo.toml)). |
| **Standalone LibTorch** | Download **2.3.x** CPU or CUDA zips from PyTorch; **do not** use unversioned “latest” LibTorch ([`DEVELOPMENT.md`](DEVELOPMENT.md) Path A). |
| **PyTorch wheel** | **`torch==2.3.0`** in the dev venv so `torch-sys` compiles against matching headers ([`scripts/bootstrap_dev_venv.sh`](scripts/bootstrap_dev_venv.sh), `BOLTR_TORCH_VERSION`). |
| **Build env** | Set **`LIBTORCH`** (Path A) **or** **`LIBTORCH_USE_PYTORCH=1`** (Path B). Optional: **`LIBTORCH_BYPASS_VERSION_CHECK=1`** when pip’s torch reports a newer *string* than libtch expects ([`scripts/with_dev_venv.sh`](scripts/with_dev_venv.sh)). |
| **GPU** | CUDA LibTorch matching the same **2.3.x** line; `LD_LIBRARY_PATH` must include PyTorch’s `lib/` when using CUDA wheels ([`scripts/with_dev_venv.sh`](scripts/with_dev_venv.sh)). |

### Python (scripts + optional venv — not used for Rust `predict` at runtime)

| Item | Notes |
|------|--------|
| **Interpreter** | **3.10, 3.11, or 3.12** for [`bootstrap_dev_venv.sh`](scripts/bootstrap_dev_venv.sh). **Avoid 3.13+** for that script (wheel / header mismatch with `tch` 0.16). Override pick with **`BOLTR_VENV_PYTHON`**. |
| **Packages** | **`torch==2.3.0`**, **`safetensors`**, **`numpy`**, **`omegaconf`**, **`setuptools`**, **`wheel`** (installed by bootstrap). |
| **Uses** | Checkpoint **`.ckpt` → `.safetensors`** ([`scripts/export_checkpoint_to_safetensors.py`](scripts/export_checkpoint_to_safetensors.py)), golden generators, optional post-`download` export in **`boltr download`**, and **`torch-sys`** discovery when **`LIBTORCH_USE_PYTORCH=1`**. |
| **Arch / PEP 668** | Do not `pip install` into system Python; use the repo **`.venv`** ([`DEVELOPMENT.md`](DEVELOPMENT.md)). |

### System / OS

| Item | Notes |
|------|--------|
| **Build tools** | A normal Rust toolchain pulls in `cc`; **`torch-sys`** may need **`cmake`**, **`pkg-config`**, and a **C++ compiler** — see build errors in [`DEVELOPMENT.md`](DEVELOPMENT.md). |
| **AUR / large builds** | If **`/tmp`** is a small **tmpfs**, set **`TMPDIR`** and **`BUILDDIR`** under `$HOME` for `makepkg` / `yay` so builds do not hit quota ([`DEVELOPMENT.md`](DEVELOPMENT.md) / Arch notes). |
| **Network** | **`boltr download`** uses **HTTPS** (reqwest) to fetch checkpoints and assets. |

### Runtime directories and env vars

| Variable | Purpose |
|----------|---------|
| **`BOLTZ_CACHE`** | Model cache directory (default: `~/.cache/boltr`). Same idea as **`boltr --cache-dir`**. |
| **`BOLTR`** | Absolute path to the **`boltr`** binary for **`boltr-web`** status (`boltr doctor --json`) and tooling. |
| **`BOLTR_REPO`** | Optional; helps tools find **`.venv/bin/python`** when probing from **`boltr-web`**. |
| **`BOLTR_BOLTZ_COMMAND`** | Full path to upstream Python **`boltz`** for **`boltr predict --preprocess boltz`** / **`auto`** when `boltz` is not on `PATH` (also settable in **boltr-web** “Bolt command”). |
| **`BOLTR_BOLTZ_VENV`** | Optional venv root for upstream Boltz. The Web UI probes `$BOLTR_BOLTZ_VENV/bin/boltz`, then `/workspace/boltr-envs/boltz-gpu/bin/boltz`, before repo-local venvs. |
| **`BOLTR_BOLTZ_USE_KERNELS`** | Set to `1` only when the selected upstream Boltz env has compatible `cuequivariance` wheels. By default Boltr passes `--no_kernels` to avoid `cuequivariance_torch` import/runtime failures. |
| **`BOLTR_DEVICE`** | Overrides CLI **`--device`** if set. |
| **`LIBTORCH`** | Root of unpacked standalone LibTorch (Path A). |
| **`LIBTORCH_USE_PYTORCH`** | Set to **`1`** when linking against PyTorch’s LibTorch (Path B). |

### One-shot full stack (venv + download + safetensors + `boltr` + `boltr-web`)

```bash
bash scripts/bootstrap_webui_env.sh
```

**Extended bootstrap** (runs the above, optional VarStore verify, cargo tests, and writes `boltr_go_bootstrap.json` for **boltr-web**): `./Boltr_Boltz_bootstrap` from the repo root (alias: `./Boltr_go`). This stages **Boltz2 model weights** in `BOLTZ_CACHE`; it does **not** install the Python **`boltz`** CLI used for `--preprocess boltz` (templates/constraints).

See [`QUICKSTART.md`](QUICKSTART.md) for **`boltr doctor`**, **`BOLTR`**, and **`with_dev_venv.sh`** when running **`boltr-web`**.

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
# or: bash scripts/build-boltr-cli-release.sh
```

**Troubleshooting:** If `cargo build -p boltr-cli --features tch` fails with **`Cannot find a libtorch install`** (often followed by `warning: build failed, waiting for other jobs to finish...`), you ran **`cargo` without LibTorch on the path**. Use **Path A** (`export LIBTORCH=...`) or **Path B** (`bash scripts/cargo-tch ...` after `bootstrap_dev_venv.sh`). Plain system `python3` usually has no `torch`, so `torch-sys` cannot probe PyTorch’s libraries.

### Common commands

| Command | Notes |
|---------|--------|
| `boltr download --version boltz2` | Checkpoints + CCD + mols into cache (URLs aligned with upstream Boltz). Best-effort **`.ckpt` → `.safetensors`** when repo + Python `torch`/`safetensors` are available; else export manually ([`DEVELOPMENT.md`](DEVELOPMENT.md)). |
| `boltr doctor` / `boltr doctor --json` | LibTorch / **`tch`** smoke (needs **`--features tch`** build for a real CPU tensor probe). |
| `boltr predict input.yaml --output ./out --device cpu --preprocess auto` | Build with **`--features tch`**. **Structure files** (mmCIF/PDB) require a Boltz-style preprocess bundle **`manifest.json` + `.npz` files next to the input YAML** — generate them with **`--preprocess auto`**, **`native`**, or **`boltz`** (see [**Predict: preprocess and structure output**](#predict-preprocess-and-structure-output) below). Without that bundle, the run completes with summary JSON only (no sampled coordinates). |
| `cargo test -p boltr-io` | I/O + featurizer tests (CI: [`.github/workflows/boltr-io-test.yml`](.github/workflows/boltr-io-test.yml)). |
| `bash scripts/cargo-tch test -p boltr-backend-tch --features tch-backend --lib` | Backend library tests (manual / dev venv). |
| `cargo build -p boltr-web && ./target/release/boltr-web` | Local **Axum** UI for cache status + YAML validation; export **`BOLTR`** to a **`tch`**-enabled `boltr` for **`doctor`** probes ([`QUICKSTART.md`](QUICKSTART.md)). |

---

## Predict: preprocess and structure output

`boltr predict` with **`--features tch`** can write **mmCIF (`.cif`)** or **PDB** under `--output` **only when** a valid **preprocess bundle** sits **in the same directory as the input YAML**:

- `manifest.json`
- Structure and MSA **`.npz`** files as referenced by that manifest (Boltz-compatible layout)

### Generating the bundle

| Flag | Behavior |
|------|----------|
| **`--preprocess off`** | No bundle generation. Unless you **manually** place `manifest.json` + npz next to the YAML, you will **not** get main-line structure files from the Rust bridge. |
| **`--preprocess native`** | Rust-only **protein-only** bundle (no ligands/DNA/RNA; templates/constraints must be empty/absent per validation). Does **not** use the Python `boltz` CLI. |
| **`--preprocess boltz`** | Runs upstream **`boltz predict`** into a staging dir, then copies the bundle next to your YAML. Requires the **`boltz`** executable (`pip install boltz` or conda; use **`--bolt-command`** if it is not on `PATH`). |
| **`--preprocess auto`** | Tries **native** when eligible, otherwise **`boltz`** if runnable. |

**Important:** [`Boltr_Boltz_bootstrap`](./Boltr_Boltz_bootstrap) / `./Boltr_go` / `bootstrap_webui_env.sh` stage **model weights** into `BOLTZ_CACHE` and build **`boltr`** — they do **not** install the Python **`boltz`** CLI by default. For **`--preprocess boltz`** / **`auto`** fallback to Boltz, install **`boltz`** in a venv and ensure **`boltz`** is on `PATH`, or pass **`--bolt-command`** with the full path to the `boltz` executable. The Web UI status panel checks this dependency up front. On RunPod/Blackwell GPUs, prefer `scripts/bootstrap_boltz_gpu_venv.sh`; it keeps upstream Boltz in a separate CUDA-compatible env so the repo `.venv` can remain pinned for `tch`. Set `BOLTR_INSTALL_BOLTZ=1` when running `scripts/bootstrap_dev_venv.sh` only if you want the repo venv to include the upstream CLI for CPU or compatible-GPU use.

By default Boltr adds upstream Boltz’s `--no_kernels` flag. This avoids `ModuleNotFoundError: cuequivariance_torch` and kernel-image mismatches. If you intentionally install compatible cuEquivariance wheels, set `BOLTR_BOLTZ_USE_KERNELS=1` or pass explicit upstream Boltz args.

Upstream Boltz writes preprocess artifacts under a **`boltz_results_<yaml_stem>/`** directory; Boltr discovers **`manifest.json`** under `processed/` and copies structure/MSA npz from **`structures/`**, **`msa/`**, etc., into the flat layout next to your YAML.

### What gets written on success

With a valid bundle and a successful diffusion bridge, coordinates are written as:

- `{output}/{record_id}/{record_id}_model_0.cif` (mmCIF) or `.pdb`, depending on **`--output-format`**.

`boltr_predict_complete.txt` in that folder records **`status`**: e.g. **`predict_step_complete`** (diffusion), **`preprocess_reference_structure`** (reference coordinates only if diffusion did not run but the bundle loaded), or **`pipeline_complete`** when no structure file was produced (missing bundle or `load_input` failure). See [`boltr-cli/src/predict_tch.rs`](boltr-cli/src/predict_tch.rs).

### Troubleshooting: no `.cif` / `.pdb`

1. Confirm **`manifest.json`** exists **next to the YAML** you pass to `boltr predict` (same folder).
2. Use **`--preprocess auto`** or **`boltz`** (not **`off`**) unless you pre-generated the bundle.
3. For complexes with templates, ligands, or non–protein-only inputs, prefer **`boltz`** and a working **`boltz`** CLI.
4. Ensure **`boltr` was built with `--features tch`** and LibTorch is available.

For the **web UI**, see [`boltr-web/README.md`](boltr-web/README.md): job status includes **`structure_output`** (paths + a short message before download).

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

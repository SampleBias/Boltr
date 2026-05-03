# Quick start

For a **full** project overview, checklist, and documentation index, read **[`README.md`](README.md)**. For **implementation status**, use **[`TODO.md`](TODO.md)** (master backlog).

## Clone and build (CLI + I/O only)

```bash
git clone https://github.com/SampleBias/Boltr.git
cd Boltr
cargo build --release -p boltr-cli
./target/release/boltr --help
```

No LibTorch required for this path.

## Build with LibTorch (`tch` backend)

`boltr-backend-tch` uses `torch-sys`. Prefer the repo dev venv:

```bash
bash scripts/bootstrap_dev_venv.sh
bash scripts/cargo-tch build --release -p boltr-cli --features tch
# or: bash scripts/build-boltr-cli-release.sh
```

Or set `LIBTORCH` / `LIBTORCH_USE_PYTORCH=1` and build as above. Details: **[`DEVELOPMENT.md`](DEVELOPMENT.md)**.

If **`Cannot find a libtorch install`** appears (often with `build failed, waiting for other jobs to finish...`), you used plain **`cargo`** without the venv / `LIBTORCH` — use **`scripts/cargo-tch`** or Path A in **[`README.md`](README.md)**.

## Web UI + full native cache (one script)

[`scripts/bootstrap_webui_env.sh`](scripts/bootstrap_webui_env.sh) sets up the dev venv (**including `pip install boltz`** for upstream preprocess unless `BOLTR_INSTALL_BOLTZ=0`), builds **`boltr`** and **`boltr-web`** with `--features tch`, runs **`boltr download --version boltz2`**, exports **`boltz2_conf.ckpt` → `boltz2_conf.safetensors`** (and affinity), and prints **`boltr doctor --json`**.

```bash
bash scripts/bootstrap_webui_env.sh
```

Run the Web UI (recommended: [**`scripts/run_boltr_web.sh`**](scripts/run_boltr_web.sh) sets **`BOLTR`**, **`BOLTR_REPO`**, and **`BOLTR_BOLTZ_COMMAND`**, and runs **`with_dev_venv`** so LibTorch resolves):

```bash
bash scripts/run_boltr_web.sh
```

Check LibTorch / `tch` linkage anytime:

```bash
bash scripts/with_dev_venv.sh ./target/release/boltr doctor
./target/release/boltr doctor --json
```

## Smoke tests

```bash
cargo test -p boltr-io
bash scripts/cargo-tch test -p boltr-backend-tch --features tch-backend --lib
```

## Repository layout (accurate)

```
Boltr/
├── boltr-cli/           # boltr binary: download, predict, doctor, msa-to-npz, tokens-to-npz
├── boltr-web/           # Local prerequisites + YAML validation UI
├── boltr-io/            # YAML, featurizer, collate, writers (no LibTorch)
├── boltr-backend-tch/   # Boltz2Model, tch (optional feature)
├── boltz-reference/     # Vendored Boltz *model* Python (trimmed; see README there)
├── scripts/             # Export + golden helpers (see scripts/README.md)
├── docs/                # TENSOR_CONTRACT, NUMERICAL_TOLERANCES, PYTHON_REMOVAL, …
├── README.md            # Overview + high-level checklist
├── TODO.md              # Detailed checklist
└── DEVELOPMENT.md       # LibTorch matrix
```

## Reference code

- **Boltz upstream:** https://github.com/jwohlwend/boltz  
- **Vendored in Boltr:** `boltz-reference/src/boltz/model/` — **Boltz2** model sources for parity; **not** a full `boltz` install (no `boltz.data` package in this tree). See [`docs/PYTHON_REMOVAL.md`](docs/PYTHON_REMOVAL.md).

## Common commands

```bash
boltr download --version boltz2
# After download, Boltr tries (warn-only) to run export_checkpoint_to_safetensors when a repo
# checkout + Python with torch/safetensors is found; otherwise export manually (DEVELOPMENT.md).
# Full model path (needs LibTorch — see build section above):
# bash scripts/cargo-tch build --release -p boltr-cli --features tch
boltr predict input.yaml --output ./out --device cpu
```

With **`--features tch`**, `predict` writes **PDB/mmCIF** when Boltz-style **`manifest.json`** and preprocess **`.npz`** sit next to the input YAML ([`TODO.md` §5.10](TODO.md)); otherwise see run summary / placeholder outputs.

## Resources

- Rust: https://doc.rust-lang.org/book/  
- tch-rs: https://github.com/LaurentMazare/tch-rs  
- Boltz upstream: https://github.com/jwohlwend/boltz  

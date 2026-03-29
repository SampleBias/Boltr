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
```

Or set `LIBTORCH` / `LIBTORCH_USE_PYTORCH=1` and build as above. Details: **[`DEVELOPMENT.md`](DEVELOPMENT.md)**.

## Smoke tests

```bash
cargo test -p boltr-io
bash scripts/cargo-tch test -p boltr-backend-tch --features tch-backend --lib
```

## Repository layout (accurate)

```
Boltr/
├── boltr-cli/           # boltr binary: download, predict, msa-to-npz, tokens-to-npz
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
# Full model path (needs LibTorch — see build section above):
# bash scripts/cargo-tch build --release -p boltr-cli --features tch
boltr predict input.yaml --output ./out --device cpu
```

With **`--features tch`**, `predict` writes **PDB/mmCIF** when Boltz-style **`manifest.json`** and preprocess **`.npz`** sit next to the input YAML ([`TODO.md` §5.10](TODO.md)); otherwise see run summary / placeholder outputs.

## Resources

- Rust: https://doc.rust-lang.org/book/  
- tch-rs: https://github.com/LaurentMazare/tch-rs  
- Boltz upstream: https://github.com/jwohlwend/boltz  

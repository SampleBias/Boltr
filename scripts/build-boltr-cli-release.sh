#!/usr/bin/env bash
# Release-build `boltr` with the `tch` feature using the dev venv + LIBTORCH_USE_PYTORCH=1.
# Plain `cargo build --features tch` fails with "Cannot find a libtorch install" unless
# you set LIBTORCH or use this wrapper — see README.md (LibTorch / tch backend).
#
# One-time: bash scripts/bootstrap_dev_venv.sh
# Usage: bash scripts/build-boltr-cli-release.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/cargo-tch" build --release -p boltr-cli --features tch "$@"

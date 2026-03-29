#!/usr/bin/env bash
# Interactive helper: Boltr basics, env checks, pointers to docs.
# Run from repo root:  bash scripts/boltr-menu.sh
# Or:  ./scripts/boltr-menu.sh  (after chmod +x)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

banner() {
  cat <<'EOF'
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
EOF
}

show_help() {
  banner
  cat <<'EOF'

Boltr — Rust-native Boltz2 CLI (this repo)

Quick commands (from repo root, after build)
  cargo build --release
  cargo build --release -p boltr-cli --features tch   # + LibTorch (see DEVELOPMENT.md)

  ./target/release/boltr --help
  ./target/release/boltr predict --help
  ./target/release/boltr predict input.yaml -o ./output --device cpu

GPU / LibTorch (Path A — no Python torch required)
  Unpack LibTorch 2.3.x+cu118 into third_party/libtorch (see DEVELOPMENT.md), then:
  source scripts/env_libtorch_cuda.sh
  cargo build --release -p boltr-cli --features tch
  # env script sets LIBTORCH, LD_LIBRARY_PATH, LIBTORCH_CXX11_ABI=0
  # Option [2] may still warn about pip/torch — ignore for Path A if LIBTORCH is set.

Python / venv (Path B)
  bash scripts/bootstrap_dev_venv.sh
  scripts/cargo-tch test -p boltr-backend-tch --features tch-backend

Diagnostics
  bash scripts/check_tch_prereqs.sh

Docs
  README.md, DEVELOPMENT.md, QUICKSTART.md

EOF
}

run_prereq_check() {
  echo ""
  bash "$ROOT/scripts/check_tch_prereqs.sh" || true
  echo ""
  read -r -p "Press Enter to continue..."
}

menu_loop() {
  while true; do
    clear 2>/dev/null || true
    banner
    cat <<'EOF'

  [1]  Help — commands and pointers (this text)
  [2]  Run LibTorch / Python prerequisite check (check_tch_prereqs.sh)
  [3]  Print how to source CUDA LibTorch env (env_libtorch_cuda.sh)
  [4]  Quit

EOF
    read -r -p "Choose [1-4]: " choice
    case "${choice:-}" in
      1) show_help; read -r -p "Press Enter to continue..." ;;
      2) run_prereq_check ;;
      3)
        echo ""
        echo "From repo root, before cargo build with GPU LibTorch:"
        echo "  source scripts/env_libtorch_cuda.sh"
        echo "  # Sets LIBTORCH, LD_LIBRARY_PATH, LIBTORCH_CXX11_ABI=0 (see script for details)"
        echo ""
        read -r -p "Press Enter to continue..."
        ;;
      4|q|Q) echo "Bye."; exit 0 ;;
      *) echo "Invalid choice."; sleep 1 ;;
    esac
  done
}

# Non-interactive: --help or -h
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

menu_loop

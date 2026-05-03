#!/usr/bin/env bash
# Ensure pkg-config + OpenSSL development files are present so cargo can build openssl-sys
# (native-tls / HTTPS in boltr-cli, boltr-web). Idempotent; no-op when already satisfied.
#
# Called from bootstrap_webui_env.sh. Opt out: BOLTR_SKIP_SYSTEM_DEPS=1
#
# Supports: apt (Debian/Ubuntu), dnf/yum (Fedora/RHEL-like), pacman (Arch). Other Linux: error with hint.

set -euo pipefail

if [[ "${BOLTR_SKIP_SYSTEM_DEPS:-}" == "1" ]]; then
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  exit 0
fi

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openssl 2>/dev/null; then
  exit 0
fi

echo "==> ensure_openssl_build_deps: installing pkg-config + OpenSSL headers (needed for cargo / openssl-sys)…" >&2

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "ERROR: need root or sudo to install OS packages (pkg-config, OpenSSL dev)." >&2
    return 1
  fi
}

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  run_root apt-get update -qq
  run_root apt-get install -y pkg-config libssl-dev
elif command -v dnf >/dev/null 2>&1; then
  run_root dnf install -y pkgconf-pkg-config openssl-devel
elif command -v yum >/dev/null 2>&1; then
  run_root yum install -y pkgconf-pkg-config openssl-devel
elif command -v pacman >/dev/null 2>&1; then
  run_root pacman -S --needed --noconfirm pkgconf openssl
else
  echo "ERROR: openssl-sys needs pkg-config and OpenSSL development packages." >&2
  echo "       No supported package manager found (apt-get, dnf, yum, pacman). Install them manually." >&2
  exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists openssl 2>/dev/null; then
  echo "ERROR: pkg-config still cannot find openssl after install — check OS packages." >&2
  exit 1
fi

echo "    openssl (pkg-config): $(pkg-config --modversion openssl)" >&2

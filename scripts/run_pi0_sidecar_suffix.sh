#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-build/pi0_sidecar/suffix_sidecar}"
shift || true

if [[ ! -x "${BIN}" ]]; then
  echo "suffix sidecar binary not found: ${BIN}"
  echo "build with: cmake -S native/pi0_sidecar -B build/pi0_sidecar && cmake --build build/pi0_sidecar -j"
  exit 1
fi

exec "${BIN}" "$@"


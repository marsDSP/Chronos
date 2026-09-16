#!/bin/bash

# debrel — brand-new fresh Debug + Release VST3 builds, copied to the local
# test folder. Wipes both build dirs, reconfigures from scratch with Ninja,
# builds the VST3 target in each config. The POST_BUILD copy step (enabled by
# -DCOPY_VST3_AFTER_BUILD=ON) drops Chronos-Debug.vst3 / Chronos-Release.vst3
# into VST3_COPY_DEST (default: ~/Desktop/vst test).
#
# Usage: debrel.sh [project_dir]
#   project_dir — Chronos source root (default: this script's repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)"

cd "$PROJECT_DIR"

echo "==> Wiping build-debug and build-release"
rm -rf build-debug build-release

echo "==> Configuring Debug (Ninja)"
cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCOPY_VST3_AFTER_BUILD=ON

echo "==> Configuring Release (Ninja)"
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOPY_VST3_AFTER_BUILD=ON

echo "==> Building Debug VST3"
cmake --build build-debug --target Chronos_VST3 -j"$JOBS"

echo "==> Building Release VST3"
cmake --build build-release --target Chronos_VST3 -j"$JOBS"

echo "==> Done: fresh Debug + Release VST3 built and copied"

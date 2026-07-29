#!/bin/bash
#
# run_harnesses.sh — build and run the Chronos test/benchmark harnesses.
#
# Usage:
#   ./scripts/bash/run_harnesses.sh [--arch both|x86_64|arm64]
#                                   [--bench]
#                                   [--baseline <dir>]
#
# Configures -DBUILD_TEST_HARNESSES=ON -DBUILD_AUDIO_PLUGIN_HOST=OFF into a
# per-arch build directory (build-<arch>), builds every harness target, runs
# each, tees stdout to tests/logs/<arch>/<name>.txt, and returns non-zero if
# any correctness harness exits non-zero.
#
#   --arch both|x86_64|arm64   Which architecture(s) to build and run.
#                              macOS only: uses -DCMAKE_OSX_ARCHITECTURES and
#                              `arch -<arch>` to select the slice. Default: both.
#   --bench                    Also run the three *_bench targets (tan_bench,
#                              adaa_bench, delay_line_bench) and print their
#                              output. Benches are not gated (informational).
#   --baseline <dir>           Diff each harness stdout against the recorded
#                              baseline in <dir>/<arch>/<name>.txt. Reports
#                              drift; does not fail on diff (baselines capture
#                              machine-dependent numbers).
#
# Exit code: 0 if all correctness harnesses pass on all requested arches;
#            non-zero otherwise.

set -euo pipefail

ARCH="both"
BENCH=0
BASELINE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)     ARCH="$2"; shift 2;;
        --bench)    BENCH=1; shift;;
        --baseline) BASELINE_DIR="$2"; shift 2;;
        -h|--help)  sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0;;
        *)          echo "Unknown option: $1" >&2; exit 2;;
    esac
done

# ── Resolve the repo root (this script lives in scripts/bash/) ────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── The 11 correctness harnesses (gated) and 3 benchmarks (informational) ─
CORRECTNESS_HARNESSES=(
    ring_buffer_check halfsample_fir_check short_delay_check
    align_check latency_null_check simd_delay_check
    dilog_check nonlinearity_check adaa2_check
    alias_check simd_delay_parity
)
BENCH_HARNESSES=(tan_bench adaa_bench delay_line_bench)
ALL_HARNESSES=("${CORRECTNESS_HARNESSES[@]}" "${BENCH_HARNESSES[@]}")

# ── Determine which arches to run ─────────────────────────────────────────
if [[ "$ARCH" == "both" ]]; then
    ARCHES=(x86_64 arm64)
else
    ARCHES=("$ARCH")
fi

# ── Non-macOS: ignore --arch (single-arch build, run natively) ────────────
if [[ "$(uname)" != "Darwin" ]]; then
    ARCHES=(native)
fi

LOGS_BASE="tests/logs"
ANY_FAIL=0

run_arch() {
    local arch="$1"
    local build_dir
    local arch_flag=""
    local run_prefix=""

    if [[ "$arch" == "native" ]]; then
        build_dir="build"
    else
        build_dir="build-$arch"
        arch_flag="-DCMAKE_OSX_ARCHITECTURES=$arch"
        run_prefix="arch -$arch"
    fi

    local log_dir="$LOGS_BASE/$arch"
    mkdir -p "$log_dir"

    echo "========================================================"
    echo "  Arch: $arch  (build: $build_dir)"
    echo "========================================================"

    # ── Configure ─────────────────────────────────────────────────────────
    echo "[configure] cmake -S . -B $build_dir $arch_flag ..."
    if ! cmake -S . -B "$build_dir" \
            -DBUILD_TEST_HARNESSES=ON \
            -DBUILD_AUDIO_PLUGIN_HOST=OFF \
            $arch_flag 2>&1 | tee "$log_dir/_configure.txt"; then
        echo "CONFIGURE FAILED for $arch" >&2
        ANY_FAIL=1
        return
    fi

    # ── Build all harness targets ─────────────────────────────────────────
    local targets=("${ALL_HARNESSES[@]}")
    echo "[build] cmake --build $build_dir --target ${targets[*]} ..."
    if ! cmake --build "$build_dir" --target "${targets[@]}" -j 2>&1 | tee "$log_dir/_build.txt"; then
        echo "BUILD FAILED for $arch" >&2
        ANY_FAIL=1
        return
    fi

    # ── Run correctness harnesses (gated) ─────────────────────────────────
    local harness_fail=0
    for name in "${CORRECTNESS_HARNESSES[@]}"; do
        local bin="$build_dir/tests/$name"
        local out="$log_dir/$name.txt"
        echo -n "[run] $name ... "
        if $run_prefix "$bin" > "$out" 2>&1; then
            echo "PASS"
        else
            echo "FAIL (exit $?)"
            harness_fail=1
            ANY_FAIL=1
        fi

        # Baseline diff (report only, don't gate)
        if [[ -n "$BASELINE_DIR" && -f "$BASELINE_DIR/$arch/$name.txt" ]]; then
            if ! diff -q "$BASELINE_DIR/$arch/$name.txt" "$out" >/dev/null 2>&1; then
                echo "       (differs from baseline $BASELINE_DIR/$arch/$name.txt)"
            fi
        fi
    done

    # ── Run benchmarks (informational) ────────────────────────────────────
    if [[ "$BENCH" -eq 1 ]]; then
        for name in "${BENCH_HARNESSES[@]}"; do
            local bin="$build_dir/tests/$name"
            local out="$log_dir/$name.txt"
            echo -n "[bench] $name ... "
            if $run_prefix "$bin" > "$out" 2>&1; then
                echo "done"
            else
                echo "exit $? (benches are informational)"
            fi
        done
    fi

    echo "[result] $arch: $harness_fail correctness harness(es) failed"
}

# ── Run each requested arch ───────────────────────────────────────────────
for arch in "${ARCHES[@]}"; do
    run_arch "$arch"
done

# ── Summary ──────────────────────────────────────────────────────────────
echo "========================================================"
if [[ "$ANY_FAIL" -eq 0 ]]; then
    echo "  ALL PASS — all correctness harnesses green on all arches"
    exit 0
else
    echo "  FAILURES — see logs in $LOGS_BASE/"
    exit 1
fi

#!/usr/bin/env python3
"""Benchmark regression gate for the Chronos perf harnesses.

Compares the ns_per_sample of a current bench run against a committed baseline.
A regression is a current value above baseline * (1 + tolerance/100).

Two modes:

  1. Direct compare (the owner runs this locally on a pinned core):
       bench_gate.py <baseline.json> <current.json> --tolerance 15
     The owner generates the current file with `<bench> --json current.json`.

  2. CI mode (the CTest bench_gate target):
       bench_gate.py --ci --baselines <dir> --bindir <dir> --tolerance 25
     Runs every bench with --json, then compares each against its baseline.

Baseline JSON format (produced by `<bench> --json <path> [--provisional]`):
  {
    "provisional": true,
    "records": [
      {"name": "std::tan", "config": "", "ns_per_sample": 4.4},
      ...
    ]
  }

Records are matched by (name, config). A current record with no matching
baseline is a warning, not a regression. Exit 0 = no regressions, non-zero =
at least one regression (or a bench that crashed).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# The benches bench_gate --ci runs, and the baseline file each maps to.
BENCHES = [
    "tan_bench",
    "adaa_bench",
    "delay_line_bench",
    "chain_bench",
    "fb_bench",
    "diffuser_bench",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compare(baseline, current, tol_pct):
    """Return (regressions, warnings) lists. A regression is a current
    ns_per_sample above baseline*(1+tol/100)."""
    base = {
        (r["name"], r["config"]): r["ns_per_sample"]
        for r in baseline.get("records", [])
    }
    limit = 1.0 + tol_pct / 100.0
    regressions = []
    warnings = []
    for r in current.get("records", []):
        key = (r["name"], r["config"])
        cur = float(r["ns_per_sample"])
        if key not in base:
            warnings.append((r["name"], r["config"], None, cur))
            continue
        b = float(base[key])
        if b <= 0.0:
            continue
        if cur > b * limit:
            regressions.append((r["name"], r["config"], b, cur, cur / b))
    return regressions, warnings


def report(regressions, warnings, label=""):
    for n, c, b, cur, ratio in regressions:
        print(
            f"REGRESSION {label}{n} [{c}]: {cur:.3f} > {b:.3f} ns "
            f"({ratio:.2f}x)"
        )
    for n, c, b, cur in warnings:
        print(f"WARN {label}{n} [{c}]: no baseline (cur {cur:.3f} ns)")


def run_ci(baselines_dir, bindir, tol_pct):
    any_fail = False
    for name in BENCHES:
        base_path = baselines_dir / f"{name}.json"
        exe = bindir / name
        if not base_path.exists():
            print(f"skip {name}: no baseline at {base_path}")
            continue
        if not exe.exists():
            print(f"skip {name}: no executable at {exe}")
            continue
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            r = subprocess.run(
                [str(exe), "--json", tmp],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"FAIL {name}: bench exited {r.returncode}")
                if r.stderr:
                    print(r.stderr[-500:])
                any_fail = True
                continue
            reg, warn = compare(load_json(base_path), load_json(tmp), tol_pct)
            report(reg, warn, label=f"{name}/")
            if reg:
                any_fail = True
        finally:
            os.unlink(tmp)
    if any_fail:
        print("bench_gate: REGRESSIONS DETECTED")
        return 1
    print(f"bench_gate: no regressions (tolerance {tol_pct}%)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", nargs="?", help="baseline JSON file")
    ap.add_argument("current", nargs="?", help="current JSON file")
    ap.add_argument("--tolerance", type=float, default=15.0,
                    help="regression threshold in percent (default 15)")
    ap.add_argument("--ci", action="store_true",
                    help="run every bench and compare against its baseline")
    ap.add_argument("--baselines", type=Path, help="baseline directory (--ci)")
    ap.add_argument("--bindir", type=Path, help="bench executable directory (--ci)")
    args = ap.parse_args()

    if args.ci:
        if not args.baselines or not args.bindir:
            ap.error("--ci requires --baselines and --bindir")
        return run_ci(args.baselines, args.bindir, args.tolerance)

    if not args.baseline or not args.current:
        ap.error("provide <baseline> <current>, or use --ci")

    reg, warn = compare(load_json(args.baseline), load_json(args.current),
                        args.tolerance)
    report(reg, warn)
    if reg:
        print(f"bench_gate: {len(reg)} REGRESSION(S) (tolerance {args.tolerance}%)")
        return 1
    print(f"bench_gate: no regressions ({len(reg)} of "
          f"{len(load_json(args.current).get('records', []))} records, "
          f"tolerance {args.tolerance}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

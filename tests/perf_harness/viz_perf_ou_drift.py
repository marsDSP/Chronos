#!/usr/bin/env python3
"""viz_perf_ou_drift.py

Two-panel bar chart for perf_ou_drift.csv:
  [1] ns/sample by (block_size, amount)
  [2] realtime factor (1 core at 48 kHz) on log scale

Writes tests/perf_harness/logs/perf_ou_drift.png.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR  = Path(__file__).resolve().parent / "logs"
CSV_PATH = LOG_DIR / "perf_ou_drift.csv"
OUT_PATH = LOG_DIR / "perf_ou_drift.png"

COLOUR_ACCENT = "#1971c2"
COLOUR_SECOND = "#c92a2a"
COLOUR_MUTED  = "#495057"


def load_rows() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"missing CSV: {CSV_PATH}\n"
            "Run ./cmake-build-debug/tests/perf_harness/perf_ou_drift_test first."
        )
    with CSV_PATH.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    labels     = [f"block={r['block_size']}\namt={r['amount']}" for r in rows]
    ns_values  = np.asarray([float(r["ns_per_sample"])    for r in rows])
    rt_values  = np.asarray([float(r["realtime_factor"])  for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("OU drift perf  (ns/sample, realtime factor)",
                 fontsize=13, weight="bold")

    # Panel 1: ns/sample
    ax = axes[0]
    bars = ax.bar(labels, ns_values, color=COLOUR_ACCENT,
                  edgecolor="black", linewidth=0.4)
    for bar, value in zip(bars, ns_values):
        ax.annotate(f"{value:.2f} ns",
                    xy=(bar.get_x() + bar.get_width() * 0.5,
                        bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=COLOUR_MUTED)
    ax.set_title("[1] ns per sample (lower is faster)", fontsize=11)
    ax.set_ylabel("ns / sample")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(axis="x", labelsize=8)

    # Panel 2: realtime factor
    ax = axes[1]
    bars = ax.bar(labels, rt_values, color=COLOUR_SECOND,
                  edgecolor="black", linewidth=0.4)
    for bar, value in zip(bars, rt_values):
        ax.annotate(f"{value:.0f}x",
                    xy=(bar.get_x() + bar.get_width() * 0.5,
                        bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=COLOUR_MUTED)
    ax.set_yscale("log")
    ax.set_title("[2] realtime factor at 48 kHz (higher is more headroom)",
                 fontsize=11)
    ax.set_ylabel("realtime factor (log)")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    ax.tick_params(axis="x", labelsize=8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

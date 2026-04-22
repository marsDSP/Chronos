#!/usr/bin/env python3
"""viz_perf_chronos_reverb.py

Two-panel bar chart for perf_chronos_reverb.csv:
  [1] ns/sample per configuration
  [2] realtime factor per configuration (log y)

Writes tests/perf_harness/logs/perf_chronos_reverb.png.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR  = Path(__file__).resolve().parent / "logs"
CSV_PATH = LOG_DIR / "perf_chronos_reverb.csv"
OUT_PATH = LOG_DIR / "perf_chronos_reverb.png"

COLOUR_ACCENT = "#1971c2"
COLOUR_SECOND = "#c92a2a"
COLOUR_MUTED  = "#495057"


def load_rows() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"missing CSV: {CSV_PATH}\n"
            "Run ./cmake-build-debug/tests/perf_harness/perf_chronos_reverb_test first."
        )
    with CSV_PATH.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    labels    = [r["label"].replace("_", " ") for r in rows]
    ns_values = np.asarray([float(r["ns_per_sample"])    for r in rows])
    rt_values = np.asarray([float(r["realtime_factor"])  for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("ChronosReverb perf  (ns/sample, realtime factor)",
                 fontsize=13, weight="bold")

    # Panel 1: ns/sample (horizontal bars so long labels fit)
    ax = axes[0]
    bars = ax.barh(labels, ns_values,
                   color=COLOUR_ACCENT, edgecolor="black", linewidth=0.4)
    max_width = float(ns_values.max())
    ax.set_xlim(0.0, max_width * 1.3)
    for bar, value in zip(bars, ns_values):
        ax.text(bar.get_width() + max_width * 0.01,
                bar.get_y() + bar.get_height() * 0.5,
                f"{value:.1f} ns",
                va="center", ha="left",
                fontsize=9, color=COLOUR_MUTED)
    ax.set_title("[1] ns per sample (lower is faster)", fontsize=11)
    ax.set_xlabel("ns / sample")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.tick_params(axis="y", labelsize=9)

    # Panel 2: realtime factor
    ax = axes[1]
    bars = ax.barh(labels, rt_values,
                   color=COLOUR_SECOND, edgecolor="black", linewidth=0.4)
    ax.set_xscale("log")
    for bar, value in zip(bars, rt_values):
        ax.text(bar.get_width(),
                bar.get_y() + bar.get_height() * 0.5,
                f"  {value:.0f}x",
                va="center", ha="left",
                fontsize=9, color=COLOUR_MUTED)
    ax.set_title("[2] realtime factor at 48 kHz (higher is more headroom)",
                 fontsize=11)
    ax.set_xlabel("realtime factor (log)")
    ax.grid(True, axis="x", which="both", linestyle=":", alpha=0.4)
    ax.tick_params(axis="y", labelsize=9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

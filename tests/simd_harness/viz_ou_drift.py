#!/usr/bin/env python3
"""
viz_ou_drift.py
---------------

Render a four-panel dashboard for the OU drift characterisation CSVs
produced by ou_drift_characterisation_test. Panels:

  [1] stdev / peak absolute value vs amount
  [2] Autocorrelation vs lag per amount
  [3] Distribution histogram overlay (all amount values, log y)
  [4] Mean-reversion decay trace (100 ms kick followed by 500 ms decay
      at amount = 0)

Writes tests/simd_harness/logs/ou_drift.png.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR   = REPO_ROOT / "tests" / "simd_harness" / "logs"
OUT_PATH  = LOG_DIR / "ou_drift.png"

STATS_CSV          = LOG_DIR / "ou_drift_stats.csv"
AUTOCORR_CSV       = LOG_DIR / "ou_drift_autocorr.csv"
DISTRIBUTION_CSV   = LOG_DIR / "ou_drift_distribution.csv"
MEAN_REVERSION_CSV = LOG_DIR / "ou_drift_mean_reversion.csv"

COLOUR_ACCENT   = "#1971c2"
COLOUR_SECOND   = "#c92a2a"
COLOUR_MUTED    = "#495057"
AMOUNT_CMAP     = plt.get_cmap("viridis")


def require(path: Path):
    if not path.exists():
        raise SystemExit(
            f"missing CSV: {path}\n"
            "Run ./cmake-build-debug/tests/simd_harness/ou_drift_characterisation_test "
            "first."
        )


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------
# [1] stdev / peak vs amount
# ----------------------------------------------------------------------
def draw_stats_panel(axis, stats_rows: list[dict]) -> None:
    amounts  = np.asarray([float(r["amount"])   for r in stats_rows])
    stdevs   = np.asarray([float(r["stdev"])    for r in stats_rows])
    peaks    = np.asarray([float(r["peak_abs"]) for r in stats_rows])

    axis.plot(amounts, stdevs, marker="o", color=COLOUR_ACCENT,
              label="stdev", linewidth=1.8)
    axis.plot(amounts, peaks,  marker="s", color=COLOUR_SECOND,
              label="peak |value|", linewidth=1.8)

    for x, y in zip(amounts, stdevs):
        axis.annotate(f"{y:.4f}",
                      xy=(x, y), xytext=(4, 4),
                      textcoords="offset points",
                      fontsize=8, color=COLOUR_ACCENT)
    for x, y in zip(amounts, peaks):
        axis.annotate(f"{y:.4f}",
                      xy=(x, y), xytext=(4, -10),
                      textcoords="offset points",
                      fontsize=8, color=COLOUR_SECOND)

    axis.set_xlabel("OU amount")
    axis.set_ylabel("output value")
    axis.set_title("[1] OU output magnitude vs amount knob", fontsize=11)
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(loc="upper left", fontsize=9)


# ----------------------------------------------------------------------
# [2] autocorrelation vs lag per amount
# ----------------------------------------------------------------------
def draw_autocorr_panel(axis, autocorr_rows: list[dict]) -> None:
    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in autocorr_rows:
        grouped[float(row["amount"])].append(
            (float(row["lag_ms"]), float(row["autocorr"]))
        )

    amounts = sorted(grouped.keys())
    for idx, amount in enumerate(amounts):
        points = sorted(grouped[amount], key=lambda p: p[0])
        lags     = np.asarray([p[0] for p in points])
        autocorr = np.asarray([p[1] for p in points])
        axis.plot(lags, autocorr, marker="o",
                  color=AMOUNT_CMAP(idx / max(1, len(amounts) - 1)),
                  linewidth=1.3, label=f"amount = {amount:.2f}")

    axis.axhline(1.0 / np.e, color=COLOUR_MUTED, linestyle="--", linewidth=0.8,
                 label="1/e (effective τ)")
    axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    axis.set_xscale("log")
    axis.set_xlabel("lag (ms, log)")
    axis.set_ylabel("normalised autocorrelation")
    axis.set_title("[2] OU autocorrelation vs lag", fontsize=11)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=8, loc="upper right")


# ----------------------------------------------------------------------
# [3] distribution histogram overlay (all amounts)
# ----------------------------------------------------------------------
def draw_distribution_panel(axis, distribution_rows: list[dict]) -> None:
    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in distribution_rows:
        grouped[float(row["amount"])].append(
            (float(row["bin_centre"]), float(row["count"]))
        )

    amounts = sorted(grouped.keys())
    for idx, amount in enumerate(amounts):
        if amount == 0.0:
            continue  # amount=0 is a spike at 0, not interesting here
        points = sorted(grouped[amount], key=lambda p: p[0])
        centres = np.asarray([p[0] for p in points])
        counts  = np.asarray([p[1] for p in points])
        axis.step(centres, counts, where="mid",
                  color=AMOUNT_CMAP(idx / max(1, len(amounts) - 1)),
                  linewidth=1.3, label=f"amount = {amount:.2f}")

    axis.set_yscale("log")
    axis.set_xlabel("output value bin centre")
    axis.set_ylabel("count (log)")
    axis.set_title("[3] OU output distribution per amount", fontsize=11)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=8, loc="upper right")


# ----------------------------------------------------------------------
# [4] mean-reversion decay trace
# ----------------------------------------------------------------------
def draw_mean_reversion_panel(axis, decay_rows: list[dict]) -> None:
    times  = np.asarray([float(r["time_ms"]) for r in decay_rows])
    values = np.asarray([float(r["value"])   for r in decay_rows])

    axis.plot(times, values, color=COLOUR_ACCENT, linewidth=1.4)
    axis.axhline(0.0, color=COLOUR_MUTED, linewidth=0.6, linestyle="--")
    axis.set_xlabel("time since amount→0 kick (ms)")
    axis.set_ylabel("OU state value")
    axis.set_title("[4] Mean-reversion decay (amount driven to 0 at t=0)",
                   fontsize=11)
    axis.grid(True, linestyle=":", alpha=0.5)

    # Annotate 1/e point if present.
    if values.size > 0:
        initial = values[0]
        target  = initial / np.e
        if abs(initial) > 1e-9:
            crossing = np.argmax(np.abs(values) < abs(target)) if np.any(np.abs(values) < abs(target)) else None
            if crossing:
                cross_time = times[crossing]
                axis.axvline(cross_time, color=COLOUR_SECOND,
                             linewidth=0.9, linestyle=":",
                             label=f"1/e at {cross_time:.1f} ms")
                axis.legend(fontsize=8, loc="upper right")


def main() -> None:
    require(STATS_CSV)
    require(AUTOCORR_CSV)
    require(DISTRIBUTION_CSV)
    require(MEAN_REVERSION_CSV)

    stats_rows        = load_rows(STATS_CSV)
    autocorr_rows     = load_rows(AUTOCORR_CSV)
    distribution_rows = load_rows(DISTRIBUTION_CSV)
    decay_rows        = load_rows(MEAN_REVERSION_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "OU drift characterisation  "
        "(Ornstein-Uhlenbeck output across amount sweep)",
        fontsize=14, weight="bold",
    )

    draw_stats_panel          (axes[0, 0], stats_rows)
    draw_autocorr_panel       (axes[0, 1], autocorr_rows)
    draw_distribution_panel   (axes[1, 0], distribution_rows)
    draw_mean_reversion_panel (axes[1, 1], decay_rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

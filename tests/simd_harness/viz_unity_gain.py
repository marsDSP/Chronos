#!/usr/bin/env python3
"""
viz_unity_gain.py
-----------------

Render a four-panel dashboard for tests/simd_harness/logs/unity_gain.csv
produced by unity_gain_test. Each panel shows one amount-sweep's
deviation from the dry reference RMS (in dB), with a shaded tolerance
band and per-point PASS / FAIL markers.

Designed to sit alongside viz_diffusion.py / viz_splane_filter_response.py
so the whole simd_harness dashboard family has the same look.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH  = REPO_ROOT / "tests" / "simd_harness" / "logs" / "unity_gain.csv"
OUT_PATH  = REPO_ROOT / "tests" / "simd_harness" / "logs" / "unity_gain.png"

# Panel ordering and titles. Wow/flutter modulation sweep first, then
# the reverb send sweep, then the raw delay mix sweep.
PANEL_ORDER = [
    ("wow_flutter_sweep",   "Wow + flutter depth  (mix = 1,  reverb mix = 0)"),
    ("reverb_mix_sweep",    "Reverb mix           (mix = 1,  modulation off)"),
    ("mix_sweep",           "Dry/wet mix          (reverb / modulation off)"),
]


def load_rows() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"CSV not found: {CSV_PATH}\n"
            "Run ./cmake-build-debug/tests/simd_harness/unity_gain_test first."
        )

    with CSV_PATH.open() as handle:
        return list(csv.DictReader(handle))


def group_by_sweep(rows: list[dict]) -> dict[str, list[dict]]:
    sweeps: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sweeps[row["sweep"]].append(row)
    for key in sweeps:
        sweeps[key].sort(key=lambda r: float(r["parameter_value"]))
    return sweeps


def draw_panel(axis, sweep_name: str, sweep_title: str, sweep_rows: list[dict]) -> None:
    if not sweep_rows:
        axis.text(0.5, 0.5, f"(no data for {sweep_name})",
                  ha="center", va="center", transform=axis.transAxes,
                  color="gray")
        axis.set_title(sweep_title)
        return

    parameter_values   = np.asarray([float(r["parameter_value"]) for r in sweep_rows])
    deviation_decibels = np.asarray([float(r["deviation_db"])    for r in sweep_rows])
    tolerance_decibels = float(sweep_rows[0]["tolerance_db"])
    pass_mask          = np.asarray([int(r["passed"]) == 1 for r in sweep_rows])

    axis.axhspan(-tolerance_decibels, +tolerance_decibels,
                 color="#4c9f70", alpha=0.12,
                 label=f"+/-{tolerance_decibels:.1f} dB tolerance")
    axis.axhline(0.0, color="#4c9f70", linewidth=0.8, linestyle="--")

    # Bars: green for PASS, red for FAIL.
    bar_colors = np.where(pass_mask, "#4c9f70", "#d2473a")
    axis.bar(parameter_values, deviation_decibels,
             width=0.08, color=bar_colors, edgecolor="black", linewidth=0.4)

    # Annotate deviation values above / below each bar.
    for x_value, y_value in zip(parameter_values, deviation_decibels):
        va = "bottom" if y_value >= 0.0 else "top"
        offset = 0.15 if y_value >= 0.0 else -0.15
        axis.annotate(f"{y_value:+.2f} dB",
                      xy=(x_value, y_value),
                      xytext=(x_value, y_value + offset),
                      ha="center", va=va, fontsize=8, color="#222")

    axis.set_xlim(-0.1, 1.1)
    symmetric_limit = max(6.0, float(np.abs(deviation_decibels).max()) + 1.0)
    axis.set_ylim(-symmetric_limit, +symmetric_limit)
    axis.set_title(sweep_title, fontsize=11)
    axis.set_xlabel("parameter value")
    axis.set_ylabel("deviation from dry reference (dB)")
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    axis.legend(loc="lower left", fontsize=8)


def draw_combined_worst_case(figure, combined_rows: list[dict]) -> None:
    if not combined_rows:
        return
    record = combined_rows[0]
    deviation = float(record["deviation_db"])
    tolerance = float(record["tolerance_db"])
    passed    = int(record["passed"]) == 1

    verdict_color = "#4c9f70" if passed else "#d2473a"
    verdict_label = "PASS" if passed else "FAIL"

    figure.text(
        0.5, 0.015,
        f"combined worst-case  (mix=1, wow+flutter=1, reverbMix=0.5):  "
        f"{deviation:+.2f} dB  [tolerance +/-{tolerance:.1f} dB]  -> {verdict_label}",
        ha="center", va="bottom",
        fontsize=11, color=verdict_color,
        bbox=dict(facecolor="white", edgecolor=verdict_color, linewidth=1.0, boxstyle="round,pad=0.35"),
    )


def main() -> None:
    rows = load_rows()
    sweeps = group_by_sweep(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle("Chronos delay-engine unity-gain sweeps  (wideband white-noise input)",
                 fontsize=14, weight="bold")

    flat_axes = axes.flatten()
    for axis, (sweep_name, sweep_title) in zip(flat_axes, PANEL_ORDER):
        draw_panel(axis, sweep_name, sweep_title, sweeps.get(sweep_name, []))

    draw_combined_worst_case(fig, sweeps.get("combined_worst_case", []))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

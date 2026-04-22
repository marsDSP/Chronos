#!/usr/bin/env python3
"""Chronos session perf dashboard.

Aggregates the three new perf harnesses into one PNG:

  - s-plane filter throughput vs juce::dsp::IIR
  - AlignedSIMDBuffer vs std::vector vs float[] (write+read, memcpy)
  - DiffusionChain / DiffusionProcessor / FDN / FdnStereoProcessor / full tail

Output:
  tests/perf_harness/logs/session_perf.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
OUT_PATH = os.path.join(LOG_DIR, "session_perf.png")

MUTED      = "#495057"
GOOD       = "#2b8a3e"
BAD        = "#c92a2a"
JUCE_C     = "#868e96"
CHRONOS_C  = "#1971c2"
OPT_C1     = "#e64980"
OPT_C2     = "#f59f00"


def must_read(name):
    p = os.path.join(LOG_DIR, name)
    if not os.path.exists(p):
        print(f"missing CSV: {p}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(p)


# -------------------------------------------------- splane filter panel
def panel_splane(ax, df):
    ax.set_title("[1] s-plane HPF+LPF vs juce::dsp::IIR (lower = faster)",
                 fontsize=11)
    bars = ax.barh(df["filter"], df["ns_per_sample"],
                   color=[JUCE_C, CHRONOS_C], edgecolor="black", linewidth=0.4)
    for bar_patch, ns in zip(bars, df["ns_per_sample"]):
        ax.text(bar_patch.get_width(),
                bar_patch.get_y() + bar_patch.get_height() * 0.5,
                f"  {ns:.2f} ns/sample",
                va="center", ha="left", fontsize=9, color=MUTED)
    ax.set_xlabel("ns per sample")
    ax.grid(True, axis="x", alpha=0.3)
    speedup_row = df.loc[df["speedup_vs_juce"] != 1.0]
    if len(speedup_row):
        speedup_value = speedup_row["speedup_vs_juce"].iloc[0]
        ax.text(0.98, 0.07,
                f"Chronos vs JUCE: {speedup_value:.2f}x",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          edgecolor=CHRONOS_C))


# -------------------------------------------------- aligned buffer panel
def panel_aligned_buffer(ax, df):
    ax.set_title("[2] AlignedSIMDBuffer vs std::vector vs float[] (write+read)",
                 fontsize=11)
    for container, color in (("float[]", JUCE_C),
                              ("std::vector<float>", OPT_C1),
                              ("AlignedSIMDBuffer", CHRONOS_C)):
        sub = df[df["container"] == container].sort_values("num_samples")
        ax.plot(sub["num_samples"], sub["write_read_us"],
                marker="o", color=color, label=container, linewidth=1.4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("samples per channel (log)")
    ax.set_ylabel("us per write+read sweep (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def panel_aligned_buffer_memcpy(ax, df):
    ax.set_title("[3] AlignedSIMDBuffer memcpy throughput", fontsize=11)
    for container, color in (("float[]", JUCE_C),
                              ("std::vector<float>", OPT_C1),
                              ("AlignedSIMDBuffer", CHRONOS_C)):
        sub = df[df["container"] == container].sort_values("num_samples")
        ax.plot(sub["num_samples"], sub["memcpy_us"],
                marker="s", color=color, label=container, linewidth=1.4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("samples per channel (log)")
    ax.set_ylabel("us per memcpy (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


# -------------------------------------------------- diffusion / FDN panel
def panel_diffusion(ax, df):
    ax.set_title("[4] Diffusion / FDN ns per sample  (lower = faster)",
                 fontsize=11)
    # Short y-axis labels (processor name only) so text stays inside the
    # panel; the configuration string goes inside the bar as an annotation.
    short_labels = [str(row.processor) for row in df.itertuples(index=False)]
    colors = []
    for i in range(len(df)):
        colors.append([CHRONOS_C, OPT_C1, CHRONOS_C, OPT_C1, OPT_C2][i])
    bars = ax.barh(short_labels, df["ns_per_sample"],
                   color=colors, edgecolor="black", linewidth=0.4)
    # Give the longest label room without spilling into the neighbouring
    # axes: expand the x-axis so the per-bar annotation fits on the right.
    max_width = float(df["ns_per_sample"].max())
    ax.set_xlim(0.0, max_width * 1.45)

    for bar_patch, row in zip(bars, df.itertuples(index=False)):
        ns = float(row.ns_per_sample)
        rt = float(row.realtime_factor)
        cfg = str(row.configuration)
        # Right of the bar: ns + realtime factor.
        ax.text(bar_patch.get_width() + max_width * 0.01,
                bar_patch.get_y() + bar_patch.get_height() * 0.5,
                f"{ns:.1f} ns  ({rt:.0f}x RT)",
                va="center", ha="left", fontsize=8, color=MUTED)
        # Inside the bar, at the left: the configuration label (white,
        # small) so the detail isn't lost when labels are shortened.
        ax.text(max_width * 0.01,
                bar_patch.get_y() + bar_patch.get_height() * 0.5,
                cfg,
                va="center", ha="left", fontsize=7, color="white",
                fontweight="bold")
    ax.set_xlabel("ns per sample")
    ax.grid(True, axis="x", alpha=0.3)
    # Keep y-tick label text inside the panel bounds.
    ax.tick_params(axis="y", labelsize=9)


def panel_realtime_factor(ax, df):
    ax.set_title("[5] Diffusion / FDN real-time headroom (higher = more headroom)",
                 fontsize=11)
    labels = [row.processor for row in df.itertuples(index=False)]
    realtime_values = df["realtime_factor"].tolist()
    colors = [CHRONOS_C, OPT_C1, CHRONOS_C, OPT_C1, OPT_C2]
    bars = ax.bar(labels, realtime_values, color=colors,
                  edgecolor="black", linewidth=0.4)
    for bar_patch, rt in zip(bars, realtime_values):
        ax.text(bar_patch.get_x() + bar_patch.get_width() * 0.5,
                bar_patch.get_height(),
                f"{rt:.0f}x",
                ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylabel("realtime factor (log)  @ 48 kHz")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)


def main():
    splane_df   = must_read("perf_splane_filter.csv")
    buffer_df   = must_read("perf_aligned_buffer.csv")
    diffusion_df = must_read("perf_diffusion.csv")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Chronos session perf dashboard  —  s-plane filter, aligned buffer, diffusion/FDN",
                 fontsize=15, fontweight="bold", color=MUTED, y=0.995)

    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)

    ax_splane          = fig.add_subplot(gs[0, 0])
    ax_buffer          = fig.add_subplot(gs[0, 1])
    ax_buffer_memcpy   = fig.add_subplot(gs[1, 0])
    ax_diffusion       = fig.add_subplot(gs[1, 1])
    ax_realtime        = fig.add_subplot(gs[2, :])

    panel_splane          (ax_splane, splane_df)
    panel_aligned_buffer  (ax_buffer, buffer_df)
    panel_aligned_buffer_memcpy(ax_buffer_memcpy, buffer_df)
    panel_diffusion       (ax_diffusion, diffusion_df)
    panel_realtime_factor (ax_realtime, diffusion_df)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PATH, dpi=130)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

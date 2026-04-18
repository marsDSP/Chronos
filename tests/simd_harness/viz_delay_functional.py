#!/usr/bin/env python3
"""Chronos DelayEngine functional test dashboard.

Reads the seven CSVs produced by delay_functional_test and renders a
single-page dashboard with per-test pass/fail indicators and detailed
evidence for each case.

CSVs expected (under tests/simd_harness/logs):
  func_block_sizes.csv
  func_varying_blocks.csv
  func_extreme_params.csv
  func_silence_tail.csv
  func_ringout.csv
  func_mode_switch.csv
  func_reset.csv
  func_streaming.csv
  func_summary.csv

Output:
  tests/simd_harness/logs/delay_functional.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

LOG_DIR = "tests/simd_harness/logs"
OUT_PATH = os.path.join(LOG_DIR, "delay_functional.png")

PASS_COLOR = "#2b8a3e"
FAIL_COLOR = "#c92a2a"
MUTED      = "#495057"


def must_read(name):
    p = os.path.join(LOG_DIR, name)
    if not os.path.exists(p):
        print(f"missing CSV: {p}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(p)


def status_label(ax, passed: bool, where="upper right"):
    """Draw a pass/fail pill in a corner of the axes."""
    text = "PASS" if passed else "FAIL"
    color = PASS_COLOR if passed else FAIL_COLOR
    kwargs = {
        "transform": ax.transAxes,
        "fontsize": 10, "fontweight": "bold",
        "color": "white",
        "bbox": dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="none"),
    }
    if where == "upper right":
        ax.text(0.98, 0.92, text, ha="right", va="center", **kwargs)
    else:
        ax.text(0.02, 0.92, text, ha="left", va="center", **kwargs)


# -------------- panels ----------------------------------------------------
def panel_summary(ax, summary_df):
    passed = int(summary_df["passed"].iloc[0])
    failed = int(summary_df["failed"].iloc[0])
    total  = int(summary_df["total"].iloc[0])
    ax.set_axis_off()
    headline = "ALL TESTS PASS" if failed == 0 else f"{failed} TESTS FAILED"
    color = PASS_COLOR if failed == 0 else FAIL_COLOR
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02",
        linewidth=0, facecolor=color, alpha=0.12,
        transform=ax.transAxes))
    ax.text(0.5, 0.72, "Chronos DelayEngine — Functional Test Matrix",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=MUTED)
    ax.text(0.5, 0.38, headline,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=22, fontweight="bold", color=color)
    ax.text(0.5, 0.12,
            f"Passed {passed} / {total}   •   Failed {failed}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color=MUTED)


def panel_block_sizes(ax, df):
    df = df.sort_values("block_size")
    xs = np.arange(len(df))
    colors = [PASS_COLOR if p == 1 else FAIL_COLOR for p in df["passed"]]
    ax.bar(xs, df["max_peak"], color=colors, edgecolor="black", linewidth=0.4,
           label="max peak")
    ax.bar(xs, df["mean_peak"], color=colors, alpha=0.35, edgecolor="none")
    ax.set_xticks(xs)
    ax.set_xticklabels(df["block_size"].astype(str), rotation=45, ha="right",
                       fontsize=8)
    ax.set_title("[1] Weird block sizes — max / mean peak", fontsize=11)
    ax.set_ylabel("peak |out|")
    ax.set_xlabel("block size (samples)")
    ax.grid(True, axis="y", alpha=0.3)
    status_label(ax, (df["passed"] == 1).all())


def panel_varying_blocks(ax, df):
    # Scatter steps with block size on y, color by peak
    ok = df["passed"] == 1
    sc = ax.scatter(df["step"], df["block_size"], c=df["peak"], cmap="viridis",
                    s=12, alpha=0.85)
    # Mark any failures prominently
    if (~ok).any():
        ax.scatter(df.loc[~ok, "step"], df.loc[~ok, "block_size"],
                   s=60, facecolor="none", edgecolor=FAIL_COLOR, linewidth=1.5,
                   label="fail")
        ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_title("[1b] Adversarial varying block sizes — 500 calls", fontsize=11)
    ax.set_xlabel("step")
    ax.set_ylabel("block size (samples, log)")
    ax.grid(True, alpha=0.3, which="both")
    plt.colorbar(sc, ax=ax, label="peak |out|")
    status_label(ax, ok.all())


def panel_extreme_params(ax, df):
    test_names = list(dict.fromkeys(df["test_name"].tolist()))
    # Share one axes; color-code per test
    cmap = plt.get_cmap("tab10")
    overall_ok = True
    for i, name in enumerate(test_names):
        sub = df[df["test_name"] == name]
        ok = (sub["passed"] == 1).all()
        overall_ok = overall_ok and ok
        label = f"{name}  {'\u2713' if ok else '\u2717'}"
        ax.plot(sub["step"], sub["peak"], color=cmap(i % 10),
                linewidth=1.1, label=label)
    ax.set_title("[2] Extreme parameter sweeps — per-block peak vs step", fontsize=11)
    ax.set_xlabel("step")
    ax.set_ylabel("peak |out|")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    status_label(ax, overall_ok, where="upper left")


def panel_silence_tail(ax, df):
    # Final row has sample_index == -1 and stores final peak summary.
    summary = df[df["sample_index"] == -1]
    curve   = df[df["sample_index"] >= 0]
    ringout = int(df["ringout_prediction"].iloc[0])
    ok      = int(summary["passed"].iloc[0]) == 1 if len(summary) else True

    eps = 1.0e-8
    ax.plot(curve["sample_index"], curve["peak"].clip(lower=eps),
            color="#1971c2", linewidth=1.3, label="measured peak")
    ax.axvline(ringout, color=FAIL_COLOR if not ok else PASS_COLOR,
               linestyle="--", linewidth=1.2,
               label=f"ringoutSamples() = {ringout}")
    ax.axhline(0.001, color="#666", linestyle=":", linewidth=0.9,
               label="-60 dB threshold")
    ax.set_yscale("log")
    ax.set_title("[3] Silence tail — peak decay with ringout prediction", fontsize=11)
    ax.set_xlabel("samples since silence began")
    ax.set_ylabel("peak |out|  (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    status_label(ax, ok, where="upper left")


def panel_ringout(ax, df):
    xs = np.arange(len(df))
    colors = [PASS_COLOR if p == 1 else FAIL_COLOR for p in df["passed"]]
    bars = ax.bar(xs, df["predicted_tail"], color=colors, edgecolor="black",
                  linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{d:.0f}ms\nfb={fb:.2f}" for d, fb in zip(df["delay_ms"], df["feedback"])],
        fontsize=9)
    ax.set_ylabel("predicted tail (samples)")
    ax.set_title("[4] ringoutSamples() — predicted vs. measured peak after tail",
                 fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Secondary axis: measured peak after the tail
    ax2 = ax.twinx()
    ax2.scatter(xs, df["measured_peak_at_tail"],
                color="#f59f00", marker="D", s=60, zorder=3,
                label="peak after tail")
    ax2.axhline(0.002, color="#666", linestyle=":", linewidth=0.9,
                label="soft-silent threshold")
    ax2.set_ylabel("peak after predicted tail", color="#f59f00")
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1e-9)
    ax2.tick_params(axis="y", labelcolor="#f59f00")
    ax2.legend(loc="upper right", fontsize=8)
    status_label(ax, (df["passed"] == 1).all(), where="upper left")


def panel_mode_switch(ax, df):
    # Peak time-series with mode-shaded regions
    ok = (df["passed"] == 1).all()
    modes = df["mode"].values
    # find runs of constant mode
    edges = [0] + [i for i in range(1, len(modes)) if modes[i] != modes[i-1]] + [len(modes)]
    palette = {"mono": "#51cf66", "stereo": "#74c0fc"}
    for a, b in zip(edges[:-1], edges[1:]):
        ax.axvspan(df["step"].iloc[a], df["step"].iloc[b-1],
                   color=palette.get(modes[a], "#dee2e6"), alpha=0.25)

    ax.plot(df["step"], df["peak"], color="#495057", linewidth=1.1)
    # Mark switch points
    sw = df[df["switched"] == 1]
    ax.scatter(sw["step"], sw["peak"], color="#c92a2a", s=22, zorder=3,
               label="mode switch")
    ax.set_title("[5] Mono / stereo toggling — peak with switch markers", fontsize=11)
    ax.set_xlabel("step")
    ax.set_ylabel("peak |out|")
    ax.grid(True, alpha=0.3)

    # Legend for the shading
    from matplotlib.patches import Patch
    handles = [Patch(color=palette["mono"], alpha=0.4, label="mono"),
               Patch(color=palette["stereo"], alpha=0.4, label="stereo")]
    handles += ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    status_label(ax, ok, where="upper left")


def panel_reset(ax, df):
    before = df[df["state"] == "before_reset"]["peak"].iloc[0]
    after  = df[df["state"] == "after_reset"]["peak"].iloc[0]
    ok     = int(df[df["state"] == "after_reset"]["passed"].iloc[0]) == 1
    xs = [0, 1]
    colors = ["#74c0fc", PASS_COLOR if ok else FAIL_COLOR]
    ax.bar(xs, [before, max(after, 1e-12)],
           color=colors, edgecolor="black", linewidth=0.5)
    for i, v in zip(xs, [before, after]):
        ax.text(i, max(v, 1e-12) * 1.05, f"{v:.2e}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(["before reset()", "after reset() + silence"])
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-12)
    ax.set_ylabel("peak |out| (log)")
    ax.set_title("[6] reset() clears delay-line state", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    status_label(ax, ok)


def panel_streaming(ax, df):
    version = int(df["streamingVersion"].iloc[0])
    ran     = int(df["remap_ran"].iloc[0]) == 1
    ok      = int(df["passed"].iloc[0]) == 1
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.1), 0.96, 0.8, boxstyle="round,pad=0.02",
        linewidth=1, edgecolor=MUTED, facecolor="#f8f9fa",
        transform=ax.transAxes))
    ax.text(0.5, 0.78, "[7] Streaming-version contract",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=MUTED)
    ax.text(0.5, 0.55, f"streamingVersion = {version}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=18, fontweight="bold",
            color=PASS_COLOR if ok else FAIL_COLOR)
    ax.text(0.5, 0.32,
            f"remapParametersForStreamingVersion {'ran cleanly' if ran else 'FAILED TO RUN'}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color=MUTED)
    status_label(ax, ok)


# -------------- main ------------------------------------------------------
def main():
    summary     = must_read("func_summary.csv")
    block_sizes = must_read("func_block_sizes.csv")
    varying     = must_read("func_varying_blocks.csv")
    extreme     = must_read("func_extreme_params.csv")
    silence     = must_read("func_silence_tail.csv")
    ringout     = must_read("func_ringout.csv")
    mode_sw     = must_read("func_mode_switch.csv")
    reset       = must_read("func_reset.csv")
    streaming   = must_read("func_streaming.csv")

    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(
        nrows=6, ncols=2, figure=fig,
        height_ratios=[0.6, 1.1, 1.3, 1.3, 1.3, 0.7],
        hspace=0.50, wspace=0.22,
        left=0.06, right=0.96, top=0.97, bottom=0.04)

    # Row 0: summary header (spans)
    ax_sum = fig.add_subplot(gs[0, :])
    panel_summary(ax_sum, summary)

    # Row 1: block sizes | varying blocks
    panel_block_sizes(fig.add_subplot(gs[1, 0]), block_sizes)
    panel_varying_blocks(fig.add_subplot(gs[1, 1]), varying)

    # Row 2: extreme params (spans)
    panel_extreme_params(fig.add_subplot(gs[2, :]), extreme)

    # Row 3: silence tail | ringout
    panel_silence_tail(fig.add_subplot(gs[3, 0]), silence)
    panel_ringout(fig.add_subplot(gs[3, 1]), ringout)

    # Row 4: mode switch | reset
    panel_mode_switch(fig.add_subplot(gs[4, 0]), mode_sw)
    panel_reset(fig.add_subplot(gs[4, 1]), reset)

    # Row 5: streaming (spans)
    panel_streaming(fig.add_subplot(gs[5, :]), streaming)

    fig.savefig(OUT_PATH, dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

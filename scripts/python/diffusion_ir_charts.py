#!/usr/bin/env python3
"""Diffuser IR before/after charts (C7c: diffuser moved into the feedback loop).

Reads the impulse-with-feedback CSVs dumped by tests/harnesses/dsp/
diffusion_ir_probe.cpp (`diffusion_ir_probe dump <dir>`) from the old
(post-loop, one-pass, median-anchored w*base comp) and new (in-loop,
centroid-anchored base comp) architectures, and renders PNGs into
tests/charts/:

  overview_before_s{5,0}.png / overview_after_s{5,0}.png
      Repeat train (impulse, feedback 0.5) across the diffusion sweep with
      n*delay grid lines. Before: every repeat is the same one-pass blob,
      anchored late/early by the drifting median comp. After: repeat n is
      n diffusion passes — the blob widens into a wash while its energy
      centroid stays on the grid.
  zoom_after.png
      Per-repeat windows around each grid point (repeats 1-4 x diff
      {0.5, 1.0}): the blob stays centered on the grid as it widens.
      The diff=1.0 repeat-1 front spike leading the grid is the documented
      centroid-anchoring trade-off (decays (g^8)^n per repeat).
  centroid_sync.png
      Aggregate (binning-free) energy-centroid offset vs the diffuser-off
      control, in ms, before vs after. This is the IN SYNC chart: the
      aggregate cannot be fooled by blob overlap (every sample counted once
      at its true position).
  width_growth.png
      Per-repeat RMS width (Voronoi bins 1..4, diff <= 0.75 where tail
      overlap is negligible): before is flat (one pass per repeat), after
      grows ~sqrt(n) (n passes per repeat). This is the WASH chart.

Run:  ./.venv/bin/python scripts/python/diffusion_ir_charts.py
"""

import argparse
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS = 48000.0
K_SETTLE = 12000
K_DELAY = 24000
DIFFS = [25, 50, 75, 100]          # diffusion * 100 (file naming)
SIZES = [5, 0]                     # size * 10 (file naming)
GATE_SYNC_SAMPLES = 32.0           # diffuser_loop_check aggregate gate


def load_csv(path: Path) -> np.ndarray:
    # Read a CSV, decompressing .csv.gz transparently. The frozen before-set is
    # committed gzip-compressed; the after-set is plain CSV from the probe.
    gz = path.with_suffix(".csv.gz")
    if gz.exists():
        return np.loadtxt(gzip.open(gz, "rt"))
    return np.loadtxt(path)


def load(d: Path, diff: int, size: int) -> np.ndarray:
    return load_csv(d / f"d{diff:03d}_s{size}.csv")


def grid_one(ref: np.ndarray) -> int:
    return int(np.argmax(np.abs(ref) > 1e-4))


def aggregate_centroid(x: np.ndarray, t0: int) -> float:
    e = x.astype(np.float64) ** 2
    idx = np.arange(len(x))
    m = idx >= t0
    return float((e[m] * idx[m]).sum() / e[m].sum())


def voronoi_widths(x: np.ndarray, g1: int, n_bins: int = 7):
    """Per-repeat RMS width via Voronoi binning around the grid points."""
    e = x.astype(np.float64) ** 2
    idx = np.arange(len(x))
    r = np.clip(np.round((idx - g1) / K_DELAY).astype(int) + 1, 1, n_bins)
    widths = {}
    for b in range(1, n_bins + 1):
        m = r == b
        E = e[m].sum()
        if E <= 0:
            continue
        c = (e[m] * idx[m]).sum() / E
        widths[b] = float(np.sqrt((e[m] * (idx[m] - c) ** 2).sum() / E))
    return widths


def overview(ax, x: np.ndarray, g1: int, title: str):
    t = (np.arange(len(x)) - g1) / FS * 1000.0
    ax.plot(t, x, lw=0.3, color="C0")
    for n in range(1, 6):
        ax.axvline(n * K_DELAY / FS * 1000.0, color="k", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlim(-30, 5.5 * K_DELAY / FS * 1000.0)
    ax.set_title(title, fontsize=8)
    ax.set_ylim(-1.05, 1.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, default=Path("tests/logs/diffusion_ir/before"))
    ap.add_argument("--after", type=Path, default=Path("tests/logs/diffusion_ir/after"))
    ap.add_argument("--out", type=Path, default=Path("tests/charts"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    refs = {tag: load_csv(d / "ref_off.csv") for tag, d in (("before", args.before), ("after", args.after))}
    g1 = grid_one(refs["after"])

    # ── 1. overview figures ──────────────────────────────────────────────
    for tag, d, label in (("before", args.before, "post-loop one-pass (OLD)"),
                          ("after", args.after, "in-loop C7c (NEW)")):
        for size in SIZES:
            fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
            for ax, diff in zip(axes.flat, DIFFS):
                overview(ax, load(d, diff, size), g1, f"diffusion {diff/100:.2f}")
            fig.suptitle(f"Chronos diffuser — {label} — impulse, feedback 0.5, size {size/10:.1f}\n"
                         f"dashed lines = n·delay grid (500 ms)", fontsize=10)
            fig.text(0.5, 0.02, "ms relative to repeat 1 grid", ha="center")
            fig.tight_layout(rect=(0, 0.03, 1, 0.94))
            fig.savefig(args.out / f"overview_{tag}_s{size}.png", dpi=140)
            plt.close(fig)
            print(f"wrote {args.out / f'overview_{tag}_s{size}.png'}")

    # ── 2. per-repeat zoom (after) ───────────────────────────────────────
    zoom_diffs = [50, 100]
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    half = int(0.150 * FS)  # ±150 ms window
    for col, diff in enumerate(zoom_diffs):
        x = load(args.after, diff, 5)
        for row in range(1, 5):
            ax = axes[row - 1][col]
            c = g1 + (row - 1) * K_DELAY
            seg = x[c - half:c + half]
            t = (np.arange(len(seg)) - half) / FS * 1000.0
            ax.plot(t, seg, lw=0.5, color="C0")
            ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
            ax.set_ylabel(f"repeat {row}", fontsize=8)
            if row == 1:
                ax.set_title(f"diffusion {diff/100:.2f} (size 0.5)", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle("In-loop diffuser (NEW): per-repeat windows, grid at 0 — blob stays centered as it widens\n"
                 "(diff 1.0 repeat 1: the front spike leading the grid is the documented trade-off, decays (g^8)^n)",
                 fontsize=9)
    fig.text(0.5, 0.02, "ms relative to grid", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(args.out / "zoom_after.png", dpi=140)
    plt.close(fig)
    print(f"wrote {args.out / 'zoom_after.png'}")

    # ── 3. aggregate centroid sync (before vs after) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    width = 0.35
    for ax, size in zip(axes, SIZES):
        rows = []
        for tag, d in (("before", args.before), ("after", args.after)):
            ref = load_csv(d / "ref_off.csv")
            c0 = aggregate_centroid(ref, K_SETTLE)
            rows.append([aggregate_centroid(load(d, diff, size), K_SETTLE) - c0
                         for diff in DIFFS])
        xpos = np.arange(len(DIFFS))
        ax.bar(xpos - width / 2, np.array(rows[0]) / FS * 1000.0, width,
               label="before (post-loop, w·base comp)", color="C3")
        ax.bar(xpos + width / 2, np.array(rows[1]) / FS * 1000.0, width,
               label="after (in-loop, base comp)", color="C2")
        ax.axhline(GATE_SYNC_SAMPLES / FS * 1000.0, color="k", ls=":", lw=0.8)
        ax.axhline(-GATE_SYNC_SAMPLES / FS * 1000.0, color="k", ls=":", lw=0.8)
        ax.set_xticks(xpos, [f"{d/100:.2f}" for d in DIFFS])
        ax.set_xlabel("diffusion", fontsize=8)
        ax.set_title(f"size {size/10:.1f}", fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("aggregate centroid offset (ms)\nvs diffuser-off control")
    fig.suptitle("IN SYNC: aggregate energy-centroid offset (binning-free; dotted = ±32-sample gate)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out / "centroid_sync.png", dpi=140)
    plt.close(fig)
    print(f"wrote {args.out / 'centroid_sync.png'}")

    # ── 4. per-repeat width growth (before vs after) ─────────────────────
    wash_diffs = [25, 50, 75]   # diff 1.0 excluded: blob tails genuinely overlap
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    reps = [1, 2, 3, 4]
    for ax, diff in zip(axes, wash_diffs):
        for tag, d, style, color in (("before", args.before, "o--", "C3"),
                                     ("after", args.after, "o-", "C2")):
            x = load(d, diff, 5)
            w = voronoi_widths(x, g1)
            ax.plot(reps, [w[r] / FS * 1000.0 for r in reps], style, color=color,
                    label=f"{tag} (sqrt(n) fit: {w[1]/FS*1000.0:.1f}·√n ms)" if tag == "after" else tag)
        ax.set_xticks(reps)
        ax.set_xlabel("repeat n", fontsize=8)
        ax.set_title(f"diffusion {diff/100:.2f} (size 0.5)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-repeat RMS width (ms)")
    fig.suptitle("WASH: per-repeat RMS width — before is flat (one pass), after grows ~√n (n passes)\n"
                 "(diff 1.0 omitted: at g=0.92 the blob tails genuinely overlap, binning is not meaningful)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out / "width_growth.png", dpi=140)
    plt.close(fig)
    print(f"wrote {args.out / 'width_growth.png'}")


if __name__ == "__main__":
    main()

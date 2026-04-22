#!/usr/bin/env python3
"""Chronos diffusion / FDN visualisation dashboard.

Reads the three CSVs produced by the diffusion test suite and renders a
single-page dashboard:

  - Mixing matrix norm preservation (Hadamard vs Householder, per N).
  - Diffusion chain stereo impulse response over time.
  - Diffusion chain cumulative energy (input vs post-chain).
  - FDN T60 decay envelope with the analytical T60 line.

Output:
  tests/simd_harness/logs/diffusion.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
OUT_PATH = os.path.join(LOG_DIR, "diffusion.png")

PASS_COLOR = "#2b8a3e"
FAIL_COLOR = "#c92a2a"
MUTED      = "#495057"
HIGHLIGHT  = "#f59f00"
HADAMARD_C = "#1971c2"
HOUSEH_C   = "#e64980"


def must_read(name):
    p = os.path.join(LOG_DIR, name)
    if not os.path.exists(p):
        print(f"missing CSV: {p}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(p)


def panel_matrix_norms(ax, df):
    ax.set_title("[1] Mixing matrix norm preservation",
                 fontsize=11)
    ax.set_xlabel("channel count N")
    ax.set_ylabel("|Δ||v||²| / ||v||²")
    for mixer, color in (("Hadamard", HADAMARD_C),
                         ("Householder", HOUSEH_C)):
        sub = df[df["mixer"] == mixer]
        ax.semilogy(sub["channel_count"],
                    np.maximum(sub["relative_delta"], 1e-12),
                    marker="o", color=color, label=mixer, linewidth=1.3)
    ax.axhline(1e-5, color=FAIL_COLOR, linestyle="--", linewidth=0.9,
               label="test tolerance (1e-5)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    all_pass = (df["passed"] == 1).all()
    ax.text(0.02, 0.96, "PASS" if all_pass else "FAIL",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor=(PASS_COLOR if all_pass else FAIL_COLOR),
                      edgecolor="none"))


def panel_impulse_response(ax, df):
    ax.plot(df["sample_index"], df["left_output"],
            color="#4dabf7", linewidth=0.5, alpha=0.9, label="left output")
    ax.plot(df["sample_index"], df["right_output"],
            color="#f06595", linewidth=0.5, alpha=0.7, label="right output")
    ax.set_title("[2] Diffusion chain stereo impulse response",
                 fontsize=11)
    ax.set_xlabel("samples")
    ax.set_ylabel("sample value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    # Zoom x-axis to the interesting region.
    ax.set_xlim(0, min(df["sample_index"].iloc[-1], 8000))


def panel_chain_energy(ax, df):
    ax.plot(df["sample_index"], df["post_bridge_in_energy"],
            color="#adb5bd", linewidth=1.5, label="cumulative input energy")
    ax.plot(df["sample_index"], df["post_chain_energy"],
            color=HADAMARD_C, linewidth=1.5, linestyle="--",
            label="cumulative post-chain energy")
    final_ratio = (df["post_chain_energy"].iloc[-1]
                   / max(df["post_bridge_in_energy"].iloc[-1], 1e-30))
    ax.set_title(f"[3] Chain energy preservation  "
                 f"(final ratio = {final_ratio:.6f})",
                 fontsize=11)
    ax.set_xlabel("samples")
    ax.set_ylabel("cumulative energy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ok = abs(final_ratio - 1.0) < 2e-2
    ax.text(0.02, 0.96, "PASS" if ok else "FAIL",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor=(PASS_COLOR if ok else FAIL_COLOR),
                      edgecolor="none"))


def panel_chain_density(ax, df):
    # Scatter: show individual sample magnitudes as dots so the density
    # growth across time is visually obvious.
    abs_left = np.abs(df["left_output"])
    ax.scatter(df["sample_index"], abs_left,
               s=1.0, color="#1971c2", alpha=0.4)
    ax.set_title("[4] Diffusion chain reflection density",
                 fontsize=11)
    ax.set_xlabel("samples")
    ax.set_ylabel("|left output|")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, min(df["sample_index"].iloc[-1], 16000))


def panel_fdn_decay(ax, df, expected_t60_samples):
    ax.plot(df["sample_index"], np.maximum(df["stereo_envelope"], 1e-10),
            color="#2b8a3e", linewidth=1.0)
    ax.axvline(expected_t60_samples, color=FAIL_COLOR, linestyle="--",
               linewidth=1.0, label=f"expected T60 = {expected_t60_samples}")
    # -60 dB reference line, relative to the measured peak.
    peak_envelope = df["stereo_envelope"].max()
    minus_60db = peak_envelope * (10 ** (-60 / 20))
    ax.axhline(minus_60db, color="#666", linestyle=":", linewidth=0.9,
               label="peak − 60 dB")
    ax.set_yscale("log")
    ax.set_title("[5] FDN T60 decay envelope", fontsize=11)
    ax.set_xlabel("samples")
    ax.set_ylabel("|stereo envelope|  (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def main():
    matrix_df   = must_read("diffusion_matrix.csv")
    impulse_df  = must_read("diffusion_impulse.csv")
    energy_df   = must_read("diffusion_energy.csv")
    fdn_df      = must_read("fdn_decay.csv")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Chronos diffusion engine + FDN  —  test dashboard",
        fontsize=15, fontweight="bold", color=MUTED, y=0.995)

    ax_matrix  = fig.add_subplot(2, 3, 1)
    ax_impulse = fig.add_subplot(2, 3, 2)
    ax_energy  = fig.add_subplot(2, 3, 3)
    ax_density = fig.add_subplot(2, 3, 4)
    ax_fdn     = fig.add_subplot(2, 3, 5)

    panel_matrix_norms(ax_matrix, matrix_df)
    panel_impulse_response(ax_impulse, impulse_df)
    panel_chain_energy(ax_energy, energy_df)
    panel_chain_density(ax_density, impulse_df)

    # T60 = 1500 ms at 48 kHz = 72000 samples (matches fdn_test).
    expected_t60_samples = int(48000.0 * 1.5)
    panel_fdn_decay(ax_fdn, fdn_df, expected_t60_samples)

    # Summary panel (bottom-right).
    ax_summary = fig.add_subplot(2, 3, 6)
    ax_summary.set_axis_off()
    matrix_pass  = (matrix_df["passed"] == 1).all()
    chain_ratio  = (energy_df["post_chain_energy"].iloc[-1]
                    / max(energy_df["post_bridge_in_energy"].iloc[-1], 1e-30))
    chain_pass   = abs(chain_ratio - 1.0) < 2e-2
    # FDN pass criterion: the envelope at expected T60 is > 45 dB below peak.
    peak_env = fdn_df["stereo_envelope"].max()
    sub_after_t60 = fdn_df[fdn_df["sample_index"] >= expected_t60_samples]
    env_at_t60 = (sub_after_t60["stereo_envelope"].iloc[0]
                  if len(sub_after_t60) else peak_env)
    db_at_t60  = 20 * np.log10(max(env_at_t60, 1e-12)
                               / max(peak_env, 1e-12))
    fdn_pass   = db_at_t60 < -45.0

    summary_lines = [
        ("Matrix orthogonality", matrix_pass),
        ("Chain energy preservation", chain_pass),
        ("FDN T60 decay", fdn_pass),
    ]
    overall_ok = all(ok for _, ok in summary_lines)
    ax_summary.add_patch(FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0.02",
        linewidth=0,
        facecolor=(PASS_COLOR if overall_ok else FAIL_COLOR),
        alpha=0.12,
        transform=ax_summary.transAxes))
    ax_summary.text(0.5, 0.9, "summary",
                    transform=ax_summary.transAxes,
                    ha="center", va="top", fontsize=12, fontweight="bold",
                    color=MUTED)
    for i, (name, ok) in enumerate(summary_lines):
        y = 0.75 - i * 0.16
        ax_summary.text(0.06, y, name,
                        transform=ax_summary.transAxes,
                        ha="left", va="center", fontsize=11, color=MUTED)
        ax_summary.text(0.94, y, "PASS" if ok else "FAIL",
                        transform=ax_summary.transAxes,
                        ha="right", va="center",
                        fontsize=11, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.35",
                                  facecolor=(PASS_COLOR if ok else FAIL_COLOR),
                                  edgecolor="none"))
    ax_summary.text(0.5, 0.15,
                    f"chain energy ratio = {chain_ratio:.6f}\n"
                    f"FDN envelope at T60 = {db_at_t60:+.1f} dB from peak",
                    transform=ax_summary.transAxes,
                    ha="center", va="center",
                    fontsize=9, color=MUTED)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PATH, dpi=130)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

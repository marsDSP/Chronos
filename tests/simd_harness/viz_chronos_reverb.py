#!/usr/bin/env python3
"""
viz_chronos_reverb.py
---------------------

Six-panel dashboard for the ChronosReverb characterisation CSVs produced
by chronos_reverb_characterisation_test. Panels:

  [1] Impulse-response envelope overlay (one curve per decayTime sweep point)
  [2] Measured T60 vs decayTime parameter (compares measured to target)
  [3] Magnitude spectrum of the IR at the canonical config
  [4] Stereo L/R Pearson correlation vs modulation knob
  [5] Mix-sweep peak abs sample vs mix
  [6] Mix-sweep RMS vs mix (dry reference overlaid)

Writes tests/simd_harness/logs/chronos_reverb.png.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR   = REPO_ROOT / "tests" / "simd_harness" / "logs"
OUT_PATH  = LOG_DIR / "chronos_reverb.png"

IMPULSE_CSV        = LOG_DIR / "chronos_reverb_impulse_envelope.csv"
T60_CSV            = LOG_DIR / "chronos_reverb_t60_vs_decay.csv"
FREQ_RESPONSE_CSV  = LOG_DIR / "chronos_reverb_frequency_response.csv"
STEREO_CORR_CSV    = LOG_DIR / "chronos_reverb_stereo_correlation.csv"
MIX_SWEEP_CSV      = LOG_DIR / "chronos_reverb_mix_sweep.csv"

COLOUR_ACCENT = "#1971c2"
COLOUR_SECOND = "#c92a2a"
COLOUR_MUTED  = "#495057"
DECAY_CMAP    = plt.get_cmap("plasma")
MOD_CMAP      = plt.get_cmap("viridis")


def require(path: Path):
    if not path.exists():
        raise SystemExit(
            f"missing CSV: {path}\n"
            "Run ./cmake-build-debug/tests/simd_harness/"
            "chronos_reverb_characterisation_test first."
        )


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------
# [1] impulse-response envelope overlay
# ----------------------------------------------------------------------
def draw_impulse_envelope(axis, impulse_rows: list[dict]) -> None:
    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in impulse_rows:
        grouped[float(row["decay_log2s"])].append(
            (float(row["time_ms"]), float(row["envelope"]))
        )

    decays = sorted(grouped.keys())
    for idx, decay in enumerate(decays):
        points = sorted(grouped[decay], key=lambda p: p[0])
        times   = np.asarray([p[0] for p in points])
        samples = np.asarray([p[1] for p in points])
        # Log-amplitude so T60 shows up as a straight-ish line.
        samples_db = 20.0 * np.log10(np.maximum(samples, 1e-8))
        axis.plot(times / 1000.0, samples_db,
                  color=DECAY_CMAP(idx / max(1, len(decays) - 1)),
                  linewidth=1.3,
                  label=f"decay 2^{decay:g} s "
                        f"({2**decay:.3g} s)")

    axis.axhline(-60.0, color=COLOUR_MUTED, linestyle="--",
                 linewidth=0.8, label="-60 dB")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("envelope (dB)")
    axis.set_title("[1] IR envelope across decayTime sweep", fontsize=11)
    axis.set_ylim(-110.0, 6.0)
    axis.grid(True, linestyle=":", alpha=0.4)
    axis.legend(fontsize=7, loc="upper right")


# ----------------------------------------------------------------------
# [2] measured T60 vs parameter
# ----------------------------------------------------------------------
def draw_t60_panel(axis, t60_rows: list[dict]) -> None:
    decays     = np.asarray([float(r["decay_log2s"])        for r in t60_rows])
    targets    = np.asarray([float(r["decay_target_sec"])   for r in t60_rows])
    measured   = np.asarray([float(r["measured_t60_sec"])   for r in t60_rows])

    # Mark rows where the measurement didn't reach -60 dB with a
    # hollow marker so they don't distort the trend line.
    valid = measured > 0

    axis.plot(targets[valid], measured[valid],
              marker="o", color=COLOUR_ACCENT, linewidth=1.8,
              label="measured T60")
    axis.plot(targets, targets,
              color=COLOUR_MUTED, linestyle="--", linewidth=0.9,
              label="y = x (param == measured)")

    for t, m, d in zip(targets, measured, decays):
        if m > 0:
            axis.annotate(f"{m:.2f} s",
                          xy=(t, m), xytext=(4, 4),
                          textcoords="offset points",
                          fontsize=7, color=COLOUR_ACCENT)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("decayTime parameter (= 2^decay_log2s, seconds)")
    axis.set_ylabel("measured T60 (seconds)")
    axis.set_title("[2] T60 measured vs target", fontsize=11)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=8)


# ----------------------------------------------------------------------
# [3] IR magnitude spectrum
# ----------------------------------------------------------------------
def draw_frequency_response(axis, freq_rows: list[dict]) -> None:
    frequencies = np.asarray([float(r["frequency_hz"]) for r in freq_rows])
    magnitudes  = np.asarray([float(r["magnitude_db"]) for r in freq_rows])

    # Normalise so the peak is 0 dB for readability.
    magnitudes = magnitudes - np.max(magnitudes)

    axis.plot(frequencies, magnitudes,
              color=COLOUR_ACCENT, linewidth=1.6)
    axis.axhline(-3.0, color=COLOUR_MUTED, linestyle="--", linewidth=0.8,
                 label="-3 dB")
    axis.set_xscale("log")
    axis.set_xlabel("frequency (Hz, log)")
    axis.set_ylabel("IR magnitude (dB, peak-normalised)")
    axis.set_title("[3] IR frequency response", fontsize=11)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=8, loc="lower left")


# ----------------------------------------------------------------------
# [4] stereo correlation vs modulation
# ----------------------------------------------------------------------
def draw_stereo_correlation(axis, stereo_rows: list[dict]) -> None:
    mods  = np.asarray([float(r["modulation"])            for r in stereo_rows])
    corrs = np.asarray([float(r["stereo_correlation"])    for r in stereo_rows])

    colours = [MOD_CMAP(i / max(1, len(mods) - 1)) for i in range(len(mods))]
    bars = axis.bar(mods, corrs, width=0.12, color=colours,
                    edgecolor="black", linewidth=0.4)

    for bar, value in zip(bars, corrs):
        axis.annotate(f"{value:+.3f}",
                      xy=(bar.get_x() + bar.get_width() * 0.5,
                          value),
                      xytext=(0, 4 if value >= 0 else -10),
                      textcoords="offset points",
                      ha="center",
                      fontsize=8, color=COLOUR_MUTED)

    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.axhline(1.0, color=COLOUR_MUTED, linestyle=":", linewidth=0.7,
                 label="perfectly correlated L = R")
    axis.set_ylim(-0.2, 1.1)
    axis.set_xlabel("modulation knob")
    axis.set_ylabel("Pearson r(L, R) over tail")
    axis.set_title("[4] Stereo correlation vs modulation", fontsize=11)
    axis.grid(True, axis="y", linestyle=":", alpha=0.4)
    axis.legend(fontsize=8, loc="lower left")


# ----------------------------------------------------------------------
# [5 + 6] mix sweep
# ----------------------------------------------------------------------
def draw_mix_sweep_peak(axis, mix_rows: list[dict]) -> None:
    mixes = np.asarray([float(r["mix"])                for r in mix_rows])
    peaks = np.asarray([float(r["observed_peak_abs"])  for r in mix_rows])

    axis.plot(mixes, peaks, marker="o",
              color=COLOUR_SECOND, linewidth=1.6, label="peak |sample|")
    for x, y in zip(mixes, peaks):
        axis.annotate(f"{y:.3f}",
                      xy=(x, y), xytext=(4, 4),
                      textcoords="offset points",
                      fontsize=8, color=COLOUR_SECOND)
    axis.set_xlabel("mix")
    axis.set_ylabel("peak abs sample")
    axis.set_title("[5] Mix sweep - peak abs sample", fontsize=11)
    axis.grid(True, linestyle=":", alpha=0.4)
    axis.legend(fontsize=8, loc="lower right")


def draw_mix_sweep_rms(axis, mix_rows: list[dict]) -> None:
    mixes      = np.asarray([float(r["mix"])                  for r in mix_rows])
    rms_values = np.asarray([float(r["observed_rms"])         for r in mix_rows])
    dry_ref    = float(mix_rows[0]["dry_reference_rms"])

    axis.plot(mixes, rms_values, marker="o",
              color=COLOUR_ACCENT, linewidth=1.6, label="output RMS")
    axis.axhline(dry_ref, color=COLOUR_MUTED, linestyle="--",
                 linewidth=0.9, label=f"dry reference ({dry_ref:.3f})")
    for x, y in zip(mixes, rms_values):
        axis.annotate(f"{y:.3f}",
                      xy=(x, y), xytext=(4, 4),
                      textcoords="offset points",
                      fontsize=8, color=COLOUR_ACCENT)
    axis.set_xlabel("mix")
    axis.set_ylabel("output RMS")
    axis.set_title("[6] Mix sweep - steady-state RMS", fontsize=11)
    axis.grid(True, linestyle=":", alpha=0.4)
    axis.legend(fontsize=8, loc="lower left")


def main() -> None:
    for p in (IMPULSE_CSV, T60_CSV, FREQ_RESPONSE_CSV, STEREO_CORR_CSV, MIX_SWEEP_CSV):
        require(p)

    impulse_rows      = load_rows(IMPULSE_CSV)
    t60_rows          = load_rows(T60_CSV)
    freq_rows         = load_rows(FREQ_RESPONSE_CSV)
    stereo_rows       = load_rows(STEREO_CORR_CSV)
    mix_rows          = load_rows(MIX_SWEEP_CSV)

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        "ChronosReverb characterisation  "
        "(IR envelope, T60, frequency response, stereo, mix sweep)",
        fontsize=14, weight="bold",
    )
    gs = fig.add_gridspec(3, 2)

    ax_envelope  = fig.add_subplot(gs[0, 0])
    ax_t60       = fig.add_subplot(gs[0, 1])
    ax_freq      = fig.add_subplot(gs[1, 0])
    ax_stereo    = fig.add_subplot(gs[1, 1])
    ax_mix_peak  = fig.add_subplot(gs[2, 0])
    ax_mix_rms   = fig.add_subplot(gs[2, 1])

    draw_impulse_envelope   (ax_envelope,  impulse_rows)
    draw_t60_panel          (ax_t60,       t60_rows)
    draw_frequency_response (ax_freq,      freq_rows)
    draw_stereo_correlation (ax_stereo,    stereo_rows)
    draw_mix_sweep_peak     (ax_mix_peak,  mix_rows)
    draw_mix_sweep_rms      (ax_mix_rms,   mix_rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

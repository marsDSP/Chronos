#!/usr/bin/env python3
"""Chronos s-plane curve-fit filter response dashboard.

Renders side-by-side the old impulse-invariance discretisation (which
aliased the HPF passband above 1 and blew up the feedback loop) and the
new bilinear-transform discretisation (which preserves the analog
magnitude response exactly, so max |H| = 1 for both HPF and LPF).

Both paths use the same 4th-order Butterworth analog prototype - only the
discretisation method differs, so the plot is a fair apples-to-apples
comparison of the fix made in source/dsp/filter/splane_curvefit_core.h.

Output:
  tests/simd_harness/logs/splane_filter_response.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
OUT_PATH = os.path.join(LOG_DIR, "splane_filter_response.png")

PASS_COLOR  = "#2b8a3e"
FAIL_COLOR  = "#c92a2a"
OLD_COLOR   = "#c92a2a"
NEW_COLOR   = "#1971c2"
MUTED       = "#495057"
HIGHLIGHT   = "#f59f00"


# --------------------------------------------------------------- prototype
def butterworth_lhp_poles(order: int) -> np.ndarray:
    """Left-half-plane poles of the normalised Butterworth prototype."""
    k = np.arange(order)
    # Poles spaced every pi/N on the left-half unit circle in the s-plane.
    angles = np.pi * (2 * k + order + 1) / (2 * order)
    return np.exp(1j * angles)


def lp_residues(poles: np.ndarray) -> np.ndarray:
    """Residues of the LPF partial fraction at unit cutoff.
       H_lp(s) = 1 / prod(s - pole_k) = sum_k r_k / (s - pole_k)
    """
    residues = np.empty_like(poles)
    for k, p in enumerate(poles):
        others = np.delete(poles, k)
        residues[k] = 1.0 / np.prod(p - others)
    return residues


def hp_residues(poles: np.ndarray) -> np.ndarray:
    """Direct Butterworth HPF residues used by the pre-fix impulse-
       invariance discretisation.
       H_hp(s) = s^N / prod(s - pole_k) = 1 + sum_k r_k / (s - pole_k).
    """
    N = len(poles)
    residues = np.empty_like(poles)
    for k, p in enumerate(poles):
        others = np.delete(poles, k)
        residues[k] = (p ** N) / np.prod(p - others)
    return residues


# --------------------------------------------------------- discretisation
def impulse_invariance_response(poles, residues, direct_through,
                                cutoff_hz, sample_rate, w_discrete):
    """Evaluate H(e^{j w_discrete}) under impulse-invariance (OLD).

    Each partial-fraction term r_k/(s-p_k) becomes
      r_k * Ts / (1 - exp(p_k * Ts) * z^-1).
    """
    Ts = 1.0 / sample_rate
    wc = 2.0 * np.pi * cutoff_hz
    z  = np.exp(1j * w_discrete)

    H = np.full_like(w_discrete, direct_through, dtype=complex)
    for p_unit, r_unit in zip(poles, residues):
        p_scaled = p_unit * wc
        r_scaled = r_unit * wc
        z_pole   = np.exp(p_scaled * Ts)
        gain     = r_scaled * Ts
        H += gain / (1.0 - z_pole / z)
    return H


def bilinear_response(poles, residues, direct_through,
                      cutoff_hz, sample_rate, w_discrete):
    """Evaluate H(e^{j w_discrete}) under bilinear transform (NEW), with
    cutoff pre-warping so the discrete -3 dB corner lands at cutoff_hz.

    Each partial-fraction term r_k/(s-p_k) becomes
      [r_k/(2/T - p_k)] * (1 + z^-1) / (1 - zPole_k * z^-1),
      zPole_k = (2/T + p_k)/(2/T - p_k).
    """
    Ts = 1.0 / sample_rate
    two_over_T = 2.0 / Ts

    # Pre-warp the target angular cutoff so bilinear maps it back to
    # cutoff_hz on the discrete axis.
    w_target   = 2.0 * np.pi * cutoff_hz
    w_prewarp  = (2.0 / Ts) * np.tan(w_target * Ts * 0.5)
    z          = np.exp(1j * w_discrete)

    H = np.full_like(w_discrete, direct_through, dtype=complex)
    for p_unit, r_unit in zip(poles, residues):
        p_scaled = p_unit * w_prewarp
        r_scaled = r_unit * w_prewarp
        denom    = two_over_T - p_scaled
        z_pole   = (two_over_T + p_scaled) / denom
        section_gain = r_scaled / denom
        H += section_gain * (1.0 + 1.0 / z) / (1.0 - z_pole / z)
    return H


# ------------------------------------------------------------------- main
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    sample_rate = 48000.0
    order       = 4
    freq_grid   = np.logspace(np.log10(10.0), np.log10(sample_rate * 0.499), 2048)
    w_grid      = 2.0 * np.pi * freq_grid / sample_rate

    poles = butterworth_lhp_poles(order)
    r_lp  = lp_residues(poles)
    r_hp  = hp_residues(poles)

    cutoff_sweep = [50.0, 500.0, 5000.0, 15000.0]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Chronos feedback-path s-plane curve-fit filters  \u2014  "
                 "impulse-invariance (OLD) vs bilinear (NEW)",
                 fontsize=15, fontweight="bold", color=MUTED, y=0.995)

    # --- Row 1: HPF magnitude responses, old vs new ----------------------
    ax_old_hp = fig.add_subplot(2, 3, 1)
    ax_new_hp = fig.add_subplot(2, 3, 2)

    for ax, method_fn, title_tag in (
            (ax_old_hp, impulse_invariance_response, "impulse invariance"),
            (ax_new_hp, bilinear_response,           "bilinear transform"),
    ):
        for fc in cutoff_sweep:
            H = method_fn(poles, r_hp, 1.0, fc, sample_rate, w_grid)
            mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
            ax.semilogx(freq_grid, mag_db, label=f"fc = {fc:g} Hz",
                        linewidth=1.4)
        ax.axhline(0.0, color="#666", linestyle=":", linewidth=0.9,
                   label="0 dB (unity)")
        ax.axhline(-3.0, color="#aaa", linestyle=":", linewidth=0.8)
        ax.set_title(f"HPF magnitude  \u2014  {title_tag}", fontsize=11)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("|H(e^{j\u03c9})|  (dB)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-60.0, 20.0)
        ax.set_xlim(freq_grid[0], freq_grid[-1])
        ax.legend(loc="lower right", fontsize=8)

    # Peaking annotation on the OLD plot.
    peak_db = 20.0 * np.log10(np.max(np.abs(
        impulse_invariance_response(poles, r_hp, 1.0, 5000.0,
                                    sample_rate, w_grid))))
    ax_old_hp.annotate(f"aliased peak\n\u2248 {peak_db:+.1f} dB at 5 kHz",
                       xy=(5000.0, peak_db),
                       xytext=(60.0, 10.0),
                       fontsize=9, color=OLD_COLOR,
                       arrowprops=dict(arrowstyle="->", color=OLD_COLOR,
                                       lw=1.1))

    ax_new_hp.text(0.5, 0.9,
                   "max |H| = 1 everywhere",
                   transform=ax_new_hp.transAxes, ha="center",
                   fontsize=10, color=PASS_COLOR, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                             edgecolor=PASS_COLOR))

    # --- Row 1 col 3: feedback-loop stability envelope -------------------
    ax_env = fig.add_subplot(2, 3, 3)
    fc_envelope = np.logspace(np.log10(20.0),
                              np.log10(sample_rate * 0.499), 96)
    max_gain_old = np.zeros_like(fc_envelope)
    max_gain_new = np.zeros_like(fc_envelope)
    for i, fc in enumerate(fc_envelope):
        H_old = impulse_invariance_response(poles, r_hp, 1.0, fc,
                                            sample_rate, w_grid)
        H_new = bilinear_response(poles, r_hp, 1.0, fc,
                                  sample_rate, w_grid)
        max_gain_old[i] = np.max(np.abs(H_old))
        max_gain_new[i] = np.max(np.abs(H_new))

    ax_env.semilogx(fc_envelope, max_gain_old,
                    color=OLD_COLOR, linewidth=1.8,
                    label="OLD (impulse invariance)")
    ax_env.semilogx(fc_envelope, max_gain_new,
                    color=NEW_COLOR, linewidth=1.8,
                    label="NEW (bilinear)")
    ax_env.axhline(1.0, color="#333", linestyle="--", linewidth=1.0,
                   label="stability ceiling (gain = 1)")
    ax_env.fill_between(fc_envelope, 1.0,
                        np.maximum(max_gain_old, 1.0),
                        where=max_gain_old > 1.0,
                        color=FAIL_COLOR, alpha=0.15,
                        label="feedback-unstable region")
    ax_env.set_title("HPF peak gain vs cutoff  \u2014  stability envelope",
                     fontsize=11)
    ax_env.set_xlabel("cutoff (Hz)")
    ax_env.set_ylabel("max\u2081 |H(e^{j\u03c9})|")
    ax_env.set_xlim(fc_envelope[0], fc_envelope[-1])
    ax_env.set_ylim(0.0, max(3.0, np.max(max_gain_old) * 1.1))
    ax_env.grid(True, which="both", alpha=0.3)
    ax_env.legend(loc="upper left", fontsize=8)

    # --- Row 2: LPF and band-shaped series response ----------------------
    ax_lp  = fig.add_subplot(2, 3, 4)
    ax_band = fig.add_subplot(2, 3, 5)
    ax_phase = fig.add_subplot(2, 3, 6)

    # LPF magnitude (bilinear - same for old/new since LPF rolled off
    # cleanly under impulse invariance too; we're just showing the family).
    for fc in cutoff_sweep:
        H = bilinear_response(poles, r_lp, 0.0, fc, sample_rate, w_grid)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
        ax_lp.semilogx(freq_grid, mag_db, label=f"fc = {fc:g} Hz",
                       linewidth=1.4)
    ax_lp.axhline(0.0, color="#666", linestyle=":", linewidth=0.9)
    ax_lp.axhline(-3.0, color="#aaa", linestyle=":", linewidth=0.8)
    ax_lp.set_title("LPF magnitude  \u2014  bilinear transform",
                    fontsize=11)
    ax_lp.set_xlabel("frequency (Hz)")
    ax_lp.set_ylabel("|H(e^{j\u03c9})|  (dB)")
    ax_lp.grid(True, which="both", alpha=0.3)
    ax_lp.set_ylim(-60.0, 20.0)
    ax_lp.set_xlim(freq_grid[0], freq_grid[-1])
    ax_lp.legend(loc="lower left", fontsize=8)

    # Series HP (200 Hz) + LP (5 kHz) band, both bilinear, with fb = 0.99.
    fc_hp = 200.0
    fc_lp = 5000.0
    H_hp  = bilinear_response(poles, r_hp, 1.0, fc_hp, sample_rate, w_grid)
    H_lp  = bilinear_response(poles, r_lp, 0.0, fc_lp, sample_rate, w_grid)
    H_band = H_hp * H_lp
    feedback_gain_envelope = 0.99 * np.abs(H_band)

    ax_band.semilogx(freq_grid, 20.0 * np.log10(np.maximum(np.abs(H_hp), 1e-12)),
                     color="#adb5bd", linewidth=1.0,
                     label=f"HPF {fc_hp:g} Hz")
    ax_band.semilogx(freq_grid, 20.0 * np.log10(np.maximum(np.abs(H_lp), 1e-12)),
                     color="#adb5bd", linewidth=1.0, linestyle="--",
                     label=f"LPF {fc_lp:g} Hz")
    ax_band.semilogx(freq_grid, 20.0 * np.log10(np.maximum(np.abs(H_band), 1e-12)),
                     color=NEW_COLOR, linewidth=1.8,
                     label=f"HP \u00b7 LP (series)")
    ax_band.semilogx(freq_grid,
                     20.0 * np.log10(np.maximum(feedback_gain_envelope, 1e-12)),
                     color=PASS_COLOR, linewidth=1.6, linestyle=":",
                     label="feedback \u00d7 H  (fb = 0.99)")
    ax_band.axhline(0.0, color="#333", linestyle="--", linewidth=1.0,
                    label="0 dB (unity)")
    ax_band.set_title("Series HP\u2192LP band with feedback envelope",
                      fontsize=11)
    ax_band.set_xlabel("frequency (Hz)")
    ax_band.set_ylabel("|H|  (dB)")
    ax_band.grid(True, which="both", alpha=0.3)
    ax_band.set_ylim(-60.0, 10.0)
    ax_band.set_xlim(freq_grid[0], freq_grid[-1])
    ax_band.legend(loc="lower center", fontsize=8, ncol=2)

    max_fb = np.max(feedback_gain_envelope)
    stable = max_fb < 1.0
    ax_band.text(0.02, 0.95,
                 f"max(fb \u00b7 H) = {max_fb:.4f}   \u2192  "
                 f"{'STABLE' if stable else 'UNSTABLE'}",
                 transform=ax_band.transAxes, ha="left", va="top",
                 fontsize=10, fontweight="bold",
                 color=PASS_COLOR if stable else FAIL_COLOR,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           edgecolor=PASS_COLOR if stable else FAIL_COLOR))

    # Phase response (HPF + LPF, bilinear only)
    for fc, label, style in ((fc_hp, f"HPF {fc_hp:g} Hz", "-"),
                             (fc_lp, f"LPF {fc_lp:g} Hz", "--")):
        r = r_hp if "HPF" in label else r_lp
        dt = 1.0 if "HPF" in label else 0.0
        H = bilinear_response(poles, r, dt, fc, sample_rate, w_grid)
        phase = np.unwrap(np.angle(H)) * 180.0 / np.pi
        ax_phase.semilogx(freq_grid, phase, linewidth=1.4, linestyle=style,
                          label=label)
    ax_phase.set_title("Phase response  \u2014  bilinear HPF / LPF",
                       fontsize=11)
    ax_phase.set_xlabel("frequency (Hz)")
    ax_phase.set_ylabel("phase (deg)")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.set_xlim(freq_grid[0], freq_grid[-1])
    ax_phase.legend(loc="lower left", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PATH, dpi=130)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Visualise the SinhDucker smoke-test output.

Reads:
  - logs/simd_ducker_smoke_audio.csv:   per-sample dry, wet_in, wet_out
  - logs/simd_ducker_smoke_control.csv: per-block envelope, drive, gain

Produces a stacked-panel PNG showing how the kick on the sidechain pulls
the wet sine down each time and how the bridge gain swells back up.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    audio_csv   = os.path.join(here, "logs", "simd_ducker_smoke_audio.csv")
    control_csv = os.path.join(here, "logs", "simd_ducker_smoke_control.csv")
    out_path    = os.path.join(here, "logs", "simd_ducker_smoke.png")

    if not os.path.exists(audio_csv):
        print(f"Error: {audio_csv} not found. Run simd_ducker_smoke_test first.")
        return
    if not os.path.exists(control_csv):
        print(f"Error: {control_csv} not found. Run simd_ducker_smoke_test first.")
        return

    audio   = pd.read_csv(audio_csv)
    control = pd.read_csv(control_csv)

    # -------------------------------------------------------------- #
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.5, 1.5, 1.0]})

    # Panel 1: dry sidechain
    axes[0].plot(audio["t"], audio["dryL"], color="#7f8c8d", linewidth=0.6)
    axes[0].set_title("Sidechain (dry, kick @ 80 Hz, retriggered every 250 ms)")
    axes[0].set_ylabel("amplitude")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: wet input vs wet output
    axes[1].plot(audio["t"], audio["wetL_in"],  color="#3498db", linewidth=0.5,
                 label="wet in (440 Hz sine)", alpha=0.5)
    axes[1].plot(audio["t"], audio["wetL_out"], color="#e74c3c", linewidth=0.5,
                 label="wet out (ducked)", alpha=0.9)
    axes[1].set_title("Wet path: 440 Hz sine before / after the diode-bridge ducker")
    axes[1].set_ylabel("amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: instantaneous |wet_out| envelope vs |wet_in|
    win = 64
    abs_in  = np.abs(audio["wetL_in"]).rolling(win, min_periods=1).max()
    abs_out = np.abs(audio["wetL_out"]).rolling(win, min_periods=1).max()
    axes[2].plot(audio["t"], abs_in,  color="#3498db", linewidth=1.0,
                 label="|wet in| (sliding max)", alpha=0.7)
    axes[2].plot(audio["t"], abs_out, color="#e74c3c", linewidth=1.0,
                 label="|wet out| (sliding max)", alpha=0.9)
    axes[2].set_title("Wet output envelope - swelling back during kick gaps")
    axes[2].set_ylabel("|wet|")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Panel 4: control telemetry (per-block)
    ax_ctrl = axes[3]
    ax_ctrl.plot(control["t"], control["envelope"], color="#2ecc71",
                 linewidth=1.4, label="env", drawstyle="steps-post")
    ax_ctrl.plot(control["t"], control["drive"] / np.pi, color="#9b59b6",
                 linewidth=1.4, label="drive / π", drawstyle="steps-post")
    ax_ctrl.plot(control["t"], control["gain"], color="#f39c12",
                 linewidth=1.6, label="bridge gain", drawstyle="steps-post")
    ax_ctrl.set_title("Block-rate control telemetry (one sample per host block)")
    ax_ctrl.set_xlabel("time (s)")
    ax_ctrl.set_ylabel("0..1")
    ax_ctrl.set_ylim(-0.05, 1.10)
    ax_ctrl.legend(loc="upper right")
    ax_ctrl.grid(True, alpha=0.3)

    min_gain = float(control["gain"].min())
    max_drive = float(control["drive"].max())
    fig.suptitle(
        f"SinhDucker smoke test  -  min gain = {min_gain:.3f},  "
        f"peak drive = {max_drive:.3f} rad",
        fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated {out_path} (min gain = {min_gain:.3f})")


if __name__ == "__main__":
    main()

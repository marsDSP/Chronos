"""Visualise the WDF DiodePair correctness test.

Reads:
  - logs/wdf_omega.csv:           x, omega_truth, omega3, omega4, abs_err3, abs_err4
  - logs/wdf_diode_dc_sweep.csv:  Vin, Vout_acc
  - logs/wdf_diode_audio.csv:     n, t, Vin, Vout

Produces a 4-panel PNG:
  1) Wright-Omega: scipy reference overlaid with omega3 / omega4
  2) Wright-Omega absolute error (log scale) with accuracy margins drawn in
  3) DiodePair DC voltage transfer curve (V_in -> V_out)
  4) DiodePair audio chirp (V_in vs V_out vs time)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    omega_csv = os.path.join(here, "logs", "wdf_omega.csv")
    dc_csv    = os.path.join(here, "logs", "wdf_diode_dc_sweep.csv")
    audio_csv = os.path.join(here, "logs", "wdf_diode_audio.csv")
    out_path  = os.path.join(here, "logs", "wdf_diode.png")

    for path in (omega_csv, dc_csv, audio_csv):
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run wdf_diode_pair_test first.")
            return

    omega = pd.read_csv(omega_csv).sort_values("x")
    dc    = pd.read_csv(dc_csv)
    audio = pd.read_csv(audio_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Wright-Omega overlay
    ax = axes[0, 0]
    ax.plot(omega["x"], omega["omega_truth"], color="black", linewidth=2,
            label="scipy reference", alpha=0.7)
    ax.plot(omega["x"], omega["omega3"], color="#e74c3c", linestyle="--",
            label="wrightOmega3", alpha=0.85)
    ax.plot(omega["x"], omega["omega4"], color="#2ecc71", linestyle=":",
            label="wrightOmega4", alpha=0.95, linewidth=1.6)
    ax.set_title("Wright-Omega: W(x) such that W + ln(W) = x")
    ax.set_xlabel("x")
    ax.set_ylabel("W(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # 2) Wright-Omega absolute error
    ax = axes[0, 1]
    eps = 1e-12
    ax.semilogy(omega["x"], omega["abs_err3"].clip(lower=eps), color="#e74c3c",
                label="|omega3 - ref|", alpha=0.85)
    ax.semilogy(omega["x"], omega["abs_err4"].clip(lower=eps), color="#2ecc71",
                label="|omega4 - ref|", alpha=0.95)
    ax.axhline(0.3,  color="#e74c3c", linestyle=":", linewidth=1, alpha=0.5,
               label="omega3 margin (0.3)")
    ax.axhline(0.05, color="#2ecc71", linestyle=":", linewidth=1, alpha=0.5,
               label="omega4 margin (0.05)")
    max3 = float(omega["abs_err3"].max())
    max4 = float(omega["abs_err4"].max())
    ax.set_title(f"Absolute error  -  max|omega3| = {max3:.3e}, "
                 f"max|omega4| = {max4:.3e}")
    ax.set_xlabel("x")
    ax.set_ylabel("absolute error (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize="small")

    # 3) DC transfer curve
    ax = axes[1, 0]
    ax.plot(dc["Vin"], dc["Vout_acc"], color="#3498db", linewidth=1.5)
    ax.plot([dc["Vin"].min(), dc["Vin"].max()],
            [dc["Vin"].min(), dc["Vin"].max()],
            color="grey", linestyle="--", alpha=0.4, label="V_out = V_in")
    Vmin = float(dc["Vout_acc"].min())
    Vmax = float(dc["Vout_acc"].max())
    ax.set_title(f"DiodePair DC sweep: V_out clamped to [{Vmin:.3f}, "
                 f"{Vmax:.3f}] V")
    ax.set_xlabel("V_in (V)")
    ax.set_ylabel("V_out (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # 4) Audio chirp
    ax = axes[1, 1]
    ax.plot(audio["t"] * 1000.0, audio["Vin"],  color="#3498db", linewidth=0.6,
            label="V_in (chirp 0.2 -> 4 kHz, 2 V peak)", alpha=0.5)
    ax.plot(audio["t"] * 1000.0, audio["Vout"], color="#e74c3c", linewidth=0.7,
            label="V_out (clipped + soft saturation)", alpha=0.9)
    ax.set_title("DiodePair audio chirp")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.suptitle("WDF DiodePair correctness  -  scipy reference metrics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    main()

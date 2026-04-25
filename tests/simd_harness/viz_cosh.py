"""Visualise cosh correctness output.

Reads two CSVs:
  - logs/simd_cosh_results.csv:        std vs Pade scalar/SIMD on a [-pi, pi] sweep
  - logs/simd_cosh_adaa_results.csv:   ADAA SIMD vs direct SIMD Pade on a chirp

Produces a single PNG with three panels:
  1) Top:    function values overlay (std, scalar Pade, SIMD Pade)
  2) Middle: log-scale absolute error (vs std) and SIMD-scalar diff
  3) Bottom: ADAA vs direct cosh on a swept-frequency input
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    static_csv = os.path.join(script_dir, "logs", "simd_cosh_results.csv")
    adaa_csv   = os.path.join(script_dir, "logs", "simd_cosh_adaa_results.csv")
    out_path   = os.path.join(script_dir, "logs", "simd_cosh.png")

    if not os.path.exists(static_csv):
        print(f"Error: {static_csv} not found. Run simd_cosh_test first.")
        return

    static = pd.read_csv(static_csv)

    # PASS = SIMD matches scalar within ~single ULP at cosh's value scale.
    # cosh(pi) ~= 11.59 and float eps ~= 1.19e-7, so 1 ULP ~= 1.4e-6;
    # round up to 5e-6 to allow a tiny margin for FMA/no-FMA differences.
    max_simd_scalar_diff = float(static["diff_simd_scalar"].max())
    max_approx_err       = float(static["abs_err_simd"].max())
    pass_status = max_simd_scalar_diff < 5e-6
    color = "green" if pass_status else "red"
    status_text = "PASSED" if pass_status else "FAILED"

    have_adaa = os.path.exists(adaa_csv)
    n_rows = 3 if have_adaa else 2
    height_ratios = [2, 1, 2] if have_adaa else [2, 1]

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows),
                             gridspec_kw={"height_ratios": height_ratios})
    if n_rows == 1:
        axes = [axes]

    ax_fn = axes[0]
    ax_fn.plot(static["x"], static["std_cosh"],   label="std::cosh (ref)",
               color="black", linewidth=2, alpha=0.6)
    ax_fn.plot(static["x"], static["pade_scalar"], label="Pade (scalar)",
               color="#e74c3c", linestyle="--", alpha=0.8)
    ax_fn.plot(static["x"], static["pade_simd"],   label="Pade (SIMD)",
               color="#2ecc71", linestyle=":", alpha=0.9)
    ax_fn.set_title("cosh(x) over [-pi, pi]")
    ax_fn.set_xlabel("x")
    ax_fn.set_ylabel("cosh(x)")
    ax_fn.legend()
    ax_fn.grid(True, alpha=0.3)

    ax_err = axes[1]
    eps = 1e-12
    ax_err.plot(static["x"], static["abs_err_simd"].clip(lower=eps),
                color="#3498db", label="|SIMD - std|", alpha=0.8)
    ax_err.plot(static["x"], static["diff_simd_scalar"].clip(lower=eps),
                color=color, label="|SIMD - scalar|", alpha=0.8)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("absolute error (log)")
    ax_err.set_title(f"Approximation error  -  status {status_text}  -  "
                     f"max(|SIMD-scalar|) = {max_simd_scalar_diff:.2e}, "
                     f"max(|SIMD-std|) = {max_approx_err:.2e}")
    ax_err.legend(loc="upper center", fontsize="small")
    ax_err.grid(True, which="both", alpha=0.3)

    if have_adaa:
        adaa = pd.read_csv(adaa_csv)
        ax_adaa = axes[2]
        ax_adaa.plot(adaa["n"], adaa["direct"], label="direct cosh (Pade SIMD)",
                     color="#3498db", alpha=0.6)
        ax_adaa.plot(adaa["n"], adaa["adaa"],   label="ADAA cosh (SIMD)",
                     color="#9b59b6", alpha=0.8, linewidth=0.9)
        max_diff = float(adaa["diff"].max())
        ax_adaa.set_title(f"ADAA cosh vs direct on a 100Hz->8kHz sweep "
                          f"(max |direct - ADAA| = {max_diff:.3e})")
        ax_adaa.set_xlabel("sample index")
        ax_adaa.set_ylabel("cosh(x[n])")
        ax_adaa.legend(loc="upper right")
        ax_adaa.grid(True, alpha=0.3)

    fig.suptitle(f"cosh correctness  -  {status_text}",
                 fontsize=14, fontweight="bold", color=color)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated {out_path} ({status_text})")


if __name__ == "__main__":
    main()

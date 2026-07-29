#!/usr/bin/env python3
"""
remez_dilog_psi.py -- region-II remainder kernel for F2.

    F2(a) = 0.5*(a - ln2)^2 + C2 - E(a),   E(a) = 0.5 * t * psi(t)
    psi(t) = -Li2(-t)/t = 1 - t/4 + t^2/9 - t^3/16 + ...,   t = e^(-2a)

    interval: t in [0, e^(-2*a0)] = [0, 0.13534] for a0 = 1.

This fit is why the Landen fold dies: dilogNeg branches at t = 1/2 (i.e.
a = ln2/2 ~ 0.347), and region II never sees t above 0.1354, so the fold and
its log1p never fire in the audio path. psi runs monotonically from 1.0 down
to ~0.9678 over the interval -- the easiest fit in the project.

Error budget for the relative gate: an epsilon on psi enters F2 as
E*eps/F2; at a = a0 that weight is E/F2 = 0.0655/0.218 ~ 0.30 and it only
falls as a grows, so a 5e-17 psi fit contributes < 1.5e-17 to F2.

Fit: degree 10, relative-error Remez, CONSTRAINED at t0 = e^(-2*a0) so the
region-II F2 value at the seam interpolates the same oracle point that
remez_f2_region1.py pins from the other side. The seam jump then reduces to
evaluation rounding; f2_regions.py assembles both sides in float64 and gates
the jump at <= 1 ulp once the HEADER lists are transcribed.

Exit non-zero on drift from HEADER_PSI.
"""
from __future__ import annotations

import sys

import numpy as np
from mpmath import mp, mpf, exp, fabs

from tanh_anti_common import (C2_EXACT, F2_exact, LN2, check_header, fma64,
                              horner_fma, lin_grid, print_coeff_block,
                              remez_relative, report_fit)

mp.dps = 45

A0 = mpf(1)
T0 = exp(-2 * A0)
DEGREE = 10
GRID_N = 2401

HEADER_PSI: list[float] | None = [
    1.0,
    -0.24999999999999908,
    0.11111111111083918,
    -0.062499999968526385,
    0.039999998128492524,
    -0.027777712682628235,
    0.02040674580837912,
    -0.015605040437770511,
    0.012162896722470562,
    -0.008934639114660923,
    0.0045376647188292134,
]


def main() -> int:
    print("=== region II: psi(t) = -Li2(-t)/t on [0, e^(-2*a0)] ===")
    print(f"interval t in [0, {mp.nstr(T0, 8)}], degree {DEGREE}, "
          f"seam-constrained at t0")
    print(f"Landen check: t0 = {mp.nstr(T0, 6)} < 0.5, the dilogNeg fold "
          f"never fires here")

    from tanh_anti_common import psi_exact
    grid = lin_grid(0, T0, GRID_N)
    print(f"caching oracle on {GRID_N} grid points (polylog per point, "
          f"done once)...")
    f_grid = [psi_exact(t) for t in grid]

    V = psi_exact(T0)
    print(f"seam value psi(t0) to 25 digits:\n    {mp.nstr(V, 25)}")

    coeffs, rep = remez_relative(f_grid, grid, DEGREE, constraint=(T0, V),
                                 label="psi(t)")
    ok = report_fit("psi(t), degree 10, seam-constrained", rep, coeffs, grid,
                    f_grid)

    # Series sanity: 1, -1/4, 1/9, -1/16 (1/(k+1)^2 alternating).
    print("\nleading coefficients vs the alternating series 1/(k+1)^2:")
    for k in range(4):
        t = mpf(-1) ** k / mpf((k + 1) ** 2)
        print(f"    c{k} = {mp.nstr(coeffs[k], 12):>18}   series "
              f"{mp.nstr(t, 12):>18}   diff {mp.nstr(fabs(coeffs[k] - t), 3)}")

    # Assembled region-II float64 F2 vs oracle over a in [a0, 19].
    print("\nassembled float64 region-II F2 (two-part ln2, Horner+FMA) vs "
          "oracle, a in [a0, 19]:")
    c64 = [float(c) for c in coeffs]
    ln2_hi = np.float64(float(LN2))
    ln2_lo = np.float64(float(LN2 - mpf(float(LN2))))
    c2 = np.float64(float(C2_EXACT))
    worst, wa = mpf(0), mpf(0)
    for a in lin_grid(A0, mpf(19), 1800):
        af = np.float64(float(a))
        t = np.exp(np.float64(-2.0) * af)          # np.exp: <= 1 ulp; the C++
        psi = horner_fma(c64, np.array([t]))[0]    # header owns its own exp
        E = np.float64(0.5) * t * psi
        h = (af - ln2_hi) - ln2_lo
        got = fma64(np.float64(0.5) * h, h, c2) - E
        ref = F2_exact(a)
        e = fabs((mpf(float(got)) - ref) / ref)
        if e > worst:
            worst, wa = e, a
    ulp = float(worst) / 2.0 ** -52
    print(f"    max rel err = {mp.nstr(worst, 4)} = {ulp:.3f} ulp at "
          f"a = {mp.nstr(wa, 6)}   (gate <= 2.5 ulp; completed square + "
          f"two-part ln2 doing the work)")
    ok &= ulp <= 2.5

    print_coeff_block("psi(t), low -> high", coeffs, "kF2RegionIIPsi")
    ok &= check_header(coeffs, HEADER_PSI, "psi(t)")
    print("\n=== PASS ===" if ok else "\n=== FAIL -- do not transcribe ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

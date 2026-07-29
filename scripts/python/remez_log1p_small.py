#!/usr/bin/env python3
"""
remez_log1p_small.py -- region-II remainder kernel for F1.

    F1(a) = (a - ln2) + t * L(t),   L(t) = log1p(t)/t,   t = e^(-2a)
    interval: t in [0, e^(-2*a0)] = [0, 0.13534] for a0 = 1.

This replaces the planned general-purpose SIMD log1p on [0, 1] (old SIMD-plan
Step 3.3): only this narrow interval is ever needed once region I owns
|x| <= a0, and a degree-10 polynomial covers it to ~1e-17 relative where the
[0, 1] fit would have needed the atanh transformation and more terms.

Error budget: an epsilon on L enters F1 as (t*L)*eps/F1; at a = a0 that
weight is log1p(t0)/F1(a0) = 0.1270/0.4339 ~ 0.29 and falls with a.

Fit: degree 10, relative-error Remez, CONSTRAINED at t0 = e^(-2*a0), pairing
with remez_f1_region1.py's constraint on the region-I side.

Exit non-zero on drift from HEADER_L.
"""
from __future__ import annotations

import sys

import numpy as np
from mpmath import mp, mpf, exp, fabs

from tanh_anti_common import (F1_exact, LN2, L_exact, check_header, horner_fma,
                              lin_grid, print_coeff_block, remez_relative,
                              report_fit)

mp.dps = 45

A0 = mpf(1)
T0 = exp(-2 * A0)
DEGREE = 10
GRID_N = 2401

HEADER_L: list[float] | None = [
    1.0,
    -0.49999999999998945,
    0.3333333333302044,
    -0.24999999963821531,
    0.19999997849986126,
    -0.16666591894168303,
    0.14284085417648296,
    -0.1247703631890595,
    0.10900311629907711,
    -0.08765558117293032,
    0.04729716544174639,
]


def main() -> int:
    print("=== region II: L(t) = log1p(t)/t on [0, e^(-2*a0)] ===")
    print(f"interval t in [0, {mp.nstr(T0, 8)}], degree {DEGREE}, "
          f"seam-constrained at t0")

    grid = lin_grid(0, T0, GRID_N)
    print(f"caching oracle on {GRID_N} grid points...")
    f_grid = [L_exact(t) for t in grid]

    V = L_exact(T0)
    print(f"seam value L(t0) to 25 digits:\n    {mp.nstr(V, 25)}")

    coeffs, rep = remez_relative(f_grid, grid, DEGREE, constraint=(T0, V),
                                 label="L(t)")
    ok = report_fit("L(t), degree 10, seam-constrained", rep, coeffs, grid,
                    f_grid)

    # Series sanity: log1p(t)/t = 1 - t/2 + t^2/3 - t^3/4 + ...
    print("\nleading coefficients vs the series (-1)^k/(k+1):")
    for k in range(4):
        t = mpf(-1) ** k / mpf(k + 1)
        print(f"    c{k} = {mp.nstr(coeffs[k], 12):>18}   series "
              f"{mp.nstr(t, 12):>18}   diff {mp.nstr(fabs(coeffs[k] - t), 3)}")

    # Assembled region-II float64 F1 vs oracle over a in [a0, 19].
    print("\nassembled float64 region-II F1 = (a - ln2) + t*L(t) "
          "(two-part ln2, Horner+FMA) vs oracle, a in [a0, 19]:")
    c64 = [float(c) for c in coeffs]
    ln2_hi = np.float64(float(LN2))
    ln2_lo = np.float64(float(LN2 - mpf(float(LN2))))
    worst, wa = mpf(0), mpf(0)
    for a in lin_grid(A0, mpf(19), 1800):
        af = np.float64(float(a))
        t = np.exp(np.float64(-2.0) * af)
        Lp = horner_fma(c64, np.array([t]))[0]
        got = ((af - ln2_hi) - ln2_lo) + t * Lp
        ref = F1_exact(a)
        e = fabs((mpf(float(got)) - ref) / ref)
        if e > worst:
            worst, wa = e, a
    ulp = float(worst) / 2.0 ** -52
    print(f"    max rel err = {mp.nstr(worst, 4)} = {ulp:.3f} ulp at "
          f"a = {mp.nstr(wa, 6)}   (gate <= 1.5 ulp)")
    ok &= ulp <= 1.5

    print_coeff_block("L(t), low -> high", coeffs, "kF1RegionIIL")
    ok &= check_header(coeffs, HEADER_L, "L(t)")
    print("\n=== PASS ===" if ok else "\n=== FAIL -- do not transcribe ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

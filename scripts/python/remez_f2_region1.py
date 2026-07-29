#!/usr/bin/env python3
"""
remez_f2_region1.py -- region-I kernel for F2.

    F2(x) = x * u * P(u),   u = x^2,   |x| <= a0 = 1,   P(0) = 1/6
    P analytic with radius pi^2/4 ~ 2.467 (poles of ln cosh at +-i*pi/2).

The x^3 factorization is the whole point: F2(0) == 0 exactly, odd parity is
bit-exact (the sign rides the leading x), and the assembled F2 relative
accuracy EQUALS the polynomial's relative accuracy -- no cancellation, which
is the defect this replaces (the closed form's relative error is unbounded
as x -> 0: measured 13% at x=1e-5, 170% at x=4e-6).

Fit: degree 14, relative-error Remez on u in [0, 1], CONSTRAINED to
interpolate the oracle at the seam u = a0^2 (one alternation point spent on
the constraint: 15 alternations, not 16). The seam constraint bounds the
region-I/II discontinuity to evaluation rounding; remez_dilog_psi.py carries
the matching constraint on the other side.

Report includes the log-spaced ASSEMBLED sweep x in [1e-12, a0]: float64
x*u*P(u) against the mpmath oracle. That sweep IS the headline deliverable
(<= 2 ulp relative where the old code was unbounded); if it fails, do not
transcribe anything.

Exit non-zero on drift from HEADER_P (remez_tan.py convention).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from mpmath import mp, mpf, fabs

from tanh_anti_common import (F2_exact, P_exact, basis_condition,
                              check_header, fma64, horner_fma, lin_grid,
                              log_grid, print_coeff_block, remez_relative,
                              report_fit)

# The region-I monomial basis condition number is committed here so the C++
# harness can compare its runtime measurement against it (transcription
# errors move this number even when the fit grid still looks accurate).
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "tests",
                        "logs", "baseline", "f2_region1_basis_condition.txt")

mp.dps = 45

A0 = mpf(1)
U_SEAM = A0 * A0
DEGREE = 14
GRID_N = 3001

# Committed values (drift-check targets). None until first transcription.
HEADER_P: list[float] | None = [
    0.16666666666666666,
    -0.016666666666666594,
    0.003174603174597562,
    -0.0007495590827244051,
    0.00019881352950739333,
    -5.6815587325387805e-05,
    1.710521585280172e-05,
    -5.351664370785777e-06,
    1.723185777048654e-06,
    -5.648235330986546e-07,
    1.8413852923818012e-07,
    -5.656464155399535e-08,
    1.4778914550004e-08,
    -2.772833176805685e-09,
    2.671491545614661e-10,
]


def main() -> int:
    print("=== region I: P(u) for F2(x) = x*u*P(u), u = x^2 ===")
    print(f"interval u in [0, {mp.nstr(U_SEAM, 6)}], degree {DEGREE}, "
          f"seam-constrained at u = {mp.nstr(U_SEAM, 6)}")

    grid = lin_grid(0, U_SEAM, GRID_N)
    print(f"caching oracle on {GRID_N} grid points (the slow part, done once)...")
    f_grid = [P_exact(u) for u in grid]

    V = P_exact(U_SEAM)
    print(f"seam value P(a0^2) to 25 digits (remez_dilog_psi.py pairs with "
          f"this):\n    {mp.nstr(V, 25)}")

    coeffs, rep = remez_relative(f_grid, grid, DEGREE, constraint=(U_SEAM, V),
                                 label="P(u)")
    ok = report_fit("P(u), degree 14, seam-constrained", rep, coeffs, grid,
                    f_grid)

    # Taylor sanity: leading coefficients must approach 1/6, -1/60, 1/315.
    tay = [mpf(1) / 6, -mpf(1) / 60, mpf(1) / 315, -mpf(17) / 22680]
    print("\nleading coefficients vs Taylor (informational: minimax rebalances\n"
          "the monomial basis, so agreement tightens only toward low orders):")
    for k, t in enumerate(tay):
        print(f"    c{k} = {mp.nstr(coeffs[k], 12):>18}   taylor "
              f"{mp.nstr(t, 12):>18}   diff {mp.nstr(fabs(coeffs[k] - t), 3)}")

    # THE deliverable: assembled float64 F2 on a log-spaced x sweep.
    print("\nassembled float64 F2(x) = x*u*P(u) (Horner+FMA) vs oracle, "
          "log-spaced x in [1e-12, a0]:")
    c64 = [float(c) for c in coeffs]
    xs = log_grid(mpf("1e-12"), A0, 1500)
    worst = mpf(0)
    wx = mpf(0)
    for x in xs:
        xf = np.float64(float(x))
        uf = xf * xf
        got = xf * uf * horner_fma(c64, np.array([uf]))[0]
        ref = F2_exact(x)
        e = fabs((mpf(float(got)) - ref) / ref)
        if e > worst:
            worst, wx = e, x
    ulp = float(worst) / 2.0 ** -52
    print(f"    max rel err = {mp.nstr(worst, 4)} = {ulp:.3f} ulp at "
          f"x = {mp.nstr(wx, 6)}   (gate <= 2.5 ulp; old code: unbounded)")
    ok &= ulp <= 2.5

    # Structural exactness in float64 (these are the invariants N5 gates).
    z = np.float64(0.0)
    f2z = z * (z * z) * horner_fma(c64, np.array([z * z]))[0]
    print(f"    F2(0) float64 = {float(f2z)!r}  "
          f"({'exact zero: PASS' if f2z == 0.0 else 'FAIL'})")
    ok &= f2z == 0.0

    print_coeff_block("P(u), low -> high", coeffs, "kF2RegionI")
    ok &= check_header(coeffs, HEADER_P, "P(u)")

    if ok:
        kcond, ku = basis_condition(coeffs, grid, f_grid)
        with open(LOG_PATH, "w") as fh:
            fh.write("# region-I monomial basis condition for P(u), "
                     "measured by remez_f2_region1.py\n")
            fh.write("# max_u sum_k |c_k| u^k / |P(u)| over u in [0, a0^2]\n")
            fh.write(f"{mp.nstr(kcond, 12)}   # at u = {mp.nstr(ku, 6)}\n")
        print(f"\nbasis condition {mp.nstr(kcond, 6)} written to {LOG_PATH}")

    print("\n=== PASS ===" if ok else "\n=== FAIL -- do not transcribe ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

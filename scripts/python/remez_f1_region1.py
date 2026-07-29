#!/usr/bin/env python3
"""
remez_f1_region1.py -- region-I kernel for F1.

    F1(x) = u * S(u),   u = x^2,   |x| <= a0 = 1,   S(0) = 1/2

Same structural story as P(u) (see remez_f2_region1.py): F1(0) == 0 becomes
EXACT BY CONSTRUCTION -- in the current header it is a rounding coincidence
(0 - ln2 + log1p(1) happens to round back to zero) -- even parity is
bit-exact through u = x^2, and region I needs NO transcendental at all:
no exp, no log1p.

Fit: degree 14, relative-error Remez on u in [0, 1], seam-constrained at
u = a0^2 so F1 is continuous into region II's (a - ln2) + t*L(t) form to
evaluation rounding (remez_log1p_small.py carries the matching constraint).

Exit non-zero on drift from HEADER_S.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from mpmath import mp, mpf, fabs

from tanh_anti_common import (F1_exact, S_exact, basis_condition,
                              check_header, horner_fma, lin_grid, log_grid,
                              print_coeff_block, remez_relative, report_fit)

# Same committed-record role as remez_f2_region1.py's log file: the C++
# harness compares its runtime measurement against this value.
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "tests",
                        "logs", "baseline", "f1_region1_basis_condition.txt")

mp.dps = 45

A0 = mpf(1)
U_SEAM = A0 * A0
DEGREE = 14
GRID_N = 3001

HEADER_S: list[float] | None = [
    0.5,
    -0.08333333333333116,
    0.022222222222058514,
    -0.0067460317411381305,
    0.0021869487768058212,
    -0.0007386022322834027,
    0.00025657604293932094,
    -9.097020257139513e-05,
    3.271979510502662e-05,
    -1.1823955824000437e-05,
    4.188108537857103e-06,
    -1.3734342529622273e-06,
    3.760822745593593e-07,
    -7.280182752600742e-08,
    7.156534675180096e-09,
]


def main() -> int:
    print("=== region I: S(u) for F1(x) = u*S(u), u = x^2 ===")
    print(f"interval u in [0, {mp.nstr(U_SEAM, 6)}], degree {DEGREE}, "
          f"seam-constrained at u = {mp.nstr(U_SEAM, 6)}")

    grid = lin_grid(0, U_SEAM, GRID_N)
    print(f"caching oracle on {GRID_N} grid points...")
    f_grid = [S_exact(u) for u in grid]

    V = S_exact(U_SEAM)
    print(f"seam value S(a0^2) to 25 digits (remez_log1p_small.py pairs with "
          f"this):\n    {mp.nstr(V, 25)}")

    coeffs, rep = remez_relative(f_grid, grid, DEGREE, constraint=(U_SEAM, V),
                                 label="S(u)")
    ok = report_fit("S(u), degree 14, seam-constrained", rep, coeffs, grid,
                    f_grid)

    # Taylor sanity: ln cosh = x^2/2 - x^4/12 + x^6/45 - 17x^8/2520 + ...
    tay = [mpf(1) / 2, -mpf(1) / 12, mpf(1) / 45, -mpf(17) / 2520]
    print("\nleading coefficients vs Taylor:")
    for k, t in enumerate(tay):
        print(f"    c{k} = {mp.nstr(coeffs[k], 12):>18}   taylor "
              f"{mp.nstr(t, 12):>18}   diff {mp.nstr(fabs(coeffs[k] - t), 3)}")

    print("\nassembled float64 F1(x) = u*S(u) (Horner+FMA) vs oracle, "
          "log-spaced x in [1e-12, a0]:")
    c64 = [float(c) for c in coeffs]
    xs = log_grid(mpf("1e-12"), A0, 1500)
    worst, wx = mpf(0), mpf(0)
    for x in xs:
        xf = np.float64(float(x))
        uf = xf * xf
        got = uf * horner_fma(c64, np.array([uf]))[0]
        ref = F1_exact(x)
        e = fabs((mpf(float(got)) - ref) / ref)
        if e > worst:
            worst, wx = e, x
    ulp = float(worst) / 2.0 ** -52
    # Gate derivation (a-priori, NOT tuned to a measurement): the assembled
    # value is fl(u * S(fl(x^2))). Budget: 1/2 ulp from u = x^2, 1/2 ulp from
    # the final multiply, <= 0.9 ulp measured Horner+FMA polynomial error,
    # ~0.1 ulp fit error => conditioned bound 2.0 ulp. (An earlier 1.5 guess
    # came from the spec prototype's table and had no budget behind it.)
    print(f"    max rel err = {mp.nstr(worst, 4)} = {ulp:.3f} ulp at "
          f"x = {mp.nstr(wx, 6)}   (gate <= 2.0 ulp, see budget above)")
    ok &= ulp <= 2.0

    # Invariants: F1(0) exact zero, S > 0 on the interval (F1 >= 0 gate).
    z = np.float64(0.0)
    f1z = (z * z) * horner_fma(c64, np.array([z * z]))[0]
    print(f"    F1(0) float64 = {float(f1z)!r}  "
          f"({'exact zero: PASS' if f1z == 0.0 else 'FAIL'})")
    ok &= f1z == 0.0
    smin = min(f_grid)
    print(f"    min S(u) on the interval = {mp.nstr(smin, 6)}  "
          f"({'> 0: F1 >= 0 structural, PASS' if smin > 0 else 'FAIL'})")
    ok &= smin > 0

    print_coeff_block("S(u), low -> high", coeffs, "kF1RegionI")
    ok &= check_header(coeffs, HEADER_S, "S(u)")

    if ok:
        kcond, ku = basis_condition(coeffs, grid, f_grid)
        with open(LOG_PATH, "w") as fh:
            fh.write("# region-I monomial basis condition for S(u), "
                     "measured by remez_f1_region1.py\n")
            fh.write("# max_u sum_k |c_k| u^k / |S(u)| over u in [0, a0^2]\n")
            fh.write(f"{mp.nstr(kcond, 12)}   # at u = {mp.nstr(ku, 6)}\n")
        print(f"\nbasis condition {mp.nstr(kcond, 6)} written to {LOG_PATH}")

    print("\n=== PASS ===" if ok else "\n=== FAIL -- do not transcribe ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

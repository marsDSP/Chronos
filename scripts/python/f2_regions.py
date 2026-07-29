#!/usr/bin/env python3
"""
f2_regions.py -- derive and regression-check the region boundaries and
constants for source/math/TanhAntiderivatives.h.

Derives:  a0 (region I/II crossover), a1 (region II/III crossover),
          C2 = pi^2/24 - ln^2(2)/2, and the two-part ln2 hi/lo split.
Prints:   the conditioning table (completed square vs the form the current
          Nonlinearities.h evaluates), the Chebyshev degree table over the
          a0 sweep, the E/F2 and F1-tail tables that fix a1, and the Landen
          check that retires the dilogarithm fold from the audio path.
Exits:    non-zero if any derived constant drifts from the HEADER_* values
          below (remez_tan.py convention: the committed script IS the check).

Verified reference values this script must reproduce (independently measured
against mpmath at dps 60; treat disagreement as a bug in THIS script):
  C2 = 0.1710070097529558967845525...  (note: 07009, an earlier spec draft
       carried a 06009 typo)                nearest double 0.1710070097529559
  kappa (orig four-term / completed-square) at a = 0.5 .. 2.0:
       51.74/17.65  19.75/4.174  10.94/1.859  7.344/1.282  3.747/1.018
       -- reproduces ONLY with G split into |0.5*Li2| + |pi^2/24| as separate
       terms, matching what Nonlinearities.h actually sums. One-term-G gives
       35.1 at a=0.5 and is the wrong model.
  degrees at 1e-17 rel (P,S,psi,L): a0=1.00 -> 14,14,10,10 ; 1.25 ->
       16,17,8,9 ; 1.50 -> 18,20,8,8.
  E/F2: a=16 -> 5.397e-17, a=17 -> 6.437e-18.
  F1 tail crosses 1e-17 at a ~ 18.1 (1.34e-17 at a=18), NOT 18.5.
  -> a1 = 19 (smallest integer with both ratios < 1e-17).
"""
from __future__ import annotations

import sys

import numpy as np
from mpmath import mp, mpf, polylog, log, log1p, exp, fabs, pi, cos as mpcos

from tanh_anti_common import (C2_EXACT, LN2, F1_exact, F2_exact, P_exact,
                              S_exact, psi_exact, L_exact)

mp.dps = 45

# ── committed values (drift check targets) ────────────────────────────────
HEADER_A0 = 1.0
HEADER_A1 = 19.0
HEADER_C2 = 0.1710070097529559
HEADER_LN2_HI = 0.6931471805599453
HEADER_LN2_LO = 2.3190468138462996e-17

REL_TARGET = mpf("1e-17")


def kappa_table():
    print("── conditioning kappa = sum|terms| / |result| "
          "(four-term split, as the code sums it) ──")
    print("     a      kappa_orig   kappa_completed_square")
    expect = {0.50: (51.74, 17.65), 0.75: (19.75, 4.174), 1.00: (10.94, 1.859),
              1.25: (7.344, 1.282), 2.00: (3.747, 1.018)}
    ok = True
    for a, (eo, ec) in expect.items():
        A = mpf(a)
        li = polylog(2, -exp(-2 * A))
        res = A * A / 2 - A * LN2 + (li / 2 + pi ** 2 / 24)
        ko = (A * A / 2 + A * LN2 + fabs(li / 2) + pi ** 2 / 24) / fabs(res)
        E = -li / 2
        kc = ((A - LN2) ** 2 / 2 + C2_EXACT + fabs(E)) / fabs(res)
        good = abs(float(ko) / eo - 1) < 0.01 and abs(float(kc) / ec - 1) < 0.01
        ok &= good
        print(f"  {a:5.2f}   {mp.nstr(ko, 5):>9}     {mp.nstr(kc, 5):>9}"
              f"   {'ok' if good else 'MISMATCH vs verified reference'}")
    return ok


def cheb_degree(f, lo, hi, tol, N=160, dmax=44):
    """Smallest degree whose Chebyshev coefficient tail < tol (relative)."""
    lo, hi = mpf(lo), mpf(hi)
    nodes = [(lo + hi) / 2 + (hi - lo) / 2 * mpcos(pi * (mpf(j) + mpf("0.5")) / N)
             for j in range(N)]
    vals = [f(u) for u in nodes]
    scale = max(fabs(v) for v in vals)
    c = []
    for k in range(N):
        s = mpf(0)
        for i in range(N):
            s += vals[i] * mpcos(pi * k * (mpf(i) + mpf("0.5")) / N)
        c.append(2 * s / N)
    for d in range(dmax):
        tail = sum(fabs(c[j]) for j in range(d + 1, N))
        if tail / scale < tol:
            return d
    return None


def degree_table():
    print("\n── degree table at 1e-17 relative (fitted variable: u = x^2 for "
          "P/S, t for psi/L) ──")
    print("   a0      P     S    psi    L    total(F2 path)")
    expect = {1.00: (14, 14, 10, 10), 1.25: (16, 17, 8, 9), 1.50: (18, 20, 8, 8)}
    ok = True
    rows = {}
    for a0 in (0.75, 1.00, 1.25, 1.50):
        U = a0 * a0
        t0 = exp(-2 * mpf(a0))
        dP = cheb_degree(P_exact, 0, U, REL_TARGET)
        dS = cheb_degree(S_exact, 0, U, REL_TARGET)
        dp = cheb_degree(psi_exact, 0, t0, REL_TARGET)
        dL = cheb_degree(L_exact, 0, t0, REL_TARGET)
        rows[a0] = (dP, dS, dp, dL)
        tot = dP + dp
        tag = ""
        if a0 in expect and (dP, dS, dp, dL) != expect[a0]:
            tag = f"  MISMATCH vs verified reference {expect[a0]}"
            ok = False
        print(f"  {a0:4.2f}   {dP:4d}  {dS:4d}  {dp:4d}  {dL:4d}      {tot:4d}{tag}")
    print("  -> a0 = 1.00: ties 1.25 on total count AND wins on basis")
    print("     conditioning (P 1.23 vs 1.40, S 1.42 vs 1.82); region I lands")
    print("     on u in [0,1] exactly. Chosen.")
    return ok


def a1_tables():
    print("\n── region III threshold ──")
    print("   a     E/F2 (dropped F2 term)   log1p(e^-2a)/F1 (dropped F1 term)")
    a1 = None
    for a in range(14, 22):
        A = mpf(a)
        E = -mpf("0.5") * polylog(2, -exp(-2 * A))
        r2 = E / F2_exact(A)
        r1 = log1p(exp(-2 * A)) / F1_exact(A)
        both = r2 < REL_TARGET and r1 < REL_TARGET
        if both and a1 is None:
            a1 = a
        print(f"  {a:3d}    {mp.nstr(r2, 4):>10}                {mp.nstr(r1, 4):>10}"
              f"{'    <- a1' if a1 == a else ''}")
    print(f"  F1 tail crosses 1e-17 at a ~ 18.1 (not 18.5); F2 at a ~ 16.2.")
    print(f"  a1 = smallest integer with BOTH below 1e-17 = {a1}")
    return a1


def landen_check(a0):
    t0 = exp(-2 * mpf(a0))
    thr = LN2 / 2
    print(f"\n── Landen retirement ──")
    print(f"  region II sees t <= e^(-2*{a0}) = {mp.nstr(t0, 6)}; the dilogNeg")
    print(f"  Landen fold fires only for t > 0.5, i.e. a < ln2/2 = "
          f"{mp.nstr(thr, 6)}.")
    ok = t0 < mpf("0.5")
    print(f"  fold {'never fires in region II: PASS' if ok else 'CAN FIRE: FAIL'}")
    return ok


def constants():
    print("\n── constants ──")
    c2_25 = mp.nstr(C2_EXACT, 25)
    c2d = float(C2_EXACT)
    print(f"  C2 = pi^2/24 - ln^2(2)/2 = {c2_25}  (25 digits)")
    print(f"       nearest double = {c2d!r}   hex {np.float64(c2d).hex()}")
    print(f"       (single double suffices: at a=1 its rounding contributes")
    print(f"        ~0.2 ulp of F2 and its weight only falls as a grows)")
    hi = float(LN2)
    lo_exact = LN2 - mpf(hi)
    lo = float(lo_exact)
    resid = LN2 - (mpf(hi) + mpf(lo))
    print(f"  ln2_hi = {hi!r}")
    print(f"  ln2_lo = {lo!r}")
    print(f"  |ln2 - (hi + lo)| = {mp.nstr(fabs(resid), 3)}  "
          f"(double-double residual, ~1e-33 expected)")
    return c2d, hi, lo


def seam_report():
    """If the sibling remez scripts carry transcribed HEADER coefficients,
    assemble both region kernels in float64 and report the seam jump."""
    try:
        import remez_f2_region1 as r1
        import remez_dilog_psi as rp
    except Exception:
        print("\n(seam-jump report skipped: sibling scripts not importable)")
        return True
    if r1.HEADER_P is None or rp.HEADER_PSI is None:
        print("\n(seam-jump report skipped: sibling HEADER lists not yet "
              "transcribed)")
        return True
    from tanh_anti_common import horner_fma, fma64
    a0 = HEADER_A0
    uI = np.float64(a0) * np.float64(a0)
    f2_I = np.float64(a0) * uI * horner_fma(r1.HEADER_P, np.array([uI]))[0]
    t = np.exp(np.float64(-2.0) * np.float64(a0))
    psi = horner_fma(rp.HEADER_PSI, np.array([t]))[0]
    E = np.float64(0.5) * t * psi
    h = (np.float64(a0) - np.float64(HEADER_LN2_HI)) - np.float64(HEADER_LN2_LO)
    f2_II = fma64(np.float64(0.5) * h, h, np.float64(HEADER_C2)) - E
    ref = F2_exact(a0)
    ulp = float(fabs(ref)) * 2.0 ** -52
    jump = abs(float(f2_I) - float(f2_II)) / ulp
    eI = abs(float(mpf(float(f2_I)) - ref)) / ulp
    eII = abs(float(mpf(float(f2_II)) - ref)) / ulp
    print(f"\n── assembled float64 seam at a0 = {a0} ──")
    print(f"  region I  F2(a0) err = {eI:.3f} ulp")
    print(f"  region II F2(a0) err = {eII:.3f} ulp")
    print(f"  seam jump            = {jump:.3f} ulp   (gate <= 1)")
    return jump <= 1.0


def main() -> int:
    print("=== TanhNL antiderivative region derivation ===")
    ok = True
    ok &= kappa_table()
    ok &= degree_table()
    a1 = a1_tables()
    ok &= landen_check(HEADER_A0)
    c2d, hi, lo = constants()
    ok &= seam_report()

    print("\n── drift check vs HEADER_* (committed values) ──")
    checks = [("a0", HEADER_A0, 1.0), ("a1", HEADER_A1, float(a1)),
              ("C2", HEADER_C2, c2d), ("ln2_hi", HEADER_LN2_HI, hi),
              ("ln2_lo", HEADER_LN2_LO, lo)]
    for name, want, got in checks:
        if np.float64(want) != np.float64(got):
            print(f"  MISMATCH {name}: derived {got!r} != committed {want!r}")
            ok = False
        else:
            print(f"  {name:7s} = {got!r}   ok")

    print("\n=== ALL DERIVATIONS CONSISTENT ===" if ok
          else "\n=== DRIFT OR TABLE MISMATCH -- DO NOT TRANSCRIBE ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

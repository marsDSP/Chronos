#!/usr/bin/env python3
"""
tanh_anti_common.py -- shared engine for the TanhNL antiderivative derivations.

Used by:
    f2_regions.py           region boundaries a0/a1, C2, ln2 hi/lo, tables
    remez_f2_region1.py     P(u):  F2(x) = x*u*P(u),  u = x^2,  |x| <= a0
    remez_f1_region1.py     S(u):  F1(x) = u*S(u)
    remez_dilog_psi.py      psi(t) = -Li2(-t)/t   on [0, e^(-2*a0)]
    remez_log1p_small.py    L(t)   = log1p(t)/t   on [0, e^(-2*a0)]

Why this file exists (deviation from the one-file-per-script convention of
remez_{sin,cos,tan}.py, deliberate): those fits are float32 and numpy float64
is a valid derivation precision. These kernels are float64, so the fit target
(1e-17 relative) sits BELOW float64 eps and the whole derivation must run in
mpmath. The Remez engine, the oracles, and the correctly-rounded-FMA float64
error reports are ~500 lines that four scripts would otherwise quadruplicate.

Numerical ground rules, in order of importance:
  1. The oracle is mpmath at mp.dps = 45 (Li2 via mpmath.polylog). Near u = 0
     the closed form for F2 cancels ~5 decimal digits at the smallest grid
     point; 45 digits leaves ~40, which is 23 digits of margin over the 1e-17
     fit target.
  2. Fits are RELATIVE-error minimax. For P and S that is structural: the
     factored forms make the assembled F2/F1 relative accuracy equal the
     polynomial's relative accuracy, with zero cancellation.
  3. The region-I and region-II fits are CONSTRAINED to interpolate the oracle
     at the seam (u = a0^2 for P/S, t = e^(-2*a0) for psi/L). Implemented by
     substitution: p(u) = V + (u - U)*q(u), which makes the constraint exact
     by construction and costs one alternation point (deg-d fit -> d+1
     alternations instead of d+2).
  4. float64 error reports simulate the header's evaluation: Horner-with-FMA
     and Estrin-with-FMA, using a correctly-rounded software FMA built from
     Dekker two-product + 2Sum (this box's Python has no math.fma). The
     emulation can double-round in astronomically rare halfway cases; that
     perturbs the REPORT by <= 1 ulp in the last digit and cannot flip the
     2e-16 gate, which the fits pass with >= 25% margin.
  5. The extrema scan carries a noise floor (same gotcha remez_tan.py
     documents): residuals below max|r| * 1e-9 are treated as flat, otherwise
     the region around a constrained seam (where r == 0 exactly) reports
     bogus alternations.
"""
from __future__ import annotations

import numpy as np
from mpmath import mp, mpf, polylog, log, log1p, exp, cosh, fabs, pi, sqrt as mpsqrt, cos as mpcos

mp.dps = 45

# ──────────────────────────────────────────────────────────────────────────
# Oracles (all mpf in, mpf out)
# ──────────────────────────────────────────────────────────────────────────

LN2 = log(mpf(2))
C2_EXACT = pi ** 2 / 24 - LN2 ** 2 / 2          # 0.17100700975295589678455...


def F2_exact(x):
    """
    F2(x) = int_0^x ln cosh(u) du, closed form via Li2. Odd.

    CONDITIONING: near zero the closed form cancels from terms of size
    ~a*ln2 down to a^3/6 -- about 0.62 + 2*log10(1/a) decimal digits. An
    oracle that ignores this reports ITS OWN error as the kernel's (measured:
    2e-10 apparent rel err at x = 1e-12 from a fixed-dps oracle). Work at
    dps + digits_lost + 8 so the returned value keeps full base precision.
    """
    x = mpf(x)
    a = fabs(x)
    if a == 0:
        return mpf(0)
    extra = 0
    if a < mpf("0.5"):
        extra = int(2 * float(-mp.log10(a))) + 8
    with mp.workdps(mp.dps + max(0, min(extra, 80))):
        g = mpf("0.5") * polylog(2, -exp(-2 * a)) + pi ** 2 / 24
        v = a * a / 2 - a * log(2) + g
        v = +v
    return v if x >= 0 else -v


def F1_exact(x):
    """F1(x) = ln cosh(x). Even. Same small-|x| conditioning bump as F2:
    log(cosh a) ~ a^2/2 while cosh a ~ 1, losing ~2*log10(1/a) digits."""
    x = mpf(x)
    a = fabs(x)
    if a == 0:
        return mpf(0)
    extra = 0
    if a < mpf("0.5"):
        extra = int(2 * float(-mp.log10(a))) + 8
    with mp.workdps(mp.dps + max(0, min(extra, 80))):
        v = +log(cosh(a))
    return v


def P_exact(u):
    """P(u) = F2(x)/(x*u), u = x^2. P(0) = 1/6. Analytic, radius pi^2/4."""
    u = mpf(u)
    if u == 0:
        return mpf(1) / 6
    x = mpsqrt(u)
    return F2_exact(x) / (x * u)


def S_exact(u):
    """S(u) = F1(x)/u, u = x^2. S(0) = 1/2."""
    u = mpf(u)
    if u == 0:
        return mpf("0.5")
    return F1_exact(mpsqrt(u)) / u


def psi_exact(t):
    """psi(t) = -Li2(-t)/t = 1 - t/4 + t^2/9 - ... ; psi(0) = 1."""
    t = mpf(t)
    if t == 0:
        return mpf(1)
    return -polylog(2, -t) / t


def L_exact(t):
    """L(t) = log1p(t)/t ; L(0) = 1."""
    t = mpf(t)
    if t == 0:
        return mpf(1)
    return log1p(t) / t


# ──────────────────────────────────────────────────────────────────────────
# Correctly-rounded software FMA (vectorized float64), Dekker + 2Sum.
# ──────────────────────────────────────────────────────────────────────────

_SPLITTER = np.float64(134217729.0)  # 2^27 + 1


def _two_prod(a, b):
    """p + e == a*b exactly (Dekker/Veltkamp, no hardware FMA needed)."""
    p = a * b
    ca = _SPLITTER * a
    ah = ca - (ca - a)
    al = a - ah
    cb = _SPLITTER * b
    bh = cb - (cb - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _two_sum(a, b):
    """s + e == a + b exactly (Knuth 2Sum)."""
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e


def fma64(a, b, c):
    """round-to-nearest(a*b + c) in float64 (double-rounds only on rare ties)."""
    p, ep = _two_prod(np.float64(a), np.float64(b))
    s, es = _two_sum(p, np.float64(c))
    return s + (ep + es)


def horner_fma(coeffs, u):
    """Horner in float64 with fused steps; coeffs low->high (python floats)."""
    u = np.asarray(u, dtype=np.float64)
    acc = np.full_like(u, np.float64(coeffs[-1]))
    for ck in reversed(coeffs[:-1]):
        acc = fma64(acc, u, np.float64(ck))
    return acc


def horner_nofma(coeffs, u):
    u = np.asarray(u, dtype=np.float64)
    acc = np.full_like(u, np.float64(coeffs[-1]))
    for ck in reversed(coeffs[:-1]):
        acc = acc * u + np.float64(ck)
    return acc


def estrin_fma(coeffs, u):
    """
    Estrin in float64 with fused combines. Pads to a power-of-two count with
    zeros; the header must document its own (equivalent) op order in C++.
    """
    u = np.asarray(u, dtype=np.float64)
    level = [np.full_like(u, np.float64(c)) for c in coeffs]
    n = 1
    while n < len(level):
        n *= 2
    level += [np.zeros_like(u)] * (n - len(level))
    w = u.copy()
    while len(level) > 1:
        level = [fma64(level[i + 1], w, level[i]) for i in range(0, len(level), 2)]
        w = w * w
    return level[0]


# ──────────────────────────────────────────────────────────────────────────
# Relative-error Remez (mpmath), optional seam interpolation constraint.
# ──────────────────────────────────────────────────────────────────────────

def _mp_polyval(coeffs, u):
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = acc * u + c
    return acc


def _expand_constrained(q, U, V):
    """p(u) = V + (u - U) q(u)  ->  monomial coefficients of p."""
    d = len(q)
    c = [mpf(0)] * (d + 1)
    c[0] = V - U * q[0]
    for k in range(1, d):
        c[k] = q[k - 1] - U * q[k]
    c[d] = q[d - 1]
    return c


def _alternating_extrema(grid, r, m, noise_floor):
    """
    Indices of up to m sign-alternating local extrema of the residual r
    (mpf array) over grid, largest magnitudes first when trimming.
    Same idea as remez_tan.py's scan: extrema below the floor are flat noise.
    """
    n = len(grid)
    cand = []
    for i in range(n):
        v = r[i]
        if fabs(v) < noise_floor:
            continue
        lo = r[i - 1] if i > 0 else None
        hi = r[i + 1] if i < n - 1 else None
        is_max = (lo is None or v >= lo) and (hi is None or v >= hi) and v > 0
        is_min = (lo is None or v <= lo) and (hi is None or v <= hi) and v < 0
        if is_max or is_min:
            if cand and (r[cand[-1]] > 0) == (v > 0):
                if fabs(v) > fabs(r[cand[-1]]):
                    cand[-1] = i          # same sign: keep the larger
            else:
                cand.append(i)
    while len(cand) > m:                   # trim the weaker endpoint
        if fabs(r[cand[0]]) < fabs(r[cand[-1]]):
            cand.pop(0)
        else:
            cand.pop()
    return cand


def remez_relative(f_grid, grid, degree, constraint=None, max_iter=60,
                   ripple_tol=1.02, label=""):
    """
    Minimax fit of a degree-`degree` polynomial to f on `grid` in RELATIVE
    error. `f_grid` are precomputed mpf oracle values on `grid` (the oracle is
    the expensive part; the grid is fixed, so it is evaluated exactly once --
    the same caching trick remez_tan.py uses for np.tan).

    constraint = (U, V): force p(U) == V exactly (seam interpolation).

    Returns (coeffs_low_to_high, report) where report carries the achieved
    minimax relative error, ripple ratio, extrema, and iteration count.
    """
    if constraint is None:
        nfree = degree + 1
    else:
        U, V = mpf(constraint[0]), mpf(constraint[1])
        nfree = degree                     # q has `degree` coefficients
    m = nfree + 1                          # alternation count

    # Chebyshev points of the first kind (open: never lands exactly on the
    # constrained endpoint, whose zero residual would degenerate the solve).
    lo, hi = grid[0], grid[-1]
    nodes = [lo + (hi - lo) * (1 + mpcos(pi * (mpf(2 * i + 1)) / (2 * m))) / 2
             for i in range(m)]
    nodes.sort()

    def f_at(u):
        # nearest cached grid value (grid is dense; nodes are grid-snapped
        # after iteration 1 anyway). For the initial Chebyshev nodes, snap.
        i = min(range(len(grid)), key=lambda j: abs(grid[j] - u))
        return grid[i], f_grid[i]

    nodes = [f_at(u)[0] for u in nodes]
    nodes = sorted(set(nodes))
    while len(nodes) < m:                  # dedupe collapsed nodes: pad
        nodes = sorted(set(nodes + [lo + (hi - lo) * mpf(k) / (m + 1)
                                    for k in range(1, m + 1)]))[:m]

    best = None
    it_used = 0
    for it in range(max_iter):
        it_used = it + 1
        A = mp.matrix(m, m)
        rhs = mp.matrix(m, 1)
        for i, u in enumerate(nodes):
            fu = f_at(u)[1]
            sig = mpf(-1) ** i
            if constraint is None:
                for k in range(degree + 1):
                    A[i, k] = u ** k
                A[i, m - 1] = -sig * fu
                rhs[i] = fu
            else:
                for k in range(degree):
                    A[i, k] = (u - U) * u ** k
                A[i, m - 1] = -sig * fu
                rhs[i] = fu - V
        sol = mp.lu_solve(A, rhs)
        if constraint is None:
            coeffs = [sol[k] for k in range(degree + 1)]
        else:
            coeffs = _expand_constrained([sol[k] for k in range(degree)], U, V)

        r = [(_mp_polyval(coeffs, u) - fv) / fv for u, fv in zip(grid, f_grid)]
        rmax = max(fabs(v) for v in r)
        floor = rmax * mpf("1e-9")
        ext_idx = _alternating_extrema(grid, r, m, floor)

        if best is None or rmax < best["max_rel"]:
            ext_v = [r[i] for i in ext_idx]
            ripple = (max(fabs(v) for v in ext_v) / min(fabs(v) for v in ext_v)
                      if len(ext_v) == m and min(fabs(v) for v in ext_v) > 0
                      else mpf("inf"))
            best = {"coeffs": coeffs, "max_rel": rmax, "ripple": ripple,
                    "extrema": [(grid[i], r[i]) for i in ext_idx],
                    "iterations": it_used, "label": label}

        if len(ext_idx) < m:               # pad from a uniform grid (gotcha 2)
            pad = [f_at(lo + (hi - lo) * mpf(k) / (m + 1))[0]
                   for k in range(1, m + 1)]
            nodes = sorted(set([grid[i] for i in ext_idx] + pad))[:m]
            continue
        new_nodes = [grid[i] for i in ext_idx]
        if best["ripple"] <= ripple_tol and new_nodes == nodes:
            break
        if new_nodes == nodes:
            break
        nodes = new_nodes
    best["iterations"] = it_used
    return best["coeffs"], best


# ──────────────────────────────────────────────────────────────────────────
# Reports and drift checks
# ──────────────────────────────────────────────────────────────────────────

ULP_SCALE = 2.0 ** -52       # 1 ulp of a double at magnitude ~1 (relative)


def rel_err_float64(eval_fn, coeffs64, grid_mp, f_grid):
    """max relative error of a float64 evaluation vs the mpf oracle values."""
    u64 = np.array([float(u) for u in grid_mp], dtype=np.float64)
    got = eval_fn(coeffs64, u64)
    worst, wu = mpf(0), mpf(0)
    for g, u, fv in zip(got, grid_mp, f_grid):
        e = fabs((mpf(float(g)) - fv) / fv)
        if e > worst:
            worst, wu = e, u
    return worst, wu


def basis_condition(coeffs, grid_mp, f_grid):
    """max over the interval of sum_k |c_k| u^k / |p(u)| (monomial basis)."""
    worst, wu = mpf(0), mpf(0)
    abs_c = [fabs(c) for c in coeffs]
    for u, fv in zip(grid_mp, f_grid):
        num = _mp_polyval(abs_c, fabs(u))
        k = num / fabs(fv)
        if k > worst:
            worst, wu = k, u
    return worst, wu


def print_coeff_block(name, coeffs, cpp_array_name):
    """C++ transcription block + hex so the drift check is bit-meaningful."""
    print(f"\ncoefficients for source/math/TanhAntiderivatives.h ({name}):")
    print(f"    inline constexpr std::array<double, {len(coeffs)}> {cpp_array_name} {{{{")
    for c in coeffs:
        d = float(c)
        print(f"        {d!r},   // {np.float64(d).hex()}")
    print("    }};")


def check_header(derived, header, what):
    """
    remez_tan.py convention: HEADER_* literals live in the SCRIPT and are the
    committed values; drift between a fresh derivation and them exits 1.
    header = None means the header has not been transcribed yet: print and
    pass, so the first run of a fresh checkout is not red.
    """
    if header is None:
        print(f"\n[{what}] HEADER_* not yet transcribed in this script -- "
              f"emit mode, no drift check. Paste the derived values into the "
              f"script's HEADER list AND the C++ header, then re-run.")
        return True
    ok = True
    if len(header) != len(derived):
        print(f"MISMATCH {what}: count {len(derived)} != header {len(header)}")
        return False
    for i, (g, w) in enumerate(zip(derived, header)):
        g64, w64 = np.float64(float(g)), np.float64(w)
        if g64 != w64:
            print(f"MISMATCH {what}[{i}]: derived {float(g64)!r} != header {float(w64)!r}")
            ok = False
    print(f"\n[{what}] header coefficients match this derivation." if ok
          else f"\n[{what}] header coefficients DO NOT match; update the header "
               f"and this script's HEADER list together.")
    return ok


def report_fit(name, rep, coeffs, grid_mp, f_grid, gate_rel=mpf("5e-17"),
               gate_ripple=1.05, gate_f64=2e-16, gate_estrin=4.5e-16):
    """The standard report every fit prints. Returns True when all gates hold."""
    print(f"\n=== {name} ===")
    print(f"iterations           : {rep['iterations']}")
    print(f"minimax rel err (mp) : {mp.nstr(rep['max_rel'], 6)}"
          f"   (gate < {mp.nstr(gate_rel, 3)})")
    print(f"ripple ratio         : {mp.nstr(rep['ripple'], 6)}"
          f"   (gate <= {gate_ripple}, 1.0 ideal)")
    print(f"alternating extrema  : {len(rep['extrema'])}")
    for u, v in rep["extrema"]:
        print(f"    u = {mp.nstr(u, 10):>14}   rel err = {mp.nstr(v, 4)}")

    c64 = [float(c) for c in coeffs]
    eh, uh = rel_err_float64(horner_fma, c64, grid_mp, f_grid)
    en, un = rel_err_float64(horner_nofma, c64, grid_mp, f_grid)
    ee, ue = rel_err_float64(estrin_fma, c64, grid_mp, f_grid)
    print(f"float64 max rel err  : Horner+FMA {mp.nstr(eh, 4)} at u={mp.nstr(uh, 6)}")
    print(f"                       Horner     {mp.nstr(en, 4)} at u={mp.nstr(un, 6)}")
    print(f"                       Estrin+FMA {mp.nstr(ee, 4)} at u={mp.nstr(ue, 6)}"
          f"   (gate < {gate_estrin:.1e}; this script's padded Estrin is an")
    print(f"                       op-order REPORT -- the header picks its own"
          f" split; the binding")
    print(f"                       accuracy gate is the assembled ulp sweep"
          f" below)")
    kcond, ku = basis_condition(coeffs, grid_mp, f_grid)
    print(f"monomial basis cond  : {mp.nstr(kcond, 5)} at u={mp.nstr(ku, 6)}"
          f"   (switch basis if > ~5)")

    ok = (rep["max_rel"] < gate_rel and rep["ripple"] <= gate_ripple
          and eh < gate_f64 and ee < gate_estrin)
    print(f"gates                : {'PASS' if ok else 'FAIL'}")
    return ok


def log_grid(lo, hi, n):
    """log-spaced mpf grid (lo > 0)."""
    llo, lhi = log(mpf(lo)), log(mpf(hi))
    return [exp(llo + (lhi - llo) * mpf(i) / (n - 1)) for i in range(n)]


def lin_grid(lo, hi, n):
    lo, hi = mpf(lo), mpf(hi)
    return [lo + (hi - lo) * mpf(i) / (n - 1) for i in range(n)]

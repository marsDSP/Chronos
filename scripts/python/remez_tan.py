#!/usr/bin/env python3
"""
    tan(x) ~ R(x) = x * P(u) / Q(u),  u = x^2
    P(u) = N0 + N1 u + N2 u^2 + N3 u^3
    Q(u) =  1 + D1 u + D2 u^2 + D3 u^3

    tan blows up at pi/2, so this fits RELATIVE error, not absolute. Q(u)
    keeps a root just past the interval, which is how a rational form reproduces
    the pole at all. a plain polynomial cannot.

    Interval is [0, 1.55] ~ [0, 0.4934 pi].  For TPT/SVF prewarping,
    g = tan(pi * fc / fs), that covers cutoffs out to 98.7% of Nyquist.

    In float32 this fit is not coefficient-limited. Q(u) falls to ~0.023 at x = 1.55,
    so evaluating it in float32 cancels away ~1.6 decimal digits, and that rounding
    dominates the ~1e-12 approximation error by six orders of magnitude. The minimax fit is
    ~125000x better than the Pade seed in float64 and a wash in float32. The
    knob that actually moves float32 accuracy here is B, not the coefficients.
"""
from __future__ import annotations

import sys

import numpy as np

PI = np.pi
A, B = 0.0, 1.55
NPARAM = 7
NNODES = NPARAM + 1

# The relative-error metric cancels to ~1e-16 as x -> 0, where the fit is
# essentially exact. Extrema below this floor are float64 noise, not signal;
# without the floor the scan finds thousands of bogus alternations near 0.
NOISE_FLOOR = 1e-14

HEADER_N = [1.0, -0.128538921, 0.00283448538, -7.76689558e-06]
HEADER_D = [1.0, -0.46187225, 0.0234585702, -0.000212576764]

_SCALE = -135135.0
PADE_N = np.array([-135135.0, 17325.0, -378.0, 1.0]) / _SCALE
PADE_D = np.array([-135135.0, 62370.0, -3150.0, 28.0]) / _SCALE

def tanc(x):
    """tan(x)/x, with the removable singularity at x = 0 filled in"""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0.0
    out[nz] = np.tan(x[nz]) / x[nz]
    return out

# tan() over a few million points, every iteration, is the whole runtime; the
# grids are fixed so cache them and their tanc values once.
_X_EXT = np.linspace(A, B, 200_001)
_T_EXT = tanc(_X_EXT)
_X_MAX = np.linspace(A, B, 2_000_001)
_T_MAX = tanc(_X_MAX)

def evaluate(N, D, x):
    u = x * x
    return x * np.polyval(N[::-1], u) / np.polyval(D[::-1], u)

def rel_err(N, D, x, t):
    """
    Relative error (R - tan)/tan, given precomputed t = tanc(x).  Both R and tan
    carry a factor of x; cancelling it against tanc keeps the metric finite at
    x = 0, where it is exactly N0/D0 - 1.
    """
    u = x * x
    return np.polyval(N[::-1], u) / np.polyval(D[::-1], u) / t - 1.0

def err(N, D, x):
    return rel_err(N, D, x, tanc(x))

def alternating_extrema(N, D):
    """
    Local extrema of the relative-error curve.  A is a candidate here: unlike an
    absolute-error odd fit, this metric has no forced zero at x = 0.
    """
    e = rel_err(N, D, _X_EXT, _T_EXT)
    d = np.diff(e)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
    cand = [A] + list(_X_EXT[idx]) + [B]

    h = (B - A) / (len(_X_EXT) - 1)
    refined = []
    for xi in cand:
        if A + h < xi < B - h:  # parabolic refinement of interior extrema
            xs = np.array([xi - h, xi, xi + h])
            ys = err(N, D, xs)
            den = ys[0] - 2 * ys[1] + ys[2]
            if den != 0.0:
                xi = xi + 0.5 * h * (ys[0] - ys[2]) / den
        refined.append(float(np.clip(xi, A, B)))

    deduped = []
    for xi in sorted(refined):
        if not deduped or xi - deduped[-1] > 1e-6:
            deduped.append(xi)

    xs = np.array(deduped)
    es = err(N, D, xs)
    floor = max(NOISE_FLOOR, 1e-4 * np.abs(es).max())
    keep = np.abs(es) >= floor
    xs, es = xs[keep], es[keep]

    out_x: list[float] = []
    out_e: list[float] = []
    for xi, ei in zip(xs, es):
        if out_e and np.sign(ei) == np.sign(out_e[-1]):
            if abs(ei) > abs(out_e[-1]):
                out_x[-1], out_e[-1] = xi, ei
        else:
            out_x.append(xi)
            out_e.append(ei)
    return np.array(out_x)

def trim(nodes, N, D, want: int):
    """Drop surplus extrema from whichever end carries the smaller |error|"""
    nodes = list(nodes)
    vals = list(np.abs(err(N, D, np.array(nodes))))
    while len(nodes) > want:
        if vals[0] < vals[-1]:
            nodes.pop(0)
            vals.pop(0)
        else:
            nodes.pop()
            vals.pop()
    return np.array(nodes)

def solve_step(nodes, D_prev):
    """
    One linearized Remez solve at the given alternation nodes.

        (R - tan)/tan = (-1)^i E   =>   P - tanc*Q = (-1)^i E * tanc * Q
        linearize Q on the right as Q_prev, then split off the D0 = 1 term:
        P - tanc*(D1 u + D2 u^2 + D3 u^3) - (-1)^i E tanc Q_prev = tanc

    tanc > 0 on [0, 1.55], so it doubles as its own |weight|.
    """
    x = nodes
    u = x * x
    t = tanc(x)
    q_prev = np.polyval(D_prev[::-1], u)

    M = np.zeros((NNODES, NNODES))
    M[:, 0] = 1.0           # N0
    M[:, 1] = u             # N1
    M[:, 2] = u ** 2        # N2
    M[:, 3] = u ** 3        # N3
    M[:, 4] = -t * u        # D1
    M[:, 5] = -t * u ** 2   # D2
    M[:, 6] = -t * u ** 3   # D3
    M[:, 7] = -np.array([(-1.0) ** i for i in range(NNODES)]) * t * q_prev  # E

    sol = np.linalg.solve(M, t)
    return sol[0:4], np.concatenate(([1.0], sol[4:7])), sol[7]

def max_error(N, D):
    return np.abs(rel_err(N, D, _X_MAX, _T_MAX)).max()

def max_abs_error(N, D):
    return np.abs(evaluate(N, D, _X_MAX) - _X_MAX * _T_MAX).max()

def implied_pole(D):
    """x where Q(x^2) first vanishes; the pole the rational form stands in for"""
    roots = np.roots(D[::-1])
    roots = np.sort(roots[np.isreal(roots)].real)
    roots = roots[roots > 0.0]
    return float(np.sqrt(roots[0])) if len(roots) else float("nan")

def float32_error(N, D, fused: bool, hi: float = B):
    """
    float32 max relative error on (0, hi] using the header's evaluation order.
    R and tan are both odd, so this covers [-hi, hi]; x = 0 is skipped because
    relative error is 0/0 there.
    """
    x = np.linspace(hi / 4_000_000, hi, 4_000_000).astype(np.float32)
    u = (x * x).astype(np.float32)
    N32 = [np.float32(v) for v in N]
    D32 = [np.float32(v) for v in D]

    def horner(coeffs):
        acc = coeffs[-1]
        for c in reversed(coeffs[:-1]):
            if fused:  # fma: one rounding for the whole c + u*acc
                acc = np.float32(np.float64(c) + np.float64(u) * np.float64(acc))
            else:      # mul then add, rounding after each
                acc = np.float32(c + np.float32(u * acc))
        return acc

    num = np.float32(x * horner(N32))
    den = horner(D32)
    r = np.float32(num / den)
    return np.abs(r.astype(np.float64) / np.tan(x.astype(np.float64)) - 1.0).max()

def main() -> int:
    N, D = PADE_N.copy(), PADE_D.copy()
    print(f"seed (Pade [7/6], normalized): max rel err = {max_error(N, D):.6e}")

    best = (max_error(N, D), N.copy(), D.copy())
    nodes = alternating_extrema(N, D)

    for _ in range(80):
        if len(nodes) < NNODES:
            nodes = np.unique(np.concatenate([nodes, np.linspace(A, B, NNODES)]))[:NNODES]
        nodes_use = trim(nodes, N, D, NNODES) if len(nodes) > NNODES else nodes
        if len(nodes_use) != NNODES:
            break
        try:
            N_new, D_new, _ = solve_step(nodes_use, D)
        except np.linalg.LinAlgError:
            break
        if np.polyval(D_new[::-1], _X_EXT ** 2).min() <= 0:
            break  # pole pulled inside the interval
        m = max_error(N_new, D_new)
        if not np.isfinite(m):
            break
        N, D = N_new, D_new
        if m < best[0]:
            best = (m, N.copy(), D.copy())
        nodes = alternating_extrema(N, D)

    m, N, D = best
    print(f"converged: float64 max rel err on [0, {B}] = {m:.6e}")
    print(f"           float64 max abs err on [0, {B}] = {max_abs_error(N, D):.6e}")

    ext = alternating_extrema(N, D)
    vals = [err(N, D, np.array([xi]))[0] for xi in ext]
    ripple = max(map(abs, vals)) / min(map(abs, vals))
    print(f"equioscillation: {len(ext)} alternating extrema, ripple ratio {ripple:.4f} (1.0 is ideal)")
    for xi, vi in list(zip(ext, vals))[:NNODES + 2]:
        print(f"    x = {xi:.6f}   rel err = {vi:+.4e}")

    q_min = np.polyval(D[::-1], _X_EXT ** 2).min()
    print(f"min Q(u) on [0, {B}^2] = {q_min:.6f}  (> 0 means no pole inside the interval)")
    print(f"implied pole at x = {implied_pole(D):.6f}  "
          f"(true pole pi/2 = {PI / 2:.6f}, must stay above B = {B})")

    print("\ncoefficients for source/math/Trigonometry.h:")
    for i, v in enumerate(N):
        print(f"    constexpr float N{i} = {float(np.float32(v)):.9g}f;")
    for i, v in enumerate(D):
        print(f"    constexpr float D{i} = {float(np.float32(v)):.9g}f;")

    e_fused = float32_error(N, D, fused=True)
    e_split = float32_error(N, D, fused=False)
    pade_f32 = float32_error(PADE_N, PADE_D, fused=True)
    print(f"\nfloat32 max rel err on [-{B}, {B}]:")
    print(f"    with FMA     = {e_fused:.4e}")
    print(f"    without FMA  = {e_split:.4e}")
    print(f"    Pade, with FMA = {pade_f32:.4e}")
    print("    float32 here is rounding-limited, not coefficient-limited: both fits")
    print("    sit far below float32 eps in float64, so these two are a wash.")

    print("\nfloat32 max rel err (with FMA) by sub-range, showing where it goes bad:")
    for hi in (1.0, 1.3, 1.45, B):
        print(f"    x <= {hi:.2f}  ({100 * hi / (PI / 2):5.2f}% of Nyquist)  = "
              f"{float32_error(N, D, True, hi):.4e}")

    # regression check against the values actually in the header
    ok = True
    for name, got, want in (("N", N, HEADER_N), ("D", D, HEADER_D)):
        for i, (g, w) in enumerate(zip(got, want)):
            g32, w32 = np.float32(g), np.float32(w)
            if g32 != w32:
                print(f"MISMATCH {name}{i}: derived {float(g32):.9g} != header {float(w32):.9g}")
                ok = False
    print("\nheader coefficients match this derivation." if ok
          else "\nheader coefficients DO NOT match; update source/math/Trigonometry.h.")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
    sin(x) ~ R(x) = x * P(u) / Q(u)
    P(u) = N0 + N1 u + N2 u^2 + N3 u^3
    Q(u) =  1 + D1 u + D2 u^2 + D3 u^3
"""
from __future__ import annotations

import sys

import numpy as np

PI = np.pi
A, B = 0.0, PI
NPARAM = 7
NNODES = NPARAM + 1

HEADER_N = [1.0, -0.141643688, 0.00446910504, -3.88648514e-05]
HEADER_D = [1.0, 0.0250229947, 0.000306247879, 2.07578137e-06]

_SCALE = 11511339840.0
PADE_N = np.array([11511339840.0, -1640635920.0, 52785432.0, -479249.0]) / _SCALE
PADE_D = np.array([11511339840.0, 277920720.0, 3177720.0, 18361.0]) / _SCALE

def evaluate(N, D, x):
    u = x * x
    return x * np.polyval(N[::-1], u) / np.polyval(D[::-1], u)

def err(N, D, x):
    return evaluate(N, D, x) - np.sin(x)

def alternating_extrema(N, D, ngrid: int = 200_001):
    """
    Local extrema of the error curve
    """
    x = np.linspace(A, B, ngrid)
    e = err(N, D, x)
    d = np.diff(e)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
    cand = list(x[idx]) + [B]

    h = (B - A) / (ngrid - 1)
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

    out: list[float] = []
    for xi in deduped:
        ei = err(N, D, np.array([xi]))[0]
        if abs(ei) < 1e-14:
            continue
        if out and np.sign(ei) == np.sign(err(N, D, np.array([out[-1]]))[0]):
            if abs(ei) > abs(err(N, D, np.array([out[-1]]))[0]):
                out[-1] = xi
        else:
            out.append(xi)
    return np.array(out)

def trim(nodes, N, D, want: int):
    """Drop surplus extrema from whichever end carries the smaller |error|"""
    nodes = list(nodes)
    while len(nodes) > want:
        if abs(err(N, D, np.array([nodes[0]]))[0]) < abs(err(N, D, np.array([nodes[-1]]))[0]):
            nodes.pop(0)
        else:
            nodes.pop()
    return np.array(nodes)

def solve_step(nodes, D_prev):
    """One linearized Remez solve at the given alternation nodes"""
    x = nodes
    u = x * x
    s = np.sin(x)
    q_prev = np.polyval(D_prev[::-1], u)

    M = np.zeros((NNODES, NNODES))
    M[:, 0] = x             # N0
    M[:, 1] = x * u         # N1
    M[:, 2] = x * u ** 2    # N2
    M[:, 3] = x * u ** 3    # N3
    M[:, 4] = -s * u        # D1
    M[:, 5] = -s * u ** 2   # D2
    M[:, 6] = -s * u ** 3   # D3
    M[:, 7] = -np.array([(-1.0) ** i for i in range(NNODES)]) * q_prev  # E

    sol = np.linalg.solve(M, s)
    return sol[0:4], np.concatenate(([1.0], sol[4:7])), sol[7]

def max_error(N, D, n: int = 2_000_001):
    return np.abs(err(N, D, np.linspace(A, B, n))).max()

def float32_error(N, D, fused: bool):
    """
    float32 max abs error on [-pi, pi] using the header's evaluation order
    """
    x = np.linspace(-PI, PI, 4_000_001).astype(np.float32)
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
    return np.abs(r.astype(np.float64) - np.sin(x.astype(np.float64))).max()

def main() -> int:
    N, D = PADE_N.copy(), PADE_D.copy()
    print(f"seed (Pade [7/6], normalized): max err = {max_error(N, D):.6e}")

    best = (max_error(N, D), N.copy(), D.copy())
    nodes = alternating_extrema(N, D)

    for _ in range(80):
        if len(nodes) < NNODES:
            nodes = np.unique(np.concatenate([nodes, np.linspace(A + 0.05, B, NNODES)]))[:NNODES]
        nodes_use = trim(nodes, N, D, NNODES) if len(nodes) > NNODES else nodes
        if len(nodes_use) != NNODES:
            break
        try:
            N_new, D_new, _ = solve_step(nodes_use, D)
        except np.linalg.LinAlgError:
            break
        if np.polyval(D_new[::-1], np.linspace(A, B, 20_001) ** 2).min() <= 0:
            break  # pole inside the interval
        m = max_error(N_new, D_new)
        if not np.isfinite(m):
            break
        N, D = N_new, D_new
        if m < best[0]:
            best = (m, N.copy(), D.copy())
        nodes = alternating_extrema(N, D)

    m, N, D = best
    print(f"converged: float64 max abs err on [0, pi] = {m:.6e}")

    ext = alternating_extrema(N, D)
    vals = [err(N, D, np.array([xi]))[0] for xi in ext]
    ripple = max(map(abs, vals)) / min(map(abs, vals))
    print(f"equioscillation: {len(ext)} alternating extrema, ripple ratio {ripple:.4f} (1.0 is ideal)")
    for xi, vi in zip(ext, vals):
        print(f"    x = {xi:.6f}   err = {vi:+.4e}")

    q_min = np.polyval(D[::-1], np.linspace(A, B, 200_001) ** 2).min()
    print(f"min Q(u) on [0, pi^2] = {q_min:.6f}  (> 0 means pole-free)")

    print("\ncoefficients for source/math/Trigonometry.h:")
    for i, v in enumerate(N):
        print(f"    constexpr float N{i} = {float(np.float32(v)):.9g}f;")
    for i, v in enumerate(D):
        print(f"    constexpr float D{i} = {float(np.float32(v)):.9g}f;")

    e_fused = float32_error(N, D, fused=True)
    e_split = float32_error(N, D, fused=False)
    print(f"\nfloat32 max abs err on [-pi, pi]:")
    print(f"    with FMA     = {e_fused:.4e}")
    print(f"    without FMA  = {e_split:.4e}")
    print(f"    Pade was 1.1295e-05  ->  {1.1295e-05 / e_fused:.1f}x / {1.1295e-05 / e_split:.1f}x better")

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

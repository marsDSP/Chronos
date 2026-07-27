#!/usr/bin/env python3
from __future__ import annotations

import sys

import numpy as np

N = 16

PASSBAND_EDGE = 0.42

FS = 48000.0

HEADER_COEFFS: list[float] = [
    -0.00530111231,
    0.0121372724,
    -0.02284934,
    0.0391499847,
    -0.0644291341,
    0.10737168,
    -0.200103939,
    0.63402456,
    0.63402456,
    -0.200103939,
    0.10737168,
    -0.0644291341,
    0.0391499847,
    -0.02284934,
    0.0121372724,
    -0.00530111231,
]


def sinc_np(x):
    """Unnormalised sinc, sin(pi x)/(pi x), with sinc(0) = 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sin(np.pi * x) / (np.pi * x)
    return np.where(x == 0.0, 1.0, out)


def kaiser_window(n: int, beta: float) -> np.ndarray:
    """Symmetric Kaiser window of length n (I0 form)."""
    m = np.arange(n, dtype=float)
    half = (n - 1) / 2.0
    arg = beta * np.sqrt(np.maximum(0.0, 1.0 - ((m - half) / half) ** 2))
    return np.i0(arg) / np.i0(np.array([beta]))[0]


def design(n: int, beta: float) -> np.ndarray:
    d = (n - 1) / 2.0
    m = np.arange(n, dtype=float)
    h = sinc_np(m - d) * kaiser_window(n, beta)
    return h / np.sum(h)  # unity DC gain


def magnitude_db(h: np.ndarray, f_norm: float) -> float:
    """|H(f)| in dB at normalised frequency f = f/fs in [0, 0.5]."""
    m = np.arange(len(h), dtype=float)
    w = 2.0 * np.pi * f_norm
    H = np.sum(h * np.exp(-1j * w * m))
    return 20.0 * np.log10(max(float(np.abs(H)), 1e-300))


def max_passband_dev(h: np.ndarray, edge: float) -> float:
    """Max |H| deviation from unity over [0, edge] (normalised), in dB."""
    f = np.linspace(0.0, edge, 20001)
    mags = np.array([magnitude_db(h, float(fi)) for fi in f])
    return float(np.max(np.abs(mags)))


def main() -> int:
    # Sweep beta and pick the value minimising max passband deviation.
    betas = np.linspace(0.0, 12.0, 241)
    best_beta = 0.0
    best_dev = float("inf")
    for b in betas:
        dev = max_passband_dev(design(N, float(b)), PASSBAND_EDGE)
        if dev < best_dev:
            best_dev = dev
            best_beta = float(b)

    h = design(N, best_beta)

    # Symmetry residual: h[j] == h[N-1-j] by construction.
    sym_resid = float(np.max(np.abs(h - h[::-1])))

    # Coefficient sum minus 1 (DC gain residual after normalisation).
    sum_minus_one = float(np.sum(h) - 1.0)

    # Centroid = sum(m h[m]) / sum(h) = (N-1)/2 for a symmetric FIR.
    m = np.arange(N, dtype=float)
    centroid = float(np.sum(m * h) / np.sum(h))

    print(f"half-sample FIR: N = {N}, D = {(N - 1) / 2.0:.1f}")
    print(f"chosen beta = {best_beta:.4f}")
    print(f"max passband deviation over [0, {PASSBAND_EDGE:g}*fs] = {best_dev:.6f} dB")
    print(f"centroid = {centroid:.9f}  (expected {(N - 1) / 2.0:.1f})")
    print(f"symmetry residual = {sym_resid:.3e}")
    print(f"sum(h) - 1 = {sum_minus_one:.3e}")
    print()
    print(f"magnitude |H| in dB (fs = {FS:.0f} Hz):")
    for f_khz in (1.0, 5.0, 10.0, 15.0, 20.0):
        fn = (f_khz * 1000.0) / FS
        print(f"    {f_khz:5.1f} kHz : {magnitude_db(h, fn):+.6f} dB")

    # Nyquist null: structural for even-length symmetric FIRs.
    h_nyq = float(np.abs(np.sum(h * (-1.0) ** np.arange(N))))
    print(f"\n|H(Nyquist)| = {h_nyq:.3e}  (structural null for even-length symmetric FIR)")

    print("\ncoefficients (C++ for source/dsp/align/HalfSampleFir.h):")
    print("inline constexpr std::array<float, kHalfSampleTaps> kHalfSampleCoeffs = {")
    for j, v in enumerate(h):
        print(f"    {float(np.float32(v)):.9g}f,{('  // h[' + str(j) + ']') if j in (0, N - 1) else ''}")
    print("};")

    print("\ncoefficients (Python for HEADER_COEFFS in this script):")
    print("HEADER_COEFFS = [")
    for v in h:
        print(f"    {float(np.float32(v)):.9g},")
    print("]")

    # Regression check against the header values.
    if not HEADER_COEFFS:
        print("\nHEADER_COEFFS is empty -- copy the Python block above into this")
        print("script (and the C++ block above into the header), then re-run to verify.")
        return 1

    if len(HEADER_COEFFS) != N:
        print(f"\nLENGTH MISMATCH: HEADER_COEFFS has {len(HEADER_COEFFS)} values, N = {N}")
        return 1

    ok = True
    for j, (g, w) in enumerate(zip(h, HEADER_COEFFS)):
        g32, w32 = np.float32(g), np.float32(w)
        if g32 != w32:
            print(f"MISMATCH h[{j}]: derived {float(g32):.9g} != header {float(w32):.9g}")
            ok = False
    print("\nheader coefficients match this derivation." if ok
          else "\nheader coefficients DO NOT match; update source/dsp/align/HalfSampleFir.h.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

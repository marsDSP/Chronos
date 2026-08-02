#!/usr/bin/env python3
from __future__ import annotations

import sys

import numpy as np

# Drive compensation table for the tanh saturator.
#
# rmsRatio(k) = rms(tanh(k * x_ref)) / rms(x_ref)
# where x_ref is a 0.5-amplitude sine. The grid is 65 points uniform in
# log2(k) over [0, 4], i.e. k from 1 to 16.
#
# outputMakeup(k) = pow(rmsRatio(k), -0.7)
# loopTrim(k)     = pow(rmsRatio(k), -0.5)
#
# The script prints the C++ arrays for source/math/SaturatorMakeup.h,
# prints the anchor values, and exits non-zero if a fresh derivation
# drifts from the committed constants below.

N_POINTS = 65
LOG2_MIN = 0.0
LOG2_MAX = 4.0
N_INTEGRATE = 1 << 18  # 262144 midpoint samples over one sine period

# Committed header constants.
HEADER_RMS_RATIO: list[float] = [
    0.942466974, 0.979221046, 1.01699519, 1.05576766, 1.09551024,
    1.13618743, 1.17775595, 1.22016466, 1.26335371, 1.30725491,
    1.3517915, 1.39687788, 1.44242001, 1.48831582, 1.5344559,
    1.580724, 1.62699795, 1.67315125, 1.7190541, 1.76457489,
    1.809582, 1.85394585, 1.89753985, 1.94024348, 1.98194277,
    2.02253294, 2.06191969, 2.10002017, 2.13676429, 2.17209506,
    2.20597005, 2.2383604, 2.26925111, 2.29864049, 2.32653928,
    2.35296988, 2.37796497, 2.40156555, 2.42382121, 2.44478679,
    2.46452188, 2.48308921, 2.50055337, 2.51697922, 2.53243184,
    2.54697442, 2.56066799, 2.57357121, 2.58573961, 2.59722471,
    2.60807538, 2.61833596, 2.62804794, 2.63724899, 2.64597368,
    2.65425324, 2.662117, 2.66959071, 2.67669845, 2.6834619,
    2.68990135, 2.69603539, 2.70188046, 2.70745277, 2.71276641,
]
HEADER_OUTPUT_MAKEUP: list[float] = [
    1.04235029, 1.01480711, 0.988272667, 0.962724805, 0.938141763,
    0.91450274, 0.891787291, 0.869975686, 0.849048555, 0.828987122,
    0.809772849, 0.79138732, 0.773812592, 0.757030606, 0.741023362,
    0.725772917, 0.711261153, 0.697469711, 0.684380054, 0.671973169,
    0.660229921, 0.649130583, 0.638655066, 0.628782809, 0.619492769,
    0.61076349, 0.602573156, 0.594899476, 0.587719858, 0.581011593,
    0.574751675, 0.568917096, 0.563484788, 0.558431923, 0.553735971,
    0.549374521, 0.545325994, 0.541569114, 0.538083375, 0.534849107,
    0.531847477, 0.529060543, 0.526471317, 0.524063885, 0.521823406,
    0.519735992, 0.517788887, 0.51597023, 0.514269352, 0.512676418,
    0.511182427, 0.509779334, 0.508459926, 0.507217467, 0.506046176,
    0.504940629, 0.503896117, 0.50290823, 0.501973033, 0.50108707,
    0.500247061, 0.499450088, 0.498693496, 0.497974813, 0.497291803,
]
HEADER_LOOP_TRIM: list[float] = [
    1.03007042, 1.01055419, 0.991609216, 0.973230779, 0.955414355,
    0.93815589, 0.921451211, 0.905296385, 0.889687598, 0.874620914,
    0.860092461, 0.846098244, 0.832633972, 0.819695294, 0.807277381,
    0.795375109, 0.783982754, 0.773094177, 0.762702584, 0.752800584,
    0.74338001, 0.734431803, 0.725946367, 0.717913091, 0.710320652,
    0.703156829, 0.69640857, 0.690062225, 0.68410331, 0.678516746,
    0.673286915, 0.668397784, 0.663832843, 0.659575462, 0.655608833,
    0.651916265, 0.648481011, 0.645286798, 0.642317414, 0.639557362,
    0.636991501, 0.634605527, 0.632385552, 0.630318701, 0.628392696,
    0.626596153, 0.624918461, 0.623349905, 0.621881485, 0.620504916,
    0.619212806, 0.617998362, 0.616855383, 0.615778387, 0.614762306,
    0.613802731, 0.612895489, 0.612037003, 0.611223817, 0.610453069,
    0.609721899, 0.609027922, 0.608368814, 0.607742429, 0.607146919,
]


def compute_rms_ratio(k: float, sin2t: np.ndarray, rms_ref: float) -> float:
    """rms(tanh(k * 0.5 * sin(2t))) / rms(0.5 * sin(2t))."""
    x = k * 0.5 * sin2t
    y = np.tanh(x)
    rms = float(np.sqrt(np.mean(y * y)))
    return rms / rms_ref


def main() -> int:
    log2k = np.linspace(LOG2_MIN, LOG2_MAX, N_POINTS)
    k_grid = np.power(2.0, log2k)

    t = np.pi * (np.arange(N_INTEGRATE) + 0.5) / N_INTEGRATE
    sin2t = np.sin(2.0 * t)
    rms_ref = 0.5 / np.sqrt(2.0)

    rms_ratios = np.array([compute_rms_ratio(float(k), sin2t, rms_ref) for k in k_grid])
    output_makeups = np.power(rms_ratios, -0.7)
    loop_trims = np.power(rms_ratios, -0.5)

    rms_f32 = np.float32(rms_ratios)
    makeup_f32 = np.float32(output_makeups)
    trim_f32 = np.float32(loop_trims)

    print(f"makeup table: {N_POINTS} points, log2(k) in [{LOG2_MIN}, {LOG2_MAX}], "
          f"k in [1, 16], {N_INTEGRATE} integration points")
    print()

    # Anchor values.
    print("anchor values:")
    print(f"  {'drive dB':>8}  {'k':>10}  {'rmsRatio':>10}  {'outputMakeup':>12}  "
          f"{'net dB':>8}  {'loopTrim':>10}")
    for db in (0, 3, 6, 9, 12, 15, 18, 21, 24):
        k = 10.0 ** (db / 20.0)
        r = compute_rms_ratio(k, sin2t, rms_ref)
        mk = r ** (-0.7)
        tr = r ** (-0.5)
        net = 20.0 * np.log10(r * mk)
        print(f"  {db:8d}  {k:10.6f}  {r:10.6f}  {mk:12.6f}  {net:8.3f}  {tr:10.6f}")
    print()

    # C++ arrays.
    def emit_array(name: str, arr: np.ndarray) -> None:
        print(f"inline constexpr std::array<float, {N_POINTS}> {name}{{ {{")
        for i, v in enumerate(arr):
            end = "," if i < N_POINTS - 1 else ""
            print(f"    {float(v):.9g}f{end}")
        print("} };")
        print()

    print("C++ for source/math/SaturatorMakeup.h:")
    emit_array("kRmsRatioTable", rms_f32)
    emit_array("kOutputMakeupTable", makeup_f32)
    emit_array("kLoopTrimTable", trim_f32)

    # Python constants for embedding.
    def emit_python(name: str, arr: np.ndarray) -> None:
        print(f"{name} = [")
        for v in arr:
            print(f"    {float(v):.9g},")
        print("]")
        print()

    print("Python for embedding in this script:")
    emit_python("HEADER_RMS_RATIO", rms_f32)
    emit_python("HEADER_OUTPUT_MAKEUP", makeup_f32)
    emit_python("HEADER_LOOP_TRIM", trim_f32)

    # Drift check against committed constants.
    if not HEADER_RMS_RATIO:
        print("HEADER_RMS_RATIO is empty -- copy the Python block above into")
        print("this script, then re-run to verify.")
        return 1

    ok = True
    for name, derived, header in (
        ("RMS_RATIO", rms_f32, HEADER_RMS_RATIO),
        ("OUTPUT_MAKEUP", makeup_f32, HEADER_OUTPUT_MAKEUP),
        ("LOOP_TRIM", trim_f32, HEADER_LOOP_TRIM),
    ):
        if len(header) != N_POINTS:
            print(f"LENGTH MISMATCH {name}: header has {len(header)}, expected {N_POINTS}")
            ok = False
            continue
        for i, (g, w) in enumerate(zip(derived, header)):
            g32, w32 = np.float32(g), np.float32(w)
            if g32 != w32:
                print(f"MISMATCH {name}[{i}]: derived {float(g32):.9g} != header {float(w32):.9g}")
                ok = False

    print("header constants match this derivation." if ok
          else "header constants DO NOT match; update source/math/SaturatorMakeup.h.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

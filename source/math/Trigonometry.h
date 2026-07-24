#pragma once

#ifndef CHRONOS_TRIGONOMETRY_H
#define CHRONOS_TRIGONOMETRY_H

#include <cmath>
#include <algorithm>
#include "simd/Config.h"
#include <simde/x86/fma.h>

// MSVC's <cmath> does not expose the POSIX M_PI / M_2_PI macros unless
// _USE_MATH_DEFINES is defined before include.
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#ifndef M_2_PI
    #define M_2_PI 0.63661977236758134308
#endif

// ═══════════════════════════════════════════════════════════
// sin(x) ≈ x·P(x²) / Q(x²) [7/6] odd rational minimax on [-π, π]
// ───────────────────────────────────────────────────────────
namespace MinimaxSinCoeffs {
    constexpr float N0 = 1.0f; // num x⁰
    constexpr float N1 = -0.141643688f; // num x²
    constexpr float N2 = 0.00446910504f; // num x⁴
    constexpr float N3 = -3.88648514e-05f; // num x⁶
    constexpr float D0 = 1.0f; // den x⁰
    constexpr float D1 = 0.0250229947f; // den x²
    constexpr float D2 = 0.000306247879f; // den x⁴
    constexpr float D3 = 2.07578137e-06f; // den x⁶
}

inline M128 mulAdd(const M128 a, const M128 b, const M128 c) noexcept {
    return simde_mm_fmadd_ps(a, b, c);
}

inline float minimaxSinApprox(const float x) noexcept {
    using namespace MinimaxSinCoeffs;
    const auto x2 = x * x;
    // horner evaluation inside-out in x²
    const auto num = x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));
    return num / den;
}

inline float mmSin(const float x) noexcept { return minimaxSinApprox(x); }

inline M128 mmSin(const M128 x) noexcept {
    using namespace MinimaxSinCoeffs;
    // broadcast each coeff across 4 lanes
    const auto vN0 = MM(set1_ps)(N0);
    const auto vN1 = MM(set1_ps)(N1);
    const auto vN2 = MM(set1_ps)(N2);
    const auto vN3 = MM(set1_ps)(N3);
    const auto vD0 = MM(set1_ps)(D0);
    const auto vD1 = MM(set1_ps)(D1);
    const auto vD2 = MM(set1_ps)(D2);
    const auto vD3 = MM(set1_ps)(D3);
    const auto x2 = MM(mul_ps)(x, x);
    // numerator:  x · (N0 + x²·(N1 + x²·(N2 + x²·N3))) | innermost first
    auto numInner = mulAdd(x2, vN3, vN2); // N2 + x²·N3
    numInner = mulAdd(x2, numInner, vN1); // N1 + x²·(…)
    numInner = mulAdd(x2, numInner, vN0); // N0 + x²·(…)
    const auto num = MM(mul_ps)(x, numInner);
    // denominator: D0 + x²·(D1 + x²·(D2 + x²·D3))
    auto denInner = mulAdd(x2, vD3, vD2); // D2 + x²·D3
    denInner = mulAdd(x2, denInner, vD1); // D1 + x²·(…)
    const auto den = mulAdd(x2, denInner, vD0); // D0 + x²·(…)
    return MM(div_ps)(num, den);
}

#endif

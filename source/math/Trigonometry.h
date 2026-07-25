#pragma once

#ifndef CHRONOS_TRIGONOMETRY_H
#define CHRONOS_TRIGONOMETRY_H

#include <cmath>
#include <algorithm>
#include "simd/Config.h"

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
namespace MinimaxSinCoeffs
{
    constexpr float N0 = 1.0f; // num x⁰
    constexpr float N1 = -0.141643688f; // num x²
    constexpr float N2 = 0.00446910504f; // num x⁴
    constexpr float N3 = -3.88648514e-05f; // num x⁶
    constexpr float D0 = 1.0f; // den x⁰
    constexpr float D1 = 0.0250229947f; // den x²
    constexpr float D2 = 0.000306247879f; // den x⁴
    constexpr float D3 = 2.07578137e-06f; // den x⁶
}

inline M128 mulAdd(const M128 a, const M128 b, const M128 c) noexcept
{
    return FMADD(a, b, c);
}

inline float minimaxSinApprox(const float x) noexcept
{
    using namespace MinimaxSinCoeffs;
    const auto x2 = x * x;
    // horner evaluation inside-out in x²
    const auto num = x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));
    return num / den;
}

inline float mmSin(const float x) noexcept { return minimaxSinApprox(x); }

inline M128 mmSin(const M128 x) noexcept
{
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
// ═══════════════════════════════════════════════════════════
// cos(x) ≈ P(x²) / Q(x²) [6/6] even rational minimax on [-π, π]
// ───────────────────────────────────────────────────────────
// float64 max abs err 1.49e-08
// float32 3.50e-07 with FMA
namespace MinimaxCosCoeffs
{
    // cos is even, so there is no leading x factor: everything is a series in x²
    constexpr float N0 = 1.0f; // num x⁰
    constexpr float N1 = -0.469479561f; // num x²
    constexpr float N2 = 0.0268812291f; // num x⁴
    constexpr float N3 = -0.00035018372f; // num x⁶

    constexpr float D0 = 1.0f; // den x⁰
    constexpr float D1 = 0.03052059f; // den x²
    constexpr float D2 = 0.000474582455f; // den x⁴
    constexpr float D3 = 4.48269293e-06f; // den x⁶
}

inline float minimaxCosApprox(const float x) noexcept
{
    using namespace MinimaxCosCoeffs;

    // cos depends only on x², confirming even symmetry
    const auto x2 = x * x;

    // horner evaluation inside-out in x²
    const auto num = N0 + x2 * (N1 + x2 * (N2 + x2 * N3));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));

    return num / den;
}

inline float mmCos(const float x) noexcept { return minimaxCosApprox(x); }

inline M128 mmCos(const M128 x) noexcept
{
    using namespace MinimaxCosCoeffs;

    const auto vN0 = MM(set1_ps)(N0);
    const auto vN1 = MM(set1_ps)(N1);
    const auto vN2 = MM(set1_ps)(N2);
    const auto vN3 = MM(set1_ps)(N3);

    const auto vD0 = MM(set1_ps)(D0);
    const auto vD1 = MM(set1_ps)(D1);
    const auto vD2 = MM(set1_ps)(D2);
    const auto vD3 = MM(set1_ps)(D3);

    const auto x2 = MM(mul_ps)(x, x);

    // numerator: N0 + x²·(N1 + x²·(N2 + x²·N3))
    auto numInner = mulAdd(x2, vN3, vN2); // N2 + x²·N3
    numInner = mulAdd(x2, numInner, vN1); // N1 + x²·(…)
    const auto num = mulAdd(x2, numInner, vN0); // N0 + x²·(…)

    // denominator: D0 + x²·(D1 + x²·(D2 + x²·D3))
    auto denInner = mulAdd(x2, vD3, vD2); // D2 + x²·D3
    denInner = mulAdd(x2, denInner, vD1); // D1 + x²·(…)
    const auto den = mulAdd(x2, denInner, vD0); // D0 + x²·(…)

    return MM(div_ps)(num, den);
}
// ═══════════════════════════════════════════════════════════
// tan(x) ≈ x·P(x²) / Q(x²) [7/6] odd rational minimax on [-1.55, 1.55]
// ───────────────────────────────────────────────────────────
// float32 is rounding-limited, not coefficient-limited.
// Do not expect the float64 improvement to show up in float32.
// OUT OF RANGE: |x| ≥ π/2 is meaningless here, range-reduce before calling!
// [-1.55, 1.55] fit interval, so 0.49 x pi = safe. I think it's like 1.539 or something.
namespace MinimaxTanCoeffs
{
    constexpr float N0 = 1.0f;
    constexpr float N1 = -0.128538921f; // num x²
    constexpr float N2 = 0.00283448538f; // num x⁴
    constexpr float N3 = -7.76689558e-06f; // num x⁶

    constexpr float D0 = 1.0f; // den x⁰
    constexpr float D1 = -0.46187225f; // den x²
    constexpr float D2 = 0.0234585702f; // den x⁴
    constexpr float D3 = -0.000212576764f; // den x⁶
}

inline float minimaxTanApprox(const float x) noexcept
{
    using namespace MinimaxTanCoeffs;

    const auto x2 = x * x;

    const auto num = x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));

    return num / den;
}

inline float mmTan(const float x) noexcept { return minimaxTanApprox(x); }

inline M128 mmTan(const M128 x) noexcept
{
    using namespace MinimaxTanCoeffs;

    const auto vN0 = MM(set1_ps)(N0);
    const auto vN1 = MM(set1_ps)(N1);
    const auto vN2 = MM(set1_ps)(N2);
    const auto vN3 = MM(set1_ps)(N3);

    const auto vD0 = MM(set1_ps)(D0);
    const auto vD1 = MM(set1_ps)(D1);
    const auto vD2 = MM(set1_ps)(D2);
    const auto vD3 = MM(set1_ps)(D3);

    const auto x2 = MM(mul_ps)(x, x);

    // numerator: x · (N0 + x²·(N1 + x²·(N2 + x²·N3)))
    auto numInner = mulAdd(x2, vN3, vN2); // N2 + x²·N3
    numInner = mulAdd(x2, numInner, vN1); // N1 + x²·(…)
    const auto poly = mulAdd(x2, numInner, vN0); // N0 + x²·(…)
    const auto num = MM(mul_ps)(x, poly);

    // denominator: D0 + x²·(D1 + x²·(D2 + x²·D3))
    auto denInner = mulAdd(x2, vD3, vD2); // D2 + x²·D3
    denInner = mulAdd(x2, denInner, vD1); // D1 + x²·(…)
    const auto den = mulAdd(x2, denInner, vD0); // D0 + x²·(…)

    return MM(div_ps)(num, den);
}
//==============================================================================//
namespace PadeTanCoeffs
{
    // (7,6) pade approximant of tan(x)
    constexpr float N0 = -135135.0f;
    constexpr float N1 = 17325.0f;
    constexpr float N2 = -378.0f;
    constexpr float N3 = 1.0f;

    constexpr float D0 = -135135.0f;
    constexpr float D1 = 62370.0f;
    constexpr float D2 = -3150.0f;
    constexpr float D3 = 28.0f;
}

inline float padeTanApprox(const float x) noexcept
{
    using namespace PadeTanCoeffs;

    const auto x2 = x * x;

    const auto num = x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));

    return num / den;
}

inline float fasterTan(const float x) noexcept
{
    return padeTanApprox(x);
}

inline M128 fasterTan(const M128 x) noexcept
{
    using namespace PadeTanCoeffs;

    const auto vN0 = MM(set1_ps)(N0);
    const auto vN1 = MM(set1_ps)(N1);
    const auto vN2 = MM(set1_ps)(N2);
    const auto vN3 = MM(set1_ps)(N3);

    const auto vD0 = MM(set1_ps)(D0);
    const auto vD1 = MM(set1_ps)(D1);
    const auto vD2 = MM(set1_ps)(D2);
    const auto vD3 = MM(set1_ps)(D3);

    const auto x2 = MM(mul_ps)(x, x);

    auto numInner = MM(add_ps)(vN2, MM(mul_ps)(x2, vN3)); // N2 + x²·N3
    numInner = MM(add_ps)(vN1, MM(mul_ps)(x2, numInner)); // N1 + x²·(…)
    const auto poly = MM(add_ps)(vN0, MM(mul_ps)(x2, numInner)); // N0 + x²·(…)
    const auto num = MM(mul_ps)(x, poly);

    auto denInner = MM(add_ps)(vD2, MM(mul_ps)(x2, vD3)); // D2 + x²·D3
    denInner = MM(add_ps)(vD1, MM(mul_ps)(x2, denInner)); // D1 + x²·(…)
    const auto den = MM(add_ps)(vD0, MM(mul_ps)(x2, denInner));

    return MM(div_ps)(num, den);
}
//==============================================================================//
#endif

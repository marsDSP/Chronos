#pragma once

#ifndef CHRONOS_TRIGONOMETRY_H
#define CHRONOS_TRIGONOMETRY_H

#include <cmath>
#include <algorithm>
#include "simd/Config.h"

/**
 * Rational minimax approximations for sin, cos, and tan.
 * Each kernel is a Horner series in x squared and shares the mulAdd helper.
 * Coefficients are derived and checked by scripts/python/remez_*.py.
 */
namespace MinimaxSinCoeffs
{
    constexpr float N0 = 1.0f;
    constexpr float N1 = -0.141643688f;
    constexpr float N2 = 0.00446910504f;
    constexpr float N3 = -3.88648514e-05f;
    constexpr float D0 = 1.0f;
    constexpr float D1 = 0.0250229947f;
    constexpr float D2 = 0.000306247879f;
    constexpr float D3 = 2.07578137e-06f;
}

/// Fused multiply-add alias over the SIMD layer.
inline M128 mulAdd(const M128 a, const M128 b, const M128 c) noexcept
{
    return FMADD(a, b, c);
}

/// Evaluate the [7/6] sin approximation on a scalar.
inline float minimaxSinApprox(const float x) noexcept
{
    using namespace MinimaxSinCoeffs;
    const auto x2 = x * x;
    const auto num = x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));
    return num / den;
}

inline float mmSin(const float x) noexcept { return minimaxSinApprox(x); }

/// Evaluate the [7/6] sin approximation across four lanes.
inline M128 mmSin(const M128 x) noexcept
{
    using namespace MinimaxSinCoeffs;
    const auto vN0 = MM(set1_ps)(N0);
    const auto vN1 = MM(set1_ps)(N1);
    const auto vN2 = MM(set1_ps)(N2);
    const auto vN3 = MM(set1_ps)(N3);
    const auto vD0 = MM(set1_ps)(D0);
    const auto vD1 = MM(set1_ps)(D1);
    const auto vD2 = MM(set1_ps)(D2);
    const auto vD3 = MM(set1_ps)(D3);
    const auto x2 = MM(mul_ps)(x, x);
    auto numInner = mulAdd(x2, vN3, vN2);
    numInner = mulAdd(x2, numInner, vN1);
    numInner = mulAdd(x2, numInner, vN0);
    const auto num = MM(mul_ps)(x, numInner);
    auto denInner = mulAdd(x2, vD3, vD2);
    denInner = mulAdd(x2, denInner, vD1);
    const auto den = mulAdd(x2, denInner, vD0);
    return MM(div_ps)(num, den);
}

/**
 * Cos approximation, [6/6] even rational minimax on the range [-pi, pi].
 * Float32 max abs error 3.5e-07 with FMA.
 */
namespace MinimaxCosCoeffs
{
    // Cos is even, so the whole approximant is a series in x squared.
    constexpr float N0 = 1.0f;
    constexpr float N1 = -0.469479561f;
    constexpr float N2 = 0.0268812291f;
    constexpr float N3 = -0.00035018372f;
    constexpr float D0 = 1.0f;
    constexpr float D1 = 0.03052059f;
    constexpr float D2 = 0.000474582455f;
    constexpr float D3 = 4.48269293e-06f;
}

/// Evaluate the [6/6] cos approximation on a scalar.
inline float minimaxCosApprox(const float x) noexcept
{
    using namespace MinimaxCosCoeffs;
    const auto x2 = x * x;
    const auto num = N0 + x2 * (N1 + x2 * (N2 + x2 * N3));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));
    return num / den;
}

inline float mmCos(const float x) noexcept { return minimaxCosApprox(x); }

/// Evaluate the [6/6] cos approximation across four lanes.
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
    auto numInner = mulAdd(x2, vN3, vN2);
    numInner = mulAdd(x2, numInner, vN1);
    const auto num = mulAdd(x2, numInner, vN0);
    auto denInner = mulAdd(x2, vD3, vD2);
    denInner = mulAdd(x2, denInner, vD1);
    const auto den = mulAdd(x2, denInner, vD0);
    return MM(div_ps)(num, den);
}

/**
 * Tan approximation, [7/6] odd rational minimax on [-1.55, 1.55].
 * Fitted for relative error. The denominator carries a root at pi/2
 * to reproduce the pole. Range-reduce before calling: |x| >= pi/2 is invalid.
 * Float32 is rounding-limited near the pole, not coefficient-limited.
 */
namespace MinimaxTanCoeffs
{
    constexpr float N0 = 1.0f;
    constexpr float N1 = -0.128538921f;
    constexpr float N2 = 0.00283448538f;
    constexpr float N3 = -7.76689558e-06f;
    constexpr float D0 = 1.0f;
    constexpr float D1 = -0.46187225f;
    constexpr float D2 = 0.0234585702f;
    constexpr float D3 = -0.000212576764f;
}

/// Evaluate the [7/6] tan approximation on a scalar.
inline float minimaxTanApprox(const float x) noexcept
{
    using namespace MinimaxTanCoeffs;
    const auto x2 = x * x;
    const auto num = x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));
    return num / den;
}

inline float mmTan(const float x) noexcept { return minimaxTanApprox(x); }

/// Evaluate the [7/6] tan approximation across four lanes.
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
    auto numInner = mulAdd(x2, vN3, vN2);
    numInner = mulAdd(x2, numInner, vN1);
    const auto poly = mulAdd(x2, numInner, vN0);
    const auto num = MM(mul_ps)(x, poly);
    auto denInner = mulAdd(x2, vD3, vD2);
    denInner = mulAdd(x2, denInner, vD1);
    const auto den = mulAdd(x2, denInner, vD0);
    return MM(div_ps)(num, den);
}
#endif

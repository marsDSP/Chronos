#pragma once

#ifndef CHRONOS_FASTERMATH_H
#define CHRONOS_FASTERMATH_H

#include "simd/simd_config.h"

namespace MarsDSP::inline FasterMath
{
    // pade sin(x) ≈ N(x) / D(x)
    // [7/6] approximant coefficients for sin(x)
    //
    //            -x · (N0 + x²·(N1 + x²·(N2 + x²·N3)))
    // sin(x) ≈  ──────────────────────────────────────────
    //                  D0 + x²·(D1 + x²·(D2 + x²·D3))
    //
    // N(x) = -x · (-11511339840 + x²·(1640635920 + x²·(-52785432 + x²·479249)))
    // D(x) =        11511339840 + x²·(277920720 + x²·(3177720 + x²· 18361))
    //
    // accuracy: ~24-bit float-precision over approx [-π, π]

    namespace PadeSinCoeffs
    {
        constexpr float N0 = -11511339840.0f;   // num x⁰ (before outer -x)
        constexpr float N1 =  1640635920.0f;    // num x²
        constexpr float N2 = -52785432.0f;      // num x⁴
        constexpr float N3 =  479249.0f;        // num x⁶

        constexpr float D0 =  11511339840.0f;   // den x⁰
        constexpr float D1 =  277920720.0f;     // den x²
        constexpr float D2 =  3177720.0f;       // den x⁴
        constexpr float D3 =  18361.0f;         // den x⁶
    }

    inline float padeSinApprox(const float x) noexcept
    {
        using namespace PadeSinCoeffs;

        const auto x2 = x * x;

        // horner evaluation inside-out in x²
        const auto num = -x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
        const auto den =       D0 + x2 * (D1 + x2 * (D2 + x2 * D3));

        return num / den;
    }

    inline SIMD_M128 fasterSin(const SIMD_M128 x) noexcept
    {
        using namespace PadeSinCoeffs;

        // broadcast each coeff across 4 lanes
        const auto vN0 = SIMD_MM(set_ps1)(N0);
        const auto vN1 = SIMD_MM(set_ps1)(N1);
        const auto vN2 = SIMD_MM(set_ps1)(N2);
        const auto vN3 = SIMD_MM(set_ps1)(N3);

        const auto vD0 = SIMD_MM(set_ps1)(D0);
        const auto vD1 = SIMD_MM(set_ps1)(D1);
        const auto vD2 = SIMD_MM(set_ps1)(D2);
        const auto vD3 = SIMD_MM(set_ps1)(D3);

        const auto neg = SIMD_MM(set_ps1)(-1.0f);
        const auto x2  = SIMD_MM(mul_ps)(x, x);

        // numerator:  -x · (N0 + x²·(N1 + x²·(N2 + x²·N3))) | innermost first
        auto numInner  = SIMD_MM(add_ps)(vN2, SIMD_MM(mul_ps)(x2, vN3));        // N2 + x²·N3
        numInner       = SIMD_MM(add_ps)(vN1, SIMD_MM(mul_ps)(x2, numInner));   // N1 + x²·(…)
        numInner       = SIMD_MM(add_ps)(vN0, SIMD_MM(mul_ps)(x2, numInner));   // N0 + x²·(…)
        const auto num = SIMD_MM(mul_ps)(neg, SIMD_MM(mul_ps)(x, numInner));

        // denominator: D0 + x²·(D1 + x²·(D2 + x²·D3))
        auto denInner  = SIMD_MM(add_ps)(vD2, SIMD_MM(mul_ps)(x2, vD3));        // D2 + x²·D3
        denInner       = SIMD_MM(add_ps)(vD1, SIMD_MM(mul_ps)(x2, denInner));   // D1 + x²·(…)
        const auto den = SIMD_MM(add_ps)(vD0, SIMD_MM(mul_ps)(x2, denInner));

        return SIMD_MM(div_ps)(num, den);
    }
//==============================================================================//
    namespace PadeCosCoeffs
    {
        // cos(x) is an even function so signs are flipped relative to sin(x)
        constexpr float N0 =  39251520.0f;    // num x⁰
        constexpr float N1 = -18471600.0f;    // num x²
        constexpr float N2 =  1075032.0f;     // num x⁴
        constexpr float N3 = -14615.0f;       // num x⁶

        constexpr float D0 =  39251520.0f;    // den x⁰
        constexpr float D1 =  1154160.0f;     // den x²
        constexpr float D2 =  16632.0f;       // den x⁴
        constexpr float D3 =  127.0f;         // den x⁶
    }

    inline float padeCosApprox(const float x) noexcept
    {
        using namespace PadeCosCoeffs;

        // cos depends only on x², confirming even symmetry
        const auto x2 = x * x;

        // horner evaluation inside-out in x²
        const auto num = N0 + x2 * (N1 + x2 * (N2 + x2 * N3));
        const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));

        return num / den;
    }

    inline SIMD_M128 fasterCos(const SIMD_M128 x) noexcept
    {
        using namespace PadeCosCoeffs;

        // broadcast each coeff across 4 lanes
        const auto vN0 = SIMD_MM(set_ps1)(N0);
        const auto vN1 = SIMD_MM(set_ps1)(N1);
        const auto vN2 = SIMD_MM(set_ps1)(N2);
        const auto vN3 = SIMD_MM(set_ps1)(N3);

        const auto vD0 = SIMD_MM(set_ps1)(D0);
        const auto vD1 = SIMD_MM(set_ps1)(D1);
        const auto vD2 = SIMD_MM(set_ps1)(D2);
        const auto vD3 = SIMD_MM(set_ps1)(D3);

        const auto x2 = SIMD_MM(mul_ps)(x, x);

        // numerator: N0 + x²·(N1 + x²·(N2 + x²·N3))
        auto numInner  = SIMD_MM(add_ps)(vN2, SIMD_MM(mul_ps)(x2, vN3));        // N2 + x²·N3
        numInner       = SIMD_MM(add_ps)(vN1, SIMD_MM(mul_ps)(x2, numInner));   // N1 + x²·(…)
        const auto num = SIMD_MM(add_ps)(vN0, SIMD_MM(mul_ps)(x2, numInner));   // N0 + x²·(…)

        // denominator: D0 + x²·(D1 + x²·(D2 + x²·D3))
        auto denInner  = SIMD_MM(add_ps)(vD2, SIMD_MM(mul_ps)(x2, vD3));        // D2 + x²·D3
        denInner       = SIMD_MM(add_ps)(vD1, SIMD_MM(mul_ps)(x2, denInner));   // D1 + x²·(…)
        const auto den = SIMD_MM(add_ps)(vD0, SIMD_MM(mul_ps)(x2, denInner));

        return SIMD_MM(div_ps)(num, den);
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
        const auto den =       D0 + x2 * (D1 + x2 * (D2 + x2 * D3));

        return num / den;
    }

    inline SIMD_M128 fasterTan(const SIMD_M128 x) noexcept
    {
        using namespace PadeTanCoeffs;

        const auto vN0 = SIMD_MM(set_ps1)(N0);
        const auto vN1 = SIMD_MM(set_ps1)(N1);
        const auto vN2 = SIMD_MM(set_ps1)(N2);
        const auto vN3 = SIMD_MM(set_ps1)(N3);

        const auto vD0 = SIMD_MM(set_ps1)(D0);
        const auto vD1 = SIMD_MM(set_ps1)(D1);
        const auto vD2 = SIMD_MM(set_ps1)(D2);
        const auto vD3 = SIMD_MM(set_ps1)(D3);

        const auto x2  = SIMD_MM(mul_ps)(x, x);

        auto numInner  = SIMD_MM(add_ps)(vN2, SIMD_MM(mul_ps)(x2, vN3));        // N2 + x²·N3
        numInner       = SIMD_MM(add_ps)(vN1, SIMD_MM(mul_ps)(x2, numInner));   // N1 + x²·(…)
        const auto poly = SIMD_MM(add_ps)(vN0, SIMD_MM(mul_ps)(x2, numInner));   // N0 + x²·(…)
        const auto num  = SIMD_MM(mul_ps)(x, poly);

        auto denInner  = SIMD_MM(add_ps)(vD2, SIMD_MM(mul_ps)(x2, vD3));        // D2 + x²·D3
        denInner       = SIMD_MM(add_ps)(vD1, SIMD_MM(mul_ps)(x2, denInner));   // D1 + x²·(…)
        const auto den = SIMD_MM(add_ps)(vD0, SIMD_MM(mul_ps)(x2, denInner));

        return SIMD_MM(div_ps)(num, den);
    }
//==============================================================================//
}
#endif
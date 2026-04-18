#pragma once

#ifndef CHRONOS_FASTERMATH_H
#define CHRONOS_FASTERMATH_H

#include <cmath>
#include <algorithm>
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

    inline float fasterSin(const float x) noexcept
    {
        return padeSinApprox(x);
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

    inline float fasterCos(const float x) noexcept
    {
        return padeCosApprox(x);
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

        const auto x2  = SIMD_MM(mul_ps)(x, x);

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

    inline float fasterTan(const float x) noexcept
    {
        return padeTanApprox(x);
    }

    inline SIMD_M128 fasterTan(const SIMD_M128 x) noexcept
    {
        using namespace PadeTanCoeffs;

        const auto vN0  = SIMD_MM(set_ps1)(N0);
        const auto vN1  = SIMD_MM(set_ps1)(N1);
        const auto vN2  = SIMD_MM(set_ps1)(N2);
        const auto vN3  = SIMD_MM(set_ps1)(N3);

        const auto vD0  = SIMD_MM(set_ps1)(D0);
        const auto vD1  = SIMD_MM(set_ps1)(D1);
        const auto vD2  = SIMD_MM(set_ps1)(D2);
        const auto vD3  = SIMD_MM(set_ps1)(D3);

        const auto x2   = SIMD_MM(mul_ps)(x, x);

        auto numInner   = SIMD_MM(add_ps)(vN2, SIMD_MM(mul_ps)(x2, vN3));        // N2 + x²·N3
        numInner        = SIMD_MM(add_ps)(vN1, SIMD_MM(mul_ps)(x2, numInner));   // N1 + x²·(…)
        const auto poly = SIMD_MM(add_ps)(vN0, SIMD_MM(mul_ps)(x2, numInner));   // N0 + x²·(…)
        const auto num  = SIMD_MM(mul_ps)(x, poly);

        auto denInner   = SIMD_MM(add_ps)(vD2, SIMD_MM(mul_ps)(x2, vD3));        // D2 + x²·D3
        denInner        = SIMD_MM(add_ps)(vD1, SIMD_MM(mul_ps)(x2, denInner));   // D1 + x²·(…)
        const auto den  = SIMD_MM(add_ps)(vD0, SIMD_MM(mul_ps)(x2, denInner));

        return SIMD_MM(div_ps)(num, den);
    }
//==============================================================================//
    namespace PadeTanhCoeffs
    {
        constexpr float N0 = 135135.0f;
        constexpr float N1 = 17325.0f;
        constexpr float N2 = 378.0f;
        constexpr float N3 = 1.0f;

        constexpr float D0 = 135135.0f;
        constexpr float D1 = 62370.0f;
        constexpr float D2 = 3150.0f;
        constexpr float D3 = 28.0f;
    }

    inline float padeTanhApprox(const float x) noexcept
    {
        using namespace PadeTanhCoeffs;

        const auto x2 = x * x;

        const auto num = x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
        const auto den =      D0 + x2 * (D1 + x2 * (D2 + x2 * D3));

        return num / den;
    }

    inline float fasterTanh(const float x) noexcept
    {
        return padeTanhApprox(x);
    }

    template<typename T>
    inline T fasterTanhBounded(T x) noexcept
    {
        return static_cast<T>(padeTanhApprox(std::clamp(static_cast<float>(x), -5.0f, 5.0f)));
    }

    inline SIMD_M128 fasterTanh(const SIMD_M128 x) noexcept
    {
        using namespace PadeTanhCoeffs;

        const auto vN0  = SIMD_MM(set_ps1)(N0);
        const auto vN1  = SIMD_MM(set_ps1)(N1);
        const auto vN2  = SIMD_MM(set_ps1)(N2);
        const auto vN3  = SIMD_MM(set_ps1)(N3);

        const auto vD0  = SIMD_MM(set_ps1)(D0);
        const auto vD1  = SIMD_MM(set_ps1)(D1);
        const auto vD2  = SIMD_MM(set_ps1)(D2);
        const auto vD3  = SIMD_MM(set_ps1)(D3);

        const auto x2   = SIMD_MM(mul_ps)(x, x);

        auto numInner   = SIMD_MM(add_ps)(vN2, SIMD_MM(mul_ps)(x2, vN3));        // N2 + x²·N3
        numInner        = SIMD_MM(add_ps)(vN1, SIMD_MM(mul_ps)(x2, numInner));   // N1 + x²·(…)
        const auto poly = SIMD_MM(add_ps)(vN0, SIMD_MM(mul_ps)(x2, numInner));   // N0 + x²·(…)
        const auto num  = SIMD_MM(mul_ps)(x, poly);

        auto denInner   = SIMD_MM(add_ps)(vD2, SIMD_MM(mul_ps)(x2, vD3));        // D2 + x²·D3
        denInner        = SIMD_MM(add_ps)(vD1, SIMD_MM(mul_ps)(x2, denInner));   // D1 + x²·(…)
        const auto den  = SIMD_MM(add_ps)(vD0, SIMD_MM(mul_ps)(x2, denInner));

        return SIMD_MM(div_ps)(num, den);
    }

    inline SIMD_M128 fasterTanhBounded(const SIMD_M128 x) noexcept
    {
        // clamp to [-5, 5]
        const auto v5 = SIMD_MM(set1_ps)(5.0f);
        const auto vn5 = SIMD_MM(set1_ps)(-5.0f);
        const auto xbounded = SIMD_MM(min_ps)(v5, SIMD_MM(max_ps)(vn5, x));

        return fasterTanh(xbounded);
    }
//==============================================================================//
    inline float boundToPi(const float angle)
    {
        // fast path: already in canonical range
        if (angle <= M_PI && angle >= -M_PI)
            return angle;

        // shift from [-π, π] target into [0, 2π) working range
        const float shifted = angle + M_PI;

        constexpr float invTwoPi = 1.0f / (2.0f * M_PI);

        // how many whole turns of 2π fit inside `shifted` (truncated toward zero)
        const int    wholeTurns = static_cast<int>(shifted * invTwoPi);

        // remainder after removing those whole turns; lies in (-2π, 2π)
        float        wrapped    = shifted - 2.0f * M_PI * wholeTurns;

        // fold any negative remainder up into [0, 2π)
        if (wrapped < 0.0f)
            wrapped += 2.0f * M_PI;

        // undo the initial π shift → result in [-π, π]
        return wrapped - M_PI;
    }

    inline SIMD_M128 boundToPiSIMD(const SIMD_M128 angle)
    {
        // [π, π, π, π]
        const auto vPi        = SIMD_MM(set1_ps)(M_PI);

        // [2π, 2π, 2π, 2π]
        const auto vTwoPi     = SIMD_MM(set1_ps)(2.0f * M_PI);
        const auto vInvTwoPi  = SIMD_MM(set1_ps)(1.0f / (2.0f * M_PI));
        const auto vZero      = SIMD_MM(setzero_ps)();

        // shift range so we can work in [0, 2π) per lane
        const auto shifted    = SIMD_MM(add_ps)(angle, vPi);

        // trunc(shifted / 2π) per lane, kept as float for the next multiply.
        // float → int32 (truncating) → float round-trip mimics static_cast<int>.
        const auto wholeTurns = SIMD_MM(cvtepi32_ps)(SIMD_MM(cvttps_epi32)(SIMD_MM(mul_ps)(shifted, vInvTwoPi)));

        // remainder after stripping out whole 2π turns; lies in (-2π, 2π)
        auto       wrapped    = SIMD_MM(sub_ps)(shifted, SIMD_MM(mul_ps)(vTwoPi, wholeTurns));

        // branchless "if (wrapped < 0) wrapped += 2π":
        //   cmplt_ps  → mask: all-1 bits where lane is negative, all-0 elsewhere
        //   and_ps    → picks 2π on negative lanes, 0.0 on non-negative lanes
        const auto negFixup   = SIMD_MM(and_ps)(SIMD_MM(cmplt_ps)(wrapped, vZero), vTwoPi);
        wrapped               = SIMD_MM(add_ps)(wrapped, negFixup);

        // undo the initial π shift → every lane now in [-π, π]
        return SIMD_MM(sub_ps)(wrapped, vPi);
    }
//==============================================================================//
}
#endif
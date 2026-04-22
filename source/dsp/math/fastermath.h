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
        const auto vN0 = SIMD_MM(set1_ps)(N0);
        const auto vN1 = SIMD_MM(set1_ps)(N1);
        const auto vN2 = SIMD_MM(set1_ps)(N2);
        const auto vN3 = SIMD_MM(set1_ps)(N3);

        const auto vD0 = SIMD_MM(set1_ps)(D0);
        const auto vD1 = SIMD_MM(set1_ps)(D1);
        const auto vD2 = SIMD_MM(set1_ps)(D2);
        const auto vD3 = SIMD_MM(set1_ps)(D3);

        const auto neg = SIMD_MM(set1_ps)(-1.0f);
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

        const auto vN0 = SIMD_MM(set1_ps)(N0);
        const auto vN1 = SIMD_MM(set1_ps)(N1);
        const auto vN2 = SIMD_MM(set1_ps)(N2);
        const auto vN3 = SIMD_MM(set1_ps)(N3);

        const auto vD0 = SIMD_MM(set1_ps)(D0);
        const auto vD1 = SIMD_MM(set1_ps)(D1);
        const auto vD2 = SIMD_MM(set1_ps)(D2);
        const auto vD3 = SIMD_MM(set1_ps)(D3);

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

        const auto vN0  = SIMD_MM(set1_ps)(N0);
        const auto vN1  = SIMD_MM(set1_ps)(N1);
        const auto vN2  = SIMD_MM(set1_ps)(N2);
        const auto vN3  = SIMD_MM(set1_ps)(N3);

        const auto vD0  = SIMD_MM(set1_ps)(D0);
        const auto vD1  = SIMD_MM(set1_ps)(D1);
        const auto vD2  = SIMD_MM(set1_ps)(D2);
        const auto vD3  = SIMD_MM(set1_ps)(D3);

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

    inline SIMD_M128 fasterTanh(const SIMD_M128 x) noexcept
    {
        using namespace PadeTanhCoeffs;

        const auto vN0  = SIMD_MM(set1_ps)(N0);
        const auto vN1  = SIMD_MM(set1_ps)(N1);
        const auto vN2  = SIMD_MM(set1_ps)(N2);
        const auto vN3  = SIMD_MM(set1_ps)(N3);

        const auto vD0  = SIMD_MM(set1_ps)(D0);
        const auto vD1  = SIMD_MM(set1_ps)(D1);
        const auto vD2  = SIMD_MM(set1_ps)(D2);
        const auto vD3  = SIMD_MM(set1_ps)(D3);

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

    template<typename T>
    T fasterTanhBounded(T x) noexcept
    {
        return static_cast<T>(padeTanhApprox(std::clamp(static_cast<float>(x), -5.0f, 5.0f)));
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
    // pade exp(x) ≈ N(x) / D(x)
    // [4/4] approximant coefficients for exp(x)
    //
    //             N0 + x·(N1 + x·(N2 + x·(N3 + x)))
    // exp(x) ≈  ─────────────────────────────────────
    //             D0 + x·(D1 + x·(D2 + x·(D3 + x)))
    //
    // N(x) = 1680 + x·(  840 + x·(180 + x·(  20 + x)))
    // D(x) = 1680 + x·( -840 + x·(180 + x·( -20 + x)))
    //
    // exp is neither even nor odd so unlike sin/cos/tan/tanh here we expand
    // in x directly rather than in x².
    //
    // accuracy: near-double-precision around zero; degrades for |x| beyond
    // ~5. Used where the argument is already bounded (see fastTanhIntegral
    // below) - not safe as a general-purpose exp for arbitrary-range inputs.

    namespace PadeExpCoeffs
    {
        constexpr float N0 =  1680.0f;   // num x⁰
        constexpr float N1 =   840.0f;   // num x¹
        constexpr float N2 =   180.0f;   // num x²
        constexpr float N3 =    20.0f;   // num x³ (implicit x⁴ coefficient = 1)

        constexpr float D0 =  1680.0f;   // den x⁰
        constexpr float D1 =  -840.0f;   // den x¹
        constexpr float D2 =   180.0f;   // den x²
        constexpr float D3 =   -20.0f;   // den x³ (implicit x⁴ coefficient = 1)
    }

    inline float padeExpApprox(const float x) noexcept
    {
        using namespace PadeExpCoeffs;

        // horner evaluation inside-out in x
        const auto num = N0 + x * (N1 + x * (N2 + x * (N3 + x)));
        const auto den = D0 + x * (D1 + x * (D2 + x * (D3 + x)));

        return num / den;
    }

    inline float fasterExp(const float x) noexcept
    {
        return padeExpApprox(x);
    }

    inline SIMD_M128 fasterExp(const SIMD_M128 x) noexcept
    {
        using namespace PadeExpCoeffs;

        // broadcast each coeff across 4 lanes
        const auto vN0 = SIMD_MM(set1_ps)(N0);
        const auto vN1 = SIMD_MM(set1_ps)(N1);
        const auto vN2 = SIMD_MM(set1_ps)(N2);
        const auto vN3 = SIMD_MM(set1_ps)(N3);

        const auto vD0 = SIMD_MM(set1_ps)(D0);
        const auto vD1 = SIMD_MM(set1_ps)(D1);
        const auto vD2 = SIMD_MM(set1_ps)(D2);
        const auto vD3 = SIMD_MM(set1_ps)(D3);

        // numerator:   N0 + x·(N1 + x·(N2 + x·(N3 + x))) | innermost first
        auto numInner  = SIMD_MM(add_ps)(vN3, x);                              // N3 + x
        numInner       = SIMD_MM(add_ps)(vN2, SIMD_MM(mul_ps)(x, numInner));   // N2 + x·(…)
        numInner       = SIMD_MM(add_ps)(vN1, SIMD_MM(mul_ps)(x, numInner));   // N1 + x·(…)
        const auto num = SIMD_MM(add_ps)(vN0, SIMD_MM(mul_ps)(x, numInner));   // N0 + x·(…)

        // denominator: D0 + x·(D1 + x·(D2 + x·(D3 + x)))
        auto denInner  = SIMD_MM(add_ps)(vD3, x);                              // D3 + x
        denInner       = SIMD_MM(add_ps)(vD2, SIMD_MM(mul_ps)(x, denInner));   // D2 + x·(…)
        denInner       = SIMD_MM(add_ps)(vD1, SIMD_MM(mul_ps)(x, denInner));   // D1 + x·(…)
        const auto den = SIMD_MM(add_ps)(vD0, SIMD_MM(mul_ps)(x, denInner));   // D0 + x·(…)

        return SIMD_MM(div_ps)(num, den);
    }
//==============================================================================//
    // SIMD log (Cephes-style, ~6 decimal digits for x > 0).
    // log(x) = n*ln2 + log(m)  with  x = 2^n * m,  m in [sqrt(2)/2, sqrt(2)].
    // log(m) via a Horner polynomial in (m - 1).
    inline SIMD_M128 fasterLog(const SIMD_M128 xin) noexcept
    {
        // keep inputs strictly positive to avoid NaN on zero/negatives.
        const auto xMin = SIMD_MM(set1_ps)(1.17549435e-38f);
        const auto x = SIMD_MM(max_ps)(xin, xMin);

        // exponent e and fraction m of the float.
        auto e = SIMD_MM(srli_epi32)(SIMD_MM(castps_si128)(x), 23);
        e      = SIMD_MM(sub_epi32)(e, SIMD_MM(set1_epi32)(127));
        auto ef = SIMD_MM(cvtepi32_ps)(e);

        const auto mantMask = SIMD_MM(castsi128_ps)(SIMD_MM(set1_epi32)(0x007FFFFF));
        const auto oneBits  = SIMD_MM(castsi128_ps)(SIMD_MM(set1_epi32)(0x3F800000));

        auto m = SIMD_MM(or_ps)(SIMD_MM(and_ps)(x, mantMask), oneBits);

        // Mantissa extraction gives m in [1, 2). Fold the upper half down:
        // if m > sqrt(2), halve it and bump the exponent. That lands m in
        // (sqrt(2)/2, sqrt(2)] so (m - 1) ~ [-0.293, 0.414] - the design
        // range of the minimax polynomial below. Without this fold, m-1 can
        // be as large as ~1.0, which drives the polynomial far outside its
        // sweet spot and produces audible noise in downstream consumers
        // like fastTanhIntegral.
        const auto SQRT2   = SIMD_MM(set1_ps)(1.4142135623730951f);
        const auto mask    = SIMD_MM(cmpgt_ps)(m, SQRT2);
        const auto halfM   = SIMD_MM(mul_ps)(m, SIMD_MM(set1_ps)(0.5f));

        m  = SIMD_MM(or_ps)(SIMD_MM(and_ps)(mask, halfM), SIMD_MM(andnot_ps)(mask, m));
        ef = SIMD_MM(add_ps)(ef, SIMD_MM(and_ps)(mask, SIMD_MM(set1_ps)(1.0f)));
        m  = SIMD_MM(sub_ps)(m, SIMD_MM(set1_ps)(1.0f));

        // minimax polynomial for log(1+m) * m^3 on m in ~[-0.293, 0.414].
        const auto m2 = SIMD_MM(mul_ps)(m, m);
        auto poly = SIMD_MM(set1_ps)(7.0376836292E-2f);

        poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, m), SIMD_MM(set1_ps)(-1.1514610310E-1f));
        poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, m), SIMD_MM(set1_ps)( 1.1676998740E-1f));
        poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, m), SIMD_MM(set1_ps)(-1.2420140846E-1f));
        poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, m), SIMD_MM(set1_ps)( 1.4249322787E-1f));
        poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, m), SIMD_MM(set1_ps)(-1.6668057665E-1f));
        poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, m), SIMD_MM(set1_ps)( 2.0000714765E-1f));
        poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, m), SIMD_MM(set1_ps)(-2.4999993993E-1f));
        poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, m), SIMD_MM(set1_ps)( 3.3333331174E-1f));
        poly = SIMD_MM(mul_ps)(SIMD_MM(mul_ps)(poly, m), m2);

        poly = SIMD_MM(sub_ps)(poly, SIMD_MM(mul_ps)(ef, SIMD_MM(set1_ps)(2.12194440e-4f)));
        poly = SIMD_MM(sub_ps)(poly, SIMD_MM(mul_ps)(m2, SIMD_MM(set1_ps)(0.5f)));

        auto result = SIMD_MM(add_ps)(m, poly);
        result = SIMD_MM(add_ps)(result, SIMD_MM(mul_ps)(ef, SIMD_MM(set1_ps)(0.693359375f)));

        return result;
    }

    inline float fasterLog(const float x) noexcept
    {
        alignas(16) float out[4];
        SIMD_MM(store_ps)(out, fasterLog(SIMD_MM(set1_ps)(x)));
        return out[0];
    }
//==============================================================================//
    // Bounded Padé tanh.
    //
    //   fastTanh(x) = x (27 + x^2) / (27 + 9 x^2)         for |x| < 3
    //              = sgn(x)                                for |x| >= 3
    //
    // Matches tanh(x) to roughly 1% over [-3, 3]; rails cleanly beyond.
    template <typename T>
    T fastTanh(const T x) noexcept
    {
        if (x < static_cast<T>(-3)) return static_cast<T>(-1);
        if (x > static_cast<T>( 3)) return static_cast<T>( 1);

        const T x2 = x * x;

        return x * (static_cast<T>(27) + x2) / (static_cast<T>(27) + static_cast<T>(9) * x2);
    }

    inline SIMD_M128 fastTanh(const SIMD_M128 x) noexcept
    {
        const auto three       = SIMD_MM(set1_ps)(3.0f);
        const auto negThree    = SIMD_MM(set1_ps)(-3.0f);
        const auto twentySeven = SIMD_MM(set1_ps)(27.0f);
        const auto nine        = SIMD_MM(set1_ps)(9.0f);
        const auto posOne      = SIMD_MM(set1_ps)(1.0f);
        const auto negOne      = SIMD_MM(set1_ps)(-1.0f);

        const auto x2   = SIMD_MM(mul_ps)(x, x);
        const auto num  = SIMD_MM(mul_ps)(x, SIMD_MM(add_ps)(twentySeven, x2));
        const auto den  = SIMD_MM(add_ps)(twentySeven, SIMD_MM(mul_ps)(nine, x2));
        const auto body = SIMD_MM(div_ps)(num, den);

        // clamp to saturation rails
        const auto ge3    = SIMD_MM(cmpge_ps)(x, three);
        const auto le_neg = SIMD_MM(cmple_ps)(x, negThree);

        auto result = body;

        result = SIMD_MM(or_ps)(SIMD_MM(and_ps)(ge3, posOne), SIMD_MM(andnot_ps)(ge3, result));
        result = SIMD_MM(or_ps)(SIMD_MM(and_ps)(le_neg, negOne), SIMD_MM(andnot_ps)(le_neg, result));

        return result;
    }
//==============================================================================//
    // Antiderivative of fastTanh.
    //
    //   |x| <  3:  F(x) = x^2 / 18 + (4/3) log(1 + x^2 / 3)
    //   |x| >= 3:  F(x) = |x| - c          with c = 2.5 - (8/3) ln 2 ≈ 0.6516075
    //
    // c is chosen so the two branches join with matching value at |x| = 3.
    // Derived by integrating the bounded Padé expression of fastTanh above.
    // Used by the SIMD ADAA waveshaper; the log argument 1 + x^2/3 lands in
    // [1, 4] for |x| ≤ 3, which is well inside the fasterLog Padé's sweet spot.
    inline SIMD_M128 fastTanhIntegral(const SIMD_M128 x) noexcept
    {
        const auto ABSMASK = SIMD_MM(castsi128_ps)(SIMD_MM(set1_epi32)(0x7FFFFFFF));
        const auto absX    = SIMD_MM(and_ps)(x, ABSMASK);
        const auto x2      = SIMD_MM(mul_ps)(x, x);

        // inner branch
        const auto logArg  = SIMD_MM(add_ps)(SIMD_MM(set1_ps)(1.0f), SIMD_MM(mul_ps)(x2, SIMD_MM(set1_ps)(1.0f / 3.0f)));
        const auto logPart = fasterLog(logArg);
        const auto inner   = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(x2, SIMD_MM(set1_ps)(1.0f / 18.0f)),
                                             SIMD_MM(mul_ps)(logPart, SIMD_MM(set1_ps)(4.0f / 3.0f)));

        // outer branch
        const auto outer = SIMD_MM(sub_ps)(absX, SIMD_MM(set1_ps)(0.6516075f));

        const auto mask  = SIMD_MM(cmpge_ps)(absX, SIMD_MM(set1_ps)(3.0f));
        return SIMD_MM(or_ps)(SIMD_MM(and_ps)(mask, outer), SIMD_MM(andnot_ps)(mask, inner));
    }

    inline float fastTanhIntegral(const float x) noexcept
    {
        alignas(16) float out[4];
        SIMD_MM(store_ps)(out, fastTanhIntegral(SIMD_MM(set1_ps)(x)));
        return out[0];
    }
//==============================================================================//
    // Antiderivative-antialiased tanh (first-order ADAA).
    //
    //   y[n] = (F(x[n]) - F(x[n-1])) / (x[n] - x[n-1])        with F = logcosh
    //
    // When |dx| is tiny we substitute the midpoint direct tanh to dodge 0/0.
    //
    // The SIMD version processes four consecutive samples per call. The
    // "previous sample" for lane 0 comes from the carry passed in; lanes 1..3
    // read from the quad's own earlier lanes via a one-lane left-shift:
    //
    //   xPrev = [carryX, x[0], x[1], x[2]]
    //   FPrev = [carryF, F[0], F[1], F[2]]
    //
    // On exit carryX / carryF are updated to lane-3 of the current quad so the
    // next call picks up where this one left off. Seed both to 0 at reset.
    inline SIMD_M128 fasterTanhADAA(const SIMD_M128 x, float &carryX, float &carryF) noexcept
    {
        // F is the antiderivative of the same bounded tanh used by the
        // fallback branch below - keeping both from a single primitive keeps
        // behaviour consistent at the branch transition.
        const auto F = fastTanhIntegral(x);

        // One-lane left shift (fill lane 0 with 0), then splice carry into lane 0.
        const auto xShift = SIMD_MM(castsi128_ps)(SIMD_MM(slli_si128)(SIMD_MM(castps_si128)(x), 4));
        const auto FShift = SIMD_MM(castsi128_ps)(SIMD_MM(slli_si128)(SIMD_MM(castps_si128)(F), 4));

        const auto xPrev = SIMD_MM(add_ss)(xShift, SIMD_MM(set_ss)(carryX));
        const auto FPrev = SIMD_MM(add_ss)(FShift, SIMD_MM(set_ss)(carryF));

        const auto dx = SIMD_MM(sub_ps)(x, xPrev);
        const auto dF = SIMD_MM(sub_ps)(F, FPrev);

        // Per-lane fallback when |dx| < eps: use direct tanh at the midpoint.
        const auto ABSMASK  = SIMD_MM(castsi128_ps)(SIMD_MM(set1_epi32)(0x7FFFFFFF));
        const auto absDx    = SIMD_MM(and_ps)(dx, ABSMASK);
        // Threshold chosen well above single-precision subtractive-cancellation
        // noise floor (~1 ulp of |x|) so lanes whose dx is dominated by round-off
        // fall into the direct-tanh midpoint branch instead of a noisy dF/dx.
        const auto tooSmall = SIMD_MM(cmplt_ps)(absDx, SIMD_MM(set1_ps)(1.0e-4f));
        const auto adaaQ    = SIMD_MM(div_ps)(dF, dx);
        const auto mid      = SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(0.5f), SIMD_MM(add_ps)(x, xPrev));
        const auto fallbk   = fasterTanhBounded(mid);

        const auto y = SIMD_MM(or_ps)(SIMD_MM(and_ps)   (tooSmall, fallbk),
                                      SIMD_MM(andnot_ps)(tooSmall, adaaQ));

        // Persist lane-3 as carry for the next quad.
        alignas(16) float lanesX[4], lanesF[4];

        SIMD_MM(store_ps)(lanesX, x);
        SIMD_MM(store_ps)(lanesF, F);

        carryX = lanesX[3];
        carryF = lanesF[3];

        return y;
    }

    inline float fasterTanhADAA(const float x, float &carryX, float &carryF) noexcept
    {
        const float F  = fastTanhIntegral(x);
        const float dx = x - carryX;

        float y;

        if (std::fabs(dx) < 1.0e-4f)
        {
            // same midpoint fallback as SIMD path, routed through the SIMD
            // tanhBounded so scalar/SIMD results stay bit-identical.
            alignas(16) float out[4];
            SIMD_MM(store_ps)(out, fasterTanhBounded(SIMD_MM(set1_ps)(0.5f * (x + carryX))));
            y = out[0];
        }
        else
        {
            y = (F - carryF) / dx;
        }

        carryX = x;
        carryF = F;

        return y;
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
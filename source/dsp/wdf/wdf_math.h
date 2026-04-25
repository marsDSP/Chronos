#pragma once

#ifndef CHRONOS_WDF_MATH_H
#define CHRONOS_WDF_MATH_H

// ============================================================================
//  wdf_math.h
// ----------------------------------------------------------------------------
//  Mathematical primitives the WDF diode-pair model needs to solve its
//  reflection equation in closed form. The headline routine is the Wright-
//  Omega function W(x); the diode-pair reflection uses W applied to the
//  signed forward-biased branch and reuses the result for the reverse branch
//  via the Werner symmetry trick.
//
//  Wright-Omega is the solution to W + log(W) = x (for x in R). Evaluating
//  it exactly per audio sample is too expensive, so we use a piecewise
//  combination of cheap polynomial approximations - this is the same
//  strategy used by Stefano D'Angelo's DAFx-2019 paper. Four orders are
//  exposed:
//
//      wrightOmega1: max(x, 0)                      ~6 dB error tail
//      wrightOmega2: cubic in central region        ~1% error
//      wrightOmega3: cubic + log(x) for large x     ~0.1% error
//      wrightOmega4: omega3 + one Halley correction ~near machine epsilon
//
//  Higher orders are recommended for diode clippers; the diode pair in this
//  codebase uses wrightOmega4. All routines are header-only and provide a
//  scalar (float / double) overload plus a SIMD_M128 overload that operates
//  on four float lanes in parallel using the same approximations.
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <cmath>
#include "dsp/math/simd/simd_config.h"

namespace MarsDSP::DSP::WDF
{
    // ------------------------------------------------------------------ //
    //  signum: sign(x) ∈ {-1, 0, +1}
    // ------------------------------------------------------------------ //
    template <typename T>
    inline T signum (T v) noexcept
    {
        return static_cast<T>((T{0} < v) - (v < T{0}));
    }

    inline SIMD_M128 signumSIMD (SIMD_M128 v) noexcept
    {
        const auto zero = SIMD_MM(setzero_ps)();
        const auto one  = SIMD_MM(set1_ps)(1.0f);
        const auto pos  = SIMD_MM(and_ps)(SIMD_MM(cmpgt_ps)(v, zero), one);
        const auto neg  = SIMD_MM(and_ps)(SIMD_MM(cmplt_ps)(v, zero), one);
        return SIMD_MM(sub_ps)(pos, neg);
    }

    // ------------------------------------------------------------------ //
    //  Estrin's scheme: evaluate a polynomial with degree-pairs in
    //  parallel; useful for SIMD throughput. coeffs are descending order
    //  (a_n, a_{n-1}, ..., a_0).
    // ------------------------------------------------------------------ //
    template <int N, typename T, typename X>
    inline auto estrinPoly (const T (&c)[N + 1], const X& x)
    {
        if constexpr (N == 1)
        {
            return c[1] + c[0] * x;
        }
        else
        {
            decltype (T{} * X{}) tmp[N / 2 + 1];
            for (int n = N; n >= 0; n -= 2)
                tmp[n / 2] = c[n] + c[n - 1] * x;
            if constexpr (N % 2 == 0)
                tmp[0] = c[0];
            return estrinPoly<N / 2> (tmp, x * x);
        }
    }

    // ------------------------------------------------------------------ //
    //  log2 approximation, optimised on [1, 2]
    // ------------------------------------------------------------------ //
    template <typename T>
    constexpr T log2Approx (T x) noexcept
    {
        constexpr T alpha = static_cast<T>( 0.1640425613334452);
        constexpr T beta  = static_cast<T>(-1.0988652862227440);
        constexpr T gamma = static_cast<T>( 3.1482979293341170);
        constexpr T zeta  = static_cast<T>(-2.2134752044448170);
        return estrinPoly<3, T, T> ({ alpha, beta, gamma, zeta }, x);
    }

    inline SIMD_M128 log2ApproxSIMD (SIMD_M128 x) noexcept
    {
        const auto a = SIMD_MM(set1_ps)( 0.1640425613334452f);
        const auto b = SIMD_MM(set1_ps)(-1.0988652862227440f);
        const auto g = SIMD_MM(set1_ps)( 3.1482979293341170f);
        const auto z = SIMD_MM(set1_ps)(-2.2134752044448170f);
        // Horner evaluation in x: ((a*x + b)*x + g)*x + z
        auto y = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(a, x), b);
        y      = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(y, x), g);
        y      = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(y, x), z);
        return y;
    }

    // ------------------------------------------------------------------ //
    //  log(x) approximation via mantissa / exponent split.
    // ------------------------------------------------------------------ //
    inline float logApprox (float x) noexcept
    {
        union { int32_t i; float f; } v {};
        v.f = x;
        const int32_t ex = v.i & 0x7F800000;
        const int32_t e  = (ex >> 23) - 127;
        v.i = (v.i - ex) | 0x3F800000;
        return 0.6931471805599453f * (static_cast<float>(e) + log2Approx<float>(v.f));
    }

    inline double logApprox (double x) noexcept
    {
        union { int64_t i; double d; } v {};
        v.d = x;
        const int64_t ex = v.i & 0x7FF0000000000000LL;
        const int64_t e  = (ex >> 52) - 1023;
        v.i = (v.i - ex) | 0x3FF0000000000000LL;
        return 0.6931471805599453 * (static_cast<double>(e) + log2Approx<double>(v.d));
    }

    inline SIMD_M128 logApproxSIMD (SIMD_M128 x) noexcept
    {
        const auto exMask  = SIMD_MM(set1_epi32)(0x7F800000);
        const auto oneBits = SIMD_MM(set1_epi32)(0x3F800000);
        const auto bias    = SIMD_MM(set1_epi32)(127);

        const auto xi  = SIMD_MM(castps_si128)(x);
        const auto ex  = SIMD_MM(and_si128)(xi, exMask);
        const auto e   = SIMD_MM(sub_epi32)(SIMD_MM(srli_epi32)(ex, 23), bias);
        const auto eF  = SIMD_MM(cvtepi32_ps)(e);
        const auto m   = SIMD_MM(castsi128_ps)(SIMD_MM(or_si128)(SIMD_MM(sub_epi32)(xi, ex), oneBits));

        const auto ln2 = SIMD_MM(set1_ps)(0.6931471805599453f);
        return SIMD_MM(mul_ps)(ln2, SIMD_MM(add_ps)(eF, log2ApproxSIMD(m)));
    }

    // ------------------------------------------------------------------ //
    //  pow2(x) approximation, optimised on [0, 1].
    // ------------------------------------------------------------------ //
    template <typename T>
    constexpr T pow2Approx (T x) noexcept
    {
        constexpr T alpha = static_cast<T>(0.07944154167983575);
        constexpr T beta  = static_cast<T>(0.22741127776021890);
        constexpr T gamma = static_cast<T>(0.69314718055994530);
        constexpr T zeta  = static_cast<T>(1.0);
        return estrinPoly<3, T, T> ({ alpha, beta, gamma, zeta }, x);
    }

    inline SIMD_M128 pow2ApproxSIMD (SIMD_M128 x) noexcept
    {
        const auto a = SIMD_MM(set1_ps)(0.07944154167983575f);
        const auto b = SIMD_MM(set1_ps)(0.22741127776021890f);
        const auto g = SIMD_MM(set1_ps)(0.69314718055994530f);
        const auto z = SIMD_MM(set1_ps)(1.0f);
        auto y = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(a, x), b);
        y      = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(y, x), g);
        y      = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(y, x), z);
        return y;
    }

    // ------------------------------------------------------------------ //
    //  exp(x) approximation built on pow2.
    // ------------------------------------------------------------------ //
    inline float expApprox (float x) noexcept
    {
        // Clamp to keep (l + 127) inside the valid 8-bit float exponent range
        // (1..254). x = 88.7 -> l = 88 -> biased = 215, safely inside.
        x = std::clamp (1.4426950408889630f * x, -126.0f, 88.0f);
        union { int32_t i; float f; } v {};
        const int32_t xi = static_cast<int32_t>(x);
        const int32_t l  = x < 0.0f ? xi - 1 : xi;
        const float   f  = x - static_cast<float>(l);
        v.i = (l + 127) << 23;
        return v.f * pow2Approx<float>(f);
    }

    inline double expApprox (double x) noexcept
    {
        x = std::clamp (1.4426950408889630 * x, -1022.0, 1023.0);
        union { int64_t i; double d; } v {};
        const int64_t xi = static_cast<int64_t>(x);
        const int64_t l  = x < 0.0 ? xi - 1 : xi;
        const double  f  = x - static_cast<double>(l);
        v.i = (l + 1023) << 52;
        return v.d * pow2Approx<double>(f);
    }

    inline SIMD_M128 expApproxSIMD (SIMD_M128 x) noexcept
    {
        x = SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(1.4426950408889630f), x);
        x = SIMD_MM(max_ps)(SIMD_MM(set1_ps)(-126.0f), x);
        x = SIMD_MM(min_ps)(SIMD_MM(set1_ps)( 88.0f), x);

        const auto xi   = SIMD_MM(cvttps_epi32)(x);
        const auto neg  = SIMD_MM(castps_si128)(SIMD_MM(cmplt_ps)(x, SIMD_MM(setzero_ps)()));
        const auto one  = SIMD_MM(set1_epi32)(1);
        const auto l    = SIMD_MM(sub_epi32)(xi, SIMD_MM(and_si128)(neg, one));
        const auto lF   = SIMD_MM(cvtepi32_ps)(l);
        const auto f    = SIMD_MM(sub_ps)(x, lF);

        const auto bias    = SIMD_MM(set1_epi32)(127);
        const auto biased  = SIMD_MM(add_epi32)(l, bias);
        const auto twoPowL = SIMD_MM(castsi128_ps)(SIMD_MM(slli_epi32)(biased, 23));

        return SIMD_MM(mul_ps)(twoPowL, pow2ApproxSIMD(f));
    }

    // ------------------------------------------------------------------ //
    //  Wright-Omega approximations.
    //  W(x) is the solution of W + log(W) = x.
    // ------------------------------------------------------------------ //
    template <typename T>
    inline T wrightOmega1 (T x) noexcept
    {
        return std::max (x, T{0});
    }

    inline SIMD_M128 wrightOmega1SIMD (SIMD_M128 x) noexcept
    {
        return SIMD_MM(max_ps)(x, SIMD_MM(setzero_ps)());
    }

    template <typename T>
    inline T wrightOmega3 (T x) noexcept
    {
        constexpr T x1 = static_cast<T>(-3.341459552768620);
        constexpr T x2 = static_cast<T>( 8.0);
        constexpr T a  = static_cast<T>(-1.314293149877800e-3);
        constexpr T b  = static_cast<T>( 4.775931364975583e-2);
        constexpr T c  = static_cast<T>( 3.631952663804445e-1);
        constexpr T d  = static_cast<T>( 6.313183464296682e-1);

        if (x < x1) return T{0};
        if (x < x2) return estrinPoly<3, T, T> ({ a, b, c, d }, x);
        return x - logApprox (x);
    }

    inline SIMD_M128 wrightOmega3SIMD (SIMD_M128 x) noexcept
    {
        const auto vx1   = SIMD_MM(set1_ps)(-3.341459552768620f);
        const auto vx2   = SIMD_MM(set1_ps)( 8.0f);
        const auto va    = SIMD_MM(set1_ps)(-1.314293149877800e-3f);
        const auto vb    = SIMD_MM(set1_ps)( 4.775931364975583e-2f);
        const auto vc    = SIMD_MM(set1_ps)( 3.631952663804445e-1f);
        const auto vd    = SIMD_MM(set1_ps)( 6.313183464296682e-1f);
        const auto zero  = SIMD_MM(setzero_ps)();

        // central region cubic in x
        auto poly = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(va, x), vb);
        poly      = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, x), vc);
        poly      = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(poly, x), vd);

        // large-x branch: x - log(x). log only valid for x > 0; for x <= 0 we
        // never use this branch so the input to logApproxSIMD stays positive.
        const auto safeX = SIMD_MM(max_ps)(x, SIMD_MM(set1_ps)(1.0e-9f));
        const auto big   = SIMD_MM(sub_ps)(x, logApproxSIMD(safeX));

        const auto belowLow  = SIMD_MM(cmplt_ps)(x, vx1);
        const auto aboveHigh = SIMD_MM(cmpge_ps)(x, vx2);
        const auto inMid     = SIMD_MM(andnot_ps)(SIMD_MM(or_ps)(belowLow, aboveHigh),
                                                  SIMD_MM(set1_ps)(-1.0f)); // junk we just need a mask
        // pick: belowLow → 0, aboveHigh → big, else poly
        const auto pickBig = SIMD_MM(and_ps)(aboveHigh, big);
        const auto pickMid = SIMD_MM(andnot_ps)(SIMD_MM(or_ps)(belowLow, aboveHigh), poly);
        const auto pickLow = SIMD_MM(and_ps)(belowLow, zero);
        (void) inMid;

        return SIMD_MM(or_ps)(SIMD_MM(or_ps)(pickLow, pickMid), pickBig);
    }

    template <typename T>
    inline T wrightOmega4 (T x) noexcept
    {
        const T y = wrightOmega3 (x);
        return y - (y - expApprox (x - y)) / (y + T{1});
    }

    inline SIMD_M128 wrightOmega4SIMD (SIMD_M128 x) noexcept
    {
        const auto y = wrightOmega3SIMD (x);
        const auto e = expApproxSIMD (SIMD_MM(sub_ps)(x, y));
        const auto num = SIMD_MM(sub_ps)(y, e);
        const auto den = SIMD_MM(add_ps)(y, SIMD_MM(set1_ps)(1.0f));
        return SIMD_MM(sub_ps)(y, SIMD_MM(div_ps)(num, den));
    }

    // ------------------------------------------------------------------ //
    //  Wide-domain sinh / cosh built on the bit-twiddle expApprox.
    //  Unlike fastermath.h's [-π, π]-Padé fasterSinh/fasterCosh, these are
    //  valid for any |x| up to ~88 in float (~709 in double), so they can
    //  be used inside the Newton diode-bridge solver where the inner
    //  argument v/V_T can reach ~27 at saturation.
    // ------------------------------------------------------------------ //
    template <typename T>
    inline T fastSinhWide (T x) noexcept
    {
        const T e_pos = expApprox ( x);
        const T e_neg = expApprox (-x);
        return (e_pos - e_neg) * static_cast<T>(0.5);
    }

    template <typename T>
    inline T fastCoshWide (T x) noexcept
    {
        const T e_pos = expApprox ( x);
        const T e_neg = expApprox (-x);
        return (e_pos + e_neg) * static_cast<T>(0.5);
    }

    inline SIMD_M128 fastSinhWideSIMD (SIMD_M128 x) noexcept
    {
        const auto e_pos = expApproxSIMD (x);
        const auto e_neg = expApproxSIMD (SIMD_MM(sub_ps)(SIMD_MM(setzero_ps)(), x));
        return SIMD_MM(mul_ps)(SIMD_MM(sub_ps)(e_pos, e_neg), SIMD_MM(set1_ps)(0.5f));
    }

    inline SIMD_M128 fastCoshWideSIMD (SIMD_M128 x) noexcept
    {
        const auto e_pos = expApproxSIMD (x);
        const auto e_neg = expApproxSIMD (SIMD_MM(sub_ps)(SIMD_MM(setzero_ps)(), x));
        return SIMD_MM(mul_ps)(SIMD_MM(add_ps)(e_pos, e_neg), SIMD_MM(set1_ps)(0.5f));
    }
}
#endif

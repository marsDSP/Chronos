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
// in x directly rather than in x²
//
// near-double-precision around zero; degrades for |x| beyond ~5.
// not safe as a general-purpose exp for arbitrary-range inputs!

namespace PadeExpCoeffs
{
    constexpr float N0 = 1680.0f; // num x⁰
    constexpr float N1 = 840.0f; // num x¹
    constexpr float N2 = 180.0f; // num x²
    constexpr float N3 = 20.0f; // num x³ (implicit x⁴ coefficient = 1)

    constexpr float D0 = 1680.0f; // den x⁰
    constexpr float D1 = -840.0f; // den x¹
    constexpr float D2 = 180.0f; // den x²
    constexpr float D3 = -20.0f; // den x³ (implicit x⁴ coefficient = 1)
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

inline M128 fasterExp(const M128 x) noexcept
{
    using namespace PadeExpCoeffs;

    // broadcast each coeff across 4 lanes
    const auto vN0 = MM(set1_ps)(N0);
    const auto vN1 = MM(set1_ps)(N1);
    const auto vN2 = MM(set1_ps)(N2);
    const auto vN3 = MM(set1_ps)(N3);

    const auto vD0 = MM(set1_ps)(D0);
    const auto vD1 = MM(set1_ps)(D1);
    const auto vD2 = MM(set1_ps)(D2);
    const auto vD3 = MM(set1_ps)(D3);

    // numerator: N0 + x·(N1 + x·(N2 + x·(N3 + x))) | innermost first
    auto numInner = MM(add_ps)(vN3, x); // N3 + x
    numInner = MM(add_ps)(vN2, MM(mul_ps)(x, numInner)); // N2 + x·(…)
    numInner = MM(add_ps)(vN1, MM(mul_ps)(x, numInner)); // N1 + x·(…)
    const auto num = MM(add_ps)(vN0, MM(mul_ps)(x, numInner)); // N0 + x·(…)

    // denominator: D0 + x·(D1 + x·(D2 + x·(D3 + x)))
    auto denInner = MM(add_ps)(vD3, x); // D3 + x
    denInner = MM(add_ps)(vD2, MM(mul_ps)(x, denInner)); // D2 + x·(…)
    denInner = MM(add_ps)(vD1, MM(mul_ps)(x, denInner)); // D1 + x·(…)
    const auto den = MM(add_ps)(vD0, MM(mul_ps)(x, denInner)); // D0 + x·(…)

    return MM(div_ps)(num, den);
}
//==============================================================================//
// SIMD log Cephes-style, ~6 decimal digits for x > 0.
// log(x) = n*ln2 + log(m) with x = 2^n * m, m in [sqrt(2)/2, sqrt(2)].
// log(m) via a Horner polynomial in (m - 1).
inline M128 fasterLog(const M128 xin) noexcept
{
    // keep inputs strictly positive to avoid NaN on zero/negatives.
    const auto xMin = MM(set1_ps)(1.17549435e-38f);
    const auto x = MM(max_ps)(xin, xMin);

    // exponent e and fraction m of the float.
    auto e = MM(srli_epi32)(MM(castps_si128)(x), 23);
    e = MM(sub_epi32)(e, MM(set1_epi32)(127));
    auto ef = MM(cvtepi32_ps)(e);

    const auto mantMask = MM(castsi128_ps)(MM(set1_epi32)(0x007FFFFF));
    const auto oneBits = MM(castsi128_ps)(MM(set1_epi32)(0x3F800000));

    auto m = MM(or_ps)(MM(and_ps)(x, mantMask), oneBits);

    // Mantissa extraction gives m in [1, 2). Fold the upper half down:
    // if m > sqrt(2), halve it and bump the exponent. That lands m in
    // (sqrt(2)/2, sqrt(2)] so (m - 1) ~ [-0.293, 0.414]. Without this fold,
    // m-1 can be as large as ~1.0, which drives the polynomial far outside its
    // sweet spot and produces audible noise in downstream consumers!
    const auto SQRT2 = MM(set1_ps)(1.4142135623730951f);
    const auto mask = MM(cmpgt_ps)(m, SQRT2);
    const auto halfM = MM(mul_ps)(m, MM(set1_ps)(0.5f));

    m = MM(or_ps)(MM(and_ps)(mask, halfM), MM(andnot_ps)(mask, m));
    ef = MM(add_ps)(ef, MM(and_ps)(mask, MM(set1_ps)(1.0f)));
    m = MM(sub_ps)(m, MM(set1_ps)(1.0f));

    // minimax polynomial for log(1+m) * m^3 on m in ~[-0.293, 0.414].
    const auto m2 = MM(mul_ps)(m, m);
    auto poly = MM(set1_ps)(7.0376836292E-2f);

    poly = MM(add_ps)(MM(mul_ps)(poly, m), MM(set1_ps)(-1.1514610310E-1f));
    poly = MM(add_ps)(MM(mul_ps)(poly, m), MM(set1_ps)(1.1676998740E-1f));
    poly = MM(add_ps)(MM(mul_ps)(poly, m), MM(set1_ps)(-1.2420140846E-1f));
    poly = MM(add_ps)(MM(mul_ps)(poly, m), MM(set1_ps)(1.4249322787E-1f));
    poly = MM(add_ps)(MM(mul_ps)(poly, m), MM(set1_ps)(-1.6668057665E-1f));
    poly = MM(add_ps)(MM(mul_ps)(poly, m), MM(set1_ps)(2.0000714765E-1f));
    poly = MM(add_ps)(MM(mul_ps)(poly, m), MM(set1_ps)(-2.4999993993E-1f));
    poly = MM(add_ps)(MM(mul_ps)(poly, m), MM(set1_ps)(3.3333331174E-1f));
    poly = MM(mul_ps)(MM(mul_ps)(poly, m), m2);

    poly = MM(sub_ps)(poly, MM(mul_ps)(ef, MM(set1_ps)(2.12194440e-4f)));
    poly = MM(sub_ps)(poly, MM(mul_ps)(m2, MM(set1_ps)(0.5f)));

    auto result = MM(add_ps)(m, poly);
    result = MM(add_ps)(result, MM(mul_ps)(ef, MM(set1_ps)(0.693359375f)));

    return result;
}

inline float fasterLog(const float x) noexcept
{
    alignas(16) float out[4];
    MM(store_ps)(out, fasterLog(MM(set1_ps)(x)));
    return out[0];
}

//==============================================================================//
inline float boundToPi(const float angle)
{
    constexpr float kPi = static_cast<float>(M_PI);
    // fast path: already in canonical range
    if (angle <= kPi && angle >= -kPi)
        return angle;

    // shift from [-π, π] target into [0, 2π) working range
    const float shifted = angle + kPi;

    constexpr float kTwoPi = static_cast<float>(2.0 * M_PI);
    constexpr float invTwoPi = 1.0f / kTwoPi;

    // how many whole turns of 2π fit inside `shifted` (truncated toward zero)
    const int wholeTurns = static_cast<int>(shifted * invTwoPi);

    // remainder after removing those whole turns; lies in (-2π, 2π)
    float wrapped = shifted - kTwoPi * static_cast<float>(wholeTurns);

    // fold any negative remainder up into [0, 2π)
    if (wrapped < 0.0f)
        wrapped += kTwoPi;

    // undo the initial π shift → result in [-π, π]
    return wrapped - kPi;
}

inline M128 boundToPiSIMD(const M128 angle)
{
    constexpr float kPi    = static_cast<float>(M_PI);
    constexpr float kTwoPi = static_cast<float>(2.0 * M_PI);

    // [π, π, π, π]
    const auto vPi = MM(set1_ps)(kPi);

    // [2π, 2π, 2π, 2π]
    const auto vTwoPi = MM(set1_ps)(kTwoPi);
    const auto vInvTwoPi = MM(set1_ps)(1.0f / kTwoPi);
    const auto vZero = MM(setzero_ps)();

    // shift range so we can work in [0, 2π) per lane
    const auto shifted = MM(add_ps)(angle, vPi);

    // trunc(shifted / 2π) per lane, kept as float for the next multiply.
    // float → int32 (truncating) → float round-trip mimics static_cast<int>.
    const auto wholeTurns = MM(cvtepi32_ps)(MM(cvttps_epi32)(MM(mul_ps)(shifted, vInvTwoPi)));

    // remainder after stripping out whole 2π turns; lies in (-2π, 2π)
    auto wrapped = MM(sub_ps)(shifted, MM(mul_ps)(vTwoPi, wholeTurns));

    // branchless "if (wrapped < 0) wrapped += 2π":
    //   cmplt_ps  → mask: all-1 bits where lane is negative, all-0 elsewhere
    //   and_ps    → picks 2π on negative lanes, 0.0 on non-negative lanes
    const auto negFixup = MM(and_ps)(MM(cmplt_ps)(wrapped, vZero), vTwoPi);
    wrapped = MM(add_ps)(wrapped, negFixup);

    // undo the initial π shift → every lane now in [-π, π]
    return MM(sub_ps)(wrapped, vPi);
}
//==============================================================================//
#endif

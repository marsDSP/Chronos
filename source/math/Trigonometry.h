#pragma once

#ifndef CHRONOS_TRIGONOMETRY_H
#define CHRONOS_TRIGONOMETRY_H

#include <cmath>
#include <algorithm>
#include "simd/Config.h"

namespace PadeSinCoeffs {
    constexpr float N0 = -11511339840.0f; // num x⁰ (before outer -x)
    constexpr float N1 = 1640635920.0f; // num x²
    constexpr float N2 = -52785432.0f; // num x⁴
    constexpr float N3 = 479249.0f; // num x⁶
    constexpr float D0 = 11511339840.0f; // den x⁰
    constexpr float D1 = 277920720.0f; // den x²
    constexpr float D2 = 3177720.0f; // den x⁴
    constexpr float D3 = 18361.0f; // den x⁶
}

inline float fastReciprocal(const float d) noexcept {
    const auto r = MM(cvtss_f32)(MM(rcp_ss)(MM(set_ss)(d)));
    return r * (2.0f - d * r);
}

inline M128 fastReciprocal(const M128 d) noexcept {
    const auto r = MM(rcp_ps)(d);
    const auto two = MM(set1_ps)(2.0f);
    return MM(mul_ps)(r, MM(sub_ps)(two, MM(mul_ps)(d, r)));
}

inline float padeSinApprox(const float x) noexcept {
    using namespace PadeSinCoeffs;
    const auto x2 = x * x;
    // horner evaluation inside-out in x²
    const auto num = -x * (N0 + x2 * (N1 + x2 * (N2 + x2 * N3)));
    const auto den = D0 + x2 * (D1 + x2 * (D2 + x2 * D3));
    return num * fastReciprocal(den);
}

inline float pSin(const float x) noexcept { return padeSinApprox(x); }

inline M128 pSin(const M128 x) noexcept {
    using namespace PadeSinCoeffs;
    // broadcast each coeff across 4 lanes
    const auto vN0 = MM(set1_ps)(N0);
    const auto vN1 = MM(set1_ps)(N1);
    const auto vN2 = MM(set1_ps)(N2);
    const auto vN3 = MM(set1_ps)(N3);
    const auto vD0 = MM(set1_ps)(D0);
    const auto vD1 = MM(set1_ps)(D1);
    const auto vD2 = MM(set1_ps)(D2);
    const auto vD3 = MM(set1_ps)(D3);
    const auto neg = MM(set1_ps)(-1.0f);
    const auto x2 = MM(mul_ps)(x, x);
    // numerator:  -x · (N0 + x²·(N1 + x²·(N2 + x²·N3))) | innermost first
    auto numInner = MM(add_ps)(vN2, MM(mul_ps)(x2, vN3)); // N2 + x²·N3
    numInner = MM(add_ps)(vN1, MM(mul_ps)(x2, numInner)); // N1 + x²·(…)
    numInner = MM(add_ps)(vN0, MM(mul_ps)(x2, numInner)); // N0 + x²·(…)
    const auto num = MM(mul_ps)(neg, MM(mul_ps)(x, numInner));
    // denominator: D0 + x²·(D1 + x²·(D2 + x²·D3))
    auto denInner = MM(add_ps)(vD2, MM(mul_ps)(x2, vD3)); // D2 + x²·D3
    denInner = MM(add_ps)(vD1, MM(mul_ps)(x2, denInner)); // D1 + x²·(…)
    const auto den = MM(add_ps)(vD0, MM(mul_ps)(x2, denInner));
    return MM(mul_ps)(num, fastReciprocal(den));
}
#endif

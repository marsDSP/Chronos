#pragma once

#ifndef CHRONOS_DILOGARITHM_H
#define CHRONOS_DILOGARITHM_H

#include <cassert>
#include <cmath>

namespace MarsDSP::Math
{
    // Li2(v) = sum_{k>=1} v^k / k^2, evaluated as a fixed-term Horner series.
    //
    // Precondition: |v| <= 0.5. dilogNeg's Landen fold keeps every call inside
    // this bound, so the series is the only dilogarithm code the saturator runs.
    //
    // A fixed term count (not a convergence test) keeps the loop count
    // data-independent: a future hot path gets a latency-bounded evaluation
    // (a data-dependent loop count is a latency hazard). At |v| = 0.5 the
    // truncation is ~0.5^45 / 45^2 ≈ 1.4e-17, so 50 terms is past the
    // double-precision floor. The reverse accumulation (k = N -> 1) sums the
    // smallest terms first.
    inline double dilogSeries(double v) noexcept
    {
        assert(std::fabs(v) <= 0.5);
        constexpr int kTerms = 50;
        // Horner from the inside out: build sum = 1/1^2 + v*(1/2^2 + v*(1/3^2 + ...))
        // with the innermost (highest-k, smallest) term first, then multiply by v.
        double sum = 1.0 / static_cast<double>(kTerms * kTerms); // 1/N^2, innermost term
        for (int k = kTerms - 1; k >= 1; --k)
        {
            const double invK = 1.0 / static_cast<double>(k);
            sum = invK * invK + v * sum;
        }
        return v * sum; // sum_{k=1}^{N} v^k / k^2
    }

    // Li2(-t) for t in [0, 1], via Landen's transformation so the series
    // argument never leaves [-0.5, 0]:
    //
    //   Li2(z) + Li2(z/(z-1)) = -1/2 * ln^2(1 - z),   with z = -t
    //   => Li2(-t) = -1/2 * ln^2(1 + t) - Li2(t / (1 + t)),
    //      and t / (1 + t) in (0, 1/2] for t in (0, 1].
    //
    // The direct series is used for t <= 1/2 (|-t| <= 1/2); the Landen form
    // for t > 1/2. Both agree at the seam t = 1/2 (see tests/harnesses/cd/
    // dilog_check.cpp). The argument domain is deliberately narrow: TanhNL::F2
    // only ever needs Li2(-exp(-2|x|)), and exp(-2|x|) in (0, 1].
    inline double dilogNeg(double t) noexcept
    {
        assert(t >= 0.0 && t <= 1.0);
        if (t <= 0.5)
            return dilogSeries(-t);
        const double u = t / (1.0 + t);      // (0.5, 1] -> (1/3, 1/2]
        const double lt = std::log1p(t);     // ln(1 + t)
        return -0.5 * lt * lt - dilogSeries(u);
    }
}
#endif

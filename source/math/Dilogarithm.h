#pragma once

#ifndef CHRONOS_DILOGARITHM_H
#define CHRONOS_DILOGARITHM_H

#include <cassert>
#include <cmath>

namespace MarsDSP::Math {

    /// Landen-folded Li2(-t) for t in [0, 1].
    inline double dilogSeries(double v) noexcept {
        assert(std::fabs(v) <= 0.5);
        constexpr int kTerms = 50;
        double sum = 1.0 / static_cast<double>(kTerms * kTerms);
        for (int k = kTerms - 1; k >= 1; --k) {
            const double invK = 1.0 / static_cast<double>(k);
            sum = invK * invK + v * sum;
        }
        return v * sum;
    }

    /// Li2(-t) for t in [0, 1]. Uses the direct series then the Landen partner.
    inline double dilogNeg(double t) noexcept {
        assert(t >= 0.0 && t <= 1.0);
        if (t <= 0.5) return dilogSeries(-t);
        const double u = t / (1.0 + t);
        const double lt = std::log1p(t);
        return -0.5 * lt * lt - dilogSeries(u);
    }
}
#endif

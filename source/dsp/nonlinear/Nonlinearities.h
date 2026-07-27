#pragma once

#ifndef CHRONOS_NONLINEARITIES_H
#define CHRONOS_NONLINEARITIES_H

#include <cmath>
#include "math/Dilogarithm.h"

namespace MarsDSP::Nonlinear
{
    // Constants for the tanh antiderivatives
    constexpr double kLn2 = 0.6931471805599453; // ln(2)
    constexpr double kPiSqOver24 = 0.4112335167120566; // pi^2 / 24

    // sign(x): +1 / -1 / 0. Returning 0 at the origin makes F2(0) == 0 exactly
    // (0 * bracket), so the identity F2(0) = 1/2*(-pi^2/12) + pi^2/24 = 0 holds
    // even though the computed G(0) carries a ~1e-16 residual from dilogNeg(1).
    inline double signx(double x) noexcept
    {
        return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
    }

    // tanh saturator with its first and second antiderivatives.
    //
    //   f(x)  = tanh(x)
    //   F1(x) = log(cosh x) = |x| - ln2 + log1p(exp(-2|x|))   [overflow-safe, even]
    //   F2(x) = sign(x) * [ x^2/2 - |x|*ln2 + G(|x|) ]        [odd]
    //   G(a)  = 1/2 * Li2(-exp(-2a)) + pi^2/24
    //
    // F1 never forms cosh(x) (overflows around |x| ≈ 89 in double); F2's only
    // non-elementary piece is the dilogarithm. G is bounded by
    // pi^2/24 ≈ 0.411 for all x, and |G(8) - pi^2/24| < 6e-8. F1' = f and
    // F2' = F1 by construction; F1 is even, F2 is odd.
    struct TanhNL
    {
        static constexpr const char *name = "tanh";

        static double f(double x) noexcept { return std::tanh(x); }

        static double F1(double x) noexcept
        {
            const double a = std::fabs(x);
            return a - kLn2 + std::log1p(std::exp(-2.0 * a));
        }

        static double F2(double x) noexcept
        {
            const double a = std::fabs(x);
            const double t = std::exp(-2.0 * a); // t in (0, 1], =1 at x=0
            const double g = 0.5 * Math::dilogNeg(t) + kPiSqOver24;
            return signx(x) * (0.5 * a * a - a * kLn2 + g);
        }
    };

    // Algebraic saturator: f(x) = x / sqrt(1 + x^2). All antiderivatives are
    // elementary (no dilogarithm), so it is the escape hatch if the Li2 path
    // misbehaves, and a second curve for the alias harness to compare.
    //   F1(x) = sqrt(1 + x^2)                          [even, F1(0) = 1]
    //   F2(x) = 1/2 ( x*sqrt(1+x^2) + asinh(x) )       [odd,  F2(0) = 0]
    struct AlgebraicNL
    {
        static constexpr const char *name = "algebraic";

        static double f(double x) noexcept
        {
            return x / std::sqrt(1.0 + x * x);
        }

        static double F1(double x) noexcept
        {
            return std::sqrt(1.0 + x * x);
        }

        static double F2(double x) noexcept
        {
            const double s = std::sqrt(1.0 + x * x);
            return 0.5 * (x * s + std::asinh(x));
        }
    };
}
#endif

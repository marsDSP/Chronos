#pragma once

#ifndef CHRONOS_NONLINEARITIES_H
#define CHRONOS_NONLINEARITIES_H

#include <cmath>
#include "math/TanhAntiderivatives.h"

namespace MarsDSP::Nonlinear {
    // kLn2 is part of the public interface (adaa2_check.cpp uses it for errBound).
    constexpr double kLn2 = 0.6931471805599453;

    struct TanhNL
    {
        static constexpr const char *name = "tanh";

        static double f(double x) noexcept { return std::tanh(x); }

        // F1 = ln cosh(x).  Three-region minimax, no dilogarithm.
        // See source/math/TanhAntiderivatives.h.
        static double F1(double x) noexcept { return Math::f1Tanh(x); }

        // F2 = integral_0^x ln cosh(u) du.  F2(0) == 0.0 exactly.
        static double F2(double x) noexcept { return Math::f2Tanh(x); }
    };

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

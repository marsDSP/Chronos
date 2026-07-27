#pragma once

#ifndef CHRONOS_NONLINEARITIES_H
#define CHRONOS_NONLINEARITIES_H

#include <cmath>
#include "math/Dilogarithm.h"

namespace MarsDSP::Nonlinear {
    constexpr double kLn2 = 0.6931471805599453; // ln(2)
    constexpr double kPiSqOver24 = 0.4112335167120566; // pi^2 / 24

    inline double signx(double x) noexcept
    {
        return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
    }

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

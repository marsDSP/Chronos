#pragma once

#ifndef CHRONOS_NONLINEARITIES_H
#define CHRONOS_NONLINEARITIES_H

#include <cmath>
#include "math/TanhAntiderivatives.h"

namespace MarsDSP::Nonlinear {
    constexpr double kLn2 = 0.6931471805599453;
    struct TanhNL
    {
        static constexpr const char *name = "tanh";
        static double f(double x) noexcept { return std::tanh(x); }
        static double F1(double x) noexcept { return Math::f1Tanh(x); }
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

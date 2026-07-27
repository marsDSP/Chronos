#pragma once

#ifndef CHRONOS_ADAA1_H
#define CHRONOS_ADAA1_H

#include <cmath>

namespace MarsDSP::Nonlinear {
    template <typename NL>
    class ADAA1 {
    public:
        void reset() noexcept
        {
            x1_ = 0.0;
            F1x1_ = 0.0;
        }

        double process(double x0) noexcept
        {
            constexpr double kEps = 1e-4;
            const double dx = x0 - x1_;
            double y;
            if (std::fabs(dx) < kEps)
            {
                y = NL::f(0.5 * (x0 + x1_));
            }
            else
            {
                const double F1x0 = NL::F1(x0);
                y = (F1x0 - F1x1_) / dx;
                F1x1_ = F1x0;
            }
            x1_ = x0;
            return y;
        }

    private:
        double x1_{0.0};
        double F1x1_{0.0};
    };
}
#endif

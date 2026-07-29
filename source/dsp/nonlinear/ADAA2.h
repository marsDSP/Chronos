#pragma once

#ifndef CHRONOS_ADAA2_H
#define CHRONOS_ADAA2_H

#include <cmath>

namespace MarsDSP::Nonlinear {
    template <typename NL>
    class ADAA2 {
    public:
        static constexpr double kLatencySamples = 1.0;

        void reset() noexcept
        {
            x1_ = 0.0;
            x2_ = 0.0;
            F2x1_ = 0.0;
            d2_ = 0.0;
        }

        double process(double x0) noexcept
        {
            const double F2x0 = NL::F2(x0);

            const double A = std::fabs(x0 - x1_);
            const double B = std::fabs(x1_ - x2_);
            const double C = std::fabs(x0 - x2_);

            const double d1 = (A < kEpsInner)
                                  ? NL::F1(0.5 * (x0 + x1_))
                                  : (F2x0 - F2x1_) / (x0 - x1_);

            double y;
            if (A < kEpsInner && B < kEpsInner)
            {
                y = NL::f((x0 + x1_ + x2_) / 3.0);
            }
            else if (C < kEpsOuter)
            {
                const double m02 = 0.5 * (x0 + x2_);
                y = 2.0 * (NL::F1(m02) - d1) / (m02 - x1_);
            }
            else
            {
                y = 2.0 * (d1 - d2_) / (x0 - x2_);
            }

            x2_ = x1_;
            x1_ = x0;
            F2x1_ = F2x0;
            d2_ = d1;
            return y;
        }

    private:
        static constexpr double kEpsInner = 1e-4;
        static constexpr double kEpsOuter = 1e-6;

        double x1_{};
        double x2_{};
        double F2x1_{};
        double d2_{};
    };
}
#endif

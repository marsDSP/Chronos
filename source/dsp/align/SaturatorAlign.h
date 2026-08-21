#pragma once

#ifndef CHRONOS_SATURATOR_ALIGN_H
#define CHRONOS_SATURATOR_ALIGN_H

#include "HalfSampleFir.h"
#include "ShortDelay.h"

#include <cassert>
#include <cmath>

namespace MarsDSP::Align {
    class SaturatorAlign {
    public:
        static constexpr int kBudget = kHalfSampleTaps / 2;
        static_assert(kBudget >= 2, "kBudget must be >= 2 so ADAA2's integer delay (kBudget-1) stays >= 1");

        SaturatorAlign() noexcept { reset(); }

        void reset() noexcept {
            dry_.reset();
            dry_.setDelay(kBudget);
            wetInt_.reset();
            wetFir_.reset();
            setMode(mode_);
        }

        void setMode(int adaaOrder) noexcept {
            assert(adaaOrder >= 0 && adaaOrder <= 2);
            mode_ = adaaOrder;
            switch (mode_)
            {
                case 0: wetInt_.setDelay(kBudget);
                    break;
                case 1: wetInt_.setDelay(0);
                    break;
                case 2: wetInt_.setDelay(kBudget - 1);
                    break;
            }
        }

        static float scrub(const float x) noexcept { return std::isfinite(x) ? x : 0.0f; }
        float processDry(const float x) noexcept { return dry_.process(scrub(x)); }

        float processWet(const float x) noexcept {
            const float w = wetInt_.process(scrub(x));
            if (mode_ == 1)
                return wetFir_.process(w);
            return w;
        }

    private:
        ShortDelay<kBudget> dry_;
        ShortDelay<kBudget> wetInt_;
        HalfSampleFir wetFir_;
        int mode_{2};
    };
}
#endif

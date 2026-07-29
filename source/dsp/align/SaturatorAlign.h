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

        // Default-constructed in mode 2 (ADAA2, the plugin default) with
        // the delays configured and rings zeroed — immediately usable.
        SaturatorAlign() noexcept { reset(); }

        void reset() noexcept {
            dry_.reset();
            dry_.setDelay(kBudget); // dry is always the integer delay kBudget
            wetInt_.reset();
            wetFir_.reset();
            setMode(mode_); // re-apply current mode (reconfigures wetInt_)
        }

        void setMode(int adaaOrder) noexcept {
            assert(adaaOrder >= 0 && adaaOrder <= 2);
            mode_ = adaaOrder;
            switch (mode_)
            {
                case 0: wetInt_.setDelay(kBudget);
                    break; // Off:   wet integer kBudget,    FIR unused
                case 1: wetInt_.setDelay(0);
                    break; // ADAA1: wet integer 0,          FIR active
                case 2: wetInt_.setDelay(kBudget - 1);
                    break; // ADAA2: wet integer kBudget-1,  FIR unused
            }
        }

        // NaN/inf hygiene
        static float scrub(float x) noexcept { return std::isfinite(x) ? x : 0.0f; }

        // Dry path: always the integer delay kBudget (invariant I1 — at
        // mix = 0% the plugin is bit-transparent apart from this delay).
        float processDry(float x) noexcept { return dry_.process(scrub(x)); }

        // Wet path: mode-dependent. The FIR runs in mode 1 only.
        float processWet(float x) noexcept {
            const float w = wetInt_.process(scrub(x));
            if (mode_ == 1)
                return wetFir_.process(w);
            return w;
        }

    private:
        ShortDelay<kBudget> dry_; // dry: always integer delay kBudget
        ShortDelay<kBudget> wetInt_; // wet: integer padding, mode-dependent
        HalfSampleFir wetFir_; // wet: half-sample FIR, mode 1 only
        int mode_{2}; // 0=Off, 1=ADAA1, 2=ADAA2 (plugin default)
    };
}
#endif

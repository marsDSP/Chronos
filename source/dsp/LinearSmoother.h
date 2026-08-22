#pragma once

#ifndef CHRONOS_LINEAR_SMOOTHER_H
#define CHRONOS_LINEAR_SMOOTHER_H

#include <cmath>

namespace MarsDSP::Smoothers {
    template<typename T>
    class LinearSmoother {
    public:
        void reset(double sampleRate, double rampLengthSeconds) noexcept
        {
            const double fs = sampleRate > 0.0 ? sampleRate : 48000.0;
            const double dur = rampLengthSeconds > 0.0 ? rampLengthSeconds : 0.0;
            stepsToTarget_ = static_cast<int>(std::floor(dur * fs + 0.5));
        }

        void setCurrentAndTargetValue(T value) noexcept
        {
            current_ = value;
            target_ = value;
            step_ = T(0);
            countdown_ = 0;
        }

        void setTargetValue(T value) noexcept
        {
            target_ = value;
            if (stepsToTarget_ <= 0 || target_ == current_)
            {
                current_ = target_;
                step_ = T(0);
                countdown_ = 0;
                return;
            }
            step_ = (target_ - current_) / static_cast<T>(stepsToTarget_);
            countdown_ = stepsToTarget_;
        }

        T getNextValue() noexcept
        {
            if (countdown_ <= 0)
                return target_;
            --countdown_;
            current_ += step_;
            if (countdown_ <= 0)
                current_ = target_;
            return current_;
        }

        /// Advance the ramp by one sample and discard the value.
        void skip() noexcept { static_cast<void>(getNextValue()); }

        [[nodiscard]] T getCurrentValue() const noexcept { return current_; }
        [[nodiscard]] T getTargetValue() const noexcept { return target_; }
        [[nodiscard]] bool isSmoothing() const noexcept { return countdown_ > 0; }

    private:
        T current_{};
        T target_{};
        T step_{};
        int stepsToTarget_{0};
        int countdown_{0};
    };
}
#endif

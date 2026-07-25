#pragma once

#ifndef CHRONOS_ONE_POLE_SMOOTHER_H
#define CHRONOS_ONE_POLE_SMOOTHER_H

#include <cmath>

namespace MarsDSP::Smoothers {
    template <typename T>
    class OnePoleSmoother {
    public:
        void reset(const double sampleRate, const double durationSeconds) noexcept
        {
            const double fs  = sampleRate > 0.0 ? sampleRate : 48000.0;
            const double tau = durationSeconds > 0.0 ? durationSeconds : 0.02;
            alpha = static_cast<T>(1.0 - std::exp(-1.0 / (tau * fs)));
        }

        void setCurrentAndTargetValue(const T value) noexcept
        {
            target  = value;
            current = value;
        }

        void setTargetValue(const T value) noexcept
        {
            target = value;
        }

        T getNextValue() noexcept
        {
            current += alpha * (target - current);
            return current;
        }

    private:
        T alpha {};
        T current {};
        T target {};
    };
}
#endif

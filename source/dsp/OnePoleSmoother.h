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
            cachedN_ = 0;
            cachedDecay_ = T(0);
        }

        void reset(const double sampleRate, const double durationSeconds, int subBlockSize) noexcept
        {
            reset(sampleRate, durationSeconds);
            setSubBlockSize(subBlockSize);
        }

        void setSubBlockSize(int subBlockSize) noexcept
        {
            if (subBlockSize <= 0)
            {
                cachedN_ = 0;
                return;
            }
            cachedN_ = subBlockSize;
            const T lpinv = T(1) - alpha;
            cachedDecay_ = std::pow(lpinv, static_cast<T>(subBlockSize));
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

        void processN(int n) noexcept
        {
            if (n <= 0) return;
            const T decay = (n == cachedN_) ? cachedDecay_ : std::pow(T(1) - alpha, static_cast<T>(n));
            current = target + (current - target) * decay;
        }

        [[nodiscard]] T getCurrentValue() const noexcept { return current; }

    private:
        T alpha {};
        T current {};
        T target {};
        int cachedN_ = 0;
        T cachedDecay_ {};
    };
}
#endif

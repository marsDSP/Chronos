#pragma once

#ifndef CHRONOS_SHORT_DELAY_H
#define CHRONOS_SHORT_DELAY_H

#include <array>
#include <cassert>

namespace MarsDSP::Align {
    template <int MaxDelay>
    class ShortDelay {
    public:
        static constexpr int kCapacity = MaxDelay + 1;

        void reset() noexcept
        {
            z_.fill(0.0f);
            w_ = 0;
            d_ = 0;
        }

        void setDelay(int d) noexcept
        {
            assert(d >= 0 && d <= MaxDelay);
            d_ = d;
        }

        float process(float x) noexcept
        {
            if (d_ == 0)
                return x;   // bit-exact passthrough — no ring round-trip

            const float y = z_[(w_ - d_ + kCapacity) % kCapacity];
            z_[w_] = x;
            w_ = (w_ + 1) % kCapacity;
            return y;
        }

    private:
        std::array<float, kCapacity> z_ {};
        int w_{0};
        int d_{0};
    };
}
#endif

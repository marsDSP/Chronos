#pragma once

#ifndef CHRONOS_SHORT_DELAY_H
#define CHRONOS_SHORT_DELAY_H

#include <array>
#include <bit>
#include <cassert>

namespace MarsDSP::Align {
    template <int MaxDelay>
    class ShortDelay {
    public:
        static constexpr int kCapacity = static_cast<int>(std::bit_ceil(static_cast<unsigned>(MaxDelay + 1)));
        static constexpr int kMask = kCapacity - 1;

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

        float process(const float x) noexcept
        {
            if (d_ == 0) return x;

            const float y = z_[static_cast<std::size_t>((w_ - d_ + kCapacity) & kMask)];
            z_[static_cast<std::size_t>(w_)] = x;
            w_ = (w_ + 1) & kMask;
            return y;
        }

    private:
        std::array<float, kCapacity> z_ {};
        int w_{0};
        int d_{0};
    };
}
#endif

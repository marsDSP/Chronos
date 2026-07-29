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
        // Power-of-two capacity so the ring index reduces to a mask AND
        // instead of an integer modulo (~20-40 cycles on x86). With
        // MaxDelay = kBudget = 8 the old capacity was 9; bit_ceil(9) = 16.
        // Bit-exact against the old MaxDelay+1 ring — the d_ == 0 branch
        // freezes z_ and w_, so z_[(w_ - k) & mask] is "the sample written
        // k writes ago" for all k <= cap-1, and d_ <= MaxDelay <= cap-1
        // under both capacities (proof obligation §1.4(a), verified by
        // short_delay_check's power-of-two parity section).
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

        float process(float x) noexcept
        {
            if (d_ == 0)
                return x;   // bit-exact passthrough — no ring round-trip

            const float y = z_[(w_ - d_ + kCapacity) & kMask];
            z_[w_] = x;
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

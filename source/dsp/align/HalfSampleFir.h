#pragma once

#ifndef CHRONOS_HALF_SAMPLE_FIR_H
#define CHRONOS_HALF_SAMPLE_FIR_H

#include <array>
#include <cstddef>

namespace MarsDSP::Align {
    inline constexpr int kHalfSampleTaps = 16; // must be even, >= 4, power of two
    static_assert(kHalfSampleTaps % 2 == 0, "half-sample FIR tap count must be even");
    static_assert(kHalfSampleTaps >= 4, "half-sample FIR needs at least 4 taps");
    static_assert((kHalfSampleTaps & (kHalfSampleTaps - 1)) == 0,
                  "half-sample FIR tap count must be a power of two for the ring mask");

    inline constexpr std::array kHalfSampleCoeffs = {
        -0.00530111231f,
         0.0121372724f,
        -0.02284934f,
         0.0391499847f,
        -0.0644291341f,
         0.10737168f,
        -0.200103939f,
         0.63402456f,
         0.63402456f,
        -0.200103939f,
         0.10737168f,
        -0.0644291341f,
         0.0391499847f,
        -0.02284934f,
         0.0121372724f,
        -0.00530111231f
    };

    class HalfSampleFir {
    public:
        static constexpr double kBulkDelay = kHalfSampleTaps / 2.0 - 0.5;
        static constexpr int kMask = kHalfSampleTaps - 1;

        void reset() noexcept { z_.fill(0.0f); w_ = 0; }
        float process(float x) noexcept {
            // Circular buffer: write newest, advance write index. No memmove.
            // After the write, the newest sample is at (w_-1)&mask and the
            // oldest at (w_+0)&mask = w_ (the next overwrite slot). The old
            // memmove layout had z_[j] = (j+1)-th newest and z_[N-1-j] =
            // (j+1)-th oldest; the circular mapping is
            //   z_[j]      -> z_[(w_ - 1 - j) & mask]   (newer side)
            //   z_[N-1-j]  -> z_[(w_     + j) & mask]   (older side)
            // The folded accumulation order is preserved exactly: j=0 (newest
            // + oldest) first through j=N/2-1 (two middle) last, and within
            // each pair (newer + older) — so FMA reassociation cannot drift
            // the filter off linear phase (reordering it is V3, not here).
            z_[w_] = x;
            w_ = (w_ + 1) & kMask;

            float acc = 0.0f;
            for (int j = 0; j < kHalfSampleTaps / 2; ++j)
                acc += kHalfSampleCoeffs[static_cast<std::size_t>(j)]
                     * (z_[(w_ - 1 - j + kHalfSampleTaps) & kMask]
                        + z_[(w_ + j) & kMask]);
            return acc;
        }

    private:
        std::array<float, kHalfSampleTaps> z_{};
        int w_{0};
    };
}
#endif

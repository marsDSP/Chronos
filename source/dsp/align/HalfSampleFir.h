#pragma once

#ifndef CHRONOS_HALF_SAMPLE_FIR_H
#define CHRONOS_HALF_SAMPLE_FIR_H

#include <array>
#include <cstddef>
#include "simd/Config.h"

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
            z_[w_] = x;
            w_ = (w_ + 1) & kMask;

            alignas(16) float pairs[8];
            for (int j = 0; j < 8; ++j)
                pairs[j] = z_[(w_ - 1 - j + kHalfSampleTaps) & kMask]
                         + z_[(w_ + j) & kMask];

            const M128 vPairs0 = MM(loadu_ps)(pairs);
            const M128 vCoeff0 = MM(loadu_ps)(kHalfSampleCoeffs.data());
            M128 vAcc = FMADD(vCoeff0, vPairs0, MM(setzero_ps)());

            const M128 vPairs1 = MM(loadu_ps)(pairs + 4);
            const M128 vCoeff1 = MM(loadu_ps)(kHalfSampleCoeffs.data() + 4);
            vAcc = FMADD(vCoeff1, vPairs1, vAcc);

            const M128 vSwap = MM(shuffle_ps)(vAcc, vAcc, 0x4E); // swap halves
            const M128 vSum0 = MM(add_ps)(vAcc, vSwap);           // [a0+a2, a1+a3, ...]
            const M128 vSwap2 = MM(shuffle_ps)(vSum0, vSum0, 0xB1); // swap pairs
            const M128 vSum1 = MM(add_ps)(vSum0, vSwap2);

            return MM(cvtss_f32)(vSum1);
        }

    private:
        std::array<float, kHalfSampleTaps> z_{};
        int w_{0};
    };
}
#endif

#pragma once

#ifndef CHRONOS_HALF_SAMPLE_FIR_H
#define CHRONOS_HALF_SAMPLE_FIR_H

#include <array>
#include <cstddef>
#include <cstring>

namespace MarsDSP::Align {
    inline constexpr int kHalfSampleTaps = 16; // must be even, >= 4
    static_assert(kHalfSampleTaps % 2 == 0, "half-sample FIR tap count must be even");
    static_assert(kHalfSampleTaps >= 4, "half-sample FIR needs at least 4 taps");

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
        void reset() noexcept { z_.fill(0.0f); }
        float process(float x) noexcept {
            std::memmove(z_.data() + 1, z_.data(), static_cast<std::size_t>(kHalfSampleTaps - 1) * sizeof(float));
            z_.front() = x;

            float acc = 0.0f;
            for (int j = 0; j < kHalfSampleTaps / 2; ++j)
                acc += kHalfSampleCoeffs[static_cast<std::size_t>(j)]
                      * (z_[static_cast<std::size_t>(j)]
                         + z_[static_cast<std::size_t>(kHalfSampleTaps - 1 - j)]);
            return acc;
        }

    private:
        std::array<float, kHalfSampleTaps> z_{};
    };
}
#endif

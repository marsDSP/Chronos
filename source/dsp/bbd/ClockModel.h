#pragma once

#ifndef CHRONOS_BBD_CLOCK_MODEL_H
#define CHRONOS_BBD_CLOCK_MODEL_H

#include "BrigadeLine.h"
#include <algorithm>

namespace MarsDSP::BBD
{
    /**
     *  Maps the effective loop delay to the bucket-brigade clock frequency.
     *  The half-sample term accounts for the zero-order-hold midpoint of the
     *  bucket stage, so clockFor and achievedDelaySamples are exact inverses.
     */
    class ClockModel
    {
    public:
        [[nodiscard]] static constexpr float minClockHz(double sampleRate) noexcept
        {
            return static_cast<float>(sampleRate / 30.0);
        }

        [[nodiscard]] static constexpr float maxClockHz(double sampleRate) noexcept
        {
            return static_cast<float>(100.0 * sampleRate);
        }

        // The formula lives in BrigadeLine::clockForDelay so the line
        // and the loop share one clock authority.
        [[nodiscard]] static float clockFor(float dEffSamples, double sampleRate) noexcept
        {
            return BrigadeLine::clockForDelay(dEffSamples, sampleRate);
        }

        [[nodiscard]] static float achievedDelaySamples(float clockHz, double sampleRate) noexcept
        {
            if (clockHz <= 0.0f)
                return 0.0f;

            const auto kStages = static_cast<double>(BrigadeLine::kStages);
            return static_cast<float>((2.0 * kStages + 0.5) * sampleRate / static_cast<double>(clockHz));
        }

        [[nodiscard]] static float clampRemainderSamples(float dEffSamples, double sampleRate) noexcept
        {
            const float clk = clockFor(dEffSamples, sampleRate);
            const float achieved = achievedDelaySamples(clk, sampleRate);
            return dEffSamples - achieved;
        }
    };
}
#endif

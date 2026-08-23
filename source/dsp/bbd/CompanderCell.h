#pragma once

#ifndef CHRONOS_BBD_COMPANDER_CELL_H
#define CHRONOS_BBD_COMPANDER_CELL_H

#include <algorithm>
#include <cmath>

namespace MarsDSP::BBD
{
    // The envelope follower for the compander cell.
    // Full-wave rectify and asymmetric one-pole smoothing.
    struct EnvelopeFollower
    {
        static constexpr float kAttackMs = 3.0f;
        static constexpr float kReleaseMs = 13.3f;
        static constexpr float kRefLevel = 0.1f;
        static constexpr float kEnvFloor = 1e-9f;

        float aA = 0.0f;
        float aR = 0.0f;
        float env = kEnvFloor;
        bool frozen = false;

        void prepare (double sampleRate) noexcept
        {
            aA = std::exp (-1.0f / (kAttackMs * 0.001f * static_cast<float> (sampleRate)));
            aR = std::exp (-1.0f / (kReleaseMs * 0.001f * static_cast<float> (sampleRate)));
            reset();
        }

        void reset() noexcept
        {
            env = kEnvFloor;
        }

        void setFreeze (bool freeze) noexcept
        {
            frozen = freeze;
        }

        float process (float x) noexcept
        {
            if (frozen)
                return kRefLevel;

            float e = std::fabs (x);
            if (!std::isfinite (e))
            {
                env = kEnvFloor;
                return env;
            }

            const float a = (e > env) ? aA : aR;
            env = env + (1.0f - a) * (e - env);
            if (!std::isfinite (env) || env < kEnvFloor)
                env = kEnvFloor;

            return env;
        }
    };

    // 2:1 compressor above the reference level. NE570 feedback
    // topology: the rectifier senses the compressor output. The gain
    // uses the previous envelope, one sample behind. This matches the
    // expander rectifier, which senses the expander input, so the
    // comp-to-exp cascade is exact at every sample.
    class CompressorCell
    {
    public:
        static constexpr float kRefLevel = EnvelopeFollower::kRefLevel;
        static constexpr float kMinEnv = 1e-3f;

        void prepare (double sampleRate) noexcept
        {
            follower_.prepare (sampleRate);
            follower_.env = kRefLevel;
        }

        void reset() noexcept
        {
            follower_.reset();
            follower_.env = kRefLevel;
        }

        void setEnvelopeFreeze (bool freeze) noexcept
        {
            follower_.setFreeze (freeze);
        }

        [[nodiscard]] float getEnvelope() const noexcept
        {
            return follower_.env;
        }

        [[nodiscard]] float processSample (float x) noexcept
        {
            const float envPrev = std::max (follower_.env, kMinEnv);
            float y = x * kRefLevel / envPrev;
            if (!std::isfinite (y))
                y = 0.0f;
            follower_.process (std::fabs (y));
            return y;
        }

    private:
        EnvelopeFollower follower_{};
    };

    // 1:2 expander. NE570 law: y = avg(|x|) * x / VR. The gain is
    // avg(|x|)/VR at every level. Below the reference the gain is
    // below one, above it is above one. This pairs with the feedback
    // compressor for an exact comp-to-exp cascade at every level.
    class ExpanderCell
    {
    public:
        static constexpr float kRefLevel = EnvelopeFollower::kRefLevel;
        static constexpr float kMinEnv = 1e-6f;

        void prepare (double sampleRate) noexcept
        {
            follower_.prepare (sampleRate);
            follower_.env = kRefLevel;
            invRef_ = 1.0f / kRefLevel;
        }

        void reset() noexcept
        {
            follower_.reset();
            follower_.env = kRefLevel;
        }

        void setEnvelopeFreeze (bool freeze) noexcept
        {
            follower_.setFreeze (freeze);
        }

        [[nodiscard]] float getEnvelope() const noexcept
        {
            return follower_.env;
        }

        [[nodiscard]] float processSample (float x) noexcept
        {
            const float env = std::max (follower_.process (x), kMinEnv);
            float y = x * env * invRef_;
            if (!std::isfinite (y))
                y = 0.0f;
            return y;
        }

    private:
        EnvelopeFollower follower_{};
        float invRef_ = 10.0f;
    };
} // namespace MarsDSP::BBD

#endif

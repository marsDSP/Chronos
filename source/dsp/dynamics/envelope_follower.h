#pragma once

#ifndef CHRONOS_ENVELOPE_FOLLOWER_H
#define CHRONOS_ENVELOPE_FOLLOWER_H

// ============================================================================
//  envelope_follower.h
// ----------------------------------------------------------------------------
//  Asymmetric one-pole peak envelope follower used as the sidechain
//  detector for the diode-bridge ducker. State is per-channel so a stereo
//  sidechain can drive two independent envelopes, or the host can sum
//  L+R into a single mono follower upstream and use one channel.
//
//  Detector law:
//
//      a = |x[n]|
//      env[n] = env[n-1] + (a > env[n-1] ? alphaAtk : alphaRel) * (a - env[n-1])
//
//  with
//
//      alphaAtk = 1 - exp(-1 / (atk_seconds * fs))
//      alphaRel = 1 - exp(-1 / (rel_seconds * fs))
//
//  Block-rate API matches the project's wow_engine.h / flutter_engine.h
//  pattern: prepare() once, prepareBlock() per host block to latch
//  attack / release time targets, then the audio path consumes per-sample
//  or per-quad results. Per-quad output is a SIMD_M128 of envelope
//  values (state is sequential within the quad - this is unavoidable
//  because the next env value depends on the previous one).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <vector>
#include "dsp/math/simd/simd_config.h"

namespace MarsDSP::DSP::Dynamics
{
    template <typename SampleType = float>
    class EnvelopeFollower
    {
    public:
        EnvelopeFollower() = default;

        void prepare (double sampleRateHz, int maxBlockSize, int numChannels)
        {
            (void) maxBlockSize;

            sampleRate = static_cast<float>(sampleRateHz);
            envState.assign (static_cast<size_t>(std::max (numChannels, 1)),
                             SampleType{0});
            blockEndEnv.assign (envState.size(), SampleType{0});

            // Default to a sensible 1 ms / 50 ms attack/release so first
            // prepareBlock isn't required for the follower to behave.
            updateCoefficients (1.0f, 50.0f);
        }

        // attackMs / releaseMs in milliseconds; both clamped to a small
        // positive minimum so a zero-time entry from the host param
        // does not produce 1.0 alphas (which would just track the
        // instantaneous |x| with no smoothing).
        void prepareBlock (float attackMs, float releaseMs) noexcept
        {
            updateCoefficients (std::max (0.05f, attackMs),
                                std::max (0.05f, releaseMs));
        }

        // ---------------- per-sample ----------------
        inline SampleType processSample (SampleType x, size_t channelIndex) noexcept
        {
            const SampleType a = std::abs (x);
            const SampleType y = envState[channelIndex];
            const SampleType alpha = (a > y)
                                   ? static_cast<SampleType>(alphaAtk)
                                   : static_cast<SampleType>(alphaRel);
            const SampleType y_next = y + alpha * (a - y);
            envState[channelIndex] = y_next;
            return y_next;
        }

        // ---------------- per-quad (SIMD storage out, scalar inside) ----------------
        // Processes 4 consecutive samples for `channelIndex` and returns a
        // SIMD_M128 packed with the four resulting envelope values, in the
        // same lane order as the input. The carry across the quad is the
        // follower's own per-channel state, so calling processQuad twice
        // in succession is bit-identical to calling processSample 8 times.
        inline SIMD_M128 processQuad (SIMD_M128 x, size_t channelIndex) noexcept
        {
            alignas(16) SampleType in[4];
            alignas(16) SampleType out[4];
            SIMD_MM(store_ps)(reinterpret_cast<float*>(in), x);

            for (int j = 0; j < 4; ++j)
                out[j] = processSample (in[j], channelIndex);

            return SIMD_MM(load_ps)(reinterpret_cast<const float*>(out));
        }

        [[nodiscard]] SampleType getEnvelope (size_t channelIndex) const noexcept
        {
            return envState[channelIndex];
        }

        [[nodiscard]] SampleType getBlockEndEnv (size_t channelIndex) const noexcept
        {
            return blockEndEnv[channelIndex];
        }

        // Snapshot the current per-channel envelope into blockEndEnv. The
        // engine usually calls this at the end of process() so the next
        // block's gain interpolation can start from the previous block's
        // last value.
        void latchBlockEnd() noexcept
        {
            std::copy (envState.begin(), envState.end(), blockEndEnv.begin());
        }

        void reset() noexcept
        {
            std::fill (envState   .begin(), envState   .end(), SampleType{0});
            std::fill (blockEndEnv.begin(), blockEndEnv.end(), SampleType{0});
        }

        [[nodiscard]] float getAlphaAtk() const noexcept { return alphaAtk; }
        [[nodiscard]] float getAlphaRel() const noexcept { return alphaRel; }

    private:
        void updateCoefficients (float attackMs, float releaseMs) noexcept
        {
            const float fs = sampleRate;
            // 1 - exp(-T_sample / tau). Branch through expm1 for the
            // small-tau end where exp(-x) is ≈ 1 - x in float and we'd
            // lose precision otherwise.
            alphaAtk = -std::expm1 (-1.0f / (attackMs  * 1.0e-3f * fs));
            alphaRel = -std::expm1 (-1.0f / (releaseMs * 1.0e-3f * fs));
        }

        float sampleRate { 48000.0f };
        float alphaAtk   { 0.0f };
        float alphaRel   { 0.0f };

        std::vector<SampleType> envState;
        std::vector<SampleType> blockEndEnv;

        // Non-copyable: holds per-channel state vectors that should not be
        // silently aliased between instances.
        EnvelopeFollower (const EnvelopeFollower&)            = delete;
        EnvelopeFollower& operator= (const EnvelopeFollower&) = delete;
    };
}
#endif

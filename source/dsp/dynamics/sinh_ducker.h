#pragma once

#ifndef CHRONOS_SINH_DUCKER_H
#define CHRONOS_SINH_DUCKER_H

// ============================================================================
//  sinh_ducker.h
// ----------------------------------------------------------------------------
//  Sidechain ducker for the delay taps. Wires:
//
//      dryL,dryR  ─┬── ½(L+R) ──► EnvelopeFollower ──► env
//                  │                                    │
//                  │                                    ▼
//                  │                          (env − threshold) × ratio
//                  │                                    │
//                  │                                    ▼ drive ∈ [0, π]
//                  │                                    │
//                  ▼                                    ▼
//      wetL ──────────────────────► SinhDiodeBridge[0] ──► wetL'
//      wetR ──────────────────────► SinhDiodeBridge[1] ──► wetR'
//
//  The envelope follower runs on the (mono) sum of the dry inputs so the
//  sidechain trigger is independent of stereo image, then a single drive
//  signal is broadcast to both per-channel bridges. Block-rate parameter
//  latching (threshold, amount, attack/release, saturation, sinh shape)
//  matches wow_engine / flutter_engine - call prepareBlock once per host
//  block, then run the per-quad / per-block process functions on the
//  audio path.
//
//  Why sinh and not just (env-thr)*ratio*gain? See the notes in
//  sinh_diode_bridge.h: a sinh-mapped bias yields a near-linear response
//  for low envelope, then exponential bite once the threshold is crossed,
//  matching the way an analog diode-bridge attenuator behaves under a
//  rising control current. The bridge in turn applies vintage-style
//  even+odd harmonic colour to the ducked tail via an ADAA tanh blend.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include "dsp/math/fastermath.h"
#include "dsp/math/simd/simd_config.h"
#include "dsp/dynamics/envelope_follower.h"
#include "dsp/dynamics/sinh_diode_bridge.h"

namespace MarsDSP::DSP::Dynamics
{
    template <typename SampleType = float>
    class SinhDucker
    {
    public:
        SinhDucker() = default;

        // ------------------------------------------------------------------
        //  Lifecycle.  numChannels here refers to the *audio* path (stereo
        //  delay taps -> 2). The sidechain follower is always summed to a
        //  single mono channel internally.
        // ------------------------------------------------------------------
        void prepare (double sampleRateHz, int maxBlockSize, int numChannels)
        {
            sampleRate    = static_cast<float>(sampleRateHz);
            audioChannels = std::max (numChannels, 1);

            envFollower.prepare (sampleRateHz, maxBlockSize, /* numChannels */ 1);
            bridge     .prepare (sampleRateHz, maxBlockSize, audioChannels);

            blockEndDrive = 0.0f;
        }

        // ------------------------------------------------------------------
        //  Per-block parameter latch. All inputs are normalised in [0, 1]
        //  except the time constants which are in milliseconds.
        //
        //   thresholdNormalised : envelope value above which ducking starts
        //                         (pre-sinh). Below this, drive = 0 and the
        //                         bridge passes audio at unity gain.
        //   amountNormalised    : 0 = bridge stays open even on full env,
        //                         1 = full ducking range up to drive = π.
        //   attackMs/releaseMs  : envelope-follower time constants.
        //   saturationAmount    : 0..1, how much vintage colour to mix in
        //                         when the bridge is biting.
        //   seriesScale         : analog "R_series / R_shunt0" knob, larger
        //                         = harder ducking for the same envelope.
        // ------------------------------------------------------------------
        void prepareBlock (float thresholdNormalised,
                           float amountNormalised,
                           float attackMs,
                           float releaseMs,
                           float saturationAmount = 0.5f,
                           float seriesScale       = 1.0f) noexcept
        {
            threshold = std::clamp (thresholdNormalised, 0.0f, 0.999f);
            amount    = std::clamp (amountNormalised,    0.0f, 1.0f);

            // 1 / (1 - threshold) so a signal at unit envelope drives the
            // mapping to its full sinh range, regardless of where the user
            // parked the threshold knob.
            const float invSpan = 1.0f / std::max (1.0e-3f, 1.0f - threshold);
            // Cap the post-threshold drive at π so we stay inside the
            // design domain of fasterSinh.
            mapScale = invSpan * std::numbers::pi_v<float> * amount;

            envFollower.prepareBlock (attackMs, releaseMs);
            bridge     .setSeriesScale       (seriesScale);
            bridge     .setSaturationAmount  (saturationAmount);
        }

        // ------------------------------------------------------------------
        //  Stereo process. The host passes raw pointers to two interleaved-
        //  apart channels of dry input + writable wet buffers; the wet
        //  buffers are ducked in place. numSamples can be any non-negative
        //  count - any quad-aligned head is processed via SIMD, the tail
        //  is finished with the scalar fallback so the caller does not
        //  have to pad to a multiple of 4.
        // ------------------------------------------------------------------
        void processBlockStereo (const SampleType* __restrict dryL,
                                 const SampleType* __restrict dryR,
                                 SampleType* __restrict wetL,
                                 SampleType* __restrict wetR,
                                 int numSamples) noexcept
        {
            assert (audioChannels >= 2);
            if (numSamples <= 0) return;

            const auto vHalf      = SIMD_MM(set1_ps)(0.5f);
            const auto vThreshold = SIMD_MM(set1_ps)(threshold);
            const auto vMapScale  = SIMD_MM(set1_ps)(mapScale);
            const auto vZero      = SIMD_MM(setzero_ps)();
            const auto vDriveMax  = SIMD_MM(set1_ps)(std::numbers::pi_v<float>);

            int i = 0;
            const int quadEnd = numSamples & ~3;

            for (; i < quadEnd; i += 4)
            {
                const auto vDryL = SIMD_MM(loadu_ps)(reinterpret_cast<const float*>(dryL + i));
                const auto vDryR = SIMD_MM(loadu_ps)(reinterpret_cast<const float*>(dryR + i));
                const auto vMono = SIMD_MM(mul_ps)(vHalf, SIMD_MM(add_ps)(vDryL, vDryR));

                // Envelope follower runs in the mono channel slot 0.
                const auto vEnv = envFollower.processQuad (vMono, /*channel*/ 0);

                // drive = max(0, env - threshold) * mapScale, clamped to π.
                auto vDrive = SIMD_MM(sub_ps)(vEnv, vThreshold);
                vDrive      = SIMD_MM(max_ps)(vDrive, vZero);
                vDrive      = SIMD_MM(mul_ps)(vDrive, vMapScale);
                vDrive      = SIMD_MM(min_ps)(vDrive, vDriveMax);

                // Apply the bridge to each audio channel, broadcasting the
                // shared drive signal so L/R are ducked in lockstep.
                const auto vWetL = SIMD_MM(loadu_ps)(reinterpret_cast<const float*>(wetL + i));
                const auto vWetR = SIMD_MM(loadu_ps)(reinterpret_cast<const float*>(wetR + i));
                const auto vOutL = bridge.processQuad (vWetL, vDrive, 0);
                const auto vOutR = bridge.processQuad (vWetR, vDrive, 1);

                SIMD_MM(storeu_ps)(reinterpret_cast<float*>(wetL + i), vOutL);
                SIMD_MM(storeu_ps)(reinterpret_cast<float*>(wetR + i), vOutR);

                // Latch the lane-3 drive for visualisation / smoothing.
                alignas(16) float lanes[4];
                SIMD_MM(store_ps)(lanes, vDrive);
                blockEndDrive = lanes[3];
            }

            // Scalar tail (0..3 samples) for non-quad-aligned blocks.
            for (; i < numSamples; ++i)
            {
                const SampleType mono = SampleType{0.5} * (dryL[i] + dryR[i]);
                const SampleType env  = envFollower.processSample (mono, /*channel*/ 0);
                const SampleType raw  = std::max (SampleType{0}, env - static_cast<SampleType>(threshold));
                SampleType drive      = raw * static_cast<SampleType>(mapScale);
                drive = std::clamp (drive,
                                    SampleType{0},
                                    static_cast<SampleType>(std::numbers::pi_v<float>));

                wetL[i] = bridge.processSample (wetL[i], drive, 0);
                wetR[i] = bridge.processSample (wetR[i], drive, 1);
                blockEndDrive = static_cast<float>(drive);
            }

            envFollower.latchBlockEnd();
        }

        // ------------------------------------------------------------------
        //  Mono process variant: useful for harnesses and any future single-
        //  channel call site (e.g. one tap of a multi-tap delay).
        // ------------------------------------------------------------------
        void processBlockMono (const SampleType* __restrict dry,
                               SampleType* __restrict wet,
                               int numSamples) noexcept
        {
            if (numSamples <= 0) return;

            const auto vThreshold = SIMD_MM(set1_ps)(threshold);
            const auto vMapScale  = SIMD_MM(set1_ps)(mapScale);
            const auto vZero      = SIMD_MM(setzero_ps)();
            const auto vDriveMax  = SIMD_MM(set1_ps)(std::numbers::pi_v<float>);

            int i = 0;
            const int quadEnd = numSamples & ~3;
            for (; i < quadEnd; i += 4)
            {
                const auto vDry = SIMD_MM(loadu_ps)(reinterpret_cast<const float*>(dry + i));
                const auto vEnv = envFollower.processQuad (vDry, 0);

                auto vDrive = SIMD_MM(sub_ps)(vEnv, vThreshold);
                vDrive      = SIMD_MM(max_ps)(vDrive, vZero);
                vDrive      = SIMD_MM(mul_ps)(vDrive, vMapScale);
                vDrive      = SIMD_MM(min_ps)(vDrive, vDriveMax);

                const auto vWet = SIMD_MM(loadu_ps)(reinterpret_cast<const float*>(wet + i));
                const auto vOut = bridge.processQuad (vWet, vDrive, 0);
                SIMD_MM(storeu_ps)(reinterpret_cast<float*>(wet + i), vOut);

                alignas(16) float lanes[4];
                SIMD_MM(store_ps)(lanes, vDrive);
                blockEndDrive = lanes[3];
            }
            for (; i < numSamples; ++i)
            {
                const SampleType env = envFollower.processSample (dry[i], 0);
                const SampleType raw = std::max (SampleType{0}, env - static_cast<SampleType>(threshold));
                SampleType drive     = raw * static_cast<SampleType>(mapScale);
                drive = std::clamp (drive,
                                    SampleType{0},
                                    static_cast<SampleType>(std::numbers::pi_v<float>));
                wet[i] = bridge.processSample (wet[i], drive, 0);
                blockEndDrive = static_cast<float>(drive);
            }
            envFollower.latchBlockEnd();
        }

        // ------------------------------------------------------------------
        //  Diagnostics / engine introspection.
        // ------------------------------------------------------------------
        [[nodiscard]] float getBlockEndDrive() const noexcept   { return blockEndDrive; }
        [[nodiscard]] float getBlockEndEnvelope() const noexcept
        {
            return static_cast<float>(envFollower.getBlockEndEnv (0));
        }
        [[nodiscard]] SampleType getBlockEndGain (size_t channelIndex) const noexcept
        {
            return bridge.getBlockEndGain (channelIndex);
        }

        void reset() noexcept
        {
            envFollower.reset();
            bridge     .reset();
            blockEndDrive = 0.0f;
        }

    private:
        float sampleRate    { 48000.0f };
        int   audioChannels { 2 };

        // Block-rate latched parameters (set by prepareBlock).
        float threshold { 0.05f };
        float amount    { 0.5f };
        float mapScale  { 0.0f }; // (π · amount) / (1 - threshold)

        // Telemetry: the lane-3 drive of the most recent quad.
        float blockEndDrive { 0.0f };

        EnvelopeFollower<SampleType> envFollower;
        SinhDiodeBridge <SampleType> bridge;

        SinhDucker (const SinhDucker&)            = delete;
        SinhDucker& operator= (const SinhDucker&) = delete;
    };
}
#endif

 #pragma once

#ifndef CHRONOS_FLUTTER_ENGINE_H
#define CHRONOS_FLUTTER_ENGINE_H

// ============================================================================
//  flutter_engine.h
// ----------------------------------------------------------------------------
//  Sum of three cosines at f, 2f, 3f with fixed phase offsets approximates
//  the fast tape-capstan wobble musicians describe as flutter. A constant
//  DC offset accompanies the AC component so callers can add it to their
//  delay read position and keep the average delay consistent with the
//  user's rate setting.
// ============================================================================

#include <JuceHeader.h>
#include <cmath>
#include <vector>

namespace MarsDSP::DSP::Modulation
{
    class FlutterEngine
    {
    public:
        FlutterEngine() = default;

        void prepare (double sampleRateHz, int maxBlockSize, int numChannels)
        {
            ignoreUnused (maxBlockSize);

            sampleRate = static_cast<float>(sampleRateHz);

            depthSlew.resize (static_cast<size_t>(numChannels));
            for (auto& slew : depthSlew)
            {
                slew.reset (sampleRateHz, 0.05);
                slew.setCurrentAndTargetValue (kDepthFloor);
            }

            phase1.assign (static_cast<size_t>(numChannels), 0.0f);
            phase2.assign (static_cast<size_t>(numChannels), 0.0f);
            phase3.assign (static_cast<size_t>(numChannels), 0.0f);
            blockEndAc.assign (static_cast<size_t>(numChannels), 0.0f);

            const float scale = 1000.0f / sampleRate;
            amplitude1 = -230.0f * scale;
            amplitude2 =  -80.0f * scale;
            amplitude3 =  -99.0f * scale;
            dcOffsetInSamples = 350.0f * scale;
        }

        // rateNormalised in [0, 1] maps to 0.1..100 Hz.
        // depthNormalised in [0, 1] maps to the aggregate flutter amplitude.
        void prepareBlock (float rateNormalised, float depthNormalised) noexcept
        {
            // depth^2 * 0.5 → depthTarget = depth * sqrt(0.5) ≈ depth * 0.707.
            // The previous formula (depth^3 * 81/625) capped depthTarget at
            // 0.36 at full knob (depth^1.5 * 0.36), leaving the effect at
            // <13% of amplitude at 50% knob. The new curve is linear in depth
            // with a 0.707 ceiling, doubling audibility across the whole range.
            const float skewedDepth = std::pow (depthNormalised, 2.0f) * 0.5f;
            const float depthTarget = std::sqrt (juce::jmax (0.0f, skewedDepth));
            for (auto& slew : depthSlew)
                slew.setTargetValue (juce::jmax (kDepthFloor, depthTarget));

            const float rateHz = 0.1f * std::pow (1000.0f, rateNormalised);
            angleDelta1 = juce::MathConstants<float>::twoPi * rateHz / sampleRate;
            angleDelta2 = 2.0f * angleDelta1;
            angleDelta3 = 3.0f * angleDelta1;
        }

        // Per-sample pair: the AC wobble and the constant DC offset.
        // Both are in samples-of-delay; callers typically add them and
        // apply to a tap-position reader. Used by characterisation
        // harnesses; the audio hot path prefers processBlock below.
        inline std::pair<float, float> nextSample (size_t channelIndex) noexcept
        {
            phase1[channelIndex] += angleDelta1;
            phase2[channelIndex] += angleDelta2;
            phase3[channelIndex] += angleDelta3;

            const float depth = depthSlew[channelIndex].getNextValue();
            const float ac = depth * (amplitude1 * std::cos (phase1[channelIndex] + kPhaseOffset1)
                                    + amplitude2 * std::cos (phase2[channelIndex] + kPhaseOffset2)
                                    + amplitude3 * std::cos (phase3[channelIndex] + kPhaseOffset3));
            blockEndAc[channelIndex] = ac;
            return { ac, dcOffsetInSamples };
        }

        // Block-rate fast path. Advances every phase by the full block's
        // worth in one step and returns the (ac, dc) pair sampled at
        // the end of the block.
        inline std::pair<float, float> processBlock (int numSamples, size_t channelIndex) noexcept
        {
            const float phaseAdvance1 = angleDelta1 * static_cast<float>(numSamples);
            phase1[channelIndex] += phaseAdvance1;
            phase2[channelIndex] += phaseAdvance1 * 2.0f;
            phase3[channelIndex] += phaseAdvance1 * 3.0f;

            const float depth = depthSlew[channelIndex].skip (numSamples);
            const float ac = depth * (amplitude1 * std::cos (phase1[channelIndex] + kPhaseOffset1)
                                    + amplitude2 * std::cos (phase2[channelIndex] + kPhaseOffset2)
                                    + amplitude3 * std::cos (phase3[channelIndex] + kPhaseOffset3));
            blockEndAc[channelIndex] = ac;
            return { ac, dcOffsetInSamples };
        }

        [[nodiscard]] float getBlockEndAc (size_t channelIndex) const noexcept
        {
            return blockEndAc[channelIndex];
        }

        [[nodiscard]] float getDcOffsetInSamples() const noexcept
        {
            return dcOffsetInSamples;
        }

        // Advance phase without emitting an output sample.
        inline void skipSample (size_t channelIndex) noexcept
        {
            phase1[channelIndex] += angleDelta1;
            phase2[channelIndex] += angleDelta2;
            phase3[channelIndex] += angleDelta3;
            depthSlew[channelIndex].getNextValue();
        }

        inline void wrapPhase (size_t channelIndex) noexcept
        {
            constexpr float twoPi = juce::MathConstants<float>::twoPi;
            for (float* p : { &phase1[channelIndex], &phase2[channelIndex], &phase3[channelIndex] })
            {
                while (*p >= twoPi) *p -= twoPi;
                while (*p <  0.0f)  *p += twoPi;
            }
        }

        [[nodiscard]] bool isSilent() const noexcept
        {
            return depthSlew.empty()
                || depthSlew.front().getTargetValue() == kDepthFloor;
        }

        void reset() noexcept
        {
            for (auto& slew : depthSlew)
                slew.setCurrentAndTargetValue (kDepthFloor);
            std::fill (phase1.begin(), phase1.end(), 0.0f);
            std::fill (phase2.begin(), phase2.end(), 0.0f);
            std::fill (phase3.begin(), phase3.end(), 0.0f);
            std::fill (blockEndAc.begin(), blockEndAc.end(), 0.0f);
        }

    private:
        static constexpr float kDepthFloor = 0.001f;
        static constexpr float kPhaseOffset1 = 0.0f;
        static constexpr float kPhaseOffset2 = 13.0f * MathConstants<float>::pi / 4.0f;
        static constexpr float kPhaseOffset3 = -MathConstants<float>::pi / 10.0f;

        float sampleRate        { 48000.0f };
        float amplitude1        { 0.0f };
        float amplitude2        { 0.0f };
        float amplitude3        { 0.0f };
        float dcOffsetInSamples { 0.0f };

        float angleDelta1 { 0.0f };
        float angleDelta2 { 0.0f };
        float angleDelta3 { 0.0f };

        std::vector<float> phase1;
        std::vector<float> phase2;
        std::vector<float> phase3;
        std::vector<float> blockEndAc;
        std::vector<juce::SmoothedValue<float, juce::ValueSmoothingTypes::Multiplicative>> depthSlew;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FlutterEngine)
    };
}
#endif

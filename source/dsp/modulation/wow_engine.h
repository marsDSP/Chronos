#pragma once

#ifndef CHRONOS_WOW_ENGINE_H
#define CHRONOS_WOW_ENGINE_H

// ============================================================================
//  wow_engine.h
// ----------------------------------------------------------------------------
//  Slow tape-style wow generator. A single cosine LFO per channel whose
//  rate is nudged by a per-block random perturbation (drift). Output is
//  a per-sample delay offset expressed directly in samples, scaled by a
//  smoothed depth value so zeroing the depth knob takes the engine out
//  of the signal cleanly.
// ============================================================================

#include <JuceHeader.h>
#include <cmath>
#include <vector>

namespace MarsDSP::DSP::Modulation
{
    class WowEngine
    {
    public:
        WowEngine() = default;

        void prepare (double sampleRateHz, int maxBlockSize, int numChannels)
        {
            juce::ignoreUnused (maxBlockSize);

            sampleRate = static_cast<float>(sampleRateHz);
            // Max depth target: 5 ms of delay modulation, scaled to samples.
            // The original formula (1e6 / sampleRate) was an inverted unit
            // conversion that produced only ~21 samples (~0.43 ms) at 48 kHz,
            // yielding <0.15% pitch deviation – below the perception threshold
            // at slow rates. 5 ms gives ~1.5% at max rate, consistent with
            // audible tape wow.
            amplitudeInSamples = 5.0f * sampleRate / 1000.0f;

            depthSlew.resize (static_cast<size_t>(numChannels));
            for (auto& slew : depthSlew)
            {
                slew.reset (sampleRateHz, 0.05);
                slew.setCurrentAndTargetValue (kDepthFloor);
            }

            phase.assign (static_cast<size_t>(numChannels), 0.0f);
            blockEndSample.assign (static_cast<size_t>(numChannels), 0.0f);
        }

        // rateNormalised in [0, 1] maps to 0..~3.5 Hz with a musical skew.
        // depthNormalised in [0, 1] maps to the cosine amplitude.
        // driftNormalised in [0, 1] adds a per-block random perturbation
        // to the instantaneous rate.
        void prepareBlock (float rateNormalised,
                           float depthNormalised,
                           float driftNormalised) noexcept
        {
            // Quadratic taper: depth^3 left the lower half of the knob nearly
            // silent (0.5 knob → 0.125 depth). Squaring still provides a
            // gentle concave feel while keeping the effect audible mid-range.
            const float depthCurve = depthNormalised * depthNormalised;
            for (auto& slew : depthSlew)
                slew.setTargetValue (juce::jmax (kDepthFloor, depthCurve));

            const float rateHz = std::pow (4.5f, rateNormalised) - 1.0f;
            const float randomUnit = randomEngine.nextFloat();
            const float drifted = rateHz
                                * (1.0f + std::pow (randomUnit, 1.25f) * driftNormalised);
            angleDelta = juce::MathConstants<float>::twoPi * drifted / sampleRate;
        }

        // Per-sample output in samples-of-delay. Advances the phase for
        // the given channel as a side effect. Used by block-by-block
        // characterisation harnesses; the audio hot path prefers
        // processBlock below.
        inline float nextSample (size_t channelIndex) noexcept
        {
            phase[channelIndex] += angleDelta;
            const float depth = depthSlew[channelIndex].getNextValue() * amplitudeInSamples;
            blockEndSample[channelIndex] = depth * std::cos (phase[channelIndex]);
            return blockEndSample[channelIndex];
        }

        // Block-rate fast path. Advances the phase by the full block's
        // worth in one step and returns the sample offset at the end of
        // the block. The caller pairs this with getBlockEndSample from
        // the previous block to drive a linear crossfade across the
        // block boundary without per-sample iteration.
        inline float processBlock (int numSamples, size_t channelIndex) noexcept
        {
            phase[channelIndex] += angleDelta * static_cast<float>(numSamples);
            const float depth = depthSlew[channelIndex].skip (numSamples) * amplitudeInSamples;
            blockEndSample[channelIndex] = depth * std::cos (phase[channelIndex]);
            return blockEndSample[channelIndex];
        }

        [[nodiscard]] float getBlockEndSample (size_t channelIndex) const noexcept
        {
            return blockEndSample[channelIndex];
        }

        // Advance phase without emitting an output sample. Used while the
        // host buffer is bypassed so the LFO keeps its continuity across
        // the bypass transition.
        inline void skipSample (size_t channelIndex) noexcept
        {
            phase[channelIndex] += angleDelta;
            depthSlew[channelIndex].getNextValue();
        }

        inline void wrapPhase (size_t channelIndex) noexcept
        {
            while (phase[channelIndex] >= juce::MathConstants<float>::twoPi)
                phase[channelIndex] -= juce::MathConstants<float>::twoPi;
            while (phase[channelIndex] < 0.0f)
                phase[channelIndex] += juce::MathConstants<float>::twoPi;
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
            std::fill (phase.begin(), phase.end(), 0.0f);
            std::fill (blockEndSample.begin(), blockEndSample.end(), 0.0f);
        }

    private:
        static constexpr float kDepthFloor = 0.001f;

        float sampleRate         { 48000.0f };
        float amplitudeInSamples { 0.0f };
        float angleDelta         { 0.0f };

        std::vector<float> phase;
        std::vector<float> blockEndSample;
        std::vector<juce::SmoothedValue<float, juce::ValueSmoothingTypes::Multiplicative>> depthSlew;

        juce::Random randomEngine;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (WowEngine)
    };
}
#endif

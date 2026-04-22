#pragma once

#ifndef CHRONOS_OU_DRIFT_PROCESS_H
#define CHRONOS_OU_DRIFT_PROCESS_H

// ============================================================================
//  ou_drift_process.h
// ----------------------------------------------------------------------------
//  Gaussian white noise generated into a pre-allocated block, then a per-
//  sample process(n, ch) call steps the OU accumulator forward with the
//  stored noise sample, applies mean-reversion, and returns the state
//  value smoothed through a 4th-order lowpass (cascade of two 2nd-order
//  state-variable lowpasses at kDriftSmoothingCutoffHz). The cascade is
//  what gives the drift its "butter smooth" character: the 4th-order
//  rolloff suppresses per-sample jitter far more aggressively than a
//  single 2nd-order stage would, without changing the underlying
//  Ornstein-Uhlenbeck mean-reversion dynamics.
// ============================================================================

#include <JuceHeader.h>
#include <cmath>
#include <vector>

#include "noise_generator_engine.h"

namespace MarsDSP::DSP::Modulation
{
    // ----------------------------------------------------------------------------
    //  OuDriftProcess
    //
    //  Per-sample use pattern:
    //
    //      ouDrift.prepare(sampleRate, maxBlockSize, numChannels); // once
    //
    //      // every processBlock:
    //      ouDrift.prepareBlock(driftAmount, numSamples); // once per block
    //
    //      for (int n = 0; n < numSamples; ++n)
    //          samplePositionDrift = ouDrift.process(n, 0) * someScale;
    // ----------------------------------------------------------------------------
    class OuDriftProcess
    {
    public:
        // Cutoff of every stage in the smoothing cascade. Lower =
        // smoother / slower drift. 1.5 Hz gives a per-stage time
        // constant of ~106 ms and a cascade group delay around
        // ~200 ms, which keeps the drift at "wow / breath" rates
        // rather than audible-LFO rates. Anything faster starts
        // sounding like a very fast LFO instead of organic drift.
        static constexpr float kDriftSmoothingCutoffHz = 1.5f;

        // Linear gain applied to the raw Uniform noise source before
        // it reaches the OU accumulator. Matches the reference
        // OUProcess implementation: uniform [-1, 1] noise scaled by
        // 1/2.33 so the per-sample kick has std dev (1/sqrt(3))/2.33
        // ~= 0.248. Uniform noise has bounded peaks (no 3+ sigma
        // Gaussian tail) which keeps the OU state from receiving
        // occasional large impulses that ring through the smoothing
        // cascade as a 1-3 Hz "fan" modulation.
        static constexpr float kRawNoiseLinearGain =
            1.0f / 2.33f;

        OuDriftProcess() = default;

        void prepare (double sampleRate,
                      int    samplesPerBlock,
                      int    numChannels)
        {
            dsp::ProcessSpec stereoSpec
            {
                sampleRate,
                static_cast<uint32>(samplesPerBlock),
                static_cast<uint32>(numChannels)
            };

            dsp::ProcessSpec monoSpec
            {
                sampleRate,
                static_cast<uint32>(samplesPerBlock), 1u
            };

            noiseGenerator.setNoiseType(
                NoiseGeneratorEngine<float>::Uniform);
            noiseGenerator.setGainLinear(kRawNoiseLinearGain);
            noiseGenerator.prepare(monoSpec);

            // Cascade of two 2nd-order state-variable lowpasses ->
            // effectively a 4th-order Butterworth-style rolloff above
            // kDriftSmoothingCutoffHz. Each stage gets the full number
            // of channels so both the delay-position drift (stereo)
            // and the reverb-modulation drift (mono) work.
            driftSmoothingLowpassStageA.prepare(stereoSpec);
            driftSmoothingLowpassStageA.setType(
                dsp::StateVariableTPTFilterType::lowpass);
            driftSmoothingLowpassStageA.setCutoffFrequency(kDriftSmoothingCutoffHz);
            driftSmoothingLowpassStageA.reset();

            driftSmoothingLowpassStageB.prepare(stereoSpec);
            driftSmoothingLowpassStageB.setType(
                dsp::StateVariableTPTFilterType::lowpass);
            driftSmoothingLowpassStageB.setCutoffFrequency(kDriftSmoothingCutoffHz);
            driftSmoothingLowpassStageB.reset();

            preGeneratedNoiseBuffer.setSize(1, samplesPerBlock);
            preGeneratedNoiseReadPointer =
                preGeneratedNoiseBuffer.getReadPointer(0);

            sqrtSampleInterval = 1.0f / std::sqrt(static_cast<float>(sampleRate));
            sampleInterval     = 1.0f / static_cast<float>(sampleRate);

            // Start at the mean (zero) so the first block after prepare
            // doesn't emit a transient ramp from an arbitrary initial value.
            // The process is zero-mean, so state 0 = "no drift".
            accumulatedOrnsteinUhlenbeckStatePerChannel.assign(
                static_cast<size_t>(numChannels), 0.0f);
        }

        void prepareBlock (float driftAmountNormalised, int numSamples)
        {
            preGeneratedNoiseBuffer.setSize(1, numSamples,
                                            /*keepExistingContent=*/false,
                                            /*clearExtraSpace=*/false,
                                            /*avoidReallocating=*/true);
            preGeneratedNoiseBuffer.clear();

            dsp::AudioBlock<float> noiseBlock(preGeneratedNoiseBuffer);
            noiseGenerator.process(
                dsp::ProcessContextReplacing(noiseBlock));

            // Match the working reference: raw (clipped) amount feeds
            // both the noise injection and the damping law. The previous
            // pow(amount, 1.25f) "shaping" stacked with the inflated
            // damping coefficient to suppress the drift at sub-1.0 knob
            // positions; both are reverted here so equilibrium variance
            // tracks the original OUProcess formulation.
            const float clippedDriftAmount =
                juce::jlimit(0.0f, 1.0f, driftAmountNormalised);
            currentDriftAmount   = clippedDriftAmount;
            currentDamping       = 0.5f + clippedDriftAmount * 8.0f;

            // Zero-mean drift: callers get the raw deviation-from-rest
            // value from process(), no DC subtraction needed downstream.
            currentMeanTarget    = 0.0f;

            preGeneratedNoiseReadPointer =
                preGeneratedNoiseBuffer.getReadPointer(0);
        }

        // Per-block mean-reversion strength knob. Maps [0, 1] -> [0.5, 8.5]
        // inside process() and multiplies the damping term so the user
        // can dial how aggressively the OU state is pulled back to the
        // zero mean without also touching the noise injection amplitude.
        void setMeanReversion (float newMeanReversion) noexcept
        {
            meanReversion = juce::jlimit(0.0f, 1.0f, newMeanReversion);
        }

        float process (int sampleIndexInBlock, size_t channelIndex) noexcept
        {
            accumulatedOrnsteinUhlenbeckStatePerChannel[channelIndex] +=
                sqrtSampleInterval
                * preGeneratedNoiseReadPointer[sampleIndexInBlock]
                * currentDriftAmount;

            // Restored mean-reversion scalar: jmap(meanReversion, 0.5, 8.5)
            // multiplies the damping term so the effective pull-to-mean
            // rate matches the reference OUProcess formulation. With the
            // knob at its default of 0 this collapses to damping * 0.5
            // (i.e. effective theta up to 4.25 at amount=1), which is
            // what keeps the equilibrium variance in the audible range.
            const float driftToMean = juce::jmap(meanReversion, 0.5f, 8.5f);
            accumulatedOrnsteinUhlenbeckStatePerChannel[channelIndex] +=
                currentDamping
                * driftToMean
                * (currentMeanTarget
                   - accumulatedOrnsteinUhlenbeckStatePerChannel[channelIndex])
                * sampleInterval;

            // 4th-order cascade: stage A feeds stage B. Both stages
            // are at the same cutoff so the combined magnitude
            // response is Butterworth-like with a -24 dB/oct slope,
            // which is enough to completely flatten the per-sample
            // noise jitter that would otherwise ride on top of the
            // OU curve.
            const int channelIndexAsInt = static_cast<int>(channelIndex);
            const float stageAOutput =
                driftSmoothingLowpassStageA.processSample(
                    channelIndexAsInt,
                    accumulatedOrnsteinUhlenbeckStatePerChannel[channelIndex]);
            return driftSmoothingLowpassStageB.processSample(
                channelIndexAsInt, stageAOutput);
        }

    private:
        float sqrtSampleInterval { 1.0f / std::sqrt(48000.0f) };
        float sampleInterval     { 1.0f / 48000.0f };

        std::vector<float> accumulatedOrnsteinUhlenbeckStatePerChannel;

        float currentDriftAmount { 0.0f };
        float currentMeanTarget  { 0.0f };
        float currentDamping     { 0.0f };
        float meanReversion      { 0.0f };

        NoiseGeneratorEngine<float>        noiseGenerator;
        AudioBuffer<float>                 preGeneratedNoiseBuffer;
        const float*                       preGeneratedNoiseReadPointer { nullptr };

        // Cascade of two 2nd-order state-variable lowpasses at
        // kDriftSmoothingCutoffHz = 4 Hz. Chaining them in series
        // yields a 4th-order filter with a -24 dB/oct slope, which
        // scrubs away per-sample noise jitter and leaves the drift
        // output on the "butter smooth" side of audibly moving.
        dsp::StateVariableTPTFilter<float> driftSmoothingLowpassStageA;
        dsp::StateVariableTPTFilter<float> driftSmoothingLowpassStageB;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (OuDriftProcess)
    };
}
#endif

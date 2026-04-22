#pragma once

#ifndef CHRONOS_OU_DRIFT_PROCESS_H
#define CHRONOS_OU_DRIFT_PROCESS_H

// ============================================================================
//  ou_drift_process.h
// ----------------------------------------------------------------------------
//  Gaussian white noise generated into a pre-allocated block, then a per-sample
//  process(n, ch) call steps the OU accumulator forward with the stored
//  noise sample, applies mean-reversion, and returns the 10 Hz-lowpassed
//  state value.
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
                NoiseGeneratorEngine<float>::Normal);
            noiseGenerator.setGainLinear(1.0f / 2.33f);
            noiseGenerator.prepare(monoSpec);

            driftSmoothingLowpass.prepare(stereoSpec);
            driftSmoothingLowpass.setType(
                dsp::StateVariableTPTFilterType::lowpass);
            driftSmoothingLowpass.setCutoffFrequency(10.0f);
            driftSmoothingLowpass.reset();

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

            const float shapedAmount = std::pow(driftAmountNormalised, 1.25f);
            currentDriftAmount   = shapedAmount;
            currentDamping       = shapedAmount * 20.0f + 1.0f;

            // Zero-mean drift: callers get the raw deviation-from-rest
            // value from process(), no DC subtraction needed downstream.
            currentMeanTarget    = 0.0f;

            preGeneratedNoiseReadPointer =
                preGeneratedNoiseBuffer.getReadPointer(0);
        }

        float process (int sampleIndexInBlock, size_t channelIndex) noexcept
        {
            accumulatedOrnsteinUhlenbeckStatePerChannel[channelIndex] +=
                sqrtSampleInterval
                * preGeneratedNoiseReadPointer[sampleIndexInBlock]
                * currentDriftAmount;

            accumulatedOrnsteinUhlenbeckStatePerChannel[channelIndex] +=
                currentDamping
                * (currentMeanTarget
                   - accumulatedOrnsteinUhlenbeckStatePerChannel[channelIndex])
                * sampleInterval;

            return driftSmoothingLowpass.processSample(static_cast<int>(channelIndex),
                accumulatedOrnsteinUhlenbeckStatePerChannel[channelIndex]);
        }

    private:
        float sqrtSampleInterval { 1.0f / std::sqrt(48000.0f) };
        float sampleInterval     { 1.0f / 48000.0f };

        std::vector<float> accumulatedOrnsteinUhlenbeckStatePerChannel;

        float currentDriftAmount { 0.0f };
        float currentMeanTarget  { 0.0f };
        float currentDamping     { 0.0f };

        NoiseGeneratorEngine<float>        noiseGenerator;
        AudioBuffer<float>                 preGeneratedNoiseBuffer;
        const float*                       preGeneratedNoiseReadPointer { nullptr };

        dsp::StateVariableTPTFilter<float> driftSmoothingLowpass;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (OuDriftProcess)
    };
}
#endif

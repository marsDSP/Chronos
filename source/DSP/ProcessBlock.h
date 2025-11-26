#pragma once

#include <Includes.h>
#include "Parameters.h"
#include "Smoother.h"
#include "TapeDelay/TapeDelay.h"

namespace MarsDSP::DSP {

    class ProcessBlock {

    public:

        ProcessBlock() = default;
        ~ProcessBlock() = default;

        void prepareDSP (double sampleRate, juce::uint32 samplesPerBlock, juce::uint32 numChannels, int factor, const Parameters& params)
        {
            m_spec.sampleRate = sampleRate * static_cast<double>(1 << factor);
            m_spec.maximumBlockSize = samplesPerBlock * (1 << factor);
            m_spec.numChannels = numChannels;

            smoother = std::make_unique<Smoother>(params);
            smoother->prepare(m_spec);
            smoother->reset();

            tapeDelay.prepare(m_spec);
            m_oversample = std::make_unique<juce::dsp::Oversampling<float>>(
                           static_cast<int>(m_spec.numChannels), factor,
                           juce::dsp::Oversampling<float>::FilterType::filterHalfBandPolyphaseIIR,
                           true);

            m_oversample->initProcessing(m_spec.maximumBlockSize);
        }

        void processDSP (juce::AudioBuffer<float>& buffer, const juce::uint32 numSamples)
        {
            if (m_oversample == nullptr)
                return;

            juce::dsp::AudioBlock<float> block (buffer);

            const int osFactor = static_cast<int>(m_oversample->getOversamplingFactor());
            const int numOversampled = numSamples * osFactor;

            // Upsample
            auto up = m_oversample->processSamplesUp(block);
            const int upNumChannels = static_cast<int>(up.getNumChannels());
            float* outL = up.getChannelPointer(0);
            float* outR = upNumChannels > 1 ? up.getChannelPointer(1) : up.getChannelPointer(0);
            const float* inL = outL;
            const float* inR = outR;

            // Process oversampled stereo block
            if (smoother)
                tapeDelay.processTape(inL, inR, outL, outR, numOversampled, *smoother);

            // Downsample
            m_oversample->processSamplesDown(block);

        }

        [[nodiscard]] double getSampleRate() const noexcept
        {
            return m_spec.sampleRate;
        }

        void updateParams(const Parameters&)
        {
            if (smoother)
                smoother->update();
        }

    private:

        juce::dsp::ProcessSpec m_spec {};
        std::unique_ptr<juce::dsp::Oversampling<float>> m_oversample;
        std::unique_ptr<Smoother> smoother;
        TapeDelay tapeDelay;

    };
}

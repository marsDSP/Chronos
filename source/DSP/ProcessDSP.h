#pragma once

#include <Includes.h>
#include "Parameters.h"
#include "Smoother.h"
#include "TapeDelayDSP.h"

namespace MarsDSP::DSP {

    class ProcessBlock
    {
    public:

        ProcessBlock() = default;
        ~ProcessBlock() = default;

        void prepareDSP (double sampleRate, juce::uint32 samplesPerBlock, juce::uint32 numChannels, const Parameters& params)
        {
            spec.sampleRate = sampleRate;
            spec.maximumBlockSize = samplesPerBlock;
            spec.numChannels = numChannels;

            smoother = std::make_unique<Smoother<Parameters>>(params);
            smoother->prepare(spec);
            smoother->reset();

            tape.prepare(spec);
            
            if (m_scratchBuffer.size() < spec.maximumBlockSize)
                m_scratchBuffer.resize(spec.maximumBlockSize);
        }

        void process (juce::AudioBuffer<float>& buffer)
        {
            const auto numChannels = buffer.getNumChannels();
            const auto numSamples = buffer.getNumSamples();

            if (numChannels == 0 || numSamples == 0)
                return;

            if (smoother)
                smoother->update();

            if (smoother && smoother->getBypass())
                return;

            const float* inL = buffer.getReadPointer(0);
            float* outL = buffer.getWritePointer(0);
            const float* inR = nullptr;
            float* outR = nullptr;

            if (numChannels > 1)
            {
                inR = buffer.getReadPointer(1);
                outR = buffer.getWritePointer(1);
            }

            else
            {
                inR = inL;

                if (m_scratchBuffer.size() < numSamples)
                    m_scratchBuffer.resize(numSamples);
                
                outR = m_scratchBuffer.data();
            }

            tape.processTape(inL, inR, outL, outR, numSamples, *smoother);
        }

    private:

        juce::dsp::ProcessSpec spec {};
        std::unique_ptr<Smoother<Parameters>> smoother;
        TapeDelayDSP tape;
        std::vector<float> m_scratchBuffer;
    };
}

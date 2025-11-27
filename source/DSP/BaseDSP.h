#pragma once

#include <Includes.h>

#include "Smoother.h"

namespace MarsDSP::DSP
{
    template <typename ParametersType>
    class BaseDSP
    {
    public:
        explicit BaseDSP(const ParametersType& params) : smoother(params) {}
        virtual ~BaseDSP() = default;

        virtual void prepare(const juce::dsp::ProcessSpec& spec)
        {
            smoother.prepare(spec);
            smoother.reset();
        }

        virtual void processBlock(juce::dsp::AudioBlock<float>& block, int num_samples)
        {
            smoother.update();

            for (size_t channel {}; channel < block.getNumChannels(); ++channel)
            {
                auto* data = block.getChannelPointer(channel);

                for (size_t sample {}; sample < num_samples; ++sample)
                {
                    const float xn_input = data[sample];
                    const float yn_output = processSample(xn_input, static_cast<int>(channel));
                    data[sample] = yn_output;
                }
            }
        }

        virtual float processSample(float xn_input, int channel) = 0;

        float getTime(size_t channel = 0)     { return smoother.getTime(channel); }
        float getRegen(size_t channel = 0)    { return smoother.getRegen(channel); }
        float getFreq(size_t channel = 0)     { return smoother.getFreq(channel); }
        float getReso(size_t channel = 0)     { return smoother.getReso(channel); }
        float getDryWet(size_t channel = 0)   { return smoother.getDryWet(channel); }

    private:

        Smoother<ParametersType> smoother;
    };
}

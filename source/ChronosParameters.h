#pragma once

#ifndef CHRONOS_CHRONOSPARAMETERS_H
#define CHRONOS_CHRONOSPARAMETERS_H

#include <JuceHeader.h>

const ParameterID gainParamID { "gain", 1 };
const ParameterID bitsParamID { "bits", 1 };

template<typename T>
static void castParameter(const AudioProcessorValueTreeState& apvts, const ParameterID& id, T& destination)
{
    destination = dynamic_cast<T>(apvts.getParameter(id.getParamID()));
    jassert(destination);
}

class ChronosParameters {
public:
    explicit ChronosParameters(const AudioProcessorValueTreeState& apvts)
    {
        castParameter(apvts, gainParamID, gainParam);
        castParameter(apvts, bitsParamID, bitsParam);
    }

    static AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        AudioProcessorValueTreeState::ParameterLayout layout;
        layout.add(std::make_unique<AudioParameterFloat>(gainParamID, "Output Gain", NormalisableRange{-12.0f, 12.0f}, 0.0f));
        layout.add(std::make_unique<AudioParameterInt>(bitsParamID, "Bit Depth", 1, 32, 24));
        return layout;
    }

    void prepare(const double sr) noexcept
    {
        constexpr double dur = 0.02;
        gainSmoother.reset(sr, dur);
        bitsSmoother.reset(sr, dur);
    }

    void reset() noexcept
    {
        gain = 0.0f;
        bits = 0.0f;
        if (gainParam != nullptr)
            gainSmoother.setCurrentAndTargetValue(Decibels::decibelsToGain(gainParam->get()));
        if (bitsParam != nullptr)
            bitsSmoother.setCurrentAndTargetValue(static_cast<float>(bitsParam->get()));
    }

    void update() noexcept
    {
        if (gainParam != nullptr)
            gainSmoother.setTargetValue(Decibels::decibelsToGain(gainParam->get()));
        if (bitsParam != nullptr)
            bitsSmoother.setTargetValue(static_cast<float>(bitsParam->get()));
    }

    void smoothen() noexcept
    {
        gain = gainSmoother.getNextValue();
        bits = static_cast<int>(bitsSmoother.getNextValue());
    }

    [[nodiscard]] float getGain() const noexcept { return gain; }
    [[nodiscard]] int getBits() const noexcept { return bits; }

private:
    float gain {};
    int bits {};

    AudioParameterFloat* gainParam {};
    AudioParameterInt* bitsParam {};

    LinearSmoothedValue<float> gainSmoother;
    LinearSmoothedValue<float> bitsSmoother;
};
#endif
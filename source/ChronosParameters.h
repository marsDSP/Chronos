#pragma once

#ifndef CHRONOS_CHRONOSPARAMETERS_H
#define CHRONOS_CHRONOSPARAMETERS_H

#include <JuceHeader.h>

const ParameterID gainParamID { "gain", 1 };
const ParameterID bitsParamID { "bits", 1 };
const ParameterID delayTimeParamID { "delayTime", 1 };
const ParameterID bypassParamID { "bypass", 1 };

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
        castParameter(apvts, delayTimeParamID, delayParam);
        castParameter(apvts, bypassParamID, bypassParam);
    }

    static AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        AudioProcessorValueTreeState::ParameterLayout layout;
        layout.add(std::make_unique<AudioParameterFloat>(gainParamID, "Output Gain", NormalisableRange{-12.0f, 12.0f}, 0.0f));
        layout.add(std::make_unique<AudioParameterInt>(bitsParamID, "Bit Depth", 1, 32, 24));
        layout.add(std::make_unique<AudioParameterFloat>(delayTimeParamID, "Delay Time", NormalisableRange{ minDelayTime, maxDelayTime }, 500.0f));
        layout.add(std::make_unique<AudioParameterBool>(bypassParamID, "Bypass", false));
        return layout;
    }

    void prepare(const double sr) noexcept
    {
        sampleRate = sr;
        constexpr double dur = 0.02;
        gainSmoother.reset(sr, dur);
        bitsSmoother.reset(sr, dur);
        delaySmoother.reset(sr, dur);
    }

    void reset() noexcept
    {
        gain = 0.0f;
        bits = 0.0f;
        delaySamples = 0.0f;
        if (gainParam != nullptr)
            gainSmoother.setCurrentAndTargetValue(Decibels::decibelsToGain(gainParam->get()));
        if (bitsParam != nullptr)
            bitsSmoother.setCurrentAndTargetValue(static_cast<float>(bitsParam->get()));
        if (delayParam != nullptr)
            delaySmoother.setCurrentAndTargetValue(msToSamples(delayParam->get()));
    }

    void update() noexcept
    {
        if (gainParam != nullptr)
            gainSmoother.setTargetValue(Decibels::decibelsToGain(gainParam->get()));
        if (bitsParam != nullptr)
            bitsSmoother.setTargetValue(static_cast<float>(bitsParam->get()));
        if (delayParam != nullptr)
            delaySmoother.setTargetValue(msToSamples(delayParam->get()));
    }

    void smoothen() noexcept
    {
        gain = gainSmoother.getNextValue();
        bits = static_cast<int>(bitsSmoother.getNextValue());
        delaySamples = delaySmoother.getNextValue();
    }

    [[nodiscard]] float getGain() const noexcept { return gain; }
    [[nodiscard]] int getBits() const noexcept { return bits; }
    [[nodiscard]] float getDelaySamples() const noexcept { return delaySamples; }
    [[nodiscard]] bool getBypass() const noexcept { return bypassParam != nullptr && bypassParam->get(); }
    [[nodiscard]] AudioProcessorParameter* getBypassParameter() const noexcept { return bypassParam; }

    static constexpr float minDelayTime = 5.0f;
    static constexpr float maxDelayTime = 5000.0f;

private:
    float msToSamples(float ms) const noexcept
    {
        return static_cast<float>(ms * 0.001 * sampleRate);
    }

    float gain {};
    int bits {};
    float delaySamples {};

    AudioParameterFloat* gainParam {};
    AudioParameterInt* bitsParam {};
    AudioParameterFloat* delayParam {};
    AudioParameterBool* bypassParam {};

    LinearSmoothedValue<float> gainSmoother;
    LinearSmoothedValue<float> bitsSmoother;
    LinearSmoothedValue<float> delaySmoother;

    double sampleRate {};
};
#endif
#pragma once

#ifndef CHRONOS_CHRONOSPARAMETERS_H
#define CHRONOS_CHRONOSPARAMETERS_H

#include <JuceHeader.h>
#include "dsp/DelayInterpolator.h"

const ParameterID gainParamID{"gain", 1};
const ParameterID bitsParamID{"bits", 1};
const ParameterID delayTimeParamID{"delayTime", 1};
const ParameterID bypassParamID{"bypass", 1};
const ParameterID hpfFreqParamID{"hpfFreq", 1};
const ParameterID lpfFreqParamID{"lpfFreq", 1};
const ParameterID mixParamID{"mix", 1};
const ParameterID interpolationParamID{"interpolation", 1};
const ParameterID driveParamID{"drive", 1};
const ParameterID adaaOrderParamID{"adaaOrder", 1};

template<typename T>
static void castParameter(const AudioProcessorValueTreeState &apvts, const ParameterID &id, T &destination)
{
    destination = dynamic_cast<T>(apvts.getParameter(id.getParamID()));
    jassert(destination);
}

class ChronosParameters
{
public:
    explicit ChronosParameters(const AudioProcessorValueTreeState &apvts)
    {
        castParameter(apvts, gainParamID, gainParam);
        castParameter(apvts, bitsParamID, bitsParam);
        castParameter(apvts, delayTimeParamID, delayParam);
        castParameter(apvts, bypassParamID, bypassParam);
        castParameter(apvts, hpfFreqParamID, hpfParam);
        castParameter(apvts, lpfFreqParamID, lpfParam);
        castParameter(apvts, mixParamID, mixParam);
        castParameter(apvts, interpolationParamID, interpolationParam);
        castParameter(apvts, driveParamID, driveParam);
        castParameter(apvts, adaaOrderParamID, adaaOrderParam);
    }

    static AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        AudioProcessorValueTreeState::ParameterLayout layout;
        layout.add(std::make_unique<AudioParameterFloat>(gainParamID, "Output Gain", NormalisableRange{-12.0f, 12.0f},
                                                         0.0f));
        layout.add(std::make_unique<AudioParameterInt>(bitsParamID, "Bit Depth", 1, 32, 24));
        layout.add(std::make_unique<AudioParameterFloat>(delayTimeParamID, "Delay Time",
                                                         NormalisableRange{minDelayTime, maxDelayTime}, 500.0f));
        layout.add(std::make_unique<AudioParameterBool>(bypassParamID, "Bypass", false));
        layout.add(std::make_unique<AudioParameterFloat>(hpfFreqParamID, "HPF Cutoff",
                                                         NormalisableRange{20.0f, 2000.0f}, 20.0f));
        layout.add(std::make_unique<AudioParameterFloat>(lpfFreqParamID, "LPF Cutoff",
                                                         NormalisableRange{200.0f, 20000.0f}, 20000.0f));
        layout.add(std::make_unique<AudioParameterFloat>(mixParamID, "Mix", NormalisableRange{0.0f, 100.0f}, 100.0f));
        layout.add(std::make_unique<AudioParameterChoice>(interpolationParamID, "Interpolation",
                                                          StringArray{"Linear", "Lagrange 3rd", "Lagrange 5th"}, 2));
        layout.add(std::make_unique<AudioParameterFloat>(driveParamID, "Drive", NormalisableRange{0.0f, 40.0f}, 0.0f));
        layout.add(std::make_unique<AudioParameterChoice>(adaaOrderParamID, "ADAA Order",
                                                          StringArray{"Off", "1st", "2nd"}, 2));
        return layout;
    }

    void prepare(const double sr) noexcept
    {
        sampleRate = sr;
        constexpr double dur = 0.02;
        gainSmoother.reset(sr, dur);
        bitsSmoother.reset(sr, dur);
        hpfSmoother.reset(sr, dur);
        lpfSmoother.reset(sr, dur);
        mixSmoother.reset(sr, dur);
        driveSmoother.reset(sr, dur);
    }

    void reset() noexcept
    {
        gain = 0.0f;
        bits = 0.0f;
        delaySamples = 0.0f;
        mix = 0.0f;
        drive = 0.0f;
        if (gainParam != nullptr)
            gainSmoother.setCurrentAndTargetValue(Decibels::decibelsToGain(gainParam->get()));
        if (bitsParam != nullptr)
            bitsSmoother.setCurrentAndTargetValue(static_cast<float>(bitsParam->get()));
        if (delayParam != nullptr)
            delaySamples = msToSamples(delayParam->get());
        if (hpfParam != nullptr)
            hpfSmoother.setCurrentAndTargetValue(hpfParam->get());
        if (lpfParam != nullptr)
            lpfSmoother.setCurrentAndTargetValue(lpfParam->get());
        if (mixParam != nullptr)
            mixSmoother.setCurrentAndTargetValue(mixParam->get());
        if (driveParam != nullptr)
            driveSmoother.setCurrentAndTargetValue(Decibels::decibelsToGain(driveParam->get()));
    }

    void update() noexcept
    {
        if (gainParam != nullptr)
            gainSmoother.setTargetValue(Decibels::decibelsToGain(gainParam->get()));
        if (bitsParam != nullptr)
            bitsSmoother.setTargetValue(static_cast<float>(bitsParam->get()));
        if (delayParam != nullptr)
            delaySamples = msToSamples(delayParam->get());
        if (hpfParam != nullptr)
            hpfSmoother.setTargetValue(hpfParam->get());
        if (lpfParam != nullptr)
            lpfSmoother.setTargetValue(lpfParam->get());
        if (mixParam != nullptr)
            mixSmoother.setTargetValue(mixParam->get());
        if (driveParam != nullptr)
            driveSmoother.setTargetValue(Decibels::decibelsToGain(driveParam->get()));
    }

    void smoothen() noexcept
    {
        gain = gainSmoother.getNextValue();
        bits = static_cast<int>(bitsSmoother.getNextValue());
        hpfFreq = hpfSmoother.getNextValue();
        lpfFreq = lpfSmoother.getNextValue();
        mix = mixSmoother.getNextValue();
        drive = driveSmoother.getNextValue();
    }

    [[nodiscard]] float getGain() const noexcept { return gain; }
    [[nodiscard]] int getBits() const noexcept { return bits; }
    [[nodiscard]] float getDelaySamples() const noexcept { return delaySamples; }
    [[nodiscard]] float getDelayMs() const noexcept { return delayParam != nullptr ? delayParam->get() : 500.0f; }
    [[nodiscard]] float getHPFFreq() const noexcept { return hpfFreq; }
    [[nodiscard]] float getLPFFreq() const noexcept { return lpfFreq; }
    [[nodiscard]] float getMix() const noexcept { return mix; }
    [[nodiscard]] float getDrive() const noexcept { return drive; }
    [[nodiscard]] double getSampleRate() const noexcept { return sampleRate; }
    [[nodiscard]] bool getBypass() const noexcept { return bypassParam != nullptr && bypassParam->get(); }
    [[nodiscard]] AudioProcessorParameter *getBypassParameter() const noexcept { return bypassParam; }

    [[nodiscard]] MarsDSP::Delays::Interpolation getInterpolation() const noexcept
    {
        if (interpolationParam == nullptr) return MarsDSP::Delays::Interpolation::Lagrange5th;
        switch (interpolationParam->getIndex())
        {
            case 0: return MarsDSP::Delays::Interpolation::Linear;
            case 1: return MarsDSP::Delays::Interpolation::Lagrange3rd;
            default: return MarsDSP::Delays::Interpolation::Lagrange5th;
        }
    }

    [[nodiscard]] int getADAAOrder() const noexcept
    {
        if (adaaOrderParam == nullptr) return 2;
        return adaaOrderParam->getIndex();
    }

    [[nodiscard]] float getRawGainLin() const noexcept
    {
        return gainParam ? Decibels::decibelsToGain(gainParam->get()) : 1.0f;
    }

    [[nodiscard]] float getRawDriveLin() const noexcept
    {
        return driveParam ? Decibels::decibelsToGain(driveParam->get()) : 1.0f;
    }

    [[nodiscard]] float getRawMix() const noexcept { return mixParam ? mixParam->get() : 100.0f; }
    [[nodiscard]] int getRawBits() const noexcept { return bitsParam ? bitsParam->get() : 24; }
    [[nodiscard]] float getRawHpfHz() const noexcept { return hpfParam ? hpfParam->get() : 20.0f; }
    [[nodiscard]] float getRawLpfHz() const noexcept { return lpfParam ? lpfParam->get() : 20000.0f; }

    static constexpr float minDelayTime = 5.0f;
    static constexpr float maxDelayTime = 5000.0f;

private:
    [[nodiscard]] float msToSamples(const float ms) const noexcept
    {
        return static_cast<float>(ms * 0.001 * sampleRate);
    }

    float gain{};
    int bits{};
    float delaySamples{};
    float hpfFreq{};
    float lpfFreq{};
    float mix{};
    float drive{};

    AudioParameterFloat *gainParam{};
    AudioParameterInt *bitsParam{};
    AudioParameterFloat *delayParam{};
    AudioParameterFloat *hpfParam{};
    AudioParameterFloat *lpfParam{};
    AudioParameterFloat *mixParam{};
    AudioParameterChoice *interpolationParam{};
    AudioParameterBool *bypassParam{};
    AudioParameterFloat *driveParam{};
    AudioParameterChoice *adaaOrderParam{};

    LinearSmoothedValue<float> gainSmoother;
    LinearSmoothedValue<float> bitsSmoother;
    LinearSmoothedValue<float> hpfSmoother;
    LinearSmoothedValue<float> lpfSmoother;
    LinearSmoothedValue<float> mixSmoother;
    LinearSmoothedValue<float> driveSmoother;

    double sampleRate{};
};
#endif

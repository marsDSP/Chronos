#pragma once

#ifndef CHRONOS_CHRONOSPARAMETERS_H
#define CHRONOS_CHRONOSPARAMETERS_H

#include <JuceHeader.h>

const ParameterID gainParamID{"gain", 1};
const ParameterID bitsParamID{"bits", 1};
const ParameterID delayTimeParamID{"delayTime", 1};
const ParameterID delaySyncParamID{"delaySync", 1};
const ParameterID delayDivisionParamID{"delayDivision", 1};
const ParameterID bypassParamID{"bypass", 1};
const ParameterID hpfFreqParamID{"hpfFreq", 1};
const ParameterID lpfFreqParamID{"lpfFreq", 1};
const ParameterID mixParamID{"mix", 1};
const ParameterID driveParamID{"drive", 1};
const ParameterID adaaOrderParamID{"adaaOrder", 1};
const ParameterID feedbackParamID{"feedback", 1};
const ParameterID dampHzParamID{"dampHz", 1};
const ParameterID loopCutHzParamID{"loopCutHz", 1};
const ParameterID crossFeedParamID{"crossFeed", 1};
const ParameterID loopDriveParamID{"loopDrive", 1};
const ParameterID loopSatOrderParamID{"loopSatOrder", 1};
const ParameterID delayModDepthParamID{"delayModDepth", 1};
const ParameterID delayModRateHzParamID{"delayModRateHz", 1};
const ParameterID enableDiffuserParamID{"enableDiffuser", 1};
const ParameterID diffusionParamID{"diffusion", 1};
const ParameterID diffuserSizeParamID{"diffuserSize", 1};
const ParameterID diffModDepthParamID{"diffModDepth", 1};
const ParameterID diffModRateHzParamID{"diffModRateHz", 1};

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
        castParameter(apvts, delaySyncParamID, delaySyncParam);
        castParameter(apvts, delayDivisionParamID, delayDivisionParam);
        castParameter(apvts, bypassParamID, bypassParam);
        castParameter(apvts, hpfFreqParamID, hpfParam);
        castParameter(apvts, lpfFreqParamID, lpfParam);
        castParameter(apvts, mixParamID, mixParam);
        castParameter(apvts, driveParamID, driveParam);
        castParameter(apvts, adaaOrderParamID, adaaOrderParam);
        castParameter(apvts, feedbackParamID, feedbackParam);
        castParameter(apvts, dampHzParamID, dampHzParam);
        castParameter(apvts, loopCutHzParamID, loopCutHzParam);
        castParameter(apvts, crossFeedParamID, crossFeedParam);
        castParameter(apvts, loopDriveParamID, loopDriveParam);
        castParameter(apvts, loopSatOrderParamID, loopSatOrderParam);
        castParameter(apvts, delayModDepthParamID, delayModDepthParam);
        castParameter(apvts, delayModRateHzParamID, delayModRateHzParam);
        castParameter(apvts, enableDiffuserParamID, enableDiffuserParam);
        castParameter(apvts, diffusionParamID, diffusionParam);
        castParameter(apvts, diffuserSizeParamID, diffuserSizeParam);
        castParameter(apvts, diffModDepthParamID, diffModDepthParam);
        castParameter(apvts, diffModRateHzParamID, diffModRateHzParam);
    }

    static AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        AudioProcessorValueTreeState::ParameterLayout layout;

        using Attrs = AudioParameterFloatAttributes;

        layout.add(std::make_unique<AudioParameterFloat>(delayTimeParamID, "Delay Time",
            NormalisableRange{1.0f, 5000.0f, 0.0f, 0.23108f}, 375.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 1) + " ms"; })));

        layout.add(std::make_unique<AudioParameterBool>(delaySyncParamID, "Delay Sync", false));

        layout.add(std::make_unique<AudioParameterChoice>(delayDivisionParamID, "Delay Division",
            StringArray{"1/64", "1/32T", "1/32", "1/16T", "1/32.", "1/16",
                        "1/8T", "1/16.", "1/8", "1/4T", "1/8.", "1/4",
                        "1/2T", "1/4.", "1/2", "1/1T", "1/2.", "1/1",
                        "2/1", "4/1"}, 11));

        {
            constexpr float kFbKnee = 0.90f;
            constexpr float kFbN0 = 1.5f;
            constexpr float kFbN1 = 150.0f;
            constexpr float kFbGKnee = 0.954993f;
            constexpr float kFbGMax = 1.15f;

            NormalisableRange<float> fbRange(0.0f, kFbGMax,
                [](float, float, float p) noexcept {
                    if (p <= kFbKnee)
                        return std::pow(10.0f, -3.0f / (kFbN0 * std::pow(kFbN1 / kFbN0, p / kFbKnee)));
                    return kFbGKnee + (p - kFbKnee) / (1.0f - kFbKnee) * (kFbGMax - kFbGKnee);
                },
                [](float, float, float g) noexcept {
                    if (g <= 0.01f) return 0.0f;
                    if (g <= kFbGKnee)
                        return kFbKnee * std::log((-3.0f / std::log10(g)) / kFbN0)
                               / std::log(kFbN1 / kFbN0);
                    return kFbKnee + (1.0f - kFbKnee) * (g - kFbGKnee) / (kFbGMax - kFbGKnee);
                });

            layout.add(std::make_unique<AudioParameterFloat>(feedbackParamID, "Feedback",
                std::move(fbRange), 0.42f,
                Attrs().withStringFromValueFunction(
                    [](float g, int) {
                        if (g >= 1.0f) return String("self-osc");
                        if (g < 0.01f) return String("0 repeats");
                        const float repeats = -30.0f / (20.0f * std::log10(g));
                        return String(repeats, 1) + " repeats";
                    })));
        }

        layout.add(std::make_unique<AudioParameterFloat>(dampHzParamID, "Repeat Damp",
            NormalisableRange{200.0f, 20000.0f, 0.0f, 0.32198f}, 8000.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) {
                    return v >= 1000.0f ? String(v * 0.001f, 1) + " kHz" : String(v, 1) + " Hz";
                })));

        layout.add(std::make_unique<AudioParameterFloat>(loopCutHzParamID, "Repeat Cut",
            NormalisableRange{20.0f, 2000.0f, 0.0f, 0.25452f}, 40.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) {
                    return v >= 1000.0f ? String(v * 0.001f, 1) + " kHz" : String(v, 1) + " Hz";
                })));

        layout.add(std::make_unique<AudioParameterFloat>(crossFeedParamID, "Cross Feed",
            NormalisableRange{0.0f, 1.0f}, 0.0f));

        layout.add(std::make_unique<AudioParameterFloat>(loopDriveParamID, "Repeat Drive",
            NormalisableRange{-6.0f, 24.0f}, 0.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 1) + " dB"; })));

        layout.add(std::make_unique<AudioParameterChoice>(loopSatOrderParamID, "Repeat Sat",
            StringArray{"Off", "1st", "2nd"}, 2));

        layout.add(std::make_unique<AudioParameterFloat>(delayModDepthParamID, "Delay Mod Depth",
            NormalisableRange{0.0f, 50.0f, 0.0f, 0.37824f}, 0.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 1) + " cents"; })));

        layout.add(std::make_unique<AudioParameterFloat>(delayModRateHzParamID, "Delay Mod Rate",
            NormalisableRange{0.01f, 10.0f, 0.0f, 0.22990f}, 0.35f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 2) + " Hz"; })));

        layout.add(std::make_unique<AudioParameterBool>(enableDiffuserParamID, "Diffuser", false));

        layout.add(std::make_unique<AudioParameterFloat>(diffusionParamID, "Diffusion",
            NormalisableRange{0.0f, 1.0f, 0.0f, 2.63852f}, 0.55f));

        layout.add(std::make_unique<AudioParameterFloat>(diffuserSizeParamID, "Diffuser Size",
            NormalisableRange{0.0f, 1.0f}, 0.5f));

        layout.add(std::make_unique<AudioParameterFloat>(diffModDepthParamID, "Diffuser Mod",
            NormalisableRange{0.0f, 1.5f}, 0.30f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 2) + " ms"; })));

        layout.add(std::make_unique<AudioParameterFloat>(diffModRateHzParamID, "Diffuser Rate",
            NormalisableRange{0.01f, 8.0f, 0.0f, 0.28300f}, 0.35f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 2) + " Hz"; })));

        layout.add(std::make_unique<AudioParameterFloat>(driveParamID, "Drive",
            NormalisableRange{0.0f, 24.0f}, 0.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 1) + " dB"; })));

        layout.add(std::make_unique<AudioParameterChoice>(adaaOrderParamID, "Drive Sat",
            StringArray{"Off", "1st", "2nd"}, 2));

        layout.add(std::make_unique<AudioParameterFloat>(hpfFreqParamID, "Output HPF",
            NormalisableRange{20.0f, 2000.0f, 0.0f, 0.25452f}, 20.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) {
                    return v >= 1000.0f ? String(v * 0.001f, 1) + " kHz" : String(v, 1) + " Hz";
                })));

        layout.add(std::make_unique<AudioParameterFloat>(lpfFreqParamID, "Output LPF",
            NormalisableRange{200.0f, 20000.0f, 0.0f, 0.32198f}, 20000.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) {
                    return v >= 1000.0f ? String(v * 0.001f, 1) + " kHz" : String(v, 1) + " Hz";
                })));

        layout.add(std::make_unique<AudioParameterFloat>(mixParamID, "Mix",
            NormalisableRange{0.0f, 100.0f}, 35.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 0) + " %"; })));

        layout.add(std::make_unique<AudioParameterFloat>(gainParamID, "Output Gain",
            NormalisableRange{-24.0f, 12.0f}, 0.0f,
            Attrs().withStringFromValueFunction(
                [](float v, int) { return String(v, 1) + " dB"; })));

        layout.add(std::make_unique<AudioParameterInt>(bitsParamID, "Bit Depth", 4, 32, 32));

        layout.add(std::make_unique<AudioParameterBool>(bypassParamID, "Bypass", false));

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
    [[nodiscard]] float getDelayMs() const noexcept { return delayParam != nullptr ? delayParam->get() : 375.0f; }
    [[nodiscard]] float getHPFFreq() const noexcept { return hpfFreq; }
    [[nodiscard]] float getLPFFreq() const noexcept { return lpfFreq; }
    [[nodiscard]] float getMix() const noexcept { return mix; }
    [[nodiscard]] float getDrive() const noexcept { return drive; }
    [[nodiscard]] double getSampleRate() const noexcept { return sampleRate; }
    [[nodiscard]] bool getBypass() const noexcept { return bypassParam != nullptr && bypassParam->get(); }
    [[nodiscard]] AudioProcessorParameter *getBypassParameter() const noexcept { return bypassParam; }

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

    [[nodiscard]] float getRawMix() const noexcept { return mixParam ? mixParam->get() : 35.0f; }
    [[nodiscard]] int getRawBits() const noexcept { return bitsParam ? bitsParam->get() : 32; }
    [[nodiscard]] float getRawHpfHz() const noexcept { return hpfParam ? hpfParam->get() : 20.0f; }
    [[nodiscard]] float getRawLpfHz() const noexcept { return lpfParam ? lpfParam->get() : 20000.0f; }

    [[nodiscard]] float getRawFeedback() const noexcept { return feedbackParam ? feedbackParam->get() : 0.42f; }
    [[nodiscard]] float getRawDampHz() const noexcept { return dampHzParam ? dampHzParam->get() : 8000.0f; }
    [[nodiscard]] float getRawLoopCutHz() const noexcept { return loopCutHzParam ? loopCutHzParam->get() : 40.0f; }
    [[nodiscard]] float getRawCrossFeed() const noexcept { return crossFeedParam ? crossFeedParam->get() : 0.0f; }
    [[nodiscard]] float getRawLoopDrive() const noexcept
    {
        return loopDriveParam ? Decibels::decibelsToGain(loopDriveParam->get()) : 1.0f;
    }
    [[nodiscard]] int getRawLoopSatOrder() const noexcept { return loopSatOrderParam ? loopSatOrderParam->getIndex() : 2; }
    [[nodiscard]] float getRawDelayModDepth() const noexcept { return delayModDepthParam ? delayModDepthParam->get() : 0.0f; }
    [[nodiscard]] float getRawDelayModRateHz() const noexcept { return delayModRateHzParam ? delayModRateHzParam->get() : 0.35f; }
    [[nodiscard]] bool getRawDelaySync() const noexcept { return delaySyncParam != nullptr && delaySyncParam->get(); }
    [[nodiscard]] int getRawDelayDivision() const noexcept { return delayDivisionParam ? delayDivisionParam->getIndex() : 11; }
    [[nodiscard]] float getRawDiffusion() const noexcept { return diffusionParam ? diffusionParam->get() : 0.55f; }
    [[nodiscard]] float getRawDiffuserSize() const noexcept { return diffuserSizeParam ? diffuserSizeParam->get() : 0.5f; }
    [[nodiscard]] float getRawDiffModDepth() const noexcept { return diffModDepthParam ? diffModDepthParam->get() : 0.30f; }
    [[nodiscard]] float getRawDiffModRateHz() const noexcept { return diffModRateHzParam ? diffModRateHzParam->get() : 0.35f; }
    [[nodiscard]] bool getRawEnableDiffuser() const noexcept { return enableDiffuserParam != nullptr && enableDiffuserParam->get(); }

    static constexpr float minDelayTime = 1.0f;
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
    AudioParameterBool *delaySyncParam{};
    AudioParameterChoice *delayDivisionParam{};
    AudioParameterFloat *hpfParam{};
    AudioParameterFloat *lpfParam{};
    AudioParameterFloat *mixParam{};
    AudioParameterBool *bypassParam{};
    AudioParameterFloat *driveParam{};
    AudioParameterChoice *adaaOrderParam{};
    AudioParameterFloat *feedbackParam{};
    AudioParameterFloat *dampHzParam{};
    AudioParameterFloat *loopCutHzParam{};
    AudioParameterFloat *crossFeedParam{};
    AudioParameterFloat *loopDriveParam{};
    AudioParameterChoice *loopSatOrderParam{};
    AudioParameterFloat *delayModDepthParam{};
    AudioParameterFloat *delayModRateHzParam{};
    AudioParameterBool *enableDiffuserParam{};
    AudioParameterFloat *diffusionParam{};
    AudioParameterFloat *diffuserSizeParam{};
    AudioParameterFloat *diffModDepthParam{};
    AudioParameterFloat *diffModRateHzParam{};

    LinearSmoothedValue<float> gainSmoother;
    LinearSmoothedValue<float> bitsSmoother;
    LinearSmoothedValue<float> hpfSmoother;
    LinearSmoothedValue<float> lpfSmoother;
    LinearSmoothedValue<float> mixSmoother;
    LinearSmoothedValue<float> driveSmoother;

    double sampleRate{};
};
#endif

#pragma once

#include <JuceHeader.h>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "dsp/StateVariable.h"
#include "dsp/SimdDelayLine.h"
#include "dsp/align/SaturatorAlign.h"
#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"
#include "ChronosParameters.h"
//==============================================================================
class ChronosProcessor final : public AudioProcessor
{
public:
    //==============================================================================
    ChronosProcessor();
    ~ChronosProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

protected:
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

public:
    void processBlock (AudioBuffer<float>&, MidiBuffer&) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;
    AudioProcessorParameter* getBypassParameter() const override { return parameters.getBypassParameter(); }

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const String getProgramName (int index) override;
    void changeProgramName (int index, const String& newName) override;

    //==============================================================================
    void getStateInformation (MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

private:
    AudioProcessorValueTreeState apvts {*this, nullptr, "Parameters", ChronosParameters::createParameterLayout()};
    ChronosParameters parameters {apvts};

    uint32_t xorshiftL;
    uint32_t xorshiftR;

    static float nextUniform(uint32_t& state) noexcept;

    MarsDSP::Delays::SimdDelayLine delayLine;
    std::vector<float> wetBufL_;
    std::vector<float> wetBufR_;
    int wetBufCapacity_ {0};

    using SVF = MarsDSP::Filters::SimdSVF;
    SVF hpf;
    SVF lpf;
    static constexpr double svfQ { 0.7071 };

    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1L_;
    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1R_;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2L_;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2R_;

    MarsDSP::Align::SaturatorAlign alignL_;
    MarsDSP::Align::SaturatorAlign alignR_;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosProcessor)
};

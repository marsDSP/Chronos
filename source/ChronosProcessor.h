#pragma once

#include <JuceHeader.h>
#include <array>
#include <cmath>
#include <random>

#include "dsp/SVF.h"
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

    // xorshift32 -> uniform float in [0, 1), using the high 24 bits
    static float nextUniform(uint32_t& state) noexcept;

    dsp::DelayLine<float> delayLine;

    // wet-path tone shaping: HPF then LPF on the delay taps only (dry is summed unfiltered).
    // One SimdSVF per filter type. The stereo pair is packed into lanes 0,1.
    // a single processSample(M128) advances both channels at once.
    using SVF = MarsDSP::Filters::SimdSVF;
    SVF hpf;
    SVF lpf;
    static constexpr double svfQ { 0.7071 }; // Butterworth, maximally flat passband
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosProcessor)
};

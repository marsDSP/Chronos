#pragma once

#include <JuceHeader.h>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "dsp/StateVariable.h"
#include "dsp/SimdDelayLine.h"
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

    // xorshift32 -> uniform float in [0, 1), using the high 24 bits
    static float nextUniform(uint32_t& state) noexcept;

    MarsDSP::Delays::SimdDelayLine delayLine;
    std::vector<float> wetBufL_;
    std::vector<float> wetBufR_;

    // wet-path tone shaping: HPF then LPF on the delay taps only (dry is summed unfiltered).
    // One SimdSVF per filter type. The stereo pair is packed into lanes 0,1.
    // a single processSample(M128) advances both channels at once.
    using SVF = MarsDSP::Filters::SimdSVF;
    SVF hpf;
    SVF lpf;
    static constexpr double svfQ { 0.7071 }; // Butterworth, maximally flat passband

    // Wet-path tanh saturator. ADAA1 (first-order) and ADAA2 (second-order)
    // antiderivative antialiasing, two instances per order for stereo. Scalar
    // double internally; no M128 overload (the double requirement halves lane
    // width and the branch hierarchy diverges per lane). ADAA2 is the
    // production path; ADAA1 exists for the alias_check A/B comparison and is
    // not a release-quality path (half-sample latency, see AGENTS.md).
    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1L_, adaa1R_;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2L_, adaa2R_;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosProcessor)
};

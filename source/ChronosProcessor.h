#pragma once

#include <JuceHeader.h>
#include <tracy/Tracy.hpp>
#include "dsp/engine/delay/delay_engine.h"
#include "utils/helpers/overload.h"
//==============================================================================
class ChronosProcessor final : public AudioProcessor {
public:
    //==============================================================================
    ChronosProcessor();
    ~ChronosProcessor() override;
    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout &layouts) const override;
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
    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const String getProgramName (int index) override;
    void changeProgramName (int index, const String& newName) override;
    //==============================================================================
    void getStateInformation (MemoryBlock &destData) override;
    void setStateInformation (const void *data, int sizeInBytes) override;

    AudioProcessorValueTreeState apvts;

private:
    MarsDSP::DSP::DelayEngine<float> delay;

    static AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Xorshift32 PRNG
    uint32_t xorshiftL;
    uint32_t xorshiftR;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosProcessor)
};

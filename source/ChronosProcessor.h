#pragma once

#include <JuceHeader.h>

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
    AudioProcessorValueTreeState apvts {*this, nullptr, "Parameters", createParameterLayout()};
    static AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    uint32_t xorshiftL;
    uint32_t xorshiftR;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosProcessor)
};

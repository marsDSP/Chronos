#pragma once

#include "Parameters.h"
#include "DSP/ProcessDSP.h"

class PluginProcessor final : public juce::AudioProcessor,
                              public juce::AudioProcessorValueTreeState::Listener
{
public:

    PluginProcessor();
    ~PluginProcessor() override = default;

    juce::AudioProcessorValueTreeState vts
    { *this, nullptr, "PARAMETERS", MarsDSP::Parameters::createParameterLayout() };

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout &layouts) const override;

    void processBlock(juce::AudioBuffer<float> &, juce::MidiBuffer &) override;

    juce::AudioProcessorEditor *createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String &newName) override;

    void getStateInformation(juce::MemoryBlock &destData) override;
    void setStateInformation(const void *data, int sizeInBytes) override;

private:

    MarsDSP::Parameters params;
    MarsDSP::DSP::ProcessBlock processDSP;

    void updateParameters();
    void parameterChanged(const juce::String& parameterID, float newValue) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

#pragma once

#include <JuceHeader.h>
#include <random>
#include "dsp/ChronosEngine.h"
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
    void reset() override;

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

    MarsDSP::ChronosEngine engine;

    /// Bring a stored state up to the current schema version.
    void migrateState_ (ValueTree& state, int fromVersion);

    // The last known host BPM. Held when the host gives no tempo.
    double cachedBpm_ = 120.0;

    // Name of the single factory program (returned via getProgramName).
    String programName_ { "Init" };

    // Compute the delay in samples, using tempo sync when enabled.
    float computeDelaySamples_() const;

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosProcessor)
};

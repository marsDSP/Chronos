#pragma once

#include <JuceHeader.h>
#include <random>
#include "dsp/ChronosEngine.h"
#include "ChronosParameters.h"
#include "utils/memory/SpscFifo.h"
#include "gui/tap/TapFeedFrame.h"
#include "presets/PresetManager.h"

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

    AudioProcessorValueTreeState& getAPVTS() noexcept { return apvts; }
    const AudioProcessorValueTreeState& getAPVTS() const noexcept { return apvts; }
    ChronosParameters& getParameters() noexcept { return parameters; }
    const ChronosParameters& getParameters() const noexcept { return parameters; }
    MarsDSP::Presets::PresetManager& getPresetManager() noexcept { return presetManager_; }

    [[nodiscard]] MarsDSP::Memory::SpscFifo<MarsDSP::GUI::TapFeedFrame, 256>& getTapFifo() noexcept { return tapFifo_; }

    [[nodiscard]] double getCachedBpm() const noexcept { return cachedBpm_.load(std::memory_order_relaxed); }

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
    MarsDSP::Memory::SpscFifo<MarsDSP::GUI::TapFeedFrame, 256> tapFifo_;

    // The preset layer owns the identity and the change flag.
    MarsDSP::Presets::PresetManager presetManager_ { *this, apvts };

    /// Bring a stored state up to the current schema version.
    void migrateState_ (ValueTree& state, int fromVersion);

    // The last known host BPM. Held when the host gives no tempo.
    std::atomic<double> cachedBpm_{120.0};

    // Name of the single factory program (returned via getProgramName).
    String programName_ { "Init" };

    // Compute the delay pair in samples, using tempo sync when enabled.
    std::pair<float, float> computeDelaySamples_() const;

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosProcessor)
};

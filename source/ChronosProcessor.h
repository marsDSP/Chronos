#pragma once

#include <JuceHeader.h>
#include <tracy/Tracy.hpp>
#include "dsp/engine/delay/delay_engine.h"
#include "utils/helpers/overload.h"
#include "utils/data/metering_frame.h"
#include "utils/memory/spsc_queue.h"
//==============================================================================
class ChronosProcessor final : public AudioProcessor {
public:
    //==============================================================================
    // Capacity for the DSP -> UI metering queue. ~512 frames is enough
    // headroom for the editor's 30 Hz timer to drain even at very small
    // block sizes (~1 ms blocks at 48 kHz) without ever dropping a frame.
    static constexpr std::size_t kMeteringQueueCapacity = 512;
    using MeteringQueue = Chronos::Concurrency::SpscQueue<Chronos::MeteringFrame>;

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

    // DSP -> UI metering interlink. processBlock pushes a single
    // MeteringFrame per audio block on the audio thread (try_enqueue,
    // never blocks, never allocates). The editor drains it from the
    // message thread on a juce::Timer tick.
    [[nodiscard]] MeteringQueue& getMeteringQueue() noexcept { return meteringQueue; }

private:
    MarsDSP::DSP::DelayEngine<float> delay;

    static AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Single-producer / single-consumer queue for DSP -> UI metering
    // frames. Producer = audio thread (processBlock); consumer = the
    // editor's timer callback on the message thread.
    MeteringQueue meteringQueue { kMeteringQueueCapacity };

    // Monotonic block counter, stamped onto each pushed MeteringFrame
    // so the UI can detect dropped frames if the queue ever overflows.
    std::uint64_t meteringBlockCounter = 0;

    // Xorshift32 PRNG
    uint32_t xorshiftL;
    uint32_t xorshiftR;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosProcessor)
};

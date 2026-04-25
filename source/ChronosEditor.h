#pragma once

#include "ChronosProcessor.h"
#include "gui/ChronosKnob.h"

//==============================================================================
// Grouped editor for Chronos. Sections are colour-coded:
//   - Delay         (green)
//   - Reverb        (blue)
//   - Wow & Flutter (purple)
//   - Ducker        (orange)
// A top header strip carries the master controls (Bypass / Mono /
// Sync toggle / Sync interval).
//
// The editor is also the consumer end of the DSP -> UI metering SPSC
// (see ChronosProcessor::getMeteringQueue). A juce::Timer drains the
// queue ~30 times per second from the message thread; the latest
// drained values are cached for paint() and any future meter widgets.
//==============================================================================
class ChronosEditor final : public juce::AudioProcessorEditor,
                            private juce::Timer
{
public:
    explicit ChronosEditor(ChronosProcessor&);
    ~ChronosEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    ChronosProcessor& proc;

    // Latest values pulled from the metering SPSC. Read on the message
    // thread only (timerCallback / paint), so no synchronisation needed
    // beyond the queue itself.
    float latestPeakLeft         = 0.0f;
    float latestPeakRight        = 0.0f;
    float latestDuckerGain       = 1.0f;
    std::uint64_t latestBlockIdx = 0;

    // Header
    juce::Label titleLabel;
    juce::Label syncIntervalLabel;

    // Section frames
    juce::GroupComponent delayGroup  { "delay",  "DELAY"  };
    juce::GroupComponent reverbGroup { "reverb", "REVERB" };
    juce::GroupComponent modGroup    { "mod",    "WOW & FLUTTER" };
    juce::GroupComponent duckerGroup { "ducker", "DUCKER" };

    // Delay section
    Chronos::ChronosKnob delayTimeKnob, mixKnob, feedbackKnob;
    Chronos::ChronosKnob lowCutKnob, highCutKnob, crossfeedKnob;

    // Reverb section
    Chronos::ChronosKnob revMixKnob, revRoomKnob, revDecayKnob, revPredelayKnob, revDiffKnob;
    Chronos::ChronosKnob revBuildKnob, revModKnob, revHfDampKnob, revLfDampKnob;
    juce::ToggleButton   revBypassButton { "Bypass" };

    // Wow / Flutter section
    Chronos::ChronosKnob wowRateKnob, wowDepthKnob, wowDriftKnob;
    Chronos::ChronosKnob flutterRateKnob, flutterDepthKnob;
    juce::ToggleButton   flutterEnableButton { "Flutter" };

    // Ducker section
    Chronos::ChronosKnob duckThreshKnob, duckAmountKnob, duckAttackKnob, duckReleaseKnob;
    juce::ToggleButton   duckerBypassButton { "Bypass" };

    // Header / master controls
    juce::ToggleButton bypassButton { "Bypass" };
    juce::ToggleButton monoButton   { "Mono"   };
    juce::ToggleButton syncButton   { "Sync"   };
    juce::ComboBox     syncIntervalCombo;

    using BtnAtt = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using BoxAtt = juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    std::unique_ptr<BtnAtt> bypassAtt, monoAtt, syncAtt;
    std::unique_ptr<BtnAtt> revBypassAtt, flutterAtt, duckerBypassAtt;
    std::unique_ptr<BoxAtt> syncIntervalAtt;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChronosEditor)
};

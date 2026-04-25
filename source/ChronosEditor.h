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
//==============================================================================
class ChronosEditor final : public juce::AudioProcessorEditor
{
public:
    explicit ChronosEditor(ChronosProcessor&);
    ~ChronosEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    ChronosProcessor& proc;

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

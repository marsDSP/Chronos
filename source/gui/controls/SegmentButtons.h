#pragma once

#ifndef CHRONOS_SEGMENT_BUTTONS_H
#define CHRONOS_SEGMENT_BUTTONS_H

#include <JuceHeader.h>
#include "../AccentConsumer.h"
#include "../Metrics.h"
#include "ConsoleButton.h"
#include <atomic>
#include <memory>
#include <vector>

namespace MarsDSP::GUI {

// A mutually exclusive row of segment buttons bound to a choice parameter.
class SegmentButtons : public Component,
                       private AudioProcessorValueTreeState::Listener,
                       private Timer,
                       public AccentConsumer {
public:
    // Bind to a choice parameter. Set coreLinked to follow the delay core accent.
    // items hold the APVTS choice strings. displayLabels name the buttons. tooltips
    // give each button a help string. Pass empty arrays to ignore them.
    SegmentButtons(AudioProcessorValueTreeState& apvts, const String& paramID,
                   const StringArray& items, Colour accent, bool coreLinked = false,
                   const StringArray& displayLabels = {}, const StringArray& tooltips = {});
    ~SegmentButtons() override;

    // Set the active segment fill and text colours.
    void setAccentColours(Colour activeBg, Colour activeText);

    // Set the same tooltip on every button.
    void setTooltip(const String& text);

    // Store the accent colour and repaint (AccentConsumer).
    void setAccentColour(Colour c) override;

    // Store the scale metrics and relayout.
    void setMetrics(const Metrics& m);

    void resized() override;

private:
    void parameterChanged(const String& parameterID, float newValue) override;
    void timerCallback() override;
    void syncButtons();

    ComboBox combo_;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> attach_;
    std::vector<std::unique_ptr<ConsoleButton>> buttons_;
    Colour accent_;
    bool coreLinked_ = false;
    String paramID_;
    AudioProcessorValueTreeState& apvts_;
    Metrics metrics_;

    // The audio thread stores the choice index here. A timer polls this
    // value and syncs the buttons on the message thread.
    std::atomic<float> pendingValue_ { -1.0f };
    float lastAppliedValue_ { -2.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SegmentButtons)
};

// Anti-alias mode labels and tooltips. The APVTS choice strings do not change.
inline const StringArray kAntiAliasLabels   { "OFF", "LOW", "HIGH" };
inline const StringArray kAntiAliasTooltips {
    "The saturator runs raw. This setting costs the least CPU and gives the most edge.",
    "The saturator suppresses some alias tones. This setting balances clarity and CPU.",
    "The saturator suppresses the most alias tones. This setting costs the most CPU."
};

} // namespace MarsDSP::GUI

#endif

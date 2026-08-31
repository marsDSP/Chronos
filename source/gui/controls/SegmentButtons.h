#pragma once

#ifndef CHRONOS_SEGMENT_BUTTONS_H
#define CHRONOS_SEGMENT_BUTTONS_H

#include <JuceHeader.h>
#include "../AccentConsumer.h"
#include "ConsoleButton.h"
#include <memory>
#include <vector>

namespace MarsDSP::GUI {

// A mutually exclusive row of segment buttons bound to a choice parameter.
class SegmentButtons : public Component,
                       private AudioProcessorValueTreeState::Listener,
                       public AccentConsumer {
public:
    // Bind to a choice parameter. Set coreLinked to follow the delay core accent.
    SegmentButtons(AudioProcessorValueTreeState& apvts, const String& paramID,
                   const StringArray& items, Colour accent, bool coreLinked = false);
    ~SegmentButtons() override;

    // Set the active segment fill and text colours.
    void setAccentColours(Colour activeBg, Colour activeText);

    // Store the accent colour and repaint (AccentConsumer).
    void setAccentColour(Colour c) override;

    void resized() override;

private:
    void parameterChanged(const String& parameterID, float newValue) override;
    void syncButtons();

    ComboBox combo_;
    std::unique_ptr<AudioProcessorValueTreeState::ComboBoxAttachment> attach_;
    std::vector<std::unique_ptr<ConsoleButton>> buttons_;
    Colour accent_;
    bool coreLinked_ = false;
    String paramID_;
    AudioProcessorValueTreeState& apvts_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SegmentButtons)
};

} // namespace MarsDSP::GUI

#endif

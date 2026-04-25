#pragma once

#ifndef CHRONOS_CHRONOSKNOB_H
#define CHRONOS_CHRONOSKNOB_H

#include <JuceHeader.h>
#include "ChronosColours.h"

namespace Chronos
{
    // Compact rotary knob with a label on top and an inline value text-box
    // below the dial. Wraps a juce::Slider and an APVTS::SliderAttachment
    // so the editor stays declarative.
    class ChronosKnob : public juce::Component
    {
    public:
        ChronosKnob(const juce::String& labelText,
                    juce::AudioProcessorValueTreeState& state,
                    const juce::ParameterID& pid,
                    juce::Colour arc = ChronosColours::accentGreen);

        ~ChronosKnob() override = default;

        void paint(juce::Graphics& g) override;
        void resized() override;

        juce::Slider& getSlider()                    { return slider; }
        void setLabelText(const juce::String& text)  { label.setText(text, juce::dontSendNotification); }
        void setArcColour(juce::Colour c);

    private:
        juce::Slider slider;
        juce::Label  label;
        juce::Colour arcColour;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChronosKnob)
    };
} // namespace Chronos

#endif

#include "ChronosKnob.h"

namespace Chronos
{
    ChronosKnob::ChronosKnob(const juce::String& labelText,
                             juce::AudioProcessorValueTreeState& state,
                             const juce::ParameterID& pid,
                             juce::Colour arc)
        : arcColour(arc)
    {
        // Slider: rotary, value text directly under the dial.
        slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 64, 14);
        slider.setRotaryParameters(juce::MathConstants<float>::pi * 1.25f,
                                   juce::MathConstants<float>::pi * 2.75f, true);

        slider.setColour(juce::Slider::rotarySliderFillColourId,    arcColour);
        slider.setColour(juce::Slider::rotarySliderOutlineColourId, ChronosColours::knobTrack);
        slider.setColour(juce::Slider::thumbColourId,               arcColour);
        slider.setColour(juce::Slider::textBoxTextColourId,         ChronosColours::textPrimary);
        slider.setColour(juce::Slider::textBoxOutlineColourId,      juce::Colours::transparentBlack);
        slider.setColour(juce::Slider::textBoxBackgroundColourId,   juce::Colours::transparentBlack);
        addAndMakeVisible(slider);

        // Label sits above the dial in bright text.
        label.setText(labelText, juce::dontSendNotification);
        label.setJustificationType(juce::Justification::centred);
        label.setColour(juce::Label::textColourId, ChronosColours::textBright);
        label.setFont(juce::Font(juce::FontOptions(11.0f, juce::Font::bold)));
        label.setInterceptsMouseClicks(false, false);
        addAndMakeVisible(label);

        // Bind to the APVTS parameter and wire up double-click-to-default.
        attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            state, pid.getParamID(), slider);

        if (const auto* param = state.getParameter(pid.getParamID()))
            slider.setDoubleClickReturnValue(true, param->convertFrom0to1(param->getDefaultValue()));
    }

    void ChronosKnob::paint(juce::Graphics& g)
    {
        juce::ignoreUnused(g);
    }

    void ChronosKnob::resized()
    {
        constexpr int labelHeight = 16;
        auto bounds = getLocalBounds();
        label.setBounds(bounds.removeFromTop(labelHeight));
        slider.setBounds(bounds);
    }

    void ChronosKnob::setArcColour(juce::Colour c)
    {
        arcColour = c;
        slider.setColour(juce::Slider::rotarySliderFillColourId, c);
        slider.setColour(juce::Slider::thumbColourId,            c);
        repaint();
    }
} // namespace Chronos

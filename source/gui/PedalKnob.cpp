#include "PedalKnob.h"

namespace MarsDSP::GUI::Knobs {

PDLKnob::PDLKnob(const String &labelText,
                 AudioProcessorValueTreeState &state,
                 const ParameterID &pid,
                 const Colour arc)
    : arcColour(arc)
{
    slider.setSliderStyle(Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(Slider::NoTextBox, true, 0, 0);
    slider.setLookAndFeel(&lnf);
    addAndMakeVisible(slider);

    label.setText(labelText, dontSendNotification);
    label.setJustificationType(Justification::centred);
    label.setColour(Label::textColourId, Colours::textDim);
    label.setFont(Font(FontOptions(10.0f)).boldened());
    addAndMakeVisible(label);

    attachment = std::make_unique<AudioProcessorValueTreeState::SliderAttachment>(state, pid.getParamID(), slider);
}

PDLKnob::~PDLKnob()
{
    slider.setLookAndFeel(nullptr);
}

void PDLKnob::setArcColour(const Colour c)
{
    arcColour = c;
    repaint();
}

void PDLKnob::paint(Graphics &g)
{
    ignoreUnused(g);
}

void PDLKnob::paintOverChildren(Graphics &g)
{
    ignoreUnused(g);
}

void PDLKnob::resized()
{
    auto bounds = getLocalBounds();
    constexpr int labelHeight = 14;

    label.setBounds(bounds.removeFromTop(labelHeight));

    const auto size = std::min(bounds.getWidth(), bounds.getHeight());
    slider.setBounds(bounds.withSizeKeepingCentre(size, size));
}

} // namespace MarsDSP::GUI::Knobs

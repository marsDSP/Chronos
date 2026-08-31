#include "PedalKnob.h"

namespace MarsDSP::GUI::Knobs {

PDLKnob::PDLKnob(const String &labelText,
                 AudioProcessorValueTreeState &state,
                 const ParameterID &pid,
                 const Colour arc)
    : arcColour(arc), labelText_(labelText)
{
    slider.setSliderStyle(Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(Slider::NoTextBox, true, 0, 0);
    slider.setLookAndFeel(&lnf);
    addAndMakeVisible(slider);

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
    const auto m = currentMetrics();
    const float availW = static_cast<float>(getWidth()) - m.pxf(2.0f);
    const float baseH = m.font(11.0f);
    const int labelBandH = m.px(Metrics::kLabelBandH);
    const Rectangle<float> labelArea = getLocalBounds().removeFromTop(labelBandH).toFloat();

    auto drawCentered = [&](const String& t, const Font& font)
    {
        g.setFont(font);
        const float by = labelArea.getCentreY() + (font.getAscent() - font.getDescent()) * 0.5f;
        g.drawSingleLineText(t, juce::roundToInt(labelArea.getCentreX()), juce::roundToInt(by),
                              Justification::horizontallyCentred);
    };

    g.setColour(labelColour_);

    Font f = Fonts::font(Fonts::Weight::Medium, baseH);
    String text = labelText_;

    // Step 1: full label.
    if (f.getStringWidthFloat(text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 2: short form.
    text = Fonts::shortLabel(labelText_);
    if (f.getStringWidthFloat(text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 3: reduce the font height by up to 15%.
    f = Fonts::font(Fonts::Weight::Medium, baseH * 0.85f);
    if (f.getStringWidthFloat(text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 4: wrap the full label to two lines.
    f = Fonts::font(Fonts::Weight::Medium, baseH * 0.85f);
    g.setFont(f);
    const int sep = labelText_.indexOf(" ");
    if (sep >= 0)
    {
        const String line1 = labelText_.substring(0, sep);
        const String line2 = labelText_.substring(sep + 1);
        const float by1 = labelArea.getY() + f.getAscent();
        const float by2 = labelArea.getBottom() - f.getDescent();
        g.drawSingleLineText(line1, juce::roundToInt(labelArea.getCentreX()), juce::roundToInt(by1),
                              Justification::horizontallyCentred);
        g.drawSingleLineText(line2, juce::roundToInt(labelArea.getCentreX()), juce::roundToInt(by2),
                              Justification::horizontallyCentred);
    }
    else
    {
        drawCentered(text, f);
    }
}

void PDLKnob::resized()
{
    auto bounds = getLocalBounds();
    const auto m = currentMetrics();
    const int labelHeight = m.px(static_cast<float>(Metrics::kLabelBandH));
    const int knobLabelGap = m.px(static_cast<float>(Metrics::kKnobLabelGap));

    bounds.removeFromTop(labelHeight);
    bounds.removeFromTop(knobLabelGap);

    const auto size = std::min(bounds.getWidth(), bounds.getHeight());
    slider.setBounds(bounds.withSizeKeepingCentre(size, size));
}

} // namespace MarsDSP::GUI::Knobs

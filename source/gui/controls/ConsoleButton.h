#pragma once

#ifndef CHRONOS_CONSOLE_BUTTON_H
#define CHRONOS_CONSOLE_BUTTON_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "../Fonts.h"
#include "../Metrics.h"

namespace MarsDSP::GUI {

// A push or toggle button styled for console panels.
class ConsoleButton : public Button {
public:
    explicit ConsoleButton(const String& name) : Button(name)
    {
        setClickingTogglesState(true);
    }

    void setAccentColours(Colour activeBg, Colour activeText)
    {
        activeBg_ = activeBg;
        activeText_ = activeText;
        repaint();
    }

    void paintButton(Graphics& g,
                     const bool shouldDrawButtonAsHighlighted,
                     const bool shouldDrawButtonAsDown) override
    {
        ignoreUnused(shouldDrawButtonAsDown);

        const auto m = currentMetrics();
        const auto bounds = getLocalBounds().toFloat();
        const float corner = m.pxf(Metrics::kCornerSmall);
        const float sw = m.stroke(Metrics::kHairline);
        const bool isOn = getToggleState();

        if (isOn)
        {
            g.setColour(activeBg_);
            g.fillRoundedRectangle(bounds, corner);
            g.setColour(activeText_);
        }
        else
        {
            g.setColour(shouldDrawButtonAsHighlighted ? Colours::headerBackground : Colours::panelBackground);
            g.fillRoundedRectangle(bounds, corner);
            g.setColour(Colours::panelBorder);
            g.drawRoundedRectangle(bounds.reduced(sw / 2), corner, sw);
            g.setColour(shouldDrawButtonAsHighlighted ? Colours::textPrimary : Colours::textDim);
        }

        const auto font = Fonts::font(Fonts::Weight::Medium, m.font(10.0f));
        g.setFont(font);
        g.drawText(getButtonText().toUpperCase(), bounds, Justification::centred, false);
    }

private:
    Colour activeBg_ = Colours::panelBackground;
    Colour activeText_ = Colours::accentDelayDigital;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConsoleButton)
};

} // namespace MarsDSP::GUI

#endif

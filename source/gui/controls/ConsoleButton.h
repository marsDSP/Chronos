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

        const auto bounds = getLocalBounds().toFloat();
        const bool isOn = getToggleState();

        if (isOn)
        {
            g.setColour(activeBg_);
            g.fillRoundedRectangle(bounds, 4.0f);
            g.setColour(activeText_);
        }
        else
        {
            g.setColour(shouldDrawButtonAsHighlighted ? Colours::headerBackground : Colours::panelBackground);
            g.fillRoundedRectangle(bounds, 4.0f);
            g.setColour(Colours::panelBorder);
            g.drawRoundedRectangle(bounds.reduced(0.5f), 4.0f, 1.0f);
            g.setColour(shouldDrawButtonAsHighlighted ? Colours::textPrimary : Colours::textDim);
        }

        const auto font = Fonts::font(Fonts::Weight::Medium, currentMetrics().font(10.0f));
        g.setFont(font);
        g.drawText(getButtonText().toUpperCase(), bounds, Justification::centred, false);
    }

private:
    Colour activeBg_ = Colour(0xFF382015);
    Colour activeText_ = Colours::accentDelayDigital;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConsoleButton)
};

} // namespace MarsDSP::GUI

#endif

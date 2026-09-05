#pragma once

#ifndef CHRONOS_CARD_H
#define CHRONOS_CARD_H

#include <JuceHeader.h>
#include "SubTabStrip.h"
#include "Colours.h"
#include "Metrics.h"
#include "AccentConsumer.h"
#include "EnablementConsumer.h"
#include <memory>
#include <vector>

namespace MarsDSP::GUI {

// A rounded card that holds a subtab strip and swappable content panels.
class Card : public Component {
public:
    Card();
    ~Card() override = default;

    // Set the accent colour for the card border and the subtab strip.
    void setAccentColour(Colour c);

    // Set the scale metrics for the card layout.
    void setMetrics(const Metrics& m);

    // Push the enablement state to every content panel that reads it.
    void setEnablement(const EnablementState& state);

    // Add one content panel under a subtab title.
    void addContent(const String& tabName, std::unique_ptr<Component> panel);

    // Set the visible content panel index.
    void setSelectedContent(int index);

    void paint(Graphics& g) override;
    void resized() override;

private:
    SubTabStrip subTabs_;
    std::vector<std::unique_ptr<Component>> contents_;
    Colour accent_ { Colours::accentDelayDigital };
    Metrics metrics_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Card)
};

} // namespace MarsDSP::GUI

#endif

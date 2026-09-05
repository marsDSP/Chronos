#pragma once

#ifndef CHRONOS_CARD_H
#define CHRONOS_CARD_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"
#include "AccentConsumer.h"
#include "EnablementConsumer.h"
#include <memory>

namespace MarsDSP::GUI {

// A rounded card with a painted title row and one content child.
// The card border keeps the tint law. Accent, metrics, and enablement
// push to the content child.
class Card : public Component {
public:
    explicit Card(const String& title);
    ~Card() override = default;

    // Set the content child. Replaces any prior child.
    void setContent(std::unique_ptr<Component> panel);

    // Set the accent colour for the card border and the content child.
    void setAccentColour(Colour c);

    // Set the scale metrics for the card layout and the content child.
    void setMetrics(const Metrics& m);

    // Push the enablement state to the content child when it reads it.
    void setEnablement(const EnablementState& state);

    void paint(Graphics& g) override;
    void resized() override;

private:
    String title_;
    std::unique_ptr<Component> content_;
    Colour accent_ { Colours::accentDelayDigital };
    Metrics metrics_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Card)
};

} // namespace MarsDSP::GUI

#endif

#pragma once

#ifndef CHRONOS_RAIL_H
#define CHRONOS_RAIL_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"
#include "AccentConsumer.h"
#include "EnablementConsumer.h"
#include <memory>

namespace MarsDSP::GUI {

// The output rail band. It paints the card frame law with no title row
// and owns one content child. Accent, metrics, and enablement push to
// the child.
class Rail : public Component {
public:
    Rail() = default;
    ~Rail() override = default;

    // Set the content child. Replaces any prior child.
    void setPanel(std::unique_ptr<Component> panel);

    // Set the accent colour for the rail border and the content child.
    void setAccentColour(Colour c);

    // Set the scale metrics for the rail layout and the content child.
    void setMetrics(const Metrics& m);

    // Push the enablement state to the content child when it reads it.
    void setEnablement(const EnablementState& state);

    void paint(Graphics& g) override;
    void resized() override;

private:
    std::unique_ptr<Component> content_;
    Colour accent_ { Colours::accentDelayDigital };
    Metrics metrics_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Rail)
};

} // namespace MarsDSP::GUI

#endif

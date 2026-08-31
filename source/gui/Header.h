#pragma once

#ifndef CHRONOS_HEADER_H
#define CHRONOS_HEADER_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"
#include "controls/PowerButton.h"
#include <memory>

class ChronosProcessor;

namespace MarsDSP::GUI {

// The top header band: wordmark, delay core badge, and bypass power.
class Header : public Component {
public:
    explicit Header(ChronosProcessor& proc);
    ~Header() override = default;

    // Update the core badge text and fill for the delay mode.
    void setCoreMode(int mode, Colour accent);

    // Set the scale metrics for the header layout.
    void setMetrics(const Metrics& m);

    void paint(Graphics& g) override;
    void resized() override;

private:
    class CoreBadge : public Button {
    public:
        CoreBadge();
        // Set the mode index and the badge fill colour.
        void setMode(int mode, Colour accent);
        void paintButton(Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;
    private:
        int mode_ = 0;
        Colour accent_ { Colours::accentDelayDigital };
    };

    ChronosProcessor& processorRef_;
    Metrics metrics_;
    Label wordmark_;
    Label subline_;
    CoreBadge badge_;
    PowerButton bypassButton_;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> bypassAttach_;

    void toggleDelayMode_();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Header)
};

} // namespace MarsDSP::GUI

#endif

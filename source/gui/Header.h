#pragma once

#ifndef CHRONOS_HEADER_H
#define CHRONOS_HEADER_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"
#include "controls/PowerButton.h"
#include "PresetBar.h"
#include <memory>

class ChronosProcessor;

namespace MarsDSP::GUI {

// The top header band: the wordmark and the bypass power.
class Header : public Component {
public:
    explicit Header(ChronosProcessor& proc);
    ~Header() override = default;

    // Set the scale metrics for the header layout.
    void setMetrics(const Metrics& m);

    // Store the live core accent for the bypass glyph.
    void setAccentColour(Colour c);

    void paint(Graphics& g) override;
    void resized() override;

private:
    ChronosProcessor& processorRef_;
    Metrics metrics_;
    Label wordmark_;
    PresetBar presetBar_;
    PowerButton bypassButton_;
    std::unique_ptr<AudioProcessorValueTreeState::ButtonAttachment> bypassAttach_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Header)
};

} // namespace MarsDSP::GUI

#endif

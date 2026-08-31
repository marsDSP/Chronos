#pragma once

#ifndef CHRONOS_FOOTER_H
#define CHRONOS_FOOTER_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"

class ChronosProcessor;

namespace MarsDSP::GUI {

// The bottom status footer: sample rate, BPM, and version.
class Footer : public Component,
               private Timer {
public:
    explicit Footer(ChronosProcessor& proc);
    ~Footer() override;

    // Set the scale metrics for the footer layout.
    void setMetrics(const Metrics& m);

    void paint(Graphics& g) override;
    void resized() override;

private:
    void timerCallback() override;
    void refreshText_();

    ChronosProcessor& processorRef_;
    Metrics metrics_;
    String statusText_;
    String versionText_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Footer)
};

} // namespace MarsDSP::GUI

#endif

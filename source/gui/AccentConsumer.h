#pragma once

#ifndef CHRONOS_ACCENT_CONSUMER_H
#define CHRONOS_ACCENT_CONSUMER_H

#include <JuceHeader.h>

namespace MarsDSP::GUI {

// Interface for components that receive the live core accent.
// The editor calls setAccentColour on every consumer.
struct AccentConsumer {
    virtual ~AccentConsumer() = default;

    // Store the accent colour and repaint.
    virtual void setAccentColour(Colour c) = 0;
};

} // namespace MarsDSP::GUI

#endif

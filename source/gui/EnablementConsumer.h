#pragma once

#ifndef CHRONOS_ENABLEMENT_CONSUMER_H
#define CHRONOS_ENABLEMENT_CONSUMER_H

namespace MarsDSP::GUI {

// The parameter states that make a control inert.
// The editor reads the parameters and pushes this state to every panel.
struct EnablementState {
    // The delay time comes from the host tempo. The time knobs do nothing.
    bool delaySync = false;
    // The left time drives both channels. The right time knob does nothing.
    bool timeLink = true;
    // The diffuser is off. The four diffuser knobs do nothing.
    bool enableDiffuser = false;
    // The output saturator is off. The drive knob does nothing.
    bool driveSatOff = false;
};

// The interface a panel with an inert control implements.
// The editor calls this through the card, as it does for accent and metrics.
class EnablementConsumer {
public:
    virtual ~EnablementConsumer() = default;

    // Apply the enablement state to the controls this panel owns.
    virtual void setControlsEnabled(const EnablementState& state) = 0;
};

} // namespace MarsDSP::GUI

#endif

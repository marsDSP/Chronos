#pragma once

#ifndef CHRONOS_TAP_DISPLAY_H
#define CHRONOS_TAP_DISPLAY_H

#include <JuceHeader.h>
#include "TapSimulation.h"

class ChronosProcessor;

namespace MarsDSP::GUI {

// Component that displays delay taps on a horizontal ruler.
// The top lane shows left taps.
// The bottom lane shows right taps.
class TapDisplay : public Component,
                   private Timer,
                   public AudioProcessorValueTreeState::Listener {
public:
    explicit TapDisplay(ChronosProcessor& processor);
    ~TapDisplay() override;

    void paint(Graphics& g) override;
    void resized() override;

    void parameterChanged(const String& parameterID, float newValue) override;

private:
    void timerCallback() override;

    struct DisplayTap {
        bool dry = false;
        float timeSeconds = 0.0f;
        float gain = 0.0f;
    };

    struct DisplayState {
        std::vector<DisplayTap> left;
        std::vector<DisplayTap> right;
        float totalTimeSeconds = 0.25f;
    };

    [[nodiscard]] TapSim::Parameters buildParameters_() const;
    [[nodiscard]] static DisplayState toDisplayState_(const TapSim::SimulationResult& sim);
    [[nodiscard]] static DisplayState blendDisplayState_(const DisplayState& current,
                                                        const DisplayState& target,
                                                        float blendAmount);
    [[nodiscard]] static DisplayState transitionDisplayState_(const DisplayState& current,
                                                             const DisplayState& target,
                                                             float deltaSeconds);
    void advanceDisplayState_(float deltaSeconds);

    ChronosProcessor& processorRef_;
    DisplayState displayState_;
    bool hasDisplayState_ = false;
    double lastTimeSecs_ = 0.0;
    int delayMode_ = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TapDisplay)
};

} // namespace MarsDSP::GUI

#endif

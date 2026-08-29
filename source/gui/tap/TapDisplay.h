#pragma once

#ifndef CHRONOS_TAP_DISPLAY_H
#define CHRONOS_TAP_DISPLAY_H

#include <JuceHeader.h>
#include "TapSimulation.h"

class ChronosProcessor;

namespace MarsDSP::GUI {

// Component that displays delay taps on a horizontal ruler.
// Dragging the upper half adjusts the left delay time.
// Dragging the lower half adjusts the right delay time.
class TapDisplay : public Component,
                   private Timer,
                   public AudioProcessorValueTreeState::Listener {
public:
    explicit TapDisplay(ChronosProcessor& processor);
    ~TapDisplay() override;

    void paint(Graphics& g) override;
    void resized() override;

    void mouseDown(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
    void mouseUp(const MouseEvent& e) override;
    void mouseMove(const MouseEvent& e) override;
    void mouseEnter(const MouseEvent& e) override;
    void mouseExit(const MouseEvent& e) override;

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

    // Drag and hover state
    enum class DragTarget { None, LeftTime, RightTime };
    DragTarget activeDragTarget_ = DragTarget::None;
    float dragStartX_ = 0.0f;
    float dragStartY_ = 0.0f;
    float startNormL_ = 0.0f;
    float startNormR_ = 0.0f;
    float startNormFb_ = 0.0f;
    int startDiv_ = 11;
    bool dragging_ = false;

    Point<float> hoverPos_{};
    bool isHovered_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TapDisplay)
};

} // namespace MarsDSP::GUI

#endif

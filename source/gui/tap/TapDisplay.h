#pragma once

#ifndef CHRONOS_TAP_DISPLAY_H
#define CHRONOS_TAP_DISPLAY_H

#include <JuceHeader.h>
#include "../Metrics.h"
#include "../Fonts.h"
#include "../Colours.h"
#include "TapSimulation.h"
#include "TapTracker.h"
#include <vector>

class ChronosProcessor;

namespace MarsDSP::GUI {

// Component that displays delay taps on a horizontal ruler.
// Dragging the upper half adjusts the left delay time.
// Dragging the lower half adjusts the right delay time.
class TapDisplay : public Component,
                   public SettableTooltipClient,
                   private Timer {
public:
    explicit TapDisplay(ChronosProcessor& processor);
    ~TapDisplay() override;

    void paint(Graphics& g) override;
    void resized() override;
    void visibilityChanged() override;
    void parentHierarchyChanged() override;

    void mouseDown(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
    void mouseUp(const MouseEvent& e) override;
    void mouseMove(const MouseEvent& e) override;
    void mouseEnter(const MouseEvent& e) override;
    void mouseExit(const MouseEvent& e) override;
    void mouseDoubleClick(const MouseEvent& e) override;
    void mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& wheel) override;
    bool keyPressed(const KeyPress& key) override;

    // Set the scale metrics for the label fonts.
    void setMetrics(const Metrics& m);

    // Store the live core accent and repaint.
    void setAccentColour(Colour c);

private:
    void timerCallback() override;
    void updateTimerState_();

    [[nodiscard]] TapSim::Parameters buildParameters_() const;
    [[nodiscard]] bool paramsChanged_(const TapSim::Parameters& p) const;
    void runSimulation_(const TapSim::Parameters& p);

    ChronosProcessor& processorRef_;
    Metrics metrics_;
    TapTracker tracker_;
    bool hasState_ = false;
    TapSim::Parameters lastParams_ {};
    double lastTimeSecs_ = 0.0;
    Colour accentColour_ { Colours::accentDelayDigital };
    float currentWetLevelL_ = 0.0f;
    float currentWetLevelR_ = 0.0f;

    // Previous-frame state for the paint budget gate.
    Point<float> prevHoverPos_{};
    bool prevIsHovered_ = false;
    float prevWetLevelL_ = 0.0f;
    float prevWetLevelR_ = 0.0f;

    // Drag and hover state.
    float dragStartX_ = 0.0f;
    float dragStartY_ = 0.0f;
    float startNormL_ = 0.0f;
    float startNormR_ = 0.0f;
    float startNormFb_ = 0.0f;
    int startDiv_ = 11;
    bool dragging_ = false;
    // The axis latch. Zero until the dead zone clears. Then 1 for
    // horizontal (time) or 2 for vertical (feedback).
    int dragAxis_ = 0;

    // The drag latch stores the mode and link state at mouse-down.
    bool dragSynced_ = false;
    bool dragLinked_ = false;
    bool dragIsUpper_ = false;
    std::vector<RangedAudioParameter*> dragGestures_;

    Point<float> hoverPos_{};
    bool isHovered_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TapDisplay)
};

} // namespace MarsDSP::GUI

#endif

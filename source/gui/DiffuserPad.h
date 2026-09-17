#pragma once

#ifndef CHRONOS_DIFFUSER_PAD_H
#define CHRONOS_DIFFUSER_PAD_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"
#include "AccentConsumer.h"
#include "MetricsConsumer.h"
#include "animation/Animation.h"
#include <array>
#include <atomic>

namespace MarsDSP::GUI {

// The diffuser cube pad. Four parameters on one oblique wireframe cube.
// x (size) and y (diffusion) move the handle inside a slice plane, z
// (mod depth) slides the slice along the depth diagonal, and w (mod
// rate) is the arc around the handle. Drag addresses x/y; the wheel
// addresses z, and Alt hands it to w; the arrows address x/y, or z/w
// with Alt. The pad listens to its five parameters and repaints through
// an async update, so no work runs on the audio thread.
class DiffuserPad : public Component,
                    public SettableTooltipClient,
                    public AccentConsumer,
                    public MetricsConsumer,
                    private AudioProcessorValueTreeState::Listener,
                    private AsyncUpdater,
                    private Timer {
public:
    // The four axes in cube order.
    static constexpr int kX = 0;
    static constexpr int kY = 1;
    static constexpr int kZ = 2;
    static constexpr int kW = 3;
    static constexpr int kNumAxes = 4;

    DiffuserPad(AudioProcessorValueTreeState& apvts,
                const String& sizeID,
                const String& diffusionID,
                const String& depthID,
                const String& rateID,
                const String& enableID);
    ~DiffuserPad() override;

    void setAccentColour(Colour c) override;
    void setMetrics(const Metrics& m) override;

    void paint(Graphics& g) override;
    void resized() override;

    void mouseDown(const MouseEvent& e) override;
    void mouseDrag(const MouseEvent& e) override;
    void mouseUp(const MouseEvent& e) override;
    void mouseDoubleClick(const MouseEvent& e) override;
    void mouseMove(const MouseEvent& e) override;
    void mouseExit(const MouseEvent& e) override;
    void mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& wheel) override;
    bool keyPressed(const KeyPress& key) override;

    void enablementChanged() override;

    // The hint follows the pointer: one line for the cube, one per
    // readout cell. The full gesture guide stays in the help text.
    String getTooltip() override;

private:
    // One parameter binding per cube axis.
    struct AxisBinding {
        String id;
        RangedAudioParameter* param { nullptr };
        float dragStart = 0.0f;
        bool dragOpen = false;   // gesture opened at mouseDown
        bool wheelOpen = false;  // gesture opened by a wheel burst
        bool lit = false;        // the tripod arm or ring paints in the accent
    };

    // The cube geometry in pixels for the current bounds.
    struct CubeGeometry {
        Point<float> front;        // the front square's bottom-left corner
        float side = 0.0f;         // the front square side
        Point<float> depth;        // the depth vector, front to back
        Rectangle<float> readout;  // the readout band under the cube
    };

    void parameterChanged(const String& parameterID, float newValue) override;
    void handleAsyncUpdate() override;
    void timerCallback() override;

    [[nodiscard]] CubeGeometry cubeGeometry_() const noexcept;
    // The readout cell under a point, as an axis index, or -1.
    [[nodiscard]] int readoutAxisAt_(Point<float> p) const noexcept;
    [[nodiscard]] float value_(int axis) const noexcept;
    void setValue_(int axis, float v01);
    void stepValue_(int axis, float delta01);

    // Close the gestures a mouseDown opened.
    void closeDragGestures_();
    // Close any open wheel gesture burst and unlight the burst axes.
    // Return true when something changed and a repaint is due.
    bool endWheelGestures_();
    // The horizontal and vertical axes for a gesture, plain or Alt.
    static int horizontalAxis_(bool alt) noexcept { return alt ? kW : kX; }
    static int verticalAxis_(bool alt) noexcept { return alt ? kZ : kY; }

    AudioProcessorValueTreeState& apvts_;
    std::array<AxisBinding, kNumAxes> axes_;
    String enableID_;
    RangedAudioParameter* enableParam_ { nullptr };

    // The audio thread stores the latest enable value here. The async
    // update reads it on the message thread and retargets the fade.
    std::atomic<float> pendingEnable_ { -1.0f };
    bool lastEnableOn_ = false;

    Colour accent_ { Colours::accentDelayDigital };
    Metrics metrics_;

    // Drag state. mouseDown snapshots the start values and opens the
    // gestures for the whole drag. A press on the cube drags the x/y
    // pair; a press on a readout cell drags that one axis vertically
    // (valueDragAxis_ >= 0), stepping from the last pointer y so Shift
    // can switch to fine mid-drag without a jump.
    float dragStartX_ = 0.0f;
    float dragStartY_ = 0.0f;
    float lastDragY_ = 0.0f;
    bool dragging_ = false;
    int valueDragAxis_ = -1;
    // The readout cell under the pointer, or -1.
    int hoverAxis_ = -1;
    // The Shift axis latch. Zero until the dead zone clears. Then 1 for
    // horizontal or 2 for vertical.
    int shiftLatch_ = 0;

    bool hovered_ = false;

    // The time of the last wheel or key step. A burst closes once it
    // has been idle for kWheelGestureMs; the timer checks this stamp,
    // so a burst during the fade shares the fade's timer.
    uint32 lastBurstMs_ = 0;

    float fade_ = 0.0f;
    MarsDSP::Animation::Animation<float> fadeAnim_ { MarsDSP::Animation::kSlowTimeMs,
                                                     MarsDSP::Animation::kEaseInOut,
                                                     MarsDSP::Animation::kEaseInOut };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DiffuserPad)
};

} // namespace MarsDSP::GUI

#endif

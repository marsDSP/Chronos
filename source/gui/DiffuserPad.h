#pragma once

#ifndef CHRONOS_DIFFUSER_PAD_H
#define CHRONOS_DIFFUSER_PAD_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"
#include "AccentConsumer.h"
#include "MetricsConsumer.h"
#include <atomic>

namespace MarsDSP::GUI {

// A 2D pad that sets the diffusion and the diffuser size.
// Horizontal moves the size. Vertical moves the diffusion.
// The pad listens to its three parameters and repaints through an
// async update, so no work runs on the audio thread.
class DiffuserPad : public Component,
                    public SettableTooltipClient,
                    public AccentConsumer,
                    public MetricsConsumer,
                    private AudioProcessorValueTreeState::Listener,
                    private AsyncUpdater,
                    private Timer {
public:
    DiffuserPad(AudioProcessorValueTreeState& apvts,
                const String& diffusionID,
                const String& sizeID,
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

private:
    void parameterChanged(const String& parameterID, float newValue) override;
    void handleAsyncUpdate() override;
    void timerCallback() override;

    // Close any open wheel gesture burst.
    void endWheelGestures_();

    // The active area inside the pad inset.
    Rectangle<float> activeArea_() const noexcept;

    AudioProcessorValueTreeState& apvts_;
    RangedAudioParameter* diffusionParam_ { nullptr };
    RangedAudioParameter* sizeParam_       { nullptr };
    RangedAudioParameter* enableParam_     { nullptr };

    String diffusionID_;
    String sizeID_;
    String enableID_;

    Colour accent_ { Colours::accentDelayDigital };
    Metrics metrics_;

    // The audio thread stores the latest values here. The async update
    // reads them on the message thread and repaints.
    std::atomic<float> pendingDiffusion_ { -1.0f };
    std::atomic<float> pendingSize_      { -1.0f };
    std::atomic<float> pendingEnable_    { -1.0f };

    // Drag state. mouseDown snapshots the start values and the pointers.
    RangedAudioParameter* dragDiffusion_ { nullptr };
    RangedAudioParameter* dragSize_      { nullptr };
    float startDiffusion_  = 0.0f;
    float startSize_       = 0.0f;
    float dragStartX_      = 0.0f;
    float dragStartY_      = 0.0f;
    bool dragging_         = false;
    // The Shift axis latch. Zero until the dead zone clears. Then 1 for
    // horizontal (size) or 2 for vertical (diffusion).
    int shiftLatch_ = 0;

    // One wheel gesture per burst per parameter.
    bool wheelDiffusionOpen_ = false;
    bool wheelSizeOpen_      = false;

    bool hovered_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DiffuserPad)
};

} // namespace MarsDSP::GUI

#endif

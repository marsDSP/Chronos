#pragma once

#ifndef CHRONOS_EQ_DISPLAY_H
#define CHRONOS_EQ_DISPLAY_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "../Metrics.h"
#include "../AccentConsumer.h"
#include "../MetricsConsumer.h"
#include "SpectrumAnalyser.h"
#include <array>
#include <vector>

class ChronosProcessor;

namespace MarsDSP::GUI {

// The FILTER page: a spectrum analyser under a six-band EQ drawn on one
// log-frequency / dB plot. Slot 0 is the low cut (hpfFreq) and slot 5 the
// high cut (lpfFreq), the OutputFilterStage pair; slots 1..4 are the free
// bands (eqN On/Type/Freq/Gain/Q). The response curve is the analytic sum
// of the same coefficients the audio path runs. The analyser reads the
// post-EQ wet feed from the processor's spectrum FIFO on a 30 Hz timer
// that runs only while the page is showing.
class EqDisplay : public Component,
                  public SettableTooltipClient,
                  public AccentConsumer,
                  public MetricsConsumer,
                  private AudioProcessorValueTreeState::Listener,
                  private AsyncUpdater,
                  private Timer {
public:
    static constexpr int kNumSlots = 6;
    static constexpr int kLowCut = 0;
    static constexpr int kHighCut = 5;

    // The plot's fixed scales. Model values, not dimensions.
    static constexpr float kMinFreq = 20.0f;
    static constexpr float kMaxFreq = 20000.0f;
    static constexpr float kGainMaxDb = 18.0f;
    static constexpr float kGridDb = 6.0f;

    explicit EqDisplay(ChronosProcessor& proc);
    ~EqDisplay() override;

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

    // The hint follows the pointer: a band readout over a node.
    String getTooltip() override;

private:
    // The parameters behind one slot. The cuts have only a frequency.
    struct Slot {
        RangedAudioParameter* on { nullptr };
        RangedAudioParameter* type { nullptr };
        RangedAudioParameter* freq { nullptr };
        RangedAudioParameter* gain { nullptr };
        RangedAudioParameter* q { nullptr };
    };

    void parameterChanged(const String& parameterID, float newValue) override;
    void handleAsyncUpdate() override;
    void timerCallback() override;
    void updateTimerState_();

    // A page switch shows the panel above this component, which JUCE does
    // not report as visibilityChanged on the child. The watcher reports
    // the effective (showing) state through the parent chain instead.
    class ShowingWatcher final : public ComponentMovementWatcher {
    public:
        explicit ShowingWatcher(EqDisplay& owner) : ComponentMovementWatcher(&owner), owner_(owner) {}
        void componentMovedOrResized(bool, bool) override {}
        void componentPeerChanged() override { owner_.updateTimerState_(); }
        void componentVisibilityChanged() override { owner_.updateTimerState_(); }
    private:
        EqDisplay& owner_;
    };

    // Geometry. The plot is the local bounds reduced by kPadInset.
    [[nodiscard]] Rectangle<float> plot_() const noexcept;
    [[nodiscard]] static float freqToX(float f, const Rectangle<float>& plot) noexcept;
    [[nodiscard]] static float xToFreq(float x, const Rectangle<float>& plot) noexcept;
    [[nodiscard]] static float gainToY(float db, const Rectangle<float>& plot) noexcept;
    [[nodiscard]] static float yToGain(float y, const Rectangle<float>& plot) noexcept;
    [[nodiscard]] static float specToY(float db, const Rectangle<float>& plot) noexcept;

    // The band model, read from the parameters.
    [[nodiscard]] static bool isCut(int slot) noexcept { return slot == kLowCut || slot == kHighCut; }
    [[nodiscard]] bool slotOn(int slot) const noexcept;
    [[nodiscard]] bool slotHasGain(int slot) const noexcept;
    [[nodiscard]] int slotType(int slot) const noexcept;
    [[nodiscard]] float slotFreq(int slot) const noexcept;
    [[nodiscard]] float slotGain(int slot) const noexcept;
    [[nodiscard]] float slotQ(int slot) const noexcept;
    [[nodiscard]] double sampleRate_() const noexcept;
    [[nodiscard]] double responseDb_(double f) const noexcept;
    [[nodiscard]] Point<float> nodePos_(int slot, const Rectangle<float>& plot) const noexcept;
    [[nodiscard]] int nodeAt_(Point<float> p) const noexcept;
    [[nodiscard]] int nearestBandByX_(float x) const noexcept;
    [[nodiscard]] String slotName_(int slot) const;

    // Parameter writes. Every host touch is bracketed: a drag opens its
    // gestures at mouseDown and closes them at mouseUp; a wheel burst holds
    // one gesture per parameter until kWheelGestureMs idle; a key, a menu
    // item, and a double-click bracket their own single write.
    static void setDenorm_(RangedAudioParameter* p, float denorm);
    static void writeOnce_(RangedAudioParameter* p, float norm);
    void openDragGesture_(RangedAudioParameter* p);
    void closeDragGestures_();
    bool endWheelGesture_();
    void showBandMenu_(int slot);
    void showAddMenu_(Point<float> pos);
    // Switch on an off free band at pos. type < 0 picks it from the zone.
    void enableBandAt_(Point<float> pos, int type);

    ChronosProcessor& proc_;
    AudioProcessorValueTreeState& apvts_;
    std::array<Slot, kNumSlots> slots_ {};
    std::vector<String> listened_;

    Colour accent_ { Colours::accentDelayDigital };
    Metrics metrics_;

    // The analyser and its pull buffer. The trace repaints only when it
    // moved: every kTraceSampleStride-th bin is compared with the last paint.
    static constexpr int kTraceSampleStride = 16;
    SpectrumAnalyser analyser_;
    std::vector<float> pull_;
    std::array<float, SpectrumAnalyser::kBins / kTraceSampleStride> traceSample_ {};
    // One frequency and one evaluated dB per pixel column.
    std::vector<float> colFreq_;
    std::vector<float> colDb_;

    // Selection and hover, visual only.
    int selected_ = -1;
    int hover_ = -1;

    // Drag state.
    int dragSlot_ = -1;
    Point<float> dragStart_;
    float dragStartFreq_ = 0.0f;
    float dragStartGain_ = 0.0f;
    int shiftLatch_ = 0; // 0 none, 1 horizontal (frequency), 2 vertical (gain)
    std::vector<RangedAudioParameter*> dragGestures_;

    // The wheel burst.
    RangedAudioParameter* wheelParam_ { nullptr };
    uint32 lastBurstMs_ = 0;

    // Declared last: it registers on this component at construction.
    ShowingWatcher watcher_ { *this };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(EqDisplay)
};

} // namespace MarsDSP::GUI

#endif

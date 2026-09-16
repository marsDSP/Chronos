#pragma once

#ifndef CHRONOS_EDITOR_H
#define CHRONOS_EDITOR_H

#include <JuceHeader.h>
#include <atomic>
#include "ChronosProcessor.h"
#include "gui/Colours.h"
#include "gui/Metrics.h"
#include "gui/LookAndFeel.h"
#include "gui/Card.h"
#include "gui/Rail.h"
#include "gui/Header.h"
#include "gui/Footer.h"
#include "gui/PedalKnob.h"
#include "gui/controls/PowerButton.h"
#include "gui/controls/ConsoleButton.h"
#include "gui/controls/TimeLockButton.h"
#include "gui/controls/TimeDisplay.h"
#include "gui/tap/TapDisplay.h"
#include "gui/DiffuserPad.h"

// The main plugin editor component.
// The editor shows one window with a tap display and a row of cards.
class ChronosEditor final : public AudioProcessorEditor,
                            public AudioProcessorValueTreeState::Listener,
                            private AsyncUpdater,
                            private Timer {
public:
    explicit ChronosEditor(ChronosProcessor&);
    ~ChronosEditor() override;

    void paint(Graphics&) override;
    void paintOverChildren(Graphics&) override;
    void resized() override;

    void parameterChanged(const String& parameterID, float newValue) override;

    bool keyPressed(const KeyPress& key) override;
    void mouseDown(const MouseEvent& e) override;

private:
    void timerCallback() override;

    // Apply the enablement law on the message thread.
    void handleAsyncUpdate() override;

    void updateCoreAccentColour_(float delayModeVal);
    void pollParameterChanges_();

    // Read the five mode parameters and make every inert control inert.
    void updateEnablement_();

    // Update the undo and redo buttons from the history state.
    void updateHistoryButtons_();

    // Screen-aware resize for the preset bar Zoom submenu. The cap keeps
    // the window on-screen so the drag handle always stays reachable.
    void resizeToWidth_(int targetW);
    void fitToScreen_();
    int  currentScreenMaxWidth_() const;

    ChronosProcessor& processorRef;
    MarsDSP::GUI::Metrics metrics_;
    MarsDSP::GUI::LookAndFeel lnf_;
    MarsDSP::GUI::Knobs::PedalKnob knobLnf_;

    MarsDSP::GUI::TapDisplay tapDisplay_;
    MarsDSP::GUI::Header header_;
    MarsDSP::GUI::Footer footer_;
    MarsDSP::GUI::Card leftCard_;
    MarsDSP::GUI::Card rightCard_;
    MarsDSP::GUI::Rail rail_;

    // The bypass scrim covers the tap band, the card row, and the rail.
    Rectangle<int> cardRowBounds_;
    Rectangle<int> railBounds_;

    // The audio thread stores delay-mode and bypass here. A timer polls
    // these values on the message thread and applies the visual update.
    std::atomic<int> pendingDelayMode_ { -1 };
    std::atomic<int> pendingBypass_ { -1 };
    int lastDelayMode_ { -1 };
    bool lastBypass_ { false };
    std::unique_ptr<juce::Timer> paramPoll_;

    TooltipWindow tooltipWindow_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChronosEditor)
};

#endif

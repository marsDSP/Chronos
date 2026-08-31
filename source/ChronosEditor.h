#pragma once

#ifndef CHRONOS_EDITOR_H
#define CHRONOS_EDITOR_H

#include <JuceHeader.h>
#include "ChronosProcessor.h"
#include "gui/Colours.h"
#include "gui/Metrics.h"
#include "gui/LookAndFeel.h"
#include "gui/Card.h"
#include "gui/Header.h"
#include "gui/Footer.h"
#include "gui/PedalKnob.h"
#include "gui/controls/PowerButton.h"
#include "gui/controls/ConsoleButton.h"
#include "gui/controls/TimeLockButton.h"
#include "gui/controls/TimeDisplay.h"
#include "gui/controls/DotMatrixDisplay.h"
#include "gui/tap/TapDisplay.h"

// The main plugin editor component.
// The editor shows one window with a tap display and a row of cards.
class ChronosEditor final : public AudioProcessorEditor,
                            public AudioProcessorValueTreeState::Listener {
public:
    explicit ChronosEditor(ChronosProcessor&);
    ~ChronosEditor() override;

    void paint(Graphics&) override;
    void paintOverChildren(Graphics&) override;
    void resized() override;

    void parameterChanged(const String& parameterID, float newValue) override;

private:
    void updateCoreAccentColour_(float delayModeVal);

    ChronosProcessor& processorRef;
    MarsDSP::GUI::Metrics metrics_;
    MarsDSP::GUI::LookAndFeel lnf_;

    MarsDSP::GUI::TapDisplay tapDisplay_;
    MarsDSP::GUI::Header header_;
    MarsDSP::GUI::Footer footer_;
    MarsDSP::GUI::Card timeCard_;
    MarsDSP::GUI::Card repeatsCard_;
    MarsDSP::GUI::Card characterCard_;
    MarsDSP::GUI::Card outputCard_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChronosEditor)
};

#endif

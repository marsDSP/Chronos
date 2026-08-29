#pragma once

#ifndef CHRONOS_EDITOR_H
#define CHRONOS_EDITOR_H

#include <JuceHeader.h>
#include "ChronosProcessor.h"
#include "gui/Colours.h"
#include "gui/LookAndFeel.h"
#include "gui/PanelPage.h"
#include "gui/TabBar.h"
#include "gui/PedalKnob.h"
#include "gui/controls/PowerButton.h"
#include "gui/controls/ConsoleButton.h"
#include "gui/controls/TimeLockButton.h"
#include "gui/controls/TimeDisplay.h"
#include "gui/controls/DotMatrixDisplay.h"
#include "gui/tap/TapDisplay.h"

// The main plugin editor component.
// The editor hosts the tab bar and the page views.
class ChronosEditor final : public AudioProcessorEditor,
                            public AudioProcessorValueTreeState::Listener {
public:
    explicit ChronosEditor(ChronosProcessor&);
    ~ChronosEditor() override;

    void paint(Graphics&) override;
    void resized() override;

    void parameterChanged(const String& parameterID, float newValue) override;

    // Set the visible tab index.
    void setSelectedTab(int index);

private:
    void updateCoreAccentColour_(float delayModeVal);

    ChronosProcessor& processorRef;
    MarsDSP::GUI::LookAndFeel lnf_;

    MarsDSP::GUI::TabBar tabBar_;
    MarsDSP::GUI::PanelPage delayPage_;
    MarsDSP::GUI::PanelPage characterPage_;
    MarsDSP::GUI::PanelPage outputPage_;

    std::array<MarsDSP::GUI::PanelPage*, 3> pages_{};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChronosEditor)
};

#endif

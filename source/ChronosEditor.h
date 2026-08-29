#pragma once

#ifndef CHRONOS_EDITOR_H
#define CHRONOS_EDITOR_H

#include <JuceHeader.h>
#include "ChronosProcessor.h"
#include "gui/Colours.h"
#include "gui/LookAndFeel.h"

// The main plugin editor component.
class ChronosEditor final : public AudioProcessorEditor {
public:
    explicit ChronosEditor(ChronosProcessor&);
    ~ChronosEditor() override;

    void paint(Graphics&) override;
    void resized() override;

private:
    ChronosProcessor& processorRef;
    MarsDSP::GUI::LookAndFeel lnf_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChronosEditor)
};

#endif

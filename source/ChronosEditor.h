#pragma once

#include "ChronosProcessor.h"

//==============================================================================
class ChronosEditor final : public AudioProcessorEditor
{
public:
    explicit ChronosEditor (ChronosProcessor&);
    ~ChronosEditor() override;

    //==============================================================================
    void paint (Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    ChronosProcessor& pref;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChronosEditor)
};

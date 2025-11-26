#include "PluginEditor.h"

PluginEditor::PluginEditor(PluginProcessor &p) : AudioProcessorEditor(&p), pref(p)
{
    juce::ignoreUnused(pref);
    setSize (900, 450);
}

PluginEditor::~PluginEditor()
{
}

void PluginEditor::paint(juce::Graphics &g)
{
    g.setColour(juce::Colours::black.withAlpha(0.3f));
    g.fillRect(getLocalBounds());
}

void PluginEditor::resized()
{
}
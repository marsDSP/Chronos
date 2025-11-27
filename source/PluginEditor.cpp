#include "PluginEditor.h"

PluginEditor::PluginEditor(PluginProcessor &p) : AudioProcessorEditor(&p), pref(p)
{
    juce::ignoreUnused(pref);
    setSize (900, 450);
}

PluginEditor::~PluginEditor() = default;

void PluginEditor::resized()
{

}

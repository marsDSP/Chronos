#include "PluginEditor.h"

PluginEditor::PluginEditor(PluginProcessor &p) : AudioProcessorEditor(&p), pref(p)
{
    juce::ignoreUnused(pref);
    setSize (900, 450);
}

PluginEditor::~PluginEditor()
{
}

void PluginEditor::resized()
{
    const auto padding = juce::roundToInt(getWidth() * 0.03f);
    const auto width = juce::roundToInt(getWidth() * 0.07f);
    const auto height = juce::roundToInt(getHeight() * 0.05f);

    const auto x_pos = getWidth() - width - padding;
    const auto y_pos = padding;

    m_oversampling_menu.setBounds(x_pos, y_pos, width, height);
}

void PluginEditor::setOversampleMenu(juce::ComboBox& box, const juce::StringArray& items)
{
    box.addItemList(items, 1);
    box.setSelectedId(1, juce::dontSendNotification);
    addAndMakeVisible(box);
}
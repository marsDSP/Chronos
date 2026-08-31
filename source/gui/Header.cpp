#include "Header.h"
#include "../ChronosProcessor.h"
#include "Fonts.h"

namespace MarsDSP::GUI {

Header::Header(ChronosProcessor& proc)
    : processorRef_(proc), presetBar_(proc.getPresetManager())
{
    wordmark_.setText("CHRONOS", dontSendNotification);
    wordmark_.setColour(Label::textColourId, Colours::textBright);
    wordmark_.setJustificationType(Justification::centredLeft);
    setTitle("Header");
    addAndMakeVisible(wordmark_);
    addAndMakeVisible(presetBar_);

    bypassButton_.setColours(Colours::accentDelayDigital, Colours::textDim);
    bypassButton_.setTooltip("Bypass the delay processing.");
    bypassButton_.setTitle("Bypass");
    bypassButton_.setHelpText("Bypass the delay processing.");
    bypassAttach_ = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
        processorRef_.getAPVTS(), bypassParamID.getParamID(), bypassButton_);
    addAndMakeVisible(bypassButton_);
}

void Header::setMetrics(const Metrics& m)
{
    metrics_ = m;
    presetBar_.setMetrics(m);
    resized();
    repaint();
}

void Header::setAccentColour(const Colour c)
{
    bypassButton_.setAccentColour(c);
    presetBar_.setAccentColour(c);
}

void Header::paint(Graphics& g)
{
    g.fillAll(Colours::headerBackground);
    g.setColour(Colours::panelBorder);
    g.drawHorizontalLine(getHeight() - 1, 0.0f, static_cast<float>(getWidth()));
}

void Header::resized()
{
    const int h = getHeight();
    const int left = metrics_.px(14.0f);
    const int right = getWidth() - metrics_.px(14.0f);
    const int bypassSize = metrics_.px(20.0f);

    // Centre the preset bar in the band.
    const int barW = metrics_.px(static_cast<float>(Metrics::kPresetBarW));
    const int barH = metrics_.px(static_cast<float>(Metrics::kPresetBarH));
    const int barX = (getWidth() - barW) / 2;
    const int barY = (h - barH) / 2;
    presetBar_.setBounds(barX, barY, barW, barH);

    // Wordmark fills the left gap up to the bar.
    wordmark_.setFont(Fonts::font(Fonts::Weight::Semibold, metrics_.font(15.0f)));
    wordmark_.setBounds(left, 0, barX - left - metrics_.px(8.0f), h);

    bypassButton_.setBounds(right - bypassSize, (h - bypassSize) / 2, bypassSize, bypassSize);
}

} // namespace MarsDSP::GUI

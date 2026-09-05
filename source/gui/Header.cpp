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

    bypassButton_.setColours(Colours::accentDelayDigital, Colours::textMuted);
    bypassButton_.setTooltip("Bypass the delay processing.");
    bypassButton_.setTitle("Bypass");
    bypassButton_.setHelpText("Bypass the delay processing.");
    bypassAttach_ = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
        processorRef_.getAPVTS(), bypassParamID.getParamID(), bypassButton_);
    addAndMakeVisible(bypassButton_);

    addAndMakeVisible(undoButton_);
    addAndMakeVisible(redoButton_);
    undoButton_.setTooltip("Nothing to undo.");
    redoButton_.setTooltip("Nothing to redo.");
    undoButton_.setEnabled(false);
    redoButton_.setEnabled(false);
}

void Header::setMetrics(const Metrics& m)
{
    metrics_ = m;
    presetBar_.setMetrics(m);
    bypassButton_.setMetrics(m);
    undoButton_.setMetrics(m);
    redoButton_.setMetrics(m);
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
    const int left = metrics_.px(Metrics::kHeaderSideMargin);
    const int right = getWidth() - metrics_.px(Metrics::kHeaderSideMargin);
    const int bypassSize = metrics_.px(Metrics::kHeaderBypassSize);

    // Centre the preset bar in the band.
    const int barW = metrics_.px(static_cast<float>(Metrics::kPresetBarW));
    const int barH = metrics_.px(static_cast<float>(Metrics::kPresetBarH));
    const int barX = (getWidth() - barW) / 2;
    const int barY = (h - barH) / 2;
    presetBar_.setBounds(barX, barY, barW, barH);

    // Wordmark fills the left gap up to the bar.
    wordmark_.setFont(Fonts::font(Fonts::Weight::Semibold, metrics_.font(Metrics::kWordmarkFont)));
    wordmark_.setBounds(left, 0, barX - left - metrics_.px(Metrics::kWordmarkGap), h);

    // The right cluster: bypass, gap, redo, gap, undo.
    const int histSize = metrics_.px(Metrics::kHistoryButtonSize);
    const int histGap = metrics_.px(Metrics::kHistoryButtonGap);
    const int clusterGap = metrics_.px(Metrics::kHeaderClusterGap);
    int rx = right - bypassSize;
    bypassButton_.setBounds(rx, (h - bypassSize) / 2, bypassSize, bypassSize);
    rx -= clusterGap + histSize;
    redoButton_.setBounds(rx, (h - histSize) / 2, histSize, histSize);
    rx -= histGap + histSize;
    undoButton_.setBounds(rx, (h - histSize) / 2, histSize, histSize);
}

} // namespace MarsDSP::GUI

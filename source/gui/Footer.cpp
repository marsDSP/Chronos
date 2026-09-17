#include "Footer.h"
#include "../ChronosProcessor.h"
#include "Fonts.h"

namespace MarsDSP::GUI {

Footer::Footer(ChronosProcessor& proc)
    : processorRef_(proc)
{
    versionText_ = JucePlugin_VersionString;
    refreshText_();
    startTimerHz(2);
}

Footer::~Footer()
{
    stopTimer();
}

void Footer::setMetrics(const Metrics& m)
{
    metrics_ = m;
    repaint();
}

void Footer::refreshText_()
{
    const double sr = processorRef_.getSampleRate();
    const String srStr = (sr > 0.0) ? (String(sr * 0.001, 1) + " kHz") : String("---");

    const double bpm = processorRef_.getCachedBpm();
    const String bpmStr = (bpm > 0.0) ? (String(bpm, 1) + " BPM") : String("---");

    const String dot = String::charToString(static_cast<juce_wchar>(0x00B7));
    statusText_ = srStr + " " + dot + " " + bpmStr;

    repaint();
}

void Footer::timerCallback()
{
    refreshText_();
}

void Footer::setHint(const String& hint)
{
    if (hint == hint_)
        return;
    hint_ = hint;
    repaint();
}

void Footer::paint(Graphics& g)
{
    g.fillAll(Colours::footerBackground);
    g.setColour(Colours::panelBorder);
    g.drawHorizontalLine(0, 0.0f, static_cast<float>(getWidth()));

    const Font f = Fonts::font(Fonts::Weight::Regular, metrics_.font(Metrics::kFooterFont));
    g.setFont(f);
    g.setColour(Colours::textMuted);

    const auto bounds = getLocalBounds().reduced(metrics_.px(Metrics::kFooterSideMargin), 0);
    const String version = "v" + versionText_;
    g.drawText(statusText_, bounds, Justification::centredLeft, true);
    g.drawText(version, bounds, Justification::centredRight, true);

    if (hint_.isEmpty())
        return;

    // The hint takes the width the status and the version leave free,
    // one side margin clear of each, and ellipsises past that.
    const int gap = metrics_.px(Metrics::kFooterSideMargin);
    auto area = bounds;
    area.removeFromLeft(roundToInt(Fonts::textWidth(f, statusText_)) + gap);
    area.removeFromRight(roundToInt(Fonts::textWidth(f, version)) + gap);
    if (area.getWidth() <= 0)
        return;

    g.setColour(Colours::textPrimary);
    g.drawText(hint_, area, Justification::centred, true);
}

void Footer::resized()
{
}

} // namespace MarsDSP::GUI

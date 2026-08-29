#include "Footer.h"
#include "../ChronosProcessor.h"

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

void Footer::paint(Graphics& g)
{
    g.fillAll(Colours::footerBackground);
    g.setColour(Colours::panelBorder);
    g.drawHorizontalLine(0, 0.0f, static_cast<float>(getWidth()));

    g.setColour(Colours::textDim);
    g.setFont(Font(FontOptions(11.0f)));

    const auto bounds = getLocalBounds().reduced(12, 0);
    g.drawText(statusText_, bounds, Justification::centredLeft, true);
    g.drawText("v" + versionText_, bounds, Justification::centredRight, true);
}

void Footer::resized()
{
}

} // namespace MarsDSP::GUI

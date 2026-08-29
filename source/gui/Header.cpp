#include "Header.h"
#include "../ChronosProcessor.h"

namespace MarsDSP::GUI {

Header::CoreBadge::CoreBadge()
    : Button("CoreBadge")
{
    setClickingTogglesState(false);
}

void Header::CoreBadge::setMode(const int mode, const Colour accent)
{
    mode_ = mode;
    accent_ = accent;
    repaint();
}

void Header::CoreBadge::paintButton(Graphics& g,
                                    const bool shouldDrawButtonAsHighlighted,
                                    const bool shouldDrawButtonAsDown)
{
    ignoreUnused(shouldDrawButtonAsDown);

    const auto bounds = getLocalBounds().toFloat();

    g.setColour(accent_.withMultipliedAlpha(shouldDrawButtonAsHighlighted ? 0.82f : 1.0f));
    g.fillRoundedRectangle(bounds, bounds.getHeight() * 0.5f);

    g.setColour(Colours::background);
    g.setFont(Font(FontOptions(11.0f)).boldened());
    g.drawText(mode_ == 1 ? "BBD" : "DIGITAL", bounds, Justification::centred, true);
}

Header::Header(ChronosProcessor& proc)
    : processorRef_(proc)
{
    wordmark_.setText("CHRONOS", dontSendNotification);
    wordmark_.setColour(Label::textColourId, Colours::textBright);
    wordmark_.setFont(Font(FontOptions(18.0f)).boldened());
    wordmark_.setJustificationType(Justification::centredLeft);
    addAndMakeVisible(wordmark_);

    subline_.setText("NONLINEAR DELAY ENGINE", dontSendNotification);
    subline_.setColour(Label::textColourId, Colours::textDim);
    subline_.setFont(Font(FontOptions(9.0f)));
    subline_.setJustificationType(Justification::centredLeft);
    addAndMakeVisible(subline_);

    badge_.onClick = [this] { toggleDelayMode_(); };
    addAndMakeVisible(badge_);

    bypassButton_.setColours(Colours::accentRed, Colours::textDim);
    bypassAttach_ = std::make_unique<AudioProcessorValueTreeState::ButtonAttachment>(
        processorRef_.getAPVTS(), bypassParamID.getParamID(), bypassButton_);
    addAndMakeVisible(bypassButton_);
}

void Header::setCoreMode(const int mode, const Colour accent)
{
    badge_.setMode(mode, accent);
}

void Header::toggleDelayMode_()
{
    auto* p = processorRef_.getAPVTS().getParameter("delayMode");
    if (p == nullptr)
        return;

    p->beginChangeGesture();
    const float next = (p->getValue() > 0.5f) ? 0.0f : 1.0f;
    p->setValueNotifyingHost(next);
    p->endChangeGesture();
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

    wordmark_.setBounds(16, 6, 130, 22);
    subline_.setBounds(16, 28, 180, 14);

    constexpr int badgeW = 70;
    constexpr int badgeH = 22;
    constexpr int bypassSize = 24;
    const int right = getWidth() - 12;

    bypassButton_.setBounds(right - bypassSize, (h - bypassSize) / 2, bypassSize, bypassSize);
    badge_.setBounds(right - bypassSize - 8 - badgeW, (h - badgeH) / 2, badgeW, badgeH);
}

} // namespace MarsDSP::GUI

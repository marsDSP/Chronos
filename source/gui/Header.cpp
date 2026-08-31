#include "Header.h"
#include "../ChronosProcessor.h"
#include "Fonts.h"

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
    g.setFont(Fonts::font(Fonts::Weight::Semibold, currentMetrics().font(11.0f)));
    g.drawText(mode_ == 1 ? "BBD" : "DIGITAL", bounds, Justification::centred, true);
}

Header::Header(ChronosProcessor& proc)
    : processorRef_(proc)
{
    wordmark_.setText("CHRONOS", dontSendNotification);
    wordmark_.setColour(Label::textColourId, Colours::textBright);
    wordmark_.setJustificationType(Justification::centredLeft);
    addAndMakeVisible(wordmark_);

    subline_.setText("NONLINEAR DELAY ENGINE", dontSendNotification);
    subline_.setColour(Label::textColourId, Colours::textDim);
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

void Header::setMetrics(const Metrics& m)
{
    metrics_ = m;
    resized();
    repaint();
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
    const int left = metrics_.px(16.0f);

    wordmark_.setFont(Fonts::font(Fonts::Weight::Semibold, metrics_.font(15.0f)));
    subline_.setFont(Fonts::font(Fonts::Weight::Regular, metrics_.font(9.0f)));

    wordmark_.setBounds(left, metrics_.px(6.0f), metrics_.px(130.0f), metrics_.px(22.0f));
    subline_.setBounds(left, metrics_.px(28.0f), metrics_.px(180.0f), metrics_.px(14.0f));

    const int badgeW = metrics_.px(70.0f);
    const int badgeH = metrics_.px(22.0f);
    const int bypassSize = metrics_.px(24.0f);
    const int right = getWidth() - metrics_.px(12.0f);

    bypassButton_.setBounds(right - bypassSize, (h - bypassSize) / 2, bypassSize, bypassSize);
    badge_.setBounds(right - bypassSize - metrics_.px(8.0f) - badgeW,
                     (h - badgeH) / 2, badgeW, badgeH);
}

} // namespace MarsDSP::GUI

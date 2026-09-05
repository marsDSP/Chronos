#include "Card.h"
#include "Fonts.h"
#include "MetricsConsumer.h"

namespace MarsDSP::GUI {

Card::Card(const String& title)
    : title_(title.toUpperCase())
{
}

void Card::setContent(std::unique_ptr<Component> panel)
{
    if (content_ != nullptr)
        removeChildComponent(content_.get());

    content_ = std::move(panel);
    if (content_ != nullptr)
    {
        addAndMakeVisible(*content_);
        if (auto* ac = dynamic_cast<AccentConsumer*>(content_.get()))
            ac->setAccentColour(accent_);
    }
    resized();
    repaint();
}

void Card::setAccentColour(const Colour c)
{
    accent_ = c;
    if (content_ != nullptr)
        if (auto* ac = dynamic_cast<AccentConsumer*>(content_.get()))
            ac->setAccentColour(c);
    repaint();
}

void Card::setMetrics(const Metrics& m)
{
    metrics_ = m;
    if (content_ != nullptr)
        if (auto* mc = dynamic_cast<MetricsConsumer*>(content_.get()))
            mc->setMetrics(m);
    resized();
    repaint();
}

void Card::setEnablement(const EnablementState& state)
{
    if (content_ != nullptr)
        if (auto* ec = dynamic_cast<EnablementConsumer*>(content_.get()))
            ec->setControlsEnabled(state);
}

void Card::paint(Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    const float r = metrics_.pxf(static_cast<float>(Metrics::kCardCornerRadius));
    const float sw = metrics_.stroke(static_cast<float>(Metrics::kCardBorderStroke));

    g.setColour(Colours::panelBackground);
    g.fillRoundedRectangle(bounds, r);

    g.setColour(tint(Colours::panelBackground, accent_, kTintCardBorder));
    g.drawRoundedRectangle(bounds.reduced(sw / 2), r, sw);

    // The title row. Semibold, primary ink, uppercase, tracked, left aligned.
    const int titleH = metrics_.px(static_cast<float>(Metrics::kCardTitleH));
    const int hpad = metrics_.px(static_cast<float>(Metrics::kCardHPad));
    Font titleFont = Fonts::font(Fonts::Weight::Semibold, metrics_.font(Metrics::kCardTitleFont));
    titleFont.setExtraKerningFactor(Metrics::kTitleTracking);
    g.setFont(titleFont);
    g.setColour(Colours::textPrimary);
    g.drawText(title_, hpad, 0, getWidth() - 2 * hpad, titleH, Justification::centredLeft, false);
}

void Card::resized()
{
    if (content_ == nullptr)
        return;

    const int border = metrics_.px(static_cast<float>(Metrics::kCardBorderStroke));
    const int hpad = metrics_.px(static_cast<float>(Metrics::kCardHPad));
    const int titleH = metrics_.px(static_cast<float>(Metrics::kCardTitleH));
    const int titleGap = metrics_.px(static_cast<float>(Metrics::kCardTitleGap));
    const int bottomPad = metrics_.px(static_cast<float>(Metrics::kCardBottomPad));

    const int x = border + hpad;
    const int y = border + titleH + titleGap;
    const int w = getWidth() - 2 * (border + hpad);
    const int h = getHeight() - y - bottomPad;
    content_->setBounds(x, y, w, h);
}

} // namespace MarsDSP::GUI

#include "Rail.h"
#include "MetricsConsumer.h"

namespace MarsDSP::GUI {

void Rail::setPanel(std::unique_ptr<Component> panel)
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

void Rail::setAccentColour(const Colour c)
{
    accent_ = c;
    if (content_ != nullptr)
        if (auto* ac = dynamic_cast<AccentConsumer*>(content_.get()))
            ac->setAccentColour(c);
    repaint();
}

void Rail::setMetrics(const Metrics& m)
{
    metrics_ = m;
    if (content_ != nullptr)
        if (auto* mc = dynamic_cast<MetricsConsumer*>(content_.get()))
            mc->setMetrics(m);
    resized();
    repaint();
}

void Rail::setEnablement(const EnablementState& state)
{
    if (content_ != nullptr)
        if (auto* ec = dynamic_cast<EnablementConsumer*>(content_.get()))
            ec->setControlsEnabled(state);
}

void Rail::paint(Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    const float r = metrics_.pxf(static_cast<float>(Metrics::kCardCornerRadius));
    const float sw = metrics_.stroke(static_cast<float>(Metrics::kCardBorderStroke));

    g.setColour(Colours::panelBackground);
    g.fillRoundedRectangle(bounds, r);

    g.setColour(tint(Colours::panelBackground, accent_, kTintCardBorder));
    g.drawRoundedRectangle(bounds.reduced(sw / 2), r, sw);
}

void Rail::resized()
{
    if (content_ == nullptr)
        return;

    const int border = metrics_.px(static_cast<float>(Metrics::kCardBorderStroke));
    const int hpad = metrics_.px(static_cast<float>(Metrics::kCardHPad));
    const int vpad = metrics_.px(static_cast<float>(Metrics::kRailVPad));

    const int x = border + hpad;
    const int y = border + vpad;
    const int w = getWidth() - 2 * (border + hpad);
    const int h = getHeight() - 2 * (border + vpad);
    content_->setBounds(x, y, w, h);
}

} // namespace MarsDSP::GUI

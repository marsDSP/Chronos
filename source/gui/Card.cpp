#include "Card.h"
#include "Colours.h"
#include "MetricsConsumer.h"

namespace MarsDSP::GUI {

Card::Card()
{
    subTabs_.onSubTabChanged = [this](const int index)
    {
        setSelectedContent(index);
    };
    addAndMakeVisible(subTabs_);
}

void Card::setAccentColour(const Colour c)
{
    accent_ = c;
    subTabs_.setAccentColour(c);

    for (auto& panel : contents_)
        if (auto* ac = dynamic_cast<AccentConsumer*>(panel.get()))
            ac->setAccentColour(c);

    repaint();
}

void Card::setMetrics(const Metrics& m)
{
    metrics_ = m;
    subTabs_.setMetrics(m);

    for (auto& panel : contents_)
        if (auto* mc = dynamic_cast<MetricsConsumer*>(panel.get()))
            mc->setMetrics(m);

    resized();
    repaint();
}

void Card::setEnablement(const EnablementState& state)
{
    for (auto& panel : contents_)
        if (auto* ec = dynamic_cast<EnablementConsumer*>(panel.get()))
            ec->setControlsEnabled(state);
}

void Card::addContent(const String& tabName, std::unique_ptr<Component> panel)
{
    const int index = static_cast<int>(contents_.size());
    subTabs_.addSubTab(tabName);

    if (panel != nullptr)
    {
        addChildComponent(*panel);
        contents_.push_back(std::move(panel));
    }
    else
    {
        auto placeholder = std::make_unique<Component>();
        addChildComponent(*placeholder);
        contents_.push_back(std::move(placeholder));
    }

    if (index == 0)
        setSelectedContent(0);

    resized();
}

void Card::setSelectedContent(const int index)
{
    subTabs_.setSelectedSubTab(index);
    for (int i = 0; i < static_cast<int>(contents_.size()); ++i)
    {
        const bool visible = (i == index);
        contents_[static_cast<std::size_t>(i)]->setVisible(visible);
    }
    resized();
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
}

void Card::resized()
{
    const int border = metrics_.px(static_cast<float>(Metrics::kCardBorderStroke));
    const int hpad = metrics_.px(static_cast<float>(Metrics::kCardHPad));
    const int subTabH = metrics_.px(static_cast<float>(Metrics::kSubTabStripH));
    const int gap = metrics_.px(static_cast<float>(Metrics::kSubTabGap));
    const int bottomPad = metrics_.px(static_cast<float>(Metrics::kCardBottomPad));

    const int x = border + hpad;
    const int w = getWidth() - 2 * (border + hpad);
    subTabs_.setBounds(x, border, w, subTabH);

    const int contentY = border + subTabH + gap;
    const int contentH = getHeight() - contentY - bottomPad;
    const Rectangle<int> contentArea(x, contentY, w, contentH);
    for (auto& panel : contents_)
    {
        if (panel->isVisible())
            panel->setBounds(contentArea);
    }
}

} // namespace MarsDSP::GUI

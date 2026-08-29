#include "Card.h"
#include "Colours.h"

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
    repaint();
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

    g.setColour(Colours::panelBackground);
    g.fillRoundedRectangle(bounds, 6.0f);

    g.setColour(accent_.withAlpha(0.35f));
    g.drawRoundedRectangle(bounds.reduced(0.5f), 6.0f, 1.0f);
}

void Card::resized()
{
    const auto bounds = getLocalBounds().reduced(8);
    constexpr int subTabHeight = 26;

    subTabs_.setBounds(bounds.getX(), bounds.getY(), bounds.getWidth(), subTabHeight);

    const auto contentArea = bounds.withTrimmedTop(subTabHeight + 6);
    for (auto& panel : contents_)
    {
        if (panel->isVisible())
            panel->setBounds(contentArea);
    }
}

} // namespace MarsDSP::GUI

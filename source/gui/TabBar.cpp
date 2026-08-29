#include "TabBar.h"
#include "Colours.h"

namespace MarsDSP::GUI {

TabBar::TabButton::TabButton(const String& name, const Colour dotColour)
    : Button(name), dotColour_(dotColour)
{
    setClickingTogglesState(false);
}

void TabBar::TabButton::setDotColour(const Colour c)
{
    dotColour_ = c;
    repaint();
}

void TabBar::TabButton::paintButton(Graphics& g,
                                    const bool shouldDrawButtonAsHighlighted,
                                    const bool shouldDrawButtonAsDown)
{
    ignoreUnused(shouldDrawButtonAsDown);

    const auto bounds = getLocalBounds().toFloat();
    const bool isSelected = getToggleState();

    g.setColour(Colours::panelBackground);
    g.fillRoundedRectangle(bounds, 5.0f);

    const float borderAlpha = isSelected ? 1.0f : (shouldDrawButtonAsHighlighted ? 0.6f : 0.3f);
    g.setColour(Colours::panelBorder.withAlpha(borderAlpha));
    g.drawRoundedRectangle(bounds.reduced(0.5f), 5.0f, 1.0f);

    constexpr float dotSize = 5.0f;
    const float dotX = bounds.getX() + 14.0f;
    const float dotY = bounds.getCentreY() - dotSize * 0.5f;

    const float dotAlpha = isSelected ? 1.0f : 0.3f;
    g.setColour(dotColour_.withAlpha(dotAlpha));
    g.fillEllipse(dotX, dotY, dotSize, dotSize);

    if (isSelected)
        g.setColour(Colours::textBright);
    else if (shouldDrawButtonAsHighlighted)
        g.setColour(Colours::textPrimary);
    else
        g.setColour(Colours::textDim);

    const auto titleFont = Font(FontOptions(11.0f)).boldened();
    g.setFont(titleFont);

    const float textX = dotX + dotSize + 8.0f;
    const auto textBounds = Rectangle<float>(textX, bounds.getY(), bounds.getWidth() - textX - 8.0f, bounds.getHeight());
    g.drawText(getButtonText().toUpperCase(), textBounds, Justification::centredLeft, true);
}

TabBar::TabBar() = default;

void TabBar::addTab(const String& name, const Colour dotColour)
{
    const int index = static_cast<int>(buttons_.size());
    auto btn = std::make_unique<TabButton>(name, dotColour);
    btn->onClick = [this, index]
    {
        setSelectedTab(index);
        if (onTabChanged)
            onTabChanged(index);
    };

    addAndMakeVisible(*btn);
    buttons_.push_back(std::move(btn));

    if (buttons_.size() == 1)
        setSelectedTab(0);

    resized();
}

void TabBar::setSelectedTab(const int index)
{
    selectedIndex_ = index;
    for (int i = 0; i < static_cast<int>(buttons_.size()); ++i)
    {
        buttons_[static_cast<std::size_t>(i)]->setToggleState(i == index, dontSendNotification);
        buttons_[static_cast<std::size_t>(i)]->repaint();
    }
}

void TabBar::setTabDotColour(const int index, const Colour newColour)
{
    if (index >= 0 && index < static_cast<int>(buttons_.size()))
        buttons_[static_cast<std::size_t>(index)]->setDotColour(newColour);
}

void TabBar::paint(Graphics& g)
{
    ignoreUnused(g);
}

void TabBar::resized()
{
    if (buttons_.empty())
        return;

    const auto bounds = getLocalBounds();
    constexpr int buttonWidth = 110;
    constexpr int gap = 8;
    const int totalWidth = static_cast<int>(buttons_.size()) * buttonWidth + (static_cast<int>(buttons_.size()) - 1) * gap;
    int startX = (bounds.getWidth() - totalWidth) / 2;

    for (auto& btn : buttons_)
    {
        btn->setBounds(startX, (bounds.getHeight() - 28) / 2, buttonWidth, 28);
        startX += buttonWidth + gap;
    }
}

} // namespace MarsDSP::GUI

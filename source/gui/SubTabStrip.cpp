#include "SubTabStrip.h"
#include "Colours.h"

namespace MarsDSP::GUI {

SubTabStrip::SubTabButton::SubTabButton(const String& name)
    : Button(name)
{
    setClickingTogglesState(false);
}

void SubTabStrip::SubTabButton::setAccentColour(const Colour c)
{
    accent_ = c;
    repaint();
}

void SubTabStrip::SubTabButton::paintButton(Graphics& g,
                                            const bool shouldDrawButtonAsHighlighted,
                                            const bool shouldDrawButtonAsDown)
{
    ignoreUnused(shouldDrawButtonAsDown);

    const auto bounds = getLocalBounds().toFloat();
    const bool isSelected = getToggleState();

    if (isSelected)
        g.setColour(accent_);
    else if (shouldDrawButtonAsHighlighted)
        g.setColour(Colours::textPrimary);
    else
        g.setColour(Colours::textDim);

    g.setFont(Font(FontOptions(11.0f)).boldened());
    g.drawText(getButtonText().toUpperCase(), bounds, Justification::centred, true);
}

SubTabStrip::SubTabStrip() = default;

void SubTabStrip::setAccentColour(const Colour c)
{
    accentColour_ = c;
    for (auto& btn : buttons_)
        btn->setAccentColour(c);
}

void SubTabStrip::addSubTab(const String& name)
{
    const int index = static_cast<int>(buttons_.size());
    auto btn = std::make_unique<SubTabButton>(name);
    btn->setAccentColour(accentColour_);
    btn->onClick = [this, index]
    {
        setSelectedSubTab(index);
        if (onSubTabChanged)
            onSubTabChanged(index);
    };

    addAndMakeVisible(*btn);
    buttons_.push_back(std::move(btn));

    if (buttons_.size() == 1)
        setSelectedSubTab(0);

    resized();
}

void SubTabStrip::setSelectedSubTab(const int index)
{
    selectedIndex_ = index;
    for (int i = 0; i < static_cast<int>(buttons_.size()); ++i)
    {
        buttons_[static_cast<std::size_t>(i)]->setToggleState(i == index, dontSendNotification);
        buttons_[static_cast<std::size_t>(i)]->repaint();
    }
}

void SubTabStrip::paint(Graphics& g)
{
    ignoreUnused(g);
}

void SubTabStrip::resized()
{
    if (buttons_.empty())
        return;

    const auto bounds = getLocalBounds();
    constexpr int buttonWidth = 90;
    constexpr int gap = 6;
    const int totalWidth = static_cast<int>(buttons_.size()) * buttonWidth + (static_cast<int>(buttons_.size()) - 1) * gap;
    int startX = (bounds.getWidth() - totalWidth) / 2;

    for (auto& btn : buttons_)
    {
        btn->setBounds(startX, (bounds.getHeight() - 24) / 2, buttonWidth, 24);
        startX += buttonWidth + gap;
    }
}

} // namespace MarsDSP::GUI

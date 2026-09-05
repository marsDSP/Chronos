#include "SubTabStrip.h"
#include "Colours.h"
#include "Fonts.h"

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

void SubTabStrip::SubTabButton::setMetrics(const Metrics& m)
{
    metrics_ = m;
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
        g.setColour(Colours::textMuted);

    g.setFont(Fonts::font(Fonts::Weight::Medium, metrics_.font(Metrics::kSubTabFont)));
    g.drawText(getButtonText().toUpperCase(), bounds, Justification::centred, false);
}

SubTabStrip::SubTabStrip() = default;

void SubTabStrip::setAccentColour(const Colour c)
{
    accentColour_ = c;
    for (auto& btn : buttons_)
        btn->setAccentColour(c);
}

void SubTabStrip::setMetrics(const Metrics& m)
{
    metrics_ = m;
    for (auto& btn : buttons_)
        btn->setMetrics(m);
    resized();
    repaint();
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
    const int buttonWidth = metrics_.px(Metrics::kSubTabButtonW);
    const int gap = metrics_.px(Metrics::kSubTabButtonGap);
    const int buttonH = metrics_.px(Metrics::kSubTabButtonH);
    const int n = static_cast<int>(buttons_.size());
    const int totalWidth = n * buttonWidth + (n - 1) * gap;
    int startX = (bounds.getWidth() - totalWidth) / 2;

    for (auto& btn : buttons_)
    {
        btn->setBounds(startX, (bounds.getHeight() - buttonH) / 2, buttonWidth, buttonH);
        startX += buttonWidth + gap;
    }
}

} // namespace MarsDSP::GUI

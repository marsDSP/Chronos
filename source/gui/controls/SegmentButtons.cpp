#include "SegmentButtons.h"
#include "../Colours.h"

namespace MarsDSP::GUI {

SegmentButtons::SegmentButtons(AudioProcessorValueTreeState& apvts, const String& paramID,
                               const StringArray& items, const Colour accent, const bool coreLinked,
                               const StringArray& displayLabels, const StringArray& tooltips)
    : accent_(accent), coreLinked_(coreLinked), paramID_(paramID), apvts_(apvts)
{
    // A hidden combo carries the parameter attachment. The buttons mirror it.
    combo_.addItemList(items, 1);
    addChildComponent(combo_);
    combo_.setVisible(false);

    attach_ = std::make_unique<AudioProcessorValueTreeState::ComboBoxAttachment>(apvts_, paramID_, combo_);

    for (int i = 0; i < items.size(); ++i)
    {
        // Use the display labels for the buttons when they are present.
        const String btnLabel = (i < displayLabels.size()) ? displayLabels[i] : items[i];
        auto btn = std::make_unique<ConsoleButton>(btnLabel);
        btn->setClickingTogglesState(false);
        btn->setAccentColours(accent_, Colours::background);
        if (i < tooltips.size() && tooltips[i].isNotEmpty())
            btn->setTooltip(tooltips[i]);
        btn->onClick = [this, i]
        {
            combo_.setSelectedId(i + 1, sendNotificationSync);
            syncButtons();
        };
        addAndMakeVisible(*btn);
        buttons_.push_back(std::move(btn));
    }

    apvts_.addParameterListener(paramID_, this);

    syncButtons();
}

SegmentButtons::~SegmentButtons()
{
    apvts_.removeParameterListener(paramID_, this);
}

void SegmentButtons::setAccentColours(const Colour activeBg, const Colour activeText)
{
    accent_ = activeBg;
    for (auto& btn : buttons_)
        btn->setAccentColours(activeBg, activeText);
    repaint();
}

void SegmentButtons::setAccentColour(const Colour c)
{
    setAccentColours(c, Colours::background);
}

void SegmentButtons::setMetrics(const Metrics& m)
{
    metrics_ = m;
    for (auto& btn : buttons_)
        btn->setMetrics(m);
    resized();
    repaint();
}

void SegmentButtons::setTooltip(const String& text)
{
    for (auto& btn : buttons_)
        btn->setTooltip(text);
}

void SegmentButtons::syncButtons()
{
    const int idx = combo_.getSelectedId() - 1;
    for (int i = 0; i < static_cast<int>(buttons_.size()); ++i)
        buttons_[static_cast<std::size_t>(i)]->setToggleState(i == idx, dontSendNotification);
}

void SegmentButtons::parameterChanged(const String& parameterID, const float newValue)
{
    const auto safe = Component::SafePointer<SegmentButtons>(this);

    if (parameterID == paramID_)
    {
        MessageManager::callAsync([safe] { if (safe != nullptr) safe->syncButtons(); });
    }
}

void SegmentButtons::resized()
{
    if (buttons_.empty())
        return;

    const int n = static_cast<int>(buttons_.size());
    const int gap = metrics_.px(Metrics::kSegmentGap);
    const int bw = (getWidth() - gap * (n - 1)) / n;
    int x = 0;
    for (auto& btn : buttons_)
    {
        btn->setBounds(x, 0, bw, getHeight());
        x += bw + gap;
    }
}

} // namespace MarsDSP::GUI

#pragma once

#ifndef CHRONOS_CARD_H
#define CHRONOS_CARD_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"
#include "AccentConsumer.h"
#include "EnablementConsumer.h"
#include "controls/TabButton.h"
#include <functional>
#include <memory>
#include <vector>

namespace MarsDSP::GUI {

// A rounded card that owns one or more pages. The title row paints the
// title of a single page or one tab per page. The card border keeps the
// tint law. Accent, metrics, and enablement push to every page.
class Card : public Component {
public:
    Card() = default;
    ~Card() override = default;

    // Add a page and its tab. The first page is the selected page.
    void addPage(const String& title, std::unique_ptr<Component> panel);

    // Show the page at the index. The index clamps into range.
    void setSelectedPage(int index);
    [[nodiscard]] int getSelectedPage() const noexcept { return selectedPage_; }
    [[nodiscard]] int getPageCount() const noexcept { return static_cast<int>(pages_.size()); }

    // Set the mark on a tab. The mark paints on an inactive tab only.
    // The tab repaints only on a change.
    void setPageMark(int index, bool marked);

    // Called on a user click only, never on setSelectedPage.
    std::function<void(int)> onPageChanged;

    // Set the accent colour for the card border and every page and tab.
    void setAccentColour(Colour c);

    // Set the scale metrics for the card layout and every page and tab.
    void setMetrics(const Metrics& m);

    // Push the enablement state to every page that reads it.
    void setEnablement(const EnablementState& state);

    void paint(Graphics& g) override;
    void resized() override;

private:
    struct Page {
        String title;
        std::unique_ptr<Component> panel;
        std::unique_ptr<TabButton> tab;
    };

    Page& pageAt_(int index) noexcept;

    std::vector<Page> pages_;
    int selectedPage_ = 0;
    Colour accent_ { Colours::accentDelayDigital };
    Metrics metrics_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Card)
};

} // namespace MarsDSP::GUI

#endif

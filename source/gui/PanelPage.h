#pragma once

#ifndef CHRONOS_PANEL_PAGE_H
#define CHRONOS_PANEL_PAGE_H

#include <JuceHeader.h>
#include "SubTabStrip.h"
#include <memory>
#include <vector>

namespace MarsDSP::GUI {

// A component that represents one tab page.
// The page hosts subtab sections and sub-panels.
class PanelPage : public Component {
public:
    PanelPage()
    {
        subTabs_.onSubTabChanged = [this](const int index)
        {
            setSelectedSubTab(index);
        };
        addAndMakeVisible(subTabs_);
    }

    ~PanelPage() override = default;

    // Add one sub-panel associated with a subtab title.
    void addSubPanel(const String& name, std::unique_ptr<Component> panel)
    {
        const int index = static_cast<int>(subPanels_.size());
        subTabs_.addSubTab(name);

        if (panel != nullptr)
        {
            addChildComponent(*panel);
            subPanels_.push_back(std::move(panel));
        }
        else
        {
            auto placeholder = std::make_unique<Component>();
            addChildComponent(*placeholder);
            subPanels_.push_back(std::move(placeholder));
        }

        if (index == 0)
            setSelectedSubTab(0);

        resized();
    }

    // Set the selected subtab index.
    void setSelectedSubTab(const int index)
    {
        subTabs_.setSelectedSubTab(index);
        for (int i = 0; i < static_cast<int>(subPanels_.size()); ++i)
        {
            const bool visible = (i == index);
            subPanels_[static_cast<std::size_t>(i)]->setVisible(visible);
        }
        resized();
    }

    // Return the selected subtab index.
    [[nodiscard]] int getSelectedSubTab() const noexcept
    {
        return subTabs_.getSelectedSubTab();
    }

    // Return the subtab strip.
    [[nodiscard]] SubTabStrip& getSubTabStrip() noexcept
    {
        return subTabs_;
    }

    void paint(Graphics& g) override
    {
        ignoreUnused(g);
    }

    void resized() override
    {
        const auto bounds = getLocalBounds();
        constexpr int subTabHeight = 28;
        subTabs_.setBounds(0, 0, bounds.getWidth(), subTabHeight);

        const auto contentArea = bounds.withTrimmedTop(subTabHeight + 8);
        for (auto& panel : subPanels_)
        {
            if (panel->isVisible())
                panel->setBounds(contentArea);
        }
    }

private:
    SubTabStrip subTabs_;
    std::vector<std::unique_ptr<Component>> subPanels_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PanelPage)
};

} // namespace MarsDSP::GUI

#endif

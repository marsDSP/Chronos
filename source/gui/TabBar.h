#pragma once

#ifndef CHRONOS_TAB_BAR_H
#define CHRONOS_TAB_BAR_H

#include <JuceHeader.h>
#include <functional>
#include <vector>

namespace MarsDSP::GUI {

// The top-level tab bar component.
class TabBar : public Component {
public:
    TabBar();
    ~TabBar() override = default;

    // Add one tab with the specified title and status dot colour.
    void addTab(const String& name, Colour dotColour);

    // Set the selected tab index.
    void setSelectedTab(int index);

    // Return the selected tab index.
    [[nodiscard]] int getSelectedTab() const noexcept { return selectedIndex_; }

    // Update the status dot colour for the specified tab index.
    void setTabDotColour(int index, Colour newColour);

    void paint(Graphics& g) override;
    void resized() override;

    std::function<void(int)> onTabChanged;

private:
    class TabButton : public Button {
    public:
        TabButton(const String& name, Colour dotColour);
        void setDotColour(Colour c);
        void paintButton(Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;

    private:
        Colour dotColour_;
    };

    std::vector<std::unique_ptr<TabButton>> buttons_;
    int selectedIndex_ = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TabBar)
};

} // namespace MarsDSP::GUI

#endif

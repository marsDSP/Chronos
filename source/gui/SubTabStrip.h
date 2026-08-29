#pragma once

#ifndef CHRONOS_SUB_TAB_STRIP_H
#define CHRONOS_SUB_TAB_STRIP_H

#include <JuceHeader.h>
#include "Colours.h"
#include <functional>
#include <vector>

namespace MarsDSP::GUI {

// A horizontal strip of subtab buttons.
class SubTabStrip : public Component {
public:
    SubTabStrip();
    ~SubTabStrip() override = default;

    // Add one subtab button with the specified name.
    void addSubTab(const String& name);

    // Set the accent colour for the active subtab.
    void setAccentColour(Colour c);

    // Set the selected subtab index.
    void setSelectedSubTab(int index);

    // Return the selected subtab index.
    [[nodiscard]] int getSelectedSubTab() const noexcept { return selectedIndex_; }

    void paint(Graphics& g) override;
    void resized() override;

    std::function<void(int)> onSubTabChanged;

private:
    class SubTabButton : public Button {
    public:
        explicit SubTabButton(const String& name);
        // Set the accent colour for the active state.
        void setAccentColour(Colour c);
        void paintButton(Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;
    private:
        Colour accent_ { Colours::textBright };
    };

    std::vector<std::unique_ptr<SubTabButton>> buttons_;
    int selectedIndex_ = 0;
    Colour accentColour_ { Colours::textBright };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SubTabStrip)
};

} // namespace MarsDSP::GUI

#endif

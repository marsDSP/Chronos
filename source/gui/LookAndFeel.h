#pragma once

#ifndef CHRONOS_LOOK_AND_FEEL_H
#define CHRONOS_LOOK_AND_FEEL_H

#include <JuceHeader.h>
#include "Colours.h"
#include "Metrics.h"

namespace MarsDSP::GUI {

// The look and feel for the plugin interface components.
class LookAndFeel : public LookAndFeel_V4 {
public:
    LookAndFeel();
    ~LookAndFeel() override = default;

    // Store the scale metrics for the draw hooks.
    void setMetrics(const Metrics& m);

    // Return the label font.
    Font getLabelFont(Label&) override;

    // Return the combo box font.
    Font getComboBoxFont(ComboBox&) override;

    // Return the popup menu font.
    Font getPopupMenuFont() override;

    // Draw the rotary slider.
    void drawRotarySlider(Graphics& g,
                          int x,
                          int y,
                          int width,
                          int height,
                          float sliderPos,
                          float rotaryStartAngle,
                          float rotaryEndAngle,
                          Slider& slider) override;

    // Draw the combo box background and arrow.
    void drawComboBox(Graphics& g,
                      int width,
                      int height,
                      bool isButtonDown,
                      int buttonX,
                      int buttonY,
                      int buttonW,
                      int buttonH,
                      ComboBox& box) override;

    // Draw the label text.
    void drawLabel(Graphics& g, Label& label) override;

    // Draw the popup menu background.
    void drawPopupMenuBackground(Graphics& g, int width, int height) override;

    // Draw one popup menu item.
    void drawPopupMenuItem(Graphics& g,
                           const Rectangle<int>& area,
                           bool isSeparator,
                           bool isActive,
                           bool isHighlighted,
                           bool isTicked,
                           bool hasSubMenu,
                           const String& text,
                           const String& shortcutKeyText,
                           const Drawable* icon,
                           const Colour* textColour) override;

    // Draw the tooltip background and text.
    void drawTooltip(Graphics& g, const String& text, int width, int height) override;

    bool shouldPopupMenuScaleWithTargetComponent(const PopupMenu::Options&) override
    {
        return false;
    }

private:
    Metrics metrics_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LookAndFeel)
};

} // namespace MarsDSP::GUI

#endif

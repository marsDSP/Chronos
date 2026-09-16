#pragma once

#ifndef CHRONOS_TAB_BUTTON_H
#define CHRONOS_TAB_BUTTON_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "../Fonts.h"
#include "../Metrics.h"

namespace MarsDSP::GUI
{
    // A page tab in the card title row. The active tab carries an accent
    // underline. An inactive marked tab carries an accent dot.
    class TabButton : public Button
    {
    public:
        explicit TabButton(const String& pageTitle)
            : Button(pageTitle)
        {
            setClickingTogglesState(false);
            setTitle(pageTitle + " page");
        }

        // Store the accent colour for the underline and the mark dot.
        void setAccentColour(Colour c)
        {
            accent_ = c;
            repaint();
        }

        // Store the scale metrics and repaint.
        void setMetrics(const Metrics& m)
        {
            metrics_ = m;
            repaint();
        }

        // Set the active state. The active tab shows the underline.
        void setActive(bool active)
        {
            active_ = active;
            repaint();
        }

        // Set the mark state. An inactive tab shows the dot.
        void setMarked(bool marked)
        {
            marked_ = marked;
            repaint();
        }

        // The width the label needs: the text plus the two side pads.
        [[nodiscard]] int preferredWidthPx(const Metrics& m) const
        {
            const Font f = Fonts::font(Fonts::Weight::Semibold, m.font(Metrics::kCardTitleFont));
            const float tw = Fonts::textWidth(f, getButtonText().toUpperCase());
            return roundToInt(tw) + 2 * roundToInt(m.pxf(Metrics::kTabPadX));
        }

        void paintButton(Graphics& g,
                         const bool shouldDrawButtonAsHighlighted,
                         const bool shouldDrawButtonAsDown) override
        {
            ignoreUnused(shouldDrawButtonAsDown);

            const auto m = metrics_;
            const auto bounds = getLocalBounds().toFloat();
            const String text = getButtonText().toUpperCase();
            const Font f = Fonts::font(Fonts::Weight::Semibold, m.font(Metrics::kCardTitleFont));

            g.setFont(f);
            g.setColour(active_ || shouldDrawButtonAsHighlighted ? Colours::textPrimary
                                                                 : Colours::textMuted);
            g.drawText(text, bounds, Justification::centredLeft, false);

            const float labelW = Fonts::textWidth(f, text);

            // The active tab carries the accent underline at the row bottom.
            if (active_)
            {
                const float uh = m.pxf(Metrics::kTabUnderline);
                g.setColour(accent_);
                g.fillRect(0.0f, bounds.getBottom() - uh, labelW, uh);
            }

            // An inactive marked tab carries the accent dot after the label.
            if (marked_ && ! active_)
            {
                const float r = m.pxf(Metrics::kTabDotR);
                const float gap = m.pxf(Metrics::kTabDotGap);
                // The cap spans the ascent above the text baseline.
                const float baseline = bounds.getCentreY()
                                       + (f.getAscent() - f.getDescent()) * 0.5f;
                const float cy = baseline - f.getAscent() * 0.5f;
                g.setColour(accent_);
                g.fillEllipse(labelW + gap, cy - r, r * 2.0f, r * 2.0f);
            }
        }

    private:
        Colour accent_ = Colours::accentDelayDigital;
        Metrics metrics_;
        bool active_ = false;
        bool marked_ = false;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TabButton)
    };
} // namespace MarsDSP::GUI

#endif

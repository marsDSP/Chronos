#pragma once

#ifndef CHRONOS_HISTORY_BUTTON_H
#define CHRONOS_HISTORY_BUTTON_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "../Metrics.h"

namespace MarsDSP::GUI
{
    // A button with a curved-arrow glyph for undo or redo.
    class HistoryButton : public Button
    {
    public:
        enum class Direction { Undo, Redo };

        explicit HistoryButton(Direction dir) : Button(dir == Direction::Undo ? "Undo" : "Redo"), dir_(dir) {}

        void setMetrics(const Metrics& m)
        {
            metrics_ = m;
            repaint();
        }

        void paintButton(Graphics& g,
                          const bool shouldDrawButtonAsHighlighted,
                          const bool shouldDrawButtonAsDown) override
        {
            ignoreUnused(shouldDrawButtonAsDown);

            const auto bounds = getLocalBounds().toFloat();
            const auto cx = bounds.getCentreX();
            const auto cy = bounds.getCentreY();
            constexpr float unit = 20.0f;
            const float scale = std::min(bounds.getWidth(), bounds.getHeight()) / unit;
            const float stroke = metrics_.stroke(Metrics::kIconStroke);

            const bool disabled = ! isEnabled();
            const bool hover = shouldDrawButtonAsHighlighted;

            Colour c = Colours::textMuted;
            if (disabled)
                c = Colours::textMuted.withMultipliedAlpha(kInertAlpha);
            else if (hover)
                c = Colours::textPrimary;

            g.setColour(c);

            // Draw a curved arrow in a 20 x 20 unit box.
            const float ox = cx - 10.0f * scale;
            const float oy = cy - 10.0f * scale;

            Path arrow;
            const float r = 7.0f * scale;
            const float startAngle = (dir_ == Direction::Undo)
                ? MathConstants<float>::pi + 0.4f
                : -0.4f;
            const float endAngle = (dir_ == Direction::Undo)
                ? MathConstants<float>::twoPi - 0.4f
                : MathConstants<float>::pi - 0.4f;

            arrow.addCentredArc(cx, cy, r, r, 0.0f, startAngle, endAngle, true);
            g.strokePath(arrow, PathStrokeType(stroke, PathStrokeType::mitered, PathStrokeType::rounded));

            // Arrow head.
            const float tipX = cx + r * std::cos(endAngle);
            const float tipY = cy + r * std::sin(endAngle);
            const float headLen = 4.0f * scale;
            const float headAngle = (dir_ == Direction::Undo) ? endAngle + 0.5f : endAngle - 0.5f;

            Path head;
            head.startNewSubPath(tipX, tipY);
            head.lineTo(tipX + headLen * std::cos(headAngle),
                        tipY + headLen * std::sin(headAngle));
            head.startNewSubPath(tipX, tipY);
            head.lineTo(tipX + headLen * std::cos(headAngle + 2.0f),
                        tipY + headLen * std::sin(headAngle + 2.0f));
            g.strokePath(head, PathStrokeType(stroke, PathStrokeType::mitered, PathStrokeType::rounded));
        }

    private:
        Direction dir_;
        Metrics metrics_;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HistoryButton)
    };
} // namespace MarsDSP::GUI

#endif

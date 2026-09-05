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

            // A curved arrow over the top of the circle. Undo ends on the
            // left, redo on the right; both heads point down. Redo mirrors
            // undo. The arc and the head share one angle convention
            // (cos/sin, 0 = right, y-down) so the head sits on the arc end.
            const float r = 7.0f * scale;
            const float a0 = (dir_ == Direction::Undo)
                ? 0.0f
                : MathConstants<float>::pi;
            const float a1 = (dir_ == Direction::Undo)
                ? -MathConstants<float>::pi
                : MathConstants<float>::twoPi;
            const float step = (a1 > a0) ? 0.1f : -0.1f;

            Path arrow;
            arrow.startNewSubPath(cx + r * std::cos(a0), cy + r * std::sin(a0));
            for (float a = a0 + step; (step > 0.0f ? a <= a1 : a >= a1); a += step)
                arrow.lineTo(cx + r * std::cos(a), cy + r * std::sin(a));
            arrow.lineTo(cx + r * std::cos(a1), cy + r * std::sin(a1));
            g.strokePath(arrow, PathStrokeType(stroke, PathStrokeType::mitered, PathStrokeType::rounded));

            // Arrow head at the tip, aligned to the arc tangent. dP/da is
            // (-sin a, cos a); travel flips sign when the arc runs backward,
            // and the head opens opposite the travel direction.
            const Point<float> tip(cx + r * std::cos(a1), cy + r * std::sin(a1));
            const Point<float> tangent(-std::sin(a1), std::cos(a1));
            const Point<float> backDir = (a1 > a0) ? -tangent : tangent;
            const float headLen = 4.0f * scale;
            const float spread = 0.5f;
            const float base = std::atan2(backDir.y, backDir.x);

            Path head;
            head.startNewSubPath(tip);
            head.lineTo(tip.x + headLen * std::cos(base + spread),
                        tip.y + headLen * std::sin(base + spread));
            head.startNewSubPath(tip);
            head.lineTo(tip.x + headLen * std::cos(base - spread),
                        tip.y + headLen * std::sin(base - spread));
            g.strokePath(head, PathStrokeType(stroke, PathStrokeType::mitered, PathStrokeType::rounded));
        }

    private:
        Direction dir_;
        Metrics metrics_;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HistoryButton)
    };
} // namespace MarsDSP::GUI

#endif

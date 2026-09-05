#pragma once

#ifndef CHRONOS_POWER_BUTTON_H
#define CHRONOS_POWER_BUTTON_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "../Metrics.h"

namespace MarsDSP::GUI
{
    // A toggle button with power, lock, or musical note icons.
    class PowerButton : public Button
    {
    public:
        PowerButton() : Button("PowerButton")
        {
            setClickingTogglesState(true);
        }

        // Set the colours for on and off states.
        void setColours(Colour toggledOnColour, Colour toggledOffColour)
        {
            onColour_ = toggledOnColour;
            offColour_ = toggledOffColour;
            repaint();
        }

        // Store the engaged colour. Keep the off colour.
        void setAccentColour(Colour c)
        {
            onColour_ = c;
            repaint();
        }

        // Store the scale metrics and repaint.
        void setMetrics(const Metrics &m)
        {
            metrics_ = m;
            repaint();
        }

        // Configure the button to draw a musical note icon.
        void setMusicalNote(const bool isNote)
        {
            isMusicalNote_ = isNote;
            isLock_ = false;
            repaint();
        }

        // Configure the button to draw a lock icon.
        void setLock(const bool lock)
        {
            isLock_ = lock;
            isMusicalNote_ = false;
            repaint();
        }

        void paintButton(Graphics &g,
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
            const float ox = cx - 10.0f * scale;
            const float oy = cy - 10.0f * scale;

            const bool state = getToggleState();
            g.setColour(state ? onColour_ : offColour_);

            if (shouldDrawButtonAsHighlighted)
                g.setOpacity(0.85f);

            if (isMusicalNote_)
            {
                Path note;
                note.addEllipse(6.5f * scale, 11.0f * scale, 5.0f * scale, 4.0f * scale);
                note.startNewSubPath(11.5f * scale, 13.0f * scale);
                note.lineTo(11.5f * scale, 5.0f * scale);
                note.lineTo(15.0f * scale, 8.0f * scale);
                note.applyTransform(AffineTransform::translation(ox, oy));
                g.strokePath(note, PathStrokeType(stroke, PathStrokeType::mitered, PathStrokeType::rounded));
            } else if (isLock_)
            {
                Path lock;
                lock.addRoundedRectangle(6.0f * scale, 9.0f * scale, 8.0f * scale, 6.0f * scale, scale);
                lock.startNewSubPath(8.0f * scale, 9.0f * scale);
                lock.lineTo(8.0f * scale, 7.0f * scale);
                lock.addArc(8.0f * scale, 5.0f * scale, 4.0f * scale, 4.0f * scale,
                            MathConstants<float>::pi, MathConstants<float>::twoPi, true);
                lock.lineTo(12.0f * scale, 9.0f * scale);
                lock.applyTransform(AffineTransform::translation(ox, oy));

                g.strokePath(lock, PathStrokeType(stroke, PathStrokeType::mitered, PathStrokeType::rounded));
                g.fillRoundedRectangle(cx - 4 * scale, cy - 1 * scale, 8 * scale, 6 * scale, scale);

                g.setColour(Colours::background);
                g.drawLine(cx, cy + 1 * scale, cx, cy + 3 * scale, stroke);
            } else
            {
                const float r = 5.0f * scale;
                Path powerCircle;
                powerCircle.addArc(cx - r, cy - r, r * 2, r * 2,
                                   0.6f, MathConstants<float>::twoPi - 0.6f, true);
                g.strokePath(powerCircle, PathStrokeType(stroke, PathStrokeType::mitered, PathStrokeType::rounded));
                g.drawLine(cx, cy - r - 2 * scale, cx, cy + 1 * scale, stroke);
            }
        }

    private:
        bool isMusicalNote_ = false;
        bool isLock_ = false;
        Colour onColour_ = Colours::textBright;
        Colour offColour_ = Colours::textDim;
        Metrics metrics_;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PowerButton)
    };
} // namespace MarsDSP::GUI

#endif

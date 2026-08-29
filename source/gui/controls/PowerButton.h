#pragma once

#ifndef CHRONOS_POWER_BUTTON_H
#define CHRONOS_POWER_BUTTON_H

#include <JuceHeader.h>
#include "../Colours.h"

namespace MarsDSP::GUI {

// A toggle button with power, lock, or musical note icons.
class PowerButton : public Button {
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

    void paintButton(Graphics& g,
                     const bool shouldDrawButtonAsHighlighted,
                     const bool shouldDrawButtonAsDown) override
    {
        const auto bounds = getLocalBounds().toFloat();
        const auto cx = bounds.getCentreX();
        const auto cy = bounds.getCentreY();
        const auto radius = std::min(bounds.getWidth(), bounds.getHeight()) * 0.25f;

        const bool state = getToggleState();
        g.setColour(state ? onColour_ : offColour_);

        if (shouldDrawButtonAsDown || shouldDrawButtonAsHighlighted)
            g.setOpacity(0.85f);

        if (isMusicalNote_)
        {
            Path note;
            note.addEllipse(cx - 3.5f, cy + 1.0f, 5.0f, 4.0f);
            note.startNewSubPath(cx + 1.5f, cy + 3.0f);
            note.lineTo(cx + 1.5f, cy - 5.0f);
            note.lineTo(cx + 5.0f, cy - 2.0f);
            g.strokePath(note, PathStrokeType(1.5f, PathStrokeType::mitered, PathStrokeType::rounded));
        }
        else if (isLock_)
        {
            Path lock;
            lock.addRoundedRectangle(cx - 4.0f, cy - 1.0f, 8.0f, 6.0f, 1.0f);
            lock.startNewSubPath(cx - 2.0f, cy - 1.0f);
            lock.lineTo(cx - 2.0f, cy - 3.0f);
            lock.addArc(cx - 2.0f, cy - 5.0f, 4.0f, 4.0f, MathConstants<float>::pi, MathConstants<float>::twoPi, true);
            lock.lineTo(cx + 2.0f, cy - 1.0f);

            g.strokePath(lock, PathStrokeType(1.5f, PathStrokeType::mitered, PathStrokeType::rounded));
            g.fillRoundedRectangle(cx - 4.0f, cy - 1.0f, 8.0f, 6.0f, 1.0f);

            g.setColour(Colours::background);
            g.drawLine(cx, cy + 1.0f, cx, cy + 3.0f, 1.0f);
        }
        else
        {
            Path powerCircle;
            powerCircle.addArc(cx - radius, cy - radius, radius * 2.0f, radius * 2.0f,
                               0.6f, MathConstants<float>::twoPi - 0.6f, true);
            g.strokePath(powerCircle, PathStrokeType(1.5f, PathStrokeType::mitered, PathStrokeType::rounded));
            g.drawLine(cx, cy - radius - 2.0f, cx, cy + 1.0f, 1.5f);
        }
    }

private:
    bool isMusicalNote_ = false;
    bool isLock_ = false;
    Colour onColour_ = Colours::textBright;
    Colour offColour_ = Colours::textDim;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PowerButton)
};

} // namespace MarsDSP::GUI

#endif

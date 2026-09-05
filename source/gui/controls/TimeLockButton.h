#pragma once

#ifndef CHRONOS_TIME_LOCK_BUTTON_H
#define CHRONOS_TIME_LOCK_BUTTON_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "../Metrics.h"

namespace MarsDSP::GUI {

// A toggle button for channel time linking.
class TimeLockButton : public Button {
public:
    TimeLockButton() : Button("TimeLockButton")
    {
        setClickingTogglesState(true);
    }

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

        const auto bounds = getLocalBounds().toFloat().reduced(3.0f, 1.0f);
        Colour colour = getToggleState() ? onColour_ : offColour_;

        if (shouldDrawButtonAsHighlighted)
            colour = colour.brighter(0.2f);

        Path path;
        path.startNewSubPath(0.5f, 7.37f);
        path.lineTo(0.5f, 4.81f);
        path.cubicTo(0.52f, 3.66f, 0.95f, 2.56f, 1.69f, 1.76f);
        path.cubicTo(2.44f, 0.95f, 3.45f, 0.5f, 4.50f, 0.5f);
        path.cubicTo(5.55f, 0.5f, 6.56f, 0.95f, 7.31f, 1.76f);
        path.cubicTo(8.05f, 2.56f, 8.48f, 3.66f, 8.50f, 4.81f);
        path.lineTo(8.50f, 7.37f);

        path.startNewSubPath(8.50f, 12.63f);
        path.lineTo(8.50f, 15.19f);
        path.cubicTo(8.48f, 16.34f, 8.05f, 17.44f, 7.31f, 18.24f);
        path.cubicTo(6.56f, 19.05f, 5.55f, 19.5f, 4.50f, 19.5f);
        path.cubicTo(3.45f, 19.5f, 2.44f, 19.05f, 1.69f, 18.24f);
        path.cubicTo(0.95f, 17.44f, 0.52f, 16.34f, 0.50f, 15.19f);
        path.lineTo(0.50f, 12.63f);

        path.startNewSubPath(4.50f, 6.50f);
        path.lineTo(4.50f, 13.50f);

        path.applyTransform(AffineTransform::rotation(MathConstants<float>::halfPi, 4.5f, 10.0f));

        const auto iconBounds = path.getBounds();
        const auto widthScale = bounds.getWidth() / iconBounds.getWidth();
        const auto heightScale = bounds.getHeight() / iconBounds.getHeight();
        const auto scale = std::min(widthScale, heightScale);
        const auto scaledWidth = iconBounds.getWidth() * scale;
        const auto scaledHeight = iconBounds.getHeight() * scale;
        const auto offsetX = bounds.getCentreX() - scaledWidth * 0.5f;
        const auto offsetY = bounds.getCentreY() - scaledHeight * 0.5f;

        path.applyTransform(AffineTransform::translation(-iconBounds.getX(), -iconBounds.getY()));
        path.applyTransform(AffineTransform::scale(scale));
        path.applyTransform(AffineTransform::translation(offsetX, offsetY));

        g.setColour(colour);
        g.strokePath(path, PathStrokeType(metrics_.stroke(Metrics::kLockStroke), PathStrokeType::curved, PathStrokeType::rounded));
    }

private:
    Colour onColour_ = Colours::accentDelayDigital;
    Colour offColour_ = Colours::textMuted;
    Metrics metrics_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TimeLockButton)
};

} // namespace MarsDSP::GUI

#endif

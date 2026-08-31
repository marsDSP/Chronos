#pragma once

#ifndef CHRONOS_DOT_MATRIX_DISPLAY_H
#define CHRONOS_DOT_MATRIX_DISPLAY_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "../Fonts.h"
#include "../Metrics.h"

namespace MarsDSP::GUI {

// A readout component that displays parameter values.
class DotMatrixDisplay : public Component {
public:
    DotMatrixDisplay() = default;
    ~DotMatrixDisplay() override = default;

    void setSlider(Slider* s)
    {
        slider_ = s;
        if (slider_ != nullptr)
        {
            slider_->onValueChange = [this] { repaint(); };
        }
        repaint();
    }

    void setAccentColour(const Colour c)
    {
        accentColour_ = c;
        repaint();
    }

    void paint(Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat();

        g.setColour(tintInk(accentColour_, kTintGridMinor));
        g.fillRoundedRectangle(bounds, 4.0f);

        g.setColour(accentColour_.withAlpha(0.2f));
        g.drawRoundedRectangle(bounds.reduced(0.5f), 4.0f, 1.0f);

        const String text = (slider_ != nullptr) ? slider_->getTextFromValue(slider_->getValue()) : "---";
        const Font f = Fonts::font(Fonts::Weight::Semibold, currentMetrics().font(13.0f));
        Fonts::drawFixedAdvanceText(g, f, text, bounds, accentColour_);
    }

    void mouseDown(const MouseEvent& e) override
    {
        if (slider_ == nullptr) return;
        dragStartValue_ = slider_->getValue();
        lastDragY_ = e.position.y;
    }

    void mouseDrag(const MouseEvent& e) override
    {
        if (slider_ == nullptr) return;

        const float delta = lastDragY_ - e.position.y;
        lastDragY_ = e.position.y;

        const double range = slider_->getMaximum() - slider_->getMinimum();
        const double sensitivity = range / 150.0;
        const double newVal = slider_->getValue() + delta * sensitivity;

        slider_->setValue(std::clamp(newVal, slider_->getMinimum(), slider_->getMaximum()), sendNotificationSync);
    }

    void mouseWheelMove(const MouseEvent&, const MouseWheelDetails& wheel) override
    {
        if (slider_ == nullptr) return;

        const double range = slider_->getMaximum() - slider_->getMinimum();
        const double step = range * 0.01;
        const double newVal = slider_->getValue() + wheel.deltaY * step * 5.0;

        slider_->setValue(std::clamp(newVal, slider_->getMinimum(), slider_->getMaximum()), sendNotificationSync);
    }

private:
    Slider* slider_ = nullptr;
    float lastDragY_ = 0.0f;
    double dragStartValue_ = 0.0;
    Colour accentColour_ = Colours::accentDelayDigital;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DotMatrixDisplay)
};

} // namespace MarsDSP::GUI

#endif

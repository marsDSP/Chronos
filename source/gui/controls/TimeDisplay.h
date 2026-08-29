#pragma once

#ifndef CHRONOS_TIME_DISPLAY_H
#define CHRONOS_TIME_DISPLAY_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "TimeDisplayFormatter.h"

namespace MarsDSP::GUI {

// A readout component that displays and edits delay time.
class TimeDisplay : public Component {
public:
    TimeDisplay() = default;
    ~TimeDisplay() override = default;

    void setSlider(Slider* s)
    {
        slider_ = s;
        if (slider_ != nullptr)
        {
            slider_->onValueChange = [this] { repaint(); };
        }
        repaint();
    }

    void setSyncState(const bool isSync)
    {
        syncActive_ = isSync;
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

        g.setColour(Colours::burntOrange);
        g.fillRoundedRectangle(bounds, 4.0f);

        g.setColour(accentColour_.withAlpha(0.2f));
        g.drawRoundedRectangle(bounds.reduced(0.5f), 4.0f, 1.0f);

        const String text = TimeDisplayFormatter::getDelayTimeText(slider_, syncActive_);

        g.setFont(Font(FontOptions(13.0f)).boldened());
        g.setColour(accentColour_);
        g.drawText(text, bounds, Justification::centred, false);
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
    bool syncActive_ = false;
    Colour accentColour_ = Colours::accentDelayDigital;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TimeDisplay)
};

} // namespace MarsDSP::GUI

#endif

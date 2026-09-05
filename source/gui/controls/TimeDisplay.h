#pragma once

#ifndef CHRONOS_TIME_DISPLAY_H
#define CHRONOS_TIME_DISPLAY_H

#include <JuceHeader.h>
#include "../Colours.h"
#include "../Fonts.h"
#include "../Metrics.h"
#include "TimeDisplayFormatter.h"

namespace MarsDSP::GUI {

// A readout component that displays and edits delay time.
class TimeDisplay : public Component, public SettableTooltipClient,
                    private Slider::Listener, private Timer {
public:
    TimeDisplay() { setWantsKeyboardFocus(true); }
    ~TimeDisplay() override
    {
        if (slider_ != nullptr)
            slider_->removeListener(this);
    }

    // Attach the slider and its parameter. The parameter carries the automation bracket.
    void setSlider(Slider* s, RangedAudioParameter* p)
    {
        if (slider_ != nullptr)
            slider_->removeListener(this);
        slider_ = s;
        param_ = p;
        if (slider_ != nullptr)
            slider_->addListener(this);
        repaint();
    }

    void setSyncState(const bool isSync)
    {
        syncActive_ = isSync;
        repaint();
    }

    // Store the tempo-sync division index for the sync readout.
    void setDivision(const int index)
    {
        divisionIndex_ = index;
        repaint();
    }

    void setAccentColour(const Colour c)
    {
        accentColour_ = c;
        repaint();
    }

    // Store the scale metrics and repaint.
    void setMetrics(const Metrics& m)
    {
        metrics_ = m;
        repaint();
    }

    // Repaint at the new alpha when the enablement changes.
    void enablementChanged() override
    {
        repaint();
    }

    void paint(Graphics& g) override
    {
        const auto m = metrics_;
        const auto bounds = getLocalBounds().toFloat();
        const float corner = m.pxf(Metrics::kCornerSmall);
        const float sw = m.stroke(Metrics::kHairline);

        // An inert readout draws at the inert alpha.
        const float inertA = isEnabled() ? 1.0f : kInertAlpha;

        g.setColour(tintInk(accentColour_, kTintReadoutFill).withMultipliedAlpha(inertA));
        g.fillRoundedRectangle(bounds, corner);

        g.setColour(tintInk(accentColour_, kTintReadoutBorder).withMultipliedAlpha(inertA));
        g.drawRoundedRectangle(bounds.reduced(sw / 2), corner, sw);

        const String text = TimeDisplayFormatter::getDelayTimeText(slider_, syncActive_, divisionIndex_);
        const Font f = Fonts::font(Fonts::Weight::Semibold, m.font(Metrics::kReadoutFont));
        Fonts::drawFixedAdvanceText(g, f, text, bounds, accentColour_.withMultipliedAlpha(inertA));
    }

    void mouseDown(const MouseEvent& e) override
    {
        // An inert readout takes no drag.
        if (! isEnabled()) return;
        // Close a wheel burst before the drag opens its own gesture.
        endWheelGesture_();
        if (slider_ == nullptr) return;
        dragStartValue_ = slider_->getValue();
        lastDragY_ = e.position.y;

        // Open one gesture on the parameter. mouseUp closes it on this pointer.
        dragParam_ = param_;
        if (dragParam_ != nullptr)
            dragParam_->beginChangeGesture();
    }

    void mouseDrag(const MouseEvent& e) override
    {
        if (slider_ == nullptr || dragParam_ == nullptr) return;

        const float delta = lastDragY_ - e.position.y;
        lastDragY_ = e.position.y;

        // Move in proportion space so the skew applies at low values.
        double prop = slider_->valueToProportionOfLength(slider_->getValue());
        prop = std::clamp(prop + delta / 150.0, 0.0, 1.0);

        dragParam_->setValueNotifyingHost(static_cast<float>(prop));
    }

    void mouseUp(const MouseEvent&) override
    {
        // Close the gesture on the pointer from mouseDown.
        if (dragParam_ != nullptr)
            dragParam_->endChangeGesture();
        dragParam_ = nullptr;
    }

    bool keyPressed(const KeyPress& key) override
    {
        // An inert readout takes no key.
        if (! isEnabled() || param_ == nullptr)
            return false;

        double dir = 0.0;
        if (key == KeyPress::rightKey || key == KeyPress::upKey)        dir = 1.0;
        else if (key == KeyPress::leftKey || key == KeyPress::downKey)  dir = -1.0;
        if (dir == 0.0)
            return false;

        // The same proportion law as the wheel.
        const bool fine = key.getModifiers().isShiftDown();
        const double step = dir * (fine ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse);
        double prop = slider_->valueToProportionOfLength(slider_->getValue());
        prop = std::clamp(prop + step, 0.0, 1.0);

        param_->beginChangeGesture();
        param_->setValueNotifyingHost(static_cast<float>(prop));
        param_->endChangeGesture();
        repaint();
        return true;
    }

    void mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& wheel) override
    {
        // An inert readout takes no wheel input.
        if (! isEnabled()) return;
        if (slider_ == nullptr || param_ == nullptr) return;

        const bool fine = e.mods.isShiftDown();
        const double step = fine ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse;
        double prop = slider_->valueToProportionOfLength(slider_->getValue());
        prop = std::clamp(prop + wheel.deltaY * step, 0.0, 1.0);

        // One gesture per burst. The idle timer closes it.
        if (wheelGestureParam_ == nullptr)
        {
            param_->beginChangeGesture();
            wheelGestureParam_ = param_;
        }
        param_->setValueNotifyingHost(static_cast<float>(prop));
        startTimer(Metrics::kWheelGestureMs);
    }

private:
    void timerCallback() override
    {
        stopTimer();
        endWheelGesture_();
    }

    // Close the wheel burst. mouseDown calls this before its own gesture.
    void endWheelGesture_()
    {
        stopTimer();
        if (wheelGestureParam_ != nullptr)
        {
            wheelGestureParam_->endChangeGesture();
            wheelGestureParam_ = nullptr;
        }
    }

    void sliderValueChanged(Slider*) override { repaint(); }

    Slider* slider_ = nullptr;
    // The parameter this readout edits. setSlider sets it with the slider.
    RangedAudioParameter* param_ = nullptr;
    // mouseDown stores the gesture target. mouseUp closes the gesture.
    RangedAudioParameter* dragParam_ = nullptr;
    // The open wheel burst. The idle timer closes it.
    RangedAudioParameter* wheelGestureParam_ = nullptr;
    float lastDragY_ = 0.0f;
    double dragStartValue_ = 0.0;
    bool syncActive_ = false;
    int divisionIndex_ = -1;
    Colour accentColour_ = Colours::accentDelayDigital;
    Metrics metrics_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TimeDisplay)
};

} // namespace MarsDSP::GUI

#endif

#include "PedalKnob.h"

namespace MarsDSP::GUI::Knobs {

// Value hold duration in milliseconds after the pointer leaves or a drag ends.
static constexpr int kValueHoldMs = 900;

PDLKnob::PDLKnob(const String &labelText,
                 AudioProcessorValueTreeState &state,
                 const ParameterID &pid,
                 PedalKnob& knobLnf)
    : labelText_(labelText), apvtsRef_(state), paramID_(pid.getParamID()), lnfRef_(knobLnf)
{
    slider.setSliderStyle(Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(Slider::NoTextBox, true, 0, 0);
    slider.setLookAndFeel(&lnfRef_);
    lnfRef_.registerSlider(&slider);
    addAndMakeVisible(slider);

    attachment = std::make_unique<AudioProcessorValueTreeState::SliderAttachment>(state, paramID_, slider);

    // Return the parameter default on a double-click of the knob body.
    if (auto *param = state.getParameter(paramID_))
    {
        const auto range = param->getNormalisableRange();
        slider.setDoubleClickReturnValue(true, range.convertFrom0to1(param->getDefaultValue()));
    }

    // Let the shift key swap into velocity mode for fine adjust.
    slider.setVelocityModeParameters(0.5, 1, 0.0, true, ModifierKeys::shiftModifier);

    // Handle the wheel on this component so the step is one parameter interval.
    slider.setScrollWheelEnabled(false);

    // Observe the slider through the Listener interface. Do not use onValueChange.
    slider.addListener(this);

    // Catch enter and exit on the slider through a mouse listener.
    slider.addMouseListener(this, true);

    // The value editor is a transient label over the label band.
    valueEditor_.setJustificationType(Justification::centred);
    valueEditor_.setColour(Label::backgroundColourId, Colours::panelBackground);
    valueEditor_.setColour(Label::textColourId, Colours::textBright);
    valueEditor_.setColour(Label::outlineColourId, Colours::panelBorder);
    valueEditor_.setEditable(true, false, false);
    valueEditor_.setVisible(false);
    addChildComponent(valueEditor_);
    valueEditor_.onEditorHide = [this] { applyEditorText_(); };
}

PDLKnob::~PDLKnob()
{
    slider.removeListener(this);
    slider.removeMouseListener(this);
    slider.setLookAndFeel(nullptr);
}

void PDLKnob::sliderDragStarted(Slider *)
{
    dragging_ = true;
    stopTimer();
    showValue_ = true;
    repaint();
}

void PDLKnob::sliderDragEnded(Slider *)
{
    dragging_ = false;
    if (! hovered_)
        startHoldTimer();
    else
        repaint();
}

void PDLKnob::mouseEnter(const MouseEvent &)
{
    hovered_ = true;
    stopTimer();
    showValue_ = true;
    repaint();
}

void PDLKnob::mouseExit(const MouseEvent &)
{
    hovered_ = false;
    if (! dragging_)
        startHoldTimer();
    else
        repaint();
}

void PDLKnob::mouseWheelMove(const MouseEvent &e, const MouseWheelDetails &wheel)
{
    // The slider listener and the parent bubble both call this. Drop the duplicate.
    if (e.eventTime == lastWheelTime_)
        return;
    lastWheelTime_ = e.eventTime;

    const bool fine = e.mods.isShiftDown();
    const double range = slider.getMaximum() - slider.getMinimum();
    double interval = slider.getInterval();
    if (interval <= 0.0)
        interval = range * 0.01;
    const double step = fine ? interval * 0.1 : interval;
    const double dir = (std::abs(wheel.deltaX) > std::abs(wheel.deltaY)) ? -wheel.deltaX : wheel.deltaY;
    const double newVal = std::clamp(slider.getValue() + dir * step,
                                     slider.getMinimum(), slider.getMaximum());
    if (std::abs(newVal - slider.getValue()) < 1e-12)
        return;

    if (auto *param = apvtsRef_.getParameter(paramID_))
    {
        param->beginChangeGesture();
        slider.setValue(newVal, sendNotificationSync);
        param->endChangeGesture();
    }
    else
    {
        slider.setValue(newVal, sendNotificationSync);
    }
}

void PDLKnob::mouseDoubleClick(const MouseEvent &e)
{
    // A double-click on the label band opens inline text entry.
    const auto m = metrics_;
    const int labelBandH = m.px(Metrics::kLabelBandH);
    if (e.position.y < static_cast<float>(labelBandH))
        showValueEditor_();
}

void PDLKnob::timerCallback()
{
    showValue_ = false;
    stopTimer();
    repaint();
}

void PDLKnob::startHoldTimer()
{
    showValue_ = true;
    startTimer(kValueHoldMs);
}

void PDLKnob::showValueEditor_()
{
    const auto m = metrics_;
    const int labelBandH = m.px(Metrics::kLabelBandH);
    const auto area = getLocalBounds().removeFromTop(labelBandH);

    valueEditor_.setText(slider.getTextFromValue(slider.getValue()), dontSendNotification);
    valueEditor_.setBounds(area);
    valueEditor_.setVisible(true);
    valueEditor_.showEditor();
}

void PDLKnob::applyEditorText_()
{
    valueEditor_.setVisible(false);

    const String text = valueEditor_.getText().trim();

    // Reject a string without a digit. Leave the parameter unchanged.
    if (! text.containsAnyOf("0123456789"))
        return;

    auto *param = apvtsRef_.getParameter(paramID_);
    if (param == nullptr)
        return;

    const float norm = param->getValueForText(text);

    // Reject a value outside the legal range. Leave the parameter unchanged.
    if (norm < 0.0f || norm > 1.0f)
        return;

    param->beginChangeGesture();
    param->setValueNotifyingHost(norm);
    param->endChangeGesture();
}

void PDLKnob::paint(Graphics &g)
{
    const auto m = metrics_;
    const float availW = static_cast<float>(getWidth()) - m.pxf(2.0f);
    const float baseH = m.font(11.0f);
    const int labelBandH = m.px(Metrics::kLabelBandH);
    const Rectangle<float> labelArea = getLocalBounds().removeFromTop(labelBandH).toFloat();

    // Draw the value in the accent when the pointer is over or the knob is dragged.
    if (showValue_ && ! valueEditor_.isVisible())
    {
        const String text = slider.getTextFromValue(slider.getValue());
        const Font f = Fonts::font(Fonts::Weight::Medium, baseH);
        Fonts::drawFixedAdvanceText(g, f, text, labelArea, accentColour_);
        return;
    }

    // Do not draw the label while the editor is open.
    if (valueEditor_.isVisible())
        return;

    auto drawCentered = [&](const String& t, const Font& font)
    {
        g.setFont(font);
        const float by = labelArea.getCentreY() + (font.getAscent() - font.getDescent()) * 0.5f;
        g.drawSingleLineText(t, juce::roundToInt(labelArea.getCentreX()), juce::roundToInt(by),
                              Justification::horizontallyCentred);
    };

    g.setColour(labelColour_);

    Font f = Fonts::font(Fonts::Weight::Medium, baseH);
    String text = labelText_;

    // Step 1: full label.
    if (f.getStringWidthFloat(text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 2: short form.
    text = Fonts::shortLabel(labelText_);
    if (f.getStringWidthFloat(text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 3: reduce the font height by up to 15%.
    f = Fonts::font(Fonts::Weight::Medium, baseH * 0.85f);
    if (f.getStringWidthFloat(text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 4: wrap the full label to two lines.
    f = Fonts::font(Fonts::Weight::Medium, baseH * 0.85f);
    g.setFont(f);
    const int sep = labelText_.indexOf(" ");
    if (sep >= 0)
    {
        const String line1 = labelText_.substring(0, sep);
        const String line2 = labelText_.substring(sep + 1);
        const float by1 = labelArea.getY() + f.getAscent();
        const float by2 = labelArea.getBottom() - f.getDescent();
        g.drawSingleLineText(line1, juce::roundToInt(labelArea.getCentreX()), juce::roundToInt(by1),
                              Justification::horizontallyCentred);
        g.drawSingleLineText(line2, juce::roundToInt(labelArea.getCentreX()), juce::roundToInt(by2),
                              Justification::horizontallyCentred);
    }
    else
    {
        drawCentered(text, f);
    }
}

void PDLKnob::resized()
{
    auto bounds = getLocalBounds();
    const auto m = metrics_;
    const int labelHeight = m.px(static_cast<float>(Metrics::kLabelBandH));
    const int knobLabelGap = m.px(static_cast<float>(Metrics::kKnobLabelGap));

    bounds.removeFromTop(labelHeight);
    bounds.removeFromTop(knobLabelGap);

    const auto size = std::min(bounds.getWidth(), bounds.getHeight());
    slider.setBounds(bounds.withSizeKeepingCentre(size, size));

    // Keep the value editor over the label band.
    if (valueEditor_.isVisible())
    {
        const auto area = getLocalBounds().removeFromTop(labelHeight);
        valueEditor_.setBounds(area);
    }
}

} // namespace MarsDSP::GUI::Knobs

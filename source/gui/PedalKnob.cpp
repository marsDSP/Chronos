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

    // Handle the wheel on this component so the step is the proportion law.
    slider.setScrollWheelEnabled(false);

    // The knob owns the right-click. The slider popup would double up.
    slider.setPopupMenuEnabled(false);

    // Arrow keys step the parameter through the owner's proportion law.
    slider.setWantsKeyboardFocus(true);
    slider.onKeyboardStep = [this](double dir, bool fine)
    {
        return keyboardStep_(dir, fine);
    };

    // Observe the slider through the Listener interface. Do not use onValueChange.
    slider.addListener(this);

    // Catch enter and exit on the slider through a mouse listener.
    slider.addMouseListener(this, true);

    // Set the accessible title so a knob without a tooltip still announces.
    slider.setTitle(labelText_);

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
    // Close an open wheel burst so no host touch dangles.
    endWheelGesture_();
    slider.removeListener(this);
    slider.removeMouseListener(this);
    slider.setLookAndFeel(nullptr);
}

void PDLKnob::sliderDragStarted(Slider *)
{
    // A drag takes over from a wheel burst on the same parameter.
    endWheelGesture_();
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

void PDLKnob::enablementChanged()
{
    slider.repaint();
    repaint();
}

void PDLKnob::mouseWheelMove(const MouseEvent &e, const MouseWheelDetails &wheel)
{
    // Handle only events that come from the slider.
    if (e.eventComponent != &slider)
        return;

    // An inert knob takes no wheel input.
    if (! isEnabled())
        return;

    // Move in proportion space, so the skew applies at the low end.
    const bool fine = e.mods.isShiftDown();
    const double step = fine ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse;
    double prop = slider.valueToProportionOfLength(slider.getValue());
    prop = std::clamp(prop + wheel.deltaY * step, 0.0, 1.0);

    // Snap to the interval, so an integer parameter lands on integers.
    double newVal = slider.proportionOfLengthToValue(prop);
    const double interval = slider.getInterval();
    if (interval > 0.0)
    {
        const double min = slider.getMinimum();
        newVal = min + std::round((newVal - min) / interval) * interval;
        prop = slider.valueToProportionOfLength(newVal);
    }

    auto* param = apvtsRef_.getParameter(paramID_);
    if (param == nullptr)
        return;

    // One gesture per burst. The idle timer closes it.
    if (! wheelGestureOpen_)
    {
        param->beginChangeGesture();
        wheelGestureOpen_ = true;
    }
    param->setValueNotifyingHost(static_cast<float>(prop));
    startWheelGestureTimer_();
}

void PDLKnob::mouseDown(const MouseEvent &e)
{
    // A right-click on the body or the label band opens the value entry.
    if (! e.mods.isPopupMenu() || ! isEnabled())
        return;
    if (e.eventComponent != this && e.eventComponent != &slider)
        return;

    PopupMenu m;
    m.addItem(1, "Enter value");
    const auto safe = SafePointer<PDLKnob>(this);
    m.showMenuAsync(PopupMenu::Options().withTargetComponent(this),
        [safe](int r) { if (safe != nullptr && r == 1) safe->showValueEditor_(); });
}

void PDLKnob::timerCallback()
{
    stopTimer();
    if (timerMode_ == TimerMode::WheelGesture)
    {
        endWheelGesture_();
        return;
    }
    showValue_ = false;
    repaint();
}

// One wheel burst opens one gesture. Each event restarts the idle
// timer, so a fast scroll records one automation touch.
void PDLKnob::startWheelGestureTimer_()
{
    timerMode_ = TimerMode::WheelGesture;
    startTimer(Metrics::kWheelGestureMs);
}

void PDLKnob::endWheelGesture_()
{
    timerMode_ = TimerMode::None;
    if (! wheelGestureOpen_)
        return;
    wheelGestureOpen_ = false;
    if (auto* param = apvtsRef_.getParameter(paramID_))
        param->endChangeGesture();
    showValue_ = false;
    repaint();
}

// Step the parameter in proportion space. One bracket pair per press.
bool PDLKnob::keyboardStep_(const double direction, const bool fine)
{
    if (! isEnabled())
        return false;
    auto* param = apvtsRef_.getParameter(paramID_);
    if (param == nullptr)
        return false;

    const double step = direction * (fine ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse);
    double prop = slider.valueToProportionOfLength(slider.getValue());
    prop = std::clamp(prop + step, 0.0, 1.0);

    // Snap to the interval, so an integer parameter lands on integers.
    double newVal = slider.proportionOfLengthToValue(prop);
    const double interval = slider.getInterval();
    if (interval > 0.0)
    {
        const double min = slider.getMinimum();
        newVal = min + std::round((newVal - min) / interval) * interval;
    }

    param->beginChangeGesture();
    param->setValueNotifyingHost(
        static_cast<float>(slider.valueToProportionOfLength(newVal)));
    param->endChangeGesture();
    repaint();
    return true;
}

void PDLKnob::startHoldTimer()
{
    // A wheel burst keeps the timer. The hold waits.
    if (wheelGestureOpen_)
        return;
    showValue_ = true;
    timerMode_ = TimerMode::Hold;
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
    const float availW = static_cast<float>(getWidth()) - m.pxf(Metrics::kKnobLabelInset);
    const float baseH = m.font(Metrics::kKnobLabelFont);
    const int labelBandH = m.px(Metrics::kLabelBandH);
    const Rectangle<float> labelArea = getLocalBounds().removeFromTop(labelBandH).toFloat();

    // An inert knob draws its text at the inert alpha.
    const float inertA = isEnabled() ? 1.0f : kInertAlpha;

    // Draw the value in the accent when the pointer is over or the knob is dragged.
    if (showValue_ && ! valueEditor_.isVisible())
    {
        const String text = slider.getTextFromValue(slider.getValue());
        const Font f = Fonts::font(Fonts::Weight::Medium, baseH);
        Fonts::drawFixedAdvanceText(g, f, text, labelArea, accentColour_.withMultipliedAlpha(inertA));
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

    g.setColour(labelColour_.withMultipliedAlpha(inertA));

    Font f = Fonts::font(Fonts::Weight::Medium, baseH);
    String text = labelText_;

    // Step 1: full label.
    if (Fonts::textWidth(f, text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 2: short form.
    text = Fonts::shortLabel(labelText_);
    if (Fonts::textWidth(f, text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 3: reduce the font height. Hold the result at or above the
    // minimum design height, so the shrink never makes text illegible.
    const float shrunkH = m.font(std::max(Metrics::kKnobLabelFont * 0.85f, Metrics::kFontMinDU));
    f = Fonts::font(Fonts::Weight::Medium, shrunkH);
    if (Fonts::textWidth(f, text) <= availW)
    {
        drawCentered(text, f);
        return;
    }

    // Step 4: wrap the full label to two lines.
    f = Fonts::font(Fonts::Weight::Medium, shrunkH);
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

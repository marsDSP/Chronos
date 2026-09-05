#include "DiffuserPad.h"
#include "Fonts.h"

#include <algorithm>
#include <cmath>

namespace MarsDSP::GUI {

DiffuserPad::DiffuserPad(AudioProcessorValueTreeState& apvts,
                         const String& diffusionID,
                         const String& sizeID,
                         const String& enableID)
    : apvts_(apvts),
      diffusionID_(diffusionID),
      sizeID_(sizeID),
      enableID_(enableID)
{
    diffusionParam_ = apvts.getParameter(diffusionID);
    sizeParam_      = apvts.getParameter(sizeID);
    enableParam_    = apvts.getParameter(enableID);

    apvts.addParameterListener(diffusionID, this);
    apvts.addParameterListener(sizeID, this);
    apvts.addParameterListener(enableID, this);

    setWantsKeyboardFocus(true);
    setTitle("Diffuser");
    setTooltip("Drag to set the diffusion and the size. Double-click to reset.");
    setHelpText("Drag to set the diffusion and the size. Double-click to reset.");
}

DiffuserPad::~DiffuserPad()
{
    endWheelGestures_();
    apvts_.removeParameterListener(diffusionID_, this);
    apvts_.removeParameterListener(sizeID_, this);
    apvts_.removeParameterListener(enableID_, this);
}

void DiffuserPad::setAccentColour(Colour c)
{
    accent_ = c;
    repaint();
}

void DiffuserPad::setMetrics(const Metrics& m)
{
    metrics_ = m;
    resized();
    repaint();
}

Rectangle<float> DiffuserPad::activeArea_() const noexcept
{
    const float inset = metrics_.pxf(static_cast<float>(Metrics::kPadInset));
    return getLocalBounds().toFloat().reduced(inset);
}

void DiffuserPad::parameterChanged(const String& parameterID, const float newValue)
{
    if (parameterID == diffusionID_)
        pendingDiffusion_.store(newValue, std::memory_order_relaxed);
    else if (parameterID == sizeID_)
        pendingSize_.store(newValue, std::memory_order_relaxed);
    else if (parameterID == enableID_)
        pendingEnable_.store(newValue, std::memory_order_relaxed);

    triggerAsyncUpdate();
}

void DiffuserPad::handleAsyncUpdate()
{
    repaint();
}

void DiffuserPad::timerCallback()
{
    stopTimer();
    endWheelGestures_();
}

void DiffuserPad::endWheelGestures_()
{
    if (wheelDiffusionOpen_ && diffusionParam_ != nullptr)
        diffusionParam_->endChangeGesture();
    if (wheelSizeOpen_ && sizeParam_ != nullptr)
        sizeParam_->endChangeGesture();
    wheelDiffusionOpen_ = false;
    wheelSizeOpen_ = false;
}

void DiffuserPad::paint(Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    const float corner = metrics_.pxf(Metrics::kCornerDisplay);
    const float sw = metrics_.stroke(Metrics::kHairline);
    const float inertA = isEnabled() ? 1.0f : kInertAlpha;

    // The display surface.
    g.setColour(tintInk(accent_, kTintPlotFill).withMultipliedAlpha(inertA));
    g.fillRoundedRectangle(bounds, corner);
    g.setColour(tintInk(accent_, kTintDisplayBorder).withMultipliedAlpha(inertA));
    g.drawRoundedRectangle(bounds.reduced(sw / 2), corner, sw);

    if (diffusionParam_ == nullptr || sizeParam_ == nullptr)
        return;

    const float diffusion = diffusionParam_->getValue();
    const float size = sizeParam_->getValue();
    const auto area = activeArea_();

    // The handle. A ring in the accent with a centre dot.
    const float hx = area.getX() + size * area.getWidth();
    const float hy = area.getBottom() - diffusion * area.getHeight();
    const float r = metrics_.pxf(Metrics::kPadHandleR);
    const float handleAlpha = (hovered_ || dragging_) ? 1.0f : 0.8f;

    g.setColour(accent_.withMultipliedAlpha(inertA * handleAlpha));
    g.drawEllipse(hx - r, hy - r, r * 2.0f, r * 2.0f, metrics_.stroke(Metrics::kHairline));
    g.fillEllipse(hx - r * 0.35f, hy - r * 0.35f, r * 0.7f, r * 0.7f);

    // The readout while hovered or dragged.
    if (hovered_ || dragging_)
    {
        const String text = "DIFF " + diffusionParam_->getCurrentValueAsText()
                          + "  " + String::charToString(static_cast<juce_wchar>(0x00B7))
                          + "  SIZE " + sizeParam_->getCurrentValueAsText();
        const Font f = Fonts::display(metrics_.displayFont(Metrics::kPadReadoutFont));
        g.setFont(f);
        g.setColour(accent_.withMultipliedAlpha(inertA));
        const float inset = metrics_.pxf(Metrics::kTapLabelInset);
        g.drawText(text, roundToInt(inset), roundToInt(inset),
                   getWidth() - roundToInt(2.0f * inset), roundToInt(f.getHeight()),
                   Justification::topLeft, false);
    }
}

void DiffuserPad::resized()
{
    repaint();
}

void DiffuserPad::mouseDown(const MouseEvent& e)
{
    if (! isEnabled())
        return;

    endWheelGestures_();

    dragDiffusion_ = diffusionParam_;
    dragSize_ = sizeParam_;
    if (dragDiffusion_ != nullptr)
    {
        startDiffusion_ = dragDiffusion_->getValue();
        dragDiffusion_->beginChangeGesture();
    }
    if (dragSize_ != nullptr)
    {
        startSize_ = dragSize_->getValue();
        dragSize_->beginChangeGesture();
    }

    dragStartX_ = e.position.x;
    dragStartY_ = e.position.y;
    dragging_ = true;
    shiftLatch_ = 0;
    hovered_ = true;
    repaint();
}

void DiffuserPad::mouseDrag(const MouseEvent& e)
{
    if (! dragging_ || ! isEnabled())
        return;

    const float dx = e.position.x - dragStartX_;
    const float dy = e.position.y - dragStartY_;
    const auto area = activeArea_();

    // The Shift axis latch. Hold one axis after the dead zone clears.
    bool writeSize = true;
    bool writeDiffusion = true;
    if (e.mods.isShiftDown())
    {
        if (shiftLatch_ == 0)
        {
            const float dz = metrics_.pxf(Metrics::kDragDeadZone);
            if (std::fabs(dx) >= dz || std::fabs(dy) >= dz)
                shiftLatch_ = (std::fabs(dx) >= std::fabs(dy)) ? 1 : 2;
        }
        if (shiftLatch_ == 1)
            writeDiffusion = false;
        else if (shiftLatch_ == 2)
            writeSize = false;
        else
        {
            // Inside the dead zone. Write nothing yet.
            return;
        }
    }

    if (writeSize && dragSize_ != nullptr && area.getWidth() > 0.0f)
    {
        const float next = std::clamp(startSize_ + dx / area.getWidth(), 0.0f, 1.0f);
        dragSize_->setValueNotifyingHost(next);
    }
    if (writeDiffusion && dragDiffusion_ != nullptr && area.getHeight() > 0.0f)
    {
        const float next = std::clamp(startDiffusion_ - dy / area.getHeight(), 0.0f, 1.0f);
        dragDiffusion_->setValueNotifyingHost(next);
    }
}

void DiffuserPad::mouseUp(const MouseEvent&)
{
    if (! dragging_)
        return;

    if (dragDiffusion_ != nullptr)
        dragDiffusion_->endChangeGesture();
    if (dragSize_ != nullptr)
        dragSize_->endChangeGesture();

    dragDiffusion_ = nullptr;
    dragSize_ = nullptr;
    dragging_ = false;
    shiftLatch_ = 0;
}

void DiffuserPad::mouseDoubleClick(const MouseEvent&)
{
    if (! isEnabled())
        return;

    // Reset both to their defaults. One bracket per parameter.
    if (diffusionParam_ != nullptr)
    {
        diffusionParam_->beginChangeGesture();
        diffusionParam_->setValueNotifyingHost(diffusionParam_->getDefaultValue());
        diffusionParam_->endChangeGesture();
    }
    if (sizeParam_ != nullptr)
    {
        sizeParam_->beginChangeGesture();
        sizeParam_->setValueNotifyingHost(sizeParam_->getDefaultValue());
        sizeParam_->endChangeGesture();
    }
}

void DiffuserPad::mouseMove(const MouseEvent&)
{
    if (! hovered_)
    {
        hovered_ = true;
        repaint();
    }
}

void DiffuserPad::mouseExit(const MouseEvent&)
{
    hovered_ = false;
    repaint();
}

void DiffuserPad::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& wheel)
{
    if (! isEnabled())
        return;

    const bool fine = e.mods.isShiftDown();
    const double step = fine ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse;

    if (std::fabs(wheel.deltaY) > 0.0f && diffusionParam_ != nullptr)
    {
        if (! wheelDiffusionOpen_)
        {
            diffusionParam_->beginChangeGesture();
            wheelDiffusionOpen_ = true;
        }
        const float next = std::clamp(diffusionParam_->getValue() + static_cast<float>(wheel.deltaY * step), 0.0f, 1.0f);
        diffusionParam_->setValueNotifyingHost(next);
    }

    if (std::fabs(wheel.deltaX) > 0.0f && sizeParam_ != nullptr)
    {
        if (! wheelSizeOpen_)
        {
            sizeParam_->beginChangeGesture();
            wheelSizeOpen_ = true;
        }
        const float next = std::clamp(sizeParam_->getValue() + static_cast<float>(wheel.deltaX * step), 0.0f, 1.0f);
        sizeParam_->setValueNotifyingHost(next);
    }

    startTimer(Metrics::kWheelGestureMs);
}

bool DiffuserPad::keyPressed(const KeyPress& key)
{
    if (! isEnabled())
        return false;

    double sizeDir = 0.0;
    double diffDir = 0.0;
    if (key == KeyPress::rightKey)       sizeDir = 1.0;
    else if (key == KeyPress::leftKey)   sizeDir = -1.0;
    else if (key == KeyPress::upKey)     diffDir = 1.0;
    else if (key == KeyPress::downKey)   diffDir = -1.0;
    else
        return false;

    const bool fine = key.getModifiers().isShiftDown();
    const double step = fine ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse;

    if (sizeDir != 0.0 && sizeParam_ != nullptr)
    {
        const float next = std::clamp(sizeParam_->getValue() + static_cast<float>(sizeDir * step), 0.0f, 1.0f);
        sizeParam_->beginChangeGesture();
        sizeParam_->setValueNotifyingHost(next);
        sizeParam_->endChangeGesture();
    }
    if (diffDir != 0.0 && diffusionParam_ != nullptr)
    {
        const float next = std::clamp(diffusionParam_->getValue() + static_cast<float>(diffDir * step), 0.0f, 1.0f);
        diffusionParam_->beginChangeGesture();
        diffusionParam_->setValueNotifyingHost(next);
        diffusionParam_->endChangeGesture();
    }

    return true;
}

void DiffuserPad::enablementChanged()
{
    repaint();
}

} // namespace MarsDSP::GUI

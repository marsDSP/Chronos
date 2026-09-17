#include "DiffuserPad.h"
#include "Fonts.h"

#include <algorithm>
#include <cmath>

namespace MarsDSP::GUI {

namespace {

// The readout labels in cube order: x, y, z, w.
constexpr const char* kAxisLabels[DiffuserPad::kNumAxes] = { "SIZE", "DIFF", "DEPTH", "RATE" };
// The readout cell names for the hover hint, in the same order.
constexpr const char* kAxisNames[DiffuserPad::kNumAxes] = { "diffuser size", "diffusion", "mod depth", "mod rate" };

// Draw one line in the cube's dash pattern.
void drawDashed(Graphics& g, const Line<float>& line, const Metrics& m)
{
    const float dashes[2] = { m.pxf(Metrics::kCubeDashOn), m.pxf(Metrics::kCubeDashOff) };
    g.drawDashedLine(line, dashes, 2, m.stroke(Metrics::kIconStroke));
}

// Draw the four edges of a square from its bottom-left corner, dashed.
void drawDashedSquare(Graphics& g, const Point<float> bottomLeft, const float side, const Metrics& m)
{
    const auto bl = bottomLeft;
    const auto br = bl.translated(side, 0.0f);
    const auto tl = bl.translated(0.0f, -side);
    const auto tr = bl.translated(side, -side);
    drawDashed(g, { bl, br }, m);
    drawDashed(g, { br, tr }, m);
    drawDashed(g, { tr, tl }, m);
    drawDashed(g, { tl, bl }, m);
}

} // namespace

DiffuserPad::DiffuserPad(AudioProcessorValueTreeState& apvts,
                         const String& sizeID,
                         const String& diffusionID,
                         const String& depthID,
                         const String& rateID,
                         const String& enableID)
    : apvts_(apvts),
      enableID_(enableID)
{
    axes_[kX].id = sizeID;
    axes_[kY].id = diffusionID;
    axes_[kZ].id = depthID;
    axes_[kW].id = rateID;
    for (auto& a : axes_)
    {
        a.param = apvts.getParameter(a.id);
        apvts.addParameterListener(a.id, this);
    }
    enableParam_ = apvts.getParameter(enableID);
    apvts.addParameterListener(enableID, this);

    setWantsKeyboardFocus(true);
    setTitle("Diffuser");
    // The tooltip is one footer line; the help text carries the full guide.
    setTooltip("Drag: size, diffusion. Scroll: depth. Alt-scroll: rate.");
    setHelpText("Drag to set the size and the diffusion. Scroll to slide the plane and set "
                "the modulation depth; hold Alt while scrolling for the rate. Hold Shift to "
                "lock one axis of a drag, or for a fine step. Drag a readout value to set it "
                "alone. Double-click to reset.");

    // Prime the slice fade from the live enable value and snap to rest.
    // source/target are the fixed off/on endpoints; only target(bool)
    // ever flips direction afterward, so a reversal mid-fade stays
    // continuous instead of snapping.
    if (enableParam_ != nullptr)
    {
        lastEnableOn_ = enableParam_->getValue() > 0.5f;
        fadeAnim_.setSourceValue(0.0f);
        fadeAnim_.setTargetValue(1.0f);
        fadeAnim_.target(lastEnableOn_, true);
        fade_ = fadeAnim_.value();
    }
}

DiffuserPad::~DiffuserPad()
{
    endWheelGestures_();
    closeDragGestures_();
    for (const auto& a : axes_)
        apvts_.removeParameterListener(a.id, this);
    apvts_.removeParameterListener(enableID_, this);
}

void DiffuserPad::setAccentColour(const Colour c)
{
    accent_ = c;
    repaint();
}

void DiffuserPad::setMetrics(const Metrics& m)
{
    metrics_ = m;
    repaint();
}

DiffuserPad::CubeGeometry DiffuserPad::cubeGeometry_() const noexcept
{
    CubeGeometry geo;
    auto region = getLocalBounds().toFloat().reduced(metrics_.pxf(static_cast<float>(Metrics::kPadInset)));
    geo.readout = region.removeFromBottom(2.0f * metrics_.pxf(static_cast<float>(Metrics::kPadReadoutRowH)));
    region.removeFromBottom(metrics_.pxf(static_cast<float>(Metrics::kPadReadoutGap)));

    // The oblique cube. The front square plus the depth offset spans
    // kCubeFillFrac of the smaller region dimension, centred on both axes.
    const float side = Metrics::kCubeFillFrac
                     * std::max(0.0f, std::min(region.getWidth(), region.getHeight()))
                     / (1.0f + Metrics::kCubeDepthFrac);
    const float dv = Metrics::kCubeDepthFrac * side;
    const float total = side + dv;
    const float left = region.getCentreX() - 0.5f * total;
    const float top = region.getCentreY() - 0.5f * total;
    geo.side = side;
    geo.depth = { dv, -dv };
    geo.front = { left, top + total };
    return geo;
}

int DiffuserPad::readoutAxisAt_(const Point<float> p) const noexcept
{
    const auto r = cubeGeometry_().readout;
    if (! r.contains(p))
        return -1;
    const int col = p.x < r.getCentreX() ? 0 : 1;
    const int row = p.y < r.getCentreY() ? 0 : 1;
    return row * 2 + col;
}

float DiffuserPad::value_(const int axis) const noexcept
{
    const auto* p = axes_[static_cast<std::size_t>(axis)].param;
    return p != nullptr ? p->getValue() : 0.0f;
}

void DiffuserPad::setValue_(const int axis, const float v01)
{
    if (auto* p = axes_[static_cast<std::size_t>(axis)].param)
        p->setValueNotifyingHost(std::clamp(v01, 0.0f, 1.0f));
}

void DiffuserPad::stepValue_(const int axis, const float delta01)
{
    setValue_(axis, value_(axis) + delta01);
}

void DiffuserPad::parameterChanged(const String& parameterID, const float newValue)
{
    // Store and defer. The paint reads the parameters on the message thread.
    if (parameterID == enableID_)
        pendingEnable_.store(newValue, std::memory_order_relaxed);
    triggerAsyncUpdate();
}

void DiffuserPad::handleAsyncUpdate()
{
    // Retarget the slice fade on an enable transition only. target()
    // resets the ease time base, so a repeated call would stall a fade.
    if (const float en = pendingEnable_.load(std::memory_order_relaxed); en >= 0.0f)
    {
        if (const bool on = en > 0.5f; on != lastEnableOn_)
        {
            lastEnableOn_ = on;
            fadeAnim_.target(on);
            if (fadeAnim_.isAnimating())
                startTimerHz(60);
        }
    }
    repaint();
}

void DiffuserPad::timerCallback()
{
    // Advance the slice fade. The vendored Animation is frame-rate
    // independent, so the ease lands in the same wall time at any rate.
    const bool fading = fadeAnim_.isAnimating();
    if (fading)
        fade_ = fadeAnim_.update();

    // Close a wheel or key burst once it has been idle long enough.
    bool changed = fading;
    if (Time::getMillisecondCounter() - lastBurstMs_ >= static_cast<uint32>(Metrics::kWheelGestureMs))
        changed = endWheelGestures_() || changed;

    if (changed)
        repaint();

    // Keep the timer alive while the fade eases or a burst is still open.
    const bool burstOpen = std::any_of(axes_.begin(), axes_.end(),
        [](const AxisBinding& a) { return a.wheelOpen || (a.lit && ! a.dragOpen); });
    if (fadeAnim_.isAnimating() || burstOpen)
        return;

    stopTimer();
}

void DiffuserPad::closeDragGestures_()
{
    for (auto& a : axes_)
    {
        if (a.dragOpen && a.param != nullptr)
            a.param->endChangeGesture();
        a.dragOpen = false;
        if (! a.wheelOpen)
            a.lit = false;
    }
}

bool DiffuserPad::endWheelGestures_()
{
    bool changed = false;
    for (auto& a : axes_)
    {
        if (a.wheelOpen && a.param != nullptr)
            a.param->endChangeGesture();
        changed = changed || a.wheelOpen || (a.lit && ! a.dragOpen);
        a.wheelOpen = false;
        if (! a.dragOpen)
            a.lit = false;
    }
    return changed;
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

    const auto geo = cubeGeometry_();
    if (geo.side <= 0.0f)
        return;

    const float x = value_(kX);
    const float y = value_(kY);
    const float z = value_(kZ);
    const float w = value_(kW);
    const float side = geo.side;
    const auto front = geo.front;
    const auto back = front + geo.depth;
    const bool live = hovered_ || dragging_;

    // The wireframe, painted back to front so the slice occludes what
    // lies behind it and the front face stays on top. The rear half of
    // each depth edge (back face to slice) goes under the slice fill; the
    // front half (slice to front face) goes over it. The rear geometry
    // takes the dimmer tint for depth.
    const auto slice = front + geo.depth * z;
    const Point<float> corners[4] = { { 0.0f, 0.0f }, { side, 0.0f }, { 0.0f, -side }, { side, -side } };
    const Colour rearColour = tintInk(accent_, kTintCardBorder).withMultipliedAlpha(inertA);
    const Colour nearColour = tintInk(accent_, kTintCentreLine).withMultipliedAlpha(inertA);

    g.setColour(rearColour);
    drawDashedSquare(g, back, side, metrics_);
    for (const auto& c : corners)
        drawDashed(g, { back + c, slice + c }, metrics_);

    // The z slice. A translucent plane at depth z along the diagonal. The
    // fill follows the enable fade; the outline stays so z reads when off.
    const Rectangle<float> sliceRect(slice.x, slice.y - side, side, side);
    g.setColour(accent_.withAlpha(kCubeSliceAlpha * fade_ * inertA));
    g.fillRect(sliceRect);
    g.setColour(nearColour);
    drawDashedSquare(g, slice, side, metrics_);
    for (const auto& c : corners)
        drawDashed(g, { slice + c, front + c }, metrics_);
    drawDashedSquare(g, front, side, metrics_);

    // The handle. A square in the slice at (x, y).
    const float hr = metrics_.pxf(Metrics::kPadHandleR);
    const Point<float> handle(slice.x + x * side, slice.y - y * side);
    const float handleA = (live ? 1.0f : kCubeHandleAlpha) * inertA;
    g.setColour(accent_.withMultipliedAlpha(handleA));
    g.fillRect(handle.x - hr, handle.y - hr, 2.0f * hr, 2.0f * hr);

    // The w ring. A faint track, and the accent arc sweeping w of a turn
    // clockwise from twelve o'clock.
    const float ringR = metrics_.pxf(Metrics::kCubeRingR);
    g.setColour(tintInk(accent_, kTintGridMajor).withMultipliedAlpha(inertA));
    g.drawEllipse(handle.x - ringR, handle.y - ringR, 2.0f * ringR, 2.0f * ringR, sw);
    if (w > 0.0f)
    {
        Path arc;
        arc.addArc(handle.x - ringR, handle.y - ringR, 2.0f * ringR, 2.0f * ringR,
                   0.0f, w * MathConstants<float>::twoPi, true);
        const float ringA = ((live || axes_[kW].lit) ? 1.0f : kCubeHandleAlpha) * inertA;
        g.setColour(accent_.withMultipliedAlpha(ringA));
        g.strokePath(arc, PathStrokeType(metrics_.stroke(Metrics::kCubeRingStroke),
                                         PathStrokeType::curved, PathStrokeType::rounded));
    }

    // The axis tripod at the front bottom-left corner: the projections of
    // three model vectors of equal length, so the z arm is the depth
    // vector scaled by that length over the side, foreshortened exactly
    // as the cube's own depth edges are. An arm paints in the accent
    // while its axis is being written.
    const float armLen = metrics_.pxf(Metrics::kCubeAxisLen);
    const float armSw = metrics_.stroke(Metrics::kIconStroke);
    const Point<float> armEnds[3] = { front.translated(armLen, 0.0f),
                                      front.translated(0.0f, -armLen),
                                      front + geo.depth * (armLen / side) };
    const int armAxes[3] = { kX, kY, kZ };
    for (int i = 0; i < 3; ++i)
    {
        const bool lit = axes_[static_cast<std::size_t>(armAxes[i])].lit;
        g.setColour((lit ? accent_ : Colours::textMuted).withMultipliedAlpha(inertA));
        g.drawLine({ front, armEnds[i] }, armSw);
    }

    // The readout band. Two rows of label and value, the value string
    // from the parameter at a fixed digit advance so it does not shift.
    const Font labelFont = Fonts::font(Fonts::Weight::Medium, metrics_.font(Metrics::kTapLabelFont));
    const Font valueFont = Fonts::display(metrics_.displayFont(Metrics::kPadReadoutFont));
    const float rowH = geo.readout.getHeight() * 0.5f;
    const float colW = geo.readout.getWidth() * 0.5f;
    const float labelW = Fonts::textWidth(labelFont, "DEPTH")
                       + metrics_.pxf(static_cast<float>(Metrics::kKnobLabelGap));
    const Colour labelColour = Colours::textMuted.withMultipliedAlpha(inertA);

    for (int axis = 0; axis < kNumAxes; ++axis)
    {
        const auto& a = axes_[static_cast<std::size_t>(axis)];
        Rectangle<float> cell(geo.readout.getX() + static_cast<float>(axis % 2) * colW,
                              geo.readout.getY() + static_cast<float>(axis / 2) * rowH,
                              colW, rowH);
        const auto labelArea = cell.removeFromLeft(labelW);
        g.setFont(labelFont);
        g.setColour(labelColour);
        g.drawText(kAxisLabels[axis], labelArea, Justification::centredLeft, false);

        if (a.param == nullptr)
            continue;
        const String text = a.param->getCurrentValueAsText();
        const bool valueLive = a.lit || hoverAxis_ == axis;
        const Colour valueColour = (valueLive ? accent_ : Colours::textPrimary).withMultipliedAlpha(inertA);
        // Left-align: hand the centring draw a box exactly as wide as the text.
        Fonts::drawFixedAdvanceText(g, valueFont, text,
                                    cell.withWidth(Fonts::fixedAdvanceWidth(valueFont, text)), valueColour);
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

    // Open the brackets here and close them in mouseUp, so a modifier
    // change mid-drag cannot unbalance them. A press on a readout cell
    // drags that one axis; a press on the cube drags the size/diffusion
    // pair. Depth and rate belong to the wheel.
    auto open = [&](const int axis)
    {
        auto& a = axes_[static_cast<std::size_t>(axis)];
        if (a.param == nullptr)
            return;
        a.dragStart = a.param->getValue();
        a.param->beginChangeGesture();
        a.dragOpen = true;
        a.lit = true;
    };

    valueDragAxis_ = readoutAxisAt_(e.position);
    if (valueDragAxis_ >= 0)
    {
        open(valueDragAxis_);
    }
    else
    {
        open(kX);
        open(kY);
    }

    dragStartX_ = e.position.x;
    dragStartY_ = e.position.y;
    lastDragY_ = e.position.y;
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
    const float side = cubeGeometry_().side;
    if (side <= 0.0f)
        return;

    // A readout value drag: one axis, vertical, one slice side per full
    // range, Shift fine. Incremental, so toggling Shift never jumps.
    if (valueDragAxis_ >= 0)
    {
        const float stepY = e.position.y - lastDragY_;
        lastDragY_ = e.position.y;
        const auto fine = static_cast<float>(Metrics::kWheelStepFine / Metrics::kWheelStepCoarse);
        const float gain = e.mods.isShiftDown() ? fine : 1.0f;
        if (axes_[static_cast<std::size_t>(valueDragAxis_)].dragOpen)
            stepValue_(valueDragAxis_, -gain * stepY / side);
        return;
    }

    // The Shift axis latch. Hold one axis after the dead zone clears.
    bool writeH = true;
    bool writeV = true;
    if (e.mods.isShiftDown())
    {
        if (shiftLatch_ == 0)
        {
            const float dz = metrics_.pxf(Metrics::kDragDeadZone);
            if (std::fabs(dx) >= dz || std::fabs(dy) >= dz)
                shiftLatch_ = (std::fabs(dx) >= std::fabs(dy)) ? 1 : 2;
        }
        if (shiftLatch_ == 1)
            writeV = false;
        else if (shiftLatch_ == 2)
            writeH = false;
        else
            return; // inside the dead zone, write nothing yet
    }

    auto& h = axes_[kX];
    auto& v = axes_[kY];

    // A full drag across the slice side spans the whole range.
    if (writeH && h.dragOpen)
        setValue_(kX, h.dragStart + dx / side);
    if (writeV && v.dragOpen)
        setValue_(kY, v.dragStart - dy / side);

    // Light the arms being written.
    const bool litH = writeH && h.dragOpen;
    const bool litV = writeV && v.dragOpen;
    if (h.lit != litH || v.lit != litV)
    {
        h.lit = litH;
        v.lit = litV;
        repaint();
    }
}

void DiffuserPad::mouseUp(const MouseEvent&)
{
    if (! dragging_)
        return;

    closeDragGestures_();
    dragging_ = false;
    valueDragAxis_ = -1;
    shiftLatch_ = 0;
    repaint();
}

void DiffuserPad::mouseDoubleClick(const MouseEvent& e)
{
    if (! isEnabled())
        return;

    // The second click's mouseDown opened a drag. Close it first, so the
    // reset brackets never nest inside an open gesture.
    closeDragGestures_();
    dragging_ = false;
    valueDragAxis_ = -1;
    shiftLatch_ = 0;

    // A readout cell resets its own axis; the cube resets all four.
    // One bracket per parameter.
    const int only = readoutAxisAt_(e.position);
    for (int axis = 0; axis < kNumAxes; ++axis)
    {
        if (only >= 0 && axis != only)
            continue;
        auto* p = axes_[static_cast<std::size_t>(axis)].param;
        if (p == nullptr)
            continue;
        p->beginChangeGesture();
        p->setValueNotifyingHost(p->getDefaultValue());
        p->endChangeGesture();
    }
}

void DiffuserPad::mouseMove(const MouseEvent& e)
{
    // Track the readout cell under the pointer. The cursor announces a
    // draggable value.
    const int over = readoutAxisAt_(e.position);
    const bool changed = over != hoverAxis_ || ! hovered_;
    if (over != hoverAxis_)
    {
        hoverAxis_ = over;
        setMouseCursor(over >= 0 ? MouseCursor::UpDownResizeCursor : MouseCursor::NormalCursor);
    }
    hovered_ = true;
    if (changed)
        repaint();
}

void DiffuserPad::mouseExit(const MouseEvent&)
{
    hovered_ = false;
    hoverAxis_ = -1;
    setMouseCursor(MouseCursor::NormalCursor);
    repaint();
}

void DiffuserPad::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& wheel)
{
    if (! isEnabled())
        return;

    // The wheel owns depth and rate, the pair the drag does not: a plain
    // scroll slides the plane (z), and Alt hands the scroll to the rate
    // ring (w). A sideways scroll takes the other of the two. One notch
    // steps the same proportion as one arrow press.
    const bool alt = e.mods.isAltDown();
    const int vAxis = alt ? kW : kZ;
    const int hAxis = alt ? kZ : kW;
    const double step = (e.mods.isShiftDown() ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse)
                      / Metrics::kWheelNotchDelta;

    // One bracketed gesture per burst per parameter. A drag already
    // holding the parameter's gesture is not reopened.
    auto burst = [&](const int axis, const float delta)
    {
        auto& a = axes_[static_cast<std::size_t>(axis)];
        if (a.param == nullptr || delta == 0.0f)
            return;
        if (! a.wheelOpen && ! a.dragOpen)
        {
            a.param->beginChangeGesture();
            a.wheelOpen = true;
        }
        a.lit = true;
        stepValue_(axis, static_cast<float>(static_cast<double>(delta) * step));
    };
    // Over a readout cell the wheel steps that axis alone.
    if (const int over = readoutAxisAt_(e.position); over >= 0)
    {
        burst(over, wheel.deltaY);
    }
    else
    {
        burst(vAxis, wheel.deltaY);
        burst(hAxis, wheel.deltaX);
    }

    lastBurstMs_ = Time::getMillisecondCounter();
    if (! isTimerRunning())
        startTimer(Metrics::kWheelGestureMs);
}

bool DiffuserPad::keyPressed(const KeyPress& key)
{
    if (! isEnabled())
        return false;

    // Compare key codes directly: KeyPress::operator==(int) rejects any
    // held modifier, which would drop the Shift and Alt forms.
    const int code = key.getKeyCode();
    float hDir = 0.0f;
    float vDir = 0.0f;
    if (code == KeyPress::rightKey)      hDir = 1.0f;
    else if (code == KeyPress::leftKey)  hDir = -1.0f;
    else if (code == KeyPress::upKey)    vDir = 1.0f;
    else if (code == KeyPress::downKey)  vDir = -1.0f;
    else
        return false;

    const auto mods = key.getModifiers();
    const bool alt = mods.isAltDown();
    const auto step = static_cast<float>(mods.isShiftDown() ? Metrics::kWheelStepFine
                                                            : Metrics::kWheelStepCoarse);

    // One bracket per step, unless a drag or burst already holds it.
    auto nudge = [&](const int axis, const float dir)
    {
        auto& a = axes_[static_cast<std::size_t>(axis)];
        if (a.param == nullptr || dir == 0.0f)
            return;
        const bool held = a.dragOpen || a.wheelOpen;
        if (! held) a.param->beginChangeGesture();
        stepValue_(axis, dir * step);
        if (! held) a.param->endChangeGesture();
        a.lit = true;
    };
    nudge(horizontalAxis_(alt), hDir);
    nudge(verticalAxis_(alt), vDir);

    // Light the arm for one burst interval.
    lastBurstMs_ = Time::getMillisecondCounter();
    if (! isTimerRunning())
        startTimer(Metrics::kWheelGestureMs);
    return true;
}

void DiffuserPad::enablementChanged()
{
    repaint();
}

String DiffuserPad::getTooltip()
{
    if (hoverAxis_ >= 0)
        return "Drag or scroll to set the " + String(kAxisNames[hoverAxis_]) + ". Double-click resets.";
    return SettableTooltipClient::getTooltip();
}

} // namespace MarsDSP::GUI

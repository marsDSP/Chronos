#include "EqDisplay.h"
#include "../../ChronosProcessor.h"
#include "../../ChronosParameters.h"
#include "../Fonts.h"
#include "dsp/ParametricEQ.h"

#include <algorithm>
#include <cmath>

namespace MarsDSP::GUI {

namespace {

// The labelled frequencies along the bottom of the plot: the decades. The
// grid still carries every 1..9 line; more labels crowd a card-width plot.
constexpr float kLabelFreqs[] = { 100.0f, 1000.0f, 10000.0f };
// The labelled dB lines down the right edge: every other grid line.
constexpr float kLabelDbStep = 12.0f;

String freqLabel(const float f)
{
    return f >= 1000.0f ? String(roundToInt(f / 1000.0f)) + "k" : String(roundToInt(f));
}

// Menu item ids. Types take 1..4, the rest are fixed.
constexpr int kMenuTypeBase = 1;
constexpr int kMenuRemove = 10;
constexpr int kMenuReset = 11;
constexpr int kMenuAddBase = 20;

// The zones that pick a new band's type from its frequency.
constexpr float kLowShelfZoneHz = 100.0f;
constexpr float kHighShelfZoneHz = 6000.0f;

} // namespace

EqDisplay::EqDisplay(ChronosProcessor& proc)
    : proc_(proc), apvts_(proc.getAPVTS())
{
    setWantsKeyboardFocus(true);
    setTitle("Filter EQ");
    setTooltip("Right-click to add a band. Drag a node; scroll for Q.");
    setHelpText("The output filter. Drag a node to set its frequency and gain; hold Shift to lock "
                "one axis. Scroll over a band for its Q. Right-click or double-click empty space "
                "to add a band; right-click a node for its type or to remove it; double-click a "
                "node to remove it. Arrow keys nudge the selected band.");

    slots_[kLowCut].freq = apvts_.getParameter(hpfFreqParamID.getParamID());
    slots_[kHighCut].freq = apvts_.getParameter(lpfFreqParamID.getParamID());
    for (int i = 0; i < kNumEqBands; ++i)
    {
        const auto& ids = eqBandParamIDs[static_cast<std::size_t>(i)];
        auto& s = slots_[static_cast<std::size_t>(1 + i)];
        s.on = apvts_.getParameter(ids.on.getParamID());
        s.type = apvts_.getParameter(ids.type.getParamID());
        s.freq = apvts_.getParameter(ids.freq.getParamID());
        s.gain = apvts_.getParameter(ids.gain.getParamID());
        s.q = apvts_.getParameter(ids.q.getParamID());
    }

    for (const auto& s : slots_)
        for (auto* p : { s.on, s.type, s.freq, s.gain, s.q })
            if (p != nullptr)
            {
                listened_.push_back(p->paramID);
                apvts_.addParameterListener(p->paramID, this);
            }

    pull_.resize(static_cast<std::size_t>(SpectrumAnalyser::kSize));
}

EqDisplay::~EqDisplay()
{
    stopTimer();
    closeDragGestures_();
    endWheelGesture_();
    for (const auto& id : listened_)
        apvts_.removeParameterListener(id, this);
}

void EqDisplay::setAccentColour(const Colour c)
{
    accent_ = c;
    repaint();
}

void EqDisplay::setMetrics(const Metrics& m)
{
    metrics_ = m;
    repaint();
}

void EqDisplay::resized()
{
    repaint();
}

// Run the timer only while the page is showing on a visible peer. A fresh
// start drops the FIFO backlog and the trace, so a hidden page never replays.
void EqDisplay::updateTimerState_()
{
    if (isShowing())
    {
        if (! isTimerRunning())
        {
            proc_.getSpectrumFifo().clear();
            analyser_.clear();
            traceSample_.fill(-1.0e9f);
            startTimerHz(static_cast<int>(SpectrumAnalyser::kRefreshHz));
        }
    }
    else
    {
        stopTimer();
        if (endWheelGesture_())
            repaint();
    }
}

void EqDisplay::parameterChanged(const String&, float)
{
    // Store nothing: the paint reads the parameters on the message thread.
    triggerAsyncUpdate();
}

void EqDisplay::handleAsyncUpdate()
{
    repaint();
}

void EqDisplay::timerCallback()
{
    // Follow the host sample rate.
    const double sr = sampleRate_();
    if (! analyser_.isPrepared() || std::abs(analyser_.sampleRate() - sr) > 0.1)
        analyser_.prepare(sr);

    // Drain the feed.
    auto& fifo = proc_.getSpectrumFifo();
    bool got = false;
    for (;;)
    {
        const auto n = fifo.read(pull_.data(), pull_.size());
        if (n == 0)
            break;
        analyser_.push(pull_.data(), static_cast<int>(n));
        got = true;
    }

    // Repaint only when the trace moved: compare a sampled set of bins.
    bool changed = false;
    if (got)
    {
        analyser_.update();
        for (std::size_t i = 0; i < traceSample_.size(); ++i)
        {
            const float v = analyser_.binDb(1 + static_cast<int>(i) * kTraceSampleStride);
            if (std::abs(v - traceSample_[i]) > 0.01f)
            {
                changed = true;
                traceSample_[i] = v;
            }
        }
    }

    // Close an idle wheel burst.
    if (wheelParam_ != nullptr
        && Time::getMillisecondCounter() - lastBurstMs_ >= static_cast<uint32>(Metrics::kWheelGestureMs))
        changed = endWheelGesture_() || changed;

    if (changed)
        repaint();
}

// ----------------------------------------------------------------------
// Geometry
// ----------------------------------------------------------------------

Rectangle<float> EqDisplay::plot_() const noexcept
{
    return getLocalBounds().toFloat().reduced(metrics_.pxf(static_cast<float>(Metrics::kPadInset)));
}

float EqDisplay::freqToX(float f, const Rectangle<float>& plot) noexcept
{
    f = std::clamp(f, kMinFreq, kMaxFreq);
    return plot.getX() + plot.getWidth() * (std::log(f / kMinFreq) / std::log(kMaxFreq / kMinFreq));
}

float EqDisplay::xToFreq(const float x, const Rectangle<float>& plot) noexcept
{
    const float t = std::clamp((x - plot.getX()) / std::max(1.0f, plot.getWidth()), 0.0f, 1.0f);
    return kMinFreq * std::exp(t * std::log(kMaxFreq / kMinFreq));
}

float EqDisplay::gainToY(const float db, const Rectangle<float>& plot) noexcept
{
    return jmap(db, kGainMaxDb, -kGainMaxDb, plot.getY(), plot.getBottom());
}

float EqDisplay::yToGain(const float y, const Rectangle<float>& plot) noexcept
{
    return std::clamp(jmap(y, plot.getY(), plot.getBottom(), kGainMaxDb, -kGainMaxDb), -kGainMaxDb, kGainMaxDb);
}

float EqDisplay::specToY(const float db, const Rectangle<float>& plot) noexcept
{
    return jmap(db, 0.0f, SpectrumAnalyser::kFloorDb, plot.getY(), plot.getBottom());
}

// ----------------------------------------------------------------------
// The band model
// ----------------------------------------------------------------------

bool EqDisplay::slotOn(const int slot) const noexcept
{
    if (isCut(slot)) return true;
    const auto* p = slots_[static_cast<std::size_t>(slot)].on;
    return p != nullptr && p->getValue() > 0.5f;
}

int EqDisplay::slotType(const int slot) const noexcept
{
    if (isCut(slot)) return -1;
    const auto* p = slots_[static_cast<std::size_t>(slot)].type;
    return p != nullptr ? roundToInt(p->convertFrom0to1(p->getValue())) : 0;
}

bool EqDisplay::slotHasGain(const int slot) const noexcept
{
    return ! isCut(slot) && slotType(slot) != static_cast<int>(Filters::ParametricEQ::Type::Notch);
}

float EqDisplay::slotFreq(const int slot) const noexcept
{
    const auto* p = slots_[static_cast<std::size_t>(slot)].freq;
    if (p != nullptr) return p->convertFrom0to1(p->getValue());
    return slot == kLowCut ? kMinFreq : slot == kHighCut ? kMaxFreq : kEqDefaultFreq[slot - 1];
}

float EqDisplay::slotGain(const int slot) const noexcept
{
    const auto* p = slots_[static_cast<std::size_t>(slot)].gain;
    return p != nullptr ? p->convertFrom0to1(p->getValue()) : 0.0f;
}

float EqDisplay::slotQ(const int slot) const noexcept
{
    const auto* p = slots_[static_cast<std::size_t>(slot)].q;
    return p != nullptr ? p->convertFrom0to1(p->getValue()) : kEqDefaultQ;
}

double EqDisplay::sampleRate_() const noexcept
{
    const double sr = proc_.getSampleRate();
    return sr > 0.0 ? sr : 48000.0;
}

// The whole response: the cut pair plus every free band that is on.
double EqDisplay::responseDb_(const double f) const noexcept
{
    using Filters::ParametricEQ;
    const double fs = sampleRate_();
    double db = ParametricEQ::cutMagnitudeDb(true, fs, f, slotFreq(kLowCut))
              + ParametricEQ::cutMagnitudeDb(false, fs, f, slotFreq(kHighCut));
    for (int s = 1; s < kHighCut; ++s)
        if (slotOn(s))
            db += ParametricEQ::bandMagnitudeDb(slotType(s), fs, f, slotFreq(s), slotGain(s), slotQ(s));
    return db;
}

Point<float> EqDisplay::nodePos_(const int slot, const Rectangle<float>& plot) const noexcept
{
    return { freqToX(slotFreq(slot), plot), gainToY(slotHasGain(slot) ? slotGain(slot) : 0.0f, plot) };
}

// The node under a point. Off bands are not drawn, so they are not grabbable.
int EqDisplay::nodeAt_(const Point<float> p) const noexcept
{
    const auto plot = plot_();
    const float reach = metrics_.pxf(Metrics::kEqHitRadius);
    int best = -1;
    float bestD = reach;
    for (int s = 0; s < kNumSlots; ++s)
    {
        if (! slotOn(s)) continue;
        const float d = nodePos_(s, plot).getDistanceFrom(p);
        if (d <= bestD)
        {
            bestD = d;
            best = s;
        }
    }
    return best;
}

// The nearest free band that is on, by column, within the wheel reach.
int EqDisplay::nearestBandByX_(const float x) const noexcept
{
    const auto plot = plot_();
    const float reach = metrics_.pxf(Metrics::kEqWheelReachPx);
    int best = -1;
    float bestD = reach;
    for (int s = 1; s < kHighCut; ++s)
    {
        if (! slotOn(s)) continue;
        const float d = std::abs(freqToX(slotFreq(s), plot) - x);
        if (d <= bestD)
        {
            bestD = d;
            best = s;
        }
    }
    return best;
}

String EqDisplay::slotName_(const int slot) const
{
    if (slot == kLowCut) return "Low Cut";
    if (slot == kHighCut) return "High Cut";
    const int t = std::clamp(slotType(slot), 0, kEqTypeNames.size() - 1);
    return "Band " + String(slot) + " " + String::charToString(static_cast<juce_wchar>(0x00B7)) + " " + kEqTypeNames[t];
}

// ----------------------------------------------------------------------
// Parameter writes
// ----------------------------------------------------------------------

void EqDisplay::setDenorm_(RangedAudioParameter* p, const float denorm)
{
    if (p != nullptr)
        p->setValueNotifyingHost(p->convertTo0to1(denorm));
}

void EqDisplay::writeOnce_(RangedAudioParameter* p, const float norm)
{
    if (p == nullptr) return;
    p->beginChangeGesture();
    p->setValueNotifyingHost(std::clamp(norm, 0.0f, 1.0f));
    p->endChangeGesture();
}

void EqDisplay::openDragGesture_(RangedAudioParameter* p)
{
    if (p == nullptr) return;
    p->beginChangeGesture();
    dragGestures_.push_back(p);
}

void EqDisplay::closeDragGestures_()
{
    for (auto* p : dragGestures_)
        p->endChangeGesture();
    dragGestures_.clear();
}

bool EqDisplay::endWheelGesture_()
{
    if (wheelParam_ == nullptr) return false;
    wheelParam_->endChangeGesture();
    wheelParam_ = nullptr;
    return true;
}

// ----------------------------------------------------------------------
// Paint
// ----------------------------------------------------------------------

void EqDisplay::paint(Graphics& g)
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

    const auto plot = plot_();
    if (plot.getWidth() <= 1.0f || plot.getHeight() <= 1.0f)
        return;

    // The frequency grid: every 1..9 of each decade, {1, 2, 5} major.
    const Colour gridMinor = tintInk(accent_, kTintGridMinor).withMultipliedAlpha(inertA);
    const Colour gridMajor = tintInk(accent_, kTintGridMajor).withMultipliedAlpha(inertA);
    for (const float decade : { 10.0f, 100.0f, 1000.0f, 10000.0f })
    {
        for (int m = 1; m <= 9; ++m)
        {
            const float f = decade * static_cast<float>(m);
            if (f < kMinFreq || f > kMaxFreq) continue;
            const bool major = (m == 1 || m == 2 || m == 5);
            g.setColour(major ? gridMajor : gridMinor);
            g.fillRect(freqToX(f, plot), plot.getY(), sw, plot.getHeight());
        }
    }

    // The dB grid for the EQ scale, the 0 dB line as the centre line.
    for (float db = -kGainMaxDb; db <= kGainMaxDb; db += kGridDb)
    {
        const bool zero = std::abs(db) < 0.5f * kGridDb;
        g.setColour(zero ? tintInk(accent_, kTintCentreLine).withMultipliedAlpha(inertA) : gridMinor);
        g.fillRect(plot.getX(), gainToY(db, plot), plot.getWidth(), sw);
    }

    const int cols = std::max(2, static_cast<int>(std::ceil(plot.getWidth())) + 1);
    const float x0 = plot.getX();

    // The spectrum trace: the makima spline over the bin knots evaluated
    // once per pixel column, filled to the plot bottom under a hairline edge.
    if (analyser_.isPrepared() && analyser_.hasData())
    {
        if (static_cast<int>(colFreq_.size()) != cols)
        {
            colFreq_.resize(static_cast<std::size_t>(cols));
            colDb_.resize(static_cast<std::size_t>(cols));
            for (int i = 0; i < cols; ++i)
                colFreq_[static_cast<std::size_t>(i)] = xToFreq(std::min(x0 + static_cast<float>(i), plot.getRight()), plot);
        }
        analyser_.evaluateDb(colFreq_.data(), colDb_.data(), cols);

        Path line;
        for (int i = 0; i < cols; ++i)
        {
            const float x = std::min(x0 + static_cast<float>(i), plot.getRight());
            const float y = std::clamp(specToY(colDb_[static_cast<std::size_t>(i)], plot), plot.getY(), plot.getBottom());
            if (i == 0) line.startNewSubPath(x, y);
            else        line.lineTo(x, y);
        }
        Path fill = line;
        fill.lineTo(plot.getRight(), plot.getBottom());
        fill.lineTo(x0, plot.getBottom());
        fill.closeSubPath();

        g.saveState();
        g.reduceClipRegion(plot.toNearestInt());
        g.setColour(accent_.withAlpha(kEqSpectrumFillAlpha * inertA));
        g.fillPath(fill);
        g.setColour(accent_.withAlpha(kEqSpectrumLineAlpha * inertA));
        g.strokePath(line, PathStrokeType(sw));
        g.restoreState();
    }

    // The response curve, filled to the 0 dB line.
    {
        const float zeroY = gainToY(0.0f, plot);
        Path line;
        for (int i = 0; i < cols; ++i)
        {
            const float x = std::min(x0 + static_cast<float>(i), plot.getRight());
            const auto db = static_cast<float>(responseDb_(xToFreq(x, plot)));
            const float y = gainToY(std::clamp(db, -kGainMaxDb, kGainMaxDb), plot);
            if (i == 0) line.startNewSubPath(x, y);
            else        line.lineTo(x, y);
        }
        Path fill = line;
        fill.lineTo(plot.getRight(), zeroY);
        fill.lineTo(x0, zeroY);
        fill.closeSubPath();

        g.saveState();
        g.reduceClipRegion(plot.toNearestInt());
        g.setColour(accent_.withAlpha(kEqCurveFillAlpha * inertA));
        g.fillPath(fill);
        g.setColour(accent_.withMultipliedAlpha(inertA));
        g.strokePath(line, PathStrokeType(metrics_.stroke(Metrics::kEqCurveStroke),
                                          PathStrokeType::curved, PathStrokeType::rounded));
        g.restoreState();
    }

    // The band nodes: accent discs for the bands that are on. An off band
    // is not drawn at all; a right-click brings it back. The selected node
    // carries a ring at the hit radius.
    const float r = metrics_.pxf(Metrics::kEqNodeR);
    const float ringR = metrics_.pxf(Metrics::kEqHitRadius);
    for (int s = 0; s < kNumSlots; ++s)
    {
        if (! slotOn(s)) continue;
        const auto p = nodePos_(s, plot);
        const bool live = (s == hover_ || s == dragSlot_ || s == selected_);
        g.setColour(accent_.withMultipliedAlpha((live ? 1.0f : kCubeHandleAlpha) * inertA));
        g.fillEllipse(p.x - r, p.y - r, 2.0f * r, 2.0f * r);
        if (s == selected_)
        {
            g.setColour(accent_.withMultipliedAlpha(inertA));
            g.drawEllipse(p.x - ringR, p.y - ringR, 2.0f * ringR, 2.0f * ringR, sw);
        }
    }

    // The labels: the decades along the bottom, every other dB line down
    // the right edge, both inside the plot.
    const Font labelFont = Fonts::font(Fonts::Weight::Medium, metrics_.font(Metrics::kTapLabelFont));
    g.setFont(labelFont);
    g.setColour(Colours::rulerText.withMultipliedAlpha(inertA));
    const float inset = metrics_.pxf(Metrics::kEqLabelInset);
    const float fontH = labelFont.getHeight();
    const float labelY = plot.getBottom() - inset - fontH;
    for (const float f : kLabelFreqs)
    {
        const String text = freqLabel(f);
        const float w = Fonts::textWidth(labelFont, text) + 2.0f * inset;
        g.drawText(text, Rectangle<float>(freqToX(f, plot) + inset, labelY, w, fontH), Justification::centredLeft, false);
    }
    for (float db = -kGainMaxDb + kLabelDbStep; db < kGainMaxDb; db += kLabelDbStep)
    {
        const String text = (db > 0.0f ? "+" : "") + String(roundToInt(db));
        const float w = Fonts::textWidth(labelFont, text) + 2.0f * inset;
        const float y = std::clamp(gainToY(db, plot) - 0.5f * fontH, plot.getY(), plot.getBottom() - fontH);
        g.drawText(text, Rectangle<float>(plot.getRight() - inset - w, y, w, fontH), Justification::centredRight, false);
    }
}

// ----------------------------------------------------------------------
// Interaction
// ----------------------------------------------------------------------

void EqDisplay::mouseDown(const MouseEvent& e)
{
    if (! isEnabled())
        return;

    // Right-click: a node's menu, or the add-band menu over empty plot.
    if (e.mods.isPopupMenu())
    {
        if (const int slot = nodeAt_(e.position); slot >= 0)
        {
            selected_ = slot;
            repaint();
            showBandMenu_(slot);
        }
        else if (plot_().contains(e.position))
        {
            showAddMenu_(e.position);
        }
        return;
    }

    endWheelGesture_();
    const int slot = nodeAt_(e.position);
    selected_ = slot;
    dragSlot_ = slot;
    shiftLatch_ = 0;
    if (slot >= 0)
    {
        dragStart_ = e.position;
        dragStartFreq_ = slotFreq(slot);
        dragStartGain_ = slotGain(slot);
        openDragGesture_(slots_[static_cast<std::size_t>(slot)].freq);
        if (slotHasGain(slot))
            openDragGesture_(slots_[static_cast<std::size_t>(slot)].gain);
    }
    repaint();
}

void EqDisplay::mouseDrag(const MouseEvent& e)
{
    if (dragSlot_ < 0 || ! isEnabled())
        return;

    const auto plot = plot_();
    const float dx = e.position.x - dragStart_.x;
    const float dy = e.position.y - dragStart_.y;

    // The Shift axis latch: hold one axis after the dead zone clears.
    bool writeX = true;
    bool writeY = true;
    if (e.mods.isShiftDown())
    {
        if (shiftLatch_ == 0)
        {
            const float dz = metrics_.pxf(Metrics::kDragDeadZone);
            if (std::fabs(dx) >= dz || std::fabs(dy) >= dz)
                shiftLatch_ = (std::fabs(dx) >= std::fabs(dy)) ? 1 : 2;
        }
        if (shiftLatch_ == 1)      writeY = false;
        else if (shiftLatch_ == 2) writeX = false;
        else                       return;
    }

    auto& s = slots_[static_cast<std::size_t>(dragSlot_)];
    if (writeX)
        setDenorm_(s.freq, xToFreq(freqToX(dragStartFreq_, plot) + dx, plot));
    if (writeY && slotHasGain(dragSlot_))
        setDenorm_(s.gain, yToGain(gainToY(dragStartGain_, plot) + dy, plot));
}

void EqDisplay::mouseUp(const MouseEvent&)
{
    closeDragGestures_();
    dragSlot_ = -1;
    shiftLatch_ = 0;
    repaint();
}

void EqDisplay::mouseDoubleClick(const MouseEvent& e)
{
    if (! isEnabled())
        return;

    // The second click's mouseDown opened a drag. Close it first so the
    // write below never nests inside an open gesture.
    closeDragGestures_();
    dragSlot_ = -1;

    const int slot = nodeAt_(e.position);
    if (slot < 0)
    {
        if (plot_().contains(e.position))
            enableBandAt_(e.position, -1);
        return;
    }

    // A cut goes back to its neutral end; a free band is removed.
    auto& s = slots_[static_cast<std::size_t>(slot)];
    if (isCut(slot))
        writeOnce_(s.freq, s.freq != nullptr ? s.freq->getDefaultValue() : 0.0f);
    else if (s.on != nullptr)
        writeOnce_(s.on, 0.0f);
    selected_ = isCut(slot) ? slot : -1;
    hover_ = -1;
    repaint();
}

// Switch on the free band that is off and whose default frequency sits
// nearest the click, placing it at the click. A negative type follows the
// zone: low shelf below 100 Hz, high shelf above 6 kHz, a bell between.
void EqDisplay::enableBandAt_(const Point<float> pos, int type)
{
    const auto plot = plot_();
    const float f = xToFreq(pos.x, plot);
    int best = -1;
    float bestD = 1.0e9f;
    for (int s = 1; s < kHighCut; ++s)
    {
        if (slotOn(s)) continue;
        const float d = std::abs(std::log(kEqDefaultFreq[s - 1] / f));
        if (d < bestD) { bestD = d; best = s; }
    }
    if (best < 0)
        return;

    using Filters::ParametricEQ;
    if (type < 0)
        type = static_cast<int>(f < kLowShelfZoneHz ? ParametricEQ::Type::LowShelf
                              : f > kHighShelfZoneHz ? ParametricEQ::Type::HighShelf
                                                     : ParametricEQ::Type::Bell);
    auto& s = slots_[static_cast<std::size_t>(best)];
    if (s.type != nullptr) writeOnce_(s.type, s.type->convertTo0to1(static_cast<float>(type)));
    if (s.freq != nullptr) writeOnce_(s.freq, s.freq->convertTo0to1(f));
    if (s.gain != nullptr) writeOnce_(s.gain, s.gain->convertTo0to1(yToGain(pos.y, plot)));
    if (s.on != nullptr)   writeOnce_(s.on, 1.0f);
    selected_ = best;
    repaint();
}

void EqDisplay::mouseMove(const MouseEvent& e)
{
    const int over = nodeAt_(e.position);
    if (over != hover_)
    {
        hover_ = over;
        repaint();
    }
}

void EqDisplay::mouseExit(const MouseEvent&)
{
    if (hover_ != -1)
    {
        hover_ = -1;
        repaint();
    }
}

void EqDisplay::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& wheel)
{
    if (! isEnabled())
        return;

    // The wheel sets the Q of the band under the pointer, or of the
    // nearest band by column. The cuts have no Q.
    int slot = nodeAt_(e.position);
    if (slot < 0 || isCut(slot))
        slot = nearestBandByX_(e.position.x);
    if (slot < 0)
        return;
    auto* q = slots_[static_cast<std::size_t>(slot)].q;
    if (q == nullptr || std::abs(wheel.deltaY) <= 0.0f)
        return;

    // One bracketed gesture per burst per parameter. One notch steps the
    // same proportion as one arrow press.
    if (wheelParam_ != q)
    {
        endWheelGesture_();
        q->beginChangeGesture();
        wheelParam_ = q;
    }
    const double step = (e.mods.isShiftDown() ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse)
                      / Metrics::kWheelNotchDelta;
    const float norm = std::clamp(q->getValue() + static_cast<float>(static_cast<double>(wheel.deltaY) * step), 0.0f, 1.0f);
    q->setValueNotifyingHost(norm);
    lastBurstMs_ = Time::getMillisecondCounter();
    selected_ = slot;
}

bool EqDisplay::keyPressed(const KeyPress& key)
{
    if (! isEnabled() || selected_ < 0)
        return false;

    // Compare key codes directly: KeyPress::operator==(int) rejects any
    // held modifier, which would drop the Shift form.
    const int code = key.getKeyCode();
    float xDir = 0.0f;
    float yDir = 0.0f;
    if (code == KeyPress::rightKey)      xDir = 1.0f;
    else if (code == KeyPress::leftKey)  xDir = -1.0f;
    else if (code == KeyPress::upKey)    yDir = 1.0f;
    else if (code == KeyPress::downKey)  yDir = -1.0f;
    else
        return false;

    const auto step = static_cast<float>(key.getModifiers().isShiftDown() ? Metrics::kWheelStepFine
                                                                            : Metrics::kWheelStepCoarse);
    auto& s = slots_[static_cast<std::size_t>(selected_)];
    if (xDir != 0.0f && s.freq != nullptr)
        writeOnce_(s.freq, s.freq->getValue() + xDir * step);
    if (yDir != 0.0f && slotHasGain(selected_) && s.gain != nullptr)
        writeOnce_(s.gain, s.gain->getValue() + yDir * step);
    return true;
}

void EqDisplay::enablementChanged()
{
    repaint();
}

// The right-click menu of a node: the type, remove, and reset. A cut has
// only the reset. The menu takes the plugin look and feel, as every menu.
void EqDisplay::showBandMenu_(const int slot)
{
    PopupMenu m;
    m.setLookAndFeel(&getLookAndFeel());
    if (! isCut(slot))
    {
        const int cur = slotType(slot);
        for (int t = 0; t < kEqTypeNames.size(); ++t)
            m.addItem(kMenuTypeBase + t, kEqTypeNames[t], true, t == cur);
        m.addSeparator();
        m.addItem(kMenuRemove, "Remove Band");
    }
    m.addItem(kMenuReset, "Reset");

    const auto safe = SafePointer<EqDisplay>(this);
    m.showMenuAsync(PopupMenu::Options().withTargetComponent(this), [safe, slot](int r)
    {
        if (safe == nullptr || r == 0) return;
        auto& sl = safe->slots_[static_cast<std::size_t>(slot)];
        if (r >= kMenuTypeBase && r < kMenuTypeBase + kEqTypeNames.size() && sl.type != nullptr)
            writeOnce_(sl.type, sl.type->convertTo0to1(static_cast<float>(r - kMenuTypeBase)));
        else if (r == kMenuRemove && sl.on != nullptr)
        {
            writeOnce_(sl.on, 0.0f);
            safe->selected_ = -1;
            safe->hover_ = -1;
        }
        else if (r == kMenuReset)
        {
            for (auto* p : { sl.freq, sl.gain, sl.q })
                if (p != nullptr)
                    writeOnce_(p, p->getDefaultValue());
        }
        safe->repaint();
    });
}

// The right-click menu over empty plot: add a band of a chosen type at
// the click. Greyed out once all four free bands are in use.
void EqDisplay::showAddMenu_(const Point<float> pos)
{
    bool anyFree = false;
    for (int s = 1; s < kHighCut; ++s)
        anyFree = anyFree || ! slotOn(s);

    PopupMenu m;
    m.setLookAndFeel(&getLookAndFeel());
    for (int t = 0; t < kEqTypeNames.size(); ++t)
        m.addItem(kMenuAddBase + t, "Add " + kEqTypeNames[t], anyFree);

    const auto safe = SafePointer<EqDisplay>(this);
    m.showMenuAsync(PopupMenu::Options().withTargetComponent(this), [safe, pos](int r)
    {
        if (safe == nullptr || r < kMenuAddBase || r >= kMenuAddBase + kEqTypeNames.size()) return;
        safe->enableBandAt_(pos, r - kMenuAddBase);
    });
}

String EqDisplay::getTooltip()
{
    if (hover_ < 0)
        return SettableTooltipClient::getTooltip();

    const auto& s = slots_[static_cast<std::size_t>(hover_)];
    const String dot = " " + String::charToString(static_cast<juce_wchar>(0x00B7)) + " ";
    String text = slotName_(hover_);
    if (s.freq != nullptr) text += dot + s.freq->getCurrentValueAsText();
    if (slotHasGain(hover_) && s.gain != nullptr) text += dot + s.gain->getCurrentValueAsText();
    if (! isCut(hover_) && s.q != nullptr) text += dot + "Q " + s.q->getCurrentValueAsText();
    if (! slotOn(hover_)) text += dot + "off";
    return text;
}

} // namespace MarsDSP::GUI

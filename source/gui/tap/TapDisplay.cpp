#include "TapDisplay.h"
#include "../../ChronosProcessor.h"
#include "../Colours.h"

#include <algorithm>
#include <cmath>

namespace MarsDSP::GUI {
namespace {

constexpr int kDisplayTimerHz = 30;
constexpr int kVerticalGridLines = 12;
constexpr float kDisplayTransitionSeconds = 0.18f;

String formatTimeSpanLabel(const float seconds)
{
    if (seconds < 1.0f)
        return String(std::round(seconds * 1000.0f)) + " ms";

    return String(std::round(seconds * 100.0f) / 100.0f) + " s";
}

float blendValue(const float current, const float target, const float amount)
{
    return current + (target - current) * amount;
}

} // namespace

TapDisplay::TapDisplay(ChronosProcessor& processor)
    : processorRef_(processor)
{
    delayMode_ = processorRef_.getParameters().getRawDelayMode();
    processorRef_.getAPVTS().addParameterListener("delayMode", this);

    startTimerHz(kDisplayTimerHz);
}

TapDisplay::~TapDisplay()
{
    stopTimer();
    processorRef_.getAPVTS().removeParameterListener("delayMode", this);
}

void TapDisplay::parameterChanged(const String& parameterID, const float newValue)
{
    if (parameterID == "delayMode")
    {
        MessageManager::callAsync([this, newValue]
        {
            delayMode_ = static_cast<int>(newValue + 0.5f);
            repaint();
        });
    }
}

TapSim::Parameters TapDisplay::buildParameters_() const
{
    TapSim::Parameters p;
    const auto& params = processorRef_.getParameters();

    p.timeLSeconds = params.getRawDelayTimeL() * 0.001f;
    p.timeRSeconds = params.getRawTimeLink() ? p.timeLSeconds : (params.getRawDelayTimeR() * 0.001f);
    p.feedback = params.getRawFeedback();
    p.crossFeed = params.getRawCrossFeed();
    p.mix = params.getRawMix();
    p.delaySync = params.getRawDelaySync();
    p.delayDivision = params.getRawDelayDivision();

    const double bpm = processorRef_.getCachedBpm();
    p.secondsPerBeat = static_cast<float>((bpm > 0.0) ? (60.0 / bpm) : 0.5);
    p.maxWindowSeconds = 2.0f;

    return p;
}

TapDisplay::DisplayState TapDisplay::toDisplayState_(const TapSim::SimulationResult& sim)
{
    DisplayState state;
    state.totalTimeSeconds = std::max(0.25f, sim.totalTimeSeconds);

    state.left.reserve(sim.left.size());
    for (const auto& t : sim.left)
    {
        if (!t.empty)
            state.left.push_back({ t.dry, t.timeSeconds, t.gain });
    }

    state.right.reserve(sim.right.size());
    for (const auto& t : sim.right)
    {
        if (!t.empty)
            state.right.push_back({ t.dry, t.timeSeconds, t.gain });
    }

    return state;
}

TapDisplay::DisplayState TapDisplay::blendDisplayState_(const DisplayState& current,
                                                       const DisplayState& target,
                                                       const float blendAmount)
{
    DisplayState blended;
    blended.totalTimeSeconds = std::max(0.25f, blendValue(current.totalTimeSeconds, target.totalTimeSeconds, blendAmount));

    const auto blendLane = [blendAmount](const std::vector<DisplayTap>& curLane,
                                         const std::vector<DisplayTap>& tgtLane,
                                         const float curTotalTime,
                                         std::vector<DisplayTap>& dst)
    {
        dst.clear();
        const std::size_t count = std::max(curLane.size(), tgtLane.size());
        dst.reserve(count);

        for (std::size_t i = 0; i < count; ++i)
        {
            const bool hasCur = i < curLane.size();
            const bool hasTgt = i < tgtLane.size();

            DisplayTap tap;
            if (hasCur && hasTgt)
            {
                tap.dry = tgtLane[i].dry;
                tap.timeSeconds = blendValue(curLane[i].timeSeconds, tgtLane[i].timeSeconds, blendAmount);
                tap.gain = blendValue(curLane[i].gain, tgtLane[i].gain, blendAmount);
            }
            else if (hasCur)
            {
                tap = curLane[i];
                tap.gain = blendValue(curLane[i].gain, 0.0f, blendAmount);
            }
            else
            {
                tap.dry = tgtLane[i].dry;
                tap.timeSeconds = blendValue(curTotalTime, tgtLane[i].timeSeconds, blendAmount);
                tap.gain = blendValue(0.0f, tgtLane[i].gain, blendAmount);
            }

            if (std::fabs(tap.gain) > 0.001f)
                dst.push_back(tap);
        }
    };

    blendLane(current.left, target.left, current.totalTimeSeconds, blended.left);
    blendLane(current.right, target.right, current.totalTimeSeconds, blended.right);

    return blended;
}

TapDisplay::DisplayState TapDisplay::transitionDisplayState_(const DisplayState& current,
                                                            const DisplayState& target,
                                                            const float deltaSeconds)
{
    if (deltaSeconds <= 0.0f)
        return current;

    const float blendAmount = std::clamp(deltaSeconds / kDisplayTransitionSeconds, 0.0f, 1.0f);
    return blendDisplayState_(current, target, blendAmount);
}

void TapDisplay::advanceDisplayState_(const float deltaSeconds)
{
    const auto target = toDisplayState_(TapSim::Engine::simulate(buildParameters_()));

    if (!hasDisplayState_)
    {
        displayState_ = target;
        hasDisplayState_ = true;
        return;
    }

    displayState_ = transitionDisplayState_(displayState_, target, deltaSeconds);
}

void TapDisplay::timerCallback()
{
    const double timeSecs = Time::getMillisecondCounterHiRes() * 0.001;
    if (lastTimeSecs_ <= 0.0)
    {
        lastTimeSecs_ = timeSecs;
        return;
    }

    const double dt = timeSecs - lastTimeSecs_;
    lastTimeSecs_ = timeSecs;

    advanceDisplayState_(static_cast<float>(dt));
    repaint();
}

void TapDisplay::paint(Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    const Colour accent = (delayMode_ == 1) ? Colours::accentDelayBBD : Colours::accentDelayDigital;

    g.setColour(Colours::burntOrange);
    g.fillRoundedRectangle(bounds, 5.0f);

    g.setColour(accent.withAlpha(0.25f));
    g.drawRoundedRectangle(bounds.reduced(0.5f), 5.0f, 1.0f);

    constexpr float padding = 10.0f;
    const auto displayBounds = bounds.reduced(padding);

    const float centerY = displayBounds.getCentreY();
    const float maxLaneHeight = displayBounds.getHeight() * 0.5f - 8.0f;
    if (maxLaneHeight <= 0.0f)
        return;

    advanceDisplayState_(0.0f);

    const float totalTime = std::max(0.25f, displayState_.totalTimeSeconds);

    // Center divider
    g.setColour(accent.withAlpha(0.12f));
    g.drawHorizontalLine(static_cast<int>(centerY), displayBounds.getX(), displayBounds.getRight());

    // Vertical grid lines
    for (int i = 0; i <= kVerticalGridLines; ++i)
    {
        const float x = displayBounds.getX() + displayBounds.getWidth() * (static_cast<float>(i) / static_cast<float>(kVerticalGridLines));
        const float lineAlpha = (i == kVerticalGridLines / 2) ? 0.18f : 0.07f;
        g.setColour(accent.withAlpha(lineAlpha));
        g.drawVerticalLine(static_cast<int>(x), displayBounds.getY(), displayBounds.getBottom());
    }

    // Time label in upper right
    g.setColour(accent.withAlpha(0.85f));
    g.setFont(Font(FontOptions(11.0f)).boldened());
    auto labelBounds = displayBounds;
    g.drawText(formatTimeSpanLabel(totalTime),
               labelBounds.removeFromTop(14.0f).toNearestInt(),
               Justification::topRight,
               false);

    // Draw taps
    const auto drawLane = [&](const std::vector<DisplayTap>& taps, bool isTopLane)
    {
        for (const auto& tap : taps)
        {
            const float timeNorm = std::clamp(tap.timeSeconds / totalTime, 0.0f, 1.0f);
            const float x = displayBounds.getX() + timeNorm * displayBounds.getWidth();

            const float gain = std::clamp(std::fabs(tap.gain), 0.0f, 1.0f);
            if (gain <= 0.001f)
                continue;

            const float barHeight = gain * maxLaneHeight;
            const Colour tapCol = tap.dry ? accent.darker(0.4f) : accent;

            Path barPath;
            if (isTopLane)
            {
                barPath.startNewSubPath(x, centerY);
                barPath.lineTo(x, centerY - barHeight);
            }
            else
            {
                barPath.startNewSubPath(x, centerY);
                barPath.lineTo(x, centerY + barHeight);
            }

            g.setColour(tapCol.withAlpha(tap.dry ? 0.45f : 0.85f));
            g.strokePath(barPath, PathStrokeType(2.5f, PathStrokeType::curved, PathStrokeType::rounded));

            // Head dot
            const float headY = isTopLane ? (centerY - barHeight) : (centerY + barHeight);
            constexpr float headRadius = 2.5f;
            g.setColour(tapCol);
            g.fillEllipse(x - headRadius, headY - headRadius, headRadius * 2.0f, headRadius * 2.0f);
        }
    };

    drawLane(displayState_.left, true);
    drawLane(displayState_.right, false);

    // Hover affordance: highlight active channel half and indicator
    if (isHovered_)
    {
        const bool timeLinked = processorRef_.getParameters().getRawTimeLink();
        const bool hoverUpper = (hoverPos_.y <= centerY);

        if (timeLinked)
        {
            g.setColour(accent.withAlpha(0.04f));
            g.fillRect(displayBounds);

            g.setColour(accent.withAlpha(0.6f));
            g.setFont(Font(FontOptions(10.0f)).boldened());
            g.drawText("L/R LINK", displayBounds.reduced(4.0f).toNearestInt(), Justification::topLeft, false);
        }
        else if (hoverUpper)
        {
            const auto upperArea = displayBounds.withBottom(centerY);
            g.setColour(accent.withAlpha(0.05f));
            g.fillRect(upperArea);

            g.setColour(accent.withAlpha(0.6f));
            g.setFont(Font(FontOptions(10.0f)).boldened());
            g.drawText("LEFT TIME", displayBounds.reduced(4.0f).toNearestInt(), Justification::topLeft, false);
        }
        else
        {
            const auto lowerArea = displayBounds.withTop(centerY);
            g.setColour(accent.withAlpha(0.05f));
            g.fillRect(lowerArea);

            g.setColour(accent.withAlpha(0.6f));
            g.setFont(Font(FontOptions(10.0f)).boldened());
            g.drawText("RIGHT TIME", displayBounds.reduced(4.0f).toNearestInt(), Justification::bottomLeft, false);
        }
    }
}

void TapDisplay::mouseDown(const MouseEvent& e)
{
    dragStartX_ = e.position.x;
    dragStartY_ = e.position.y;
    dragging_ = true;

    auto& apvts = processorRef_.getAPVTS();
    auto* pDelayL = apvts.getParameter("delayTime");
    auto* pDelayR = apvts.getParameter("delayTimeR");
    auto* pFb = apvts.getParameter("feedback");
    auto* pDiv = apvts.getParameter("delayDivision");

    startNormL_ = pDelayL ? pDelayL->getValue() : 0.0f;
    startNormR_ = pDelayR ? pDelayR->getValue() : 0.0f;
    startNormFb_ = pFb ? pFb->getValue() : 0.0f;
    startDiv_ = processorRef_.getParameters().getRawDelayDivision();

    const bool isUpper = (e.position.y <= getHeight() * 0.5f);
    activeDragTarget_ = isUpper ? DragTarget::LeftTime : DragTarget::RightTime;

    const bool synced = processorRef_.getParameters().getRawDelaySync();
    const bool timeLinked = processorRef_.getParameters().getRawTimeLink();

    if (synced)
    {
        if (pDiv) pDiv->beginChangeGesture();
    }
    else
    {
        if (timeLinked)
        {
            if (pDelayL) pDelayL->beginChangeGesture();
        }
        else if (isUpper)
        {
            if (pDelayL) pDelayL->beginChangeGesture();
        }
        else
        {
            if (pDelayR) pDelayR->beginChangeGesture();
        }
    }

    if (pFb) pFb->beginChangeGesture();
}

void TapDisplay::mouseDrag(const MouseEvent& e)
{
    if (!dragging_)
        return;

    const float dx = e.position.x - dragStartX_;
    const float dy = e.position.y - dragStartY_;
    const float w = static_cast<float>(std::max(1, getWidth()));
    const float h = static_cast<float>(std::max(1, getHeight()));

    auto& apvts = processorRef_.getAPVTS();
    auto* pDelayL = apvts.getParameter("delayTime");
    auto* pDelayR = apvts.getParameter("delayTimeR");
    auto* pFb = apvts.getParameter("feedback");
    auto* pDiv = apvts.getParameter("delayDivision");

    const bool synced = processorRef_.getParameters().getRawDelaySync();
    const bool timeLinked = processorRef_.getParameters().getRawTimeLink();
    const bool isUpper = (activeDragTarget_ == DragTarget::LeftTime);

    if (synced)
    {
        if (pDiv)
        {
            constexpr float pixelsPerDiv = 25.0f;
            const int step = static_cast<int>(std::round(dx / pixelsPerDiv));
            const int newDiv = std::clamp(startDiv_ + step, 0, 19);
            pDiv->setValueNotifyingHost(pDiv->getNormalisableRange().convertTo0to1(static_cast<float>(newDiv)));
        }
    }
    else
    {
        const float dNorm = (dx / w) * 1.2f;
        if (timeLinked)
        {
            if (pDelayL)
                pDelayL->setValueNotifyingHost(std::clamp(startNormL_ + dNorm, 0.0f, 1.0f));
        }
        else if (isUpper)
        {
            if (pDelayL)
                pDelayL->setValueNotifyingHost(std::clamp(startNormL_ + dNorm, 0.0f, 1.0f));
        }
        else
        {
            if (pDelayR)
                pDelayR->setValueNotifyingHost(std::clamp(startNormR_ + dNorm, 0.0f, 1.0f));
        }
    }

    // Vertical drag modulates feedback
    if (pFb)
    {
        const float dNormFb = -dy / h;
        pFb->setValueNotifyingHost(std::clamp(startNormFb_ + dNormFb, 0.0f, 1.0f));
    }
}

void TapDisplay::mouseUp(const MouseEvent&)
{
    if (!dragging_)
        return;

    auto& apvts = processorRef_.getAPVTS();
    auto* pDelayL = apvts.getParameter("delayTime");
    auto* pDelayR = apvts.getParameter("delayTimeR");
    auto* pFb = apvts.getParameter("feedback");
    auto* pDiv = apvts.getParameter("delayDivision");

    const bool synced = processorRef_.getParameters().getRawDelaySync();
    const bool timeLinked = processorRef_.getParameters().getRawTimeLink();
    const bool isUpper = (activeDragTarget_ == DragTarget::LeftTime);

    if (synced)
    {
        if (pDiv) pDiv->endChangeGesture();
    }
    else
    {
        if (timeLinked)
        {
            if (pDelayL) pDelayL->endChangeGesture();
        }
        else if (isUpper)
        {
            if (pDelayL) pDelayL->endChangeGesture();
        }
        else
        {
            if (pDelayR) pDelayR->endChangeGesture();
        }
    }

    if (pFb) pFb->endChangeGesture();

    dragging_ = false;
    activeDragTarget_ = DragTarget::None;
}

void TapDisplay::mouseMove(const MouseEvent& e)
{
    hoverPos_ = e.position;
    isHovered_ = true;
    repaint();
}

void TapDisplay::mouseEnter(const MouseEvent& e)
{
    hoverPos_ = e.position;
    isHovered_ = true;
    repaint();
}

void TapDisplay::mouseExit(const MouseEvent&)
{
    isHovered_ = false;
    repaint();
}

void TapDisplay::resized()
{
}

} // namespace MarsDSP::GUI

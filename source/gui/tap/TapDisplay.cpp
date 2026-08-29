#include "TapDisplay.h"
#include "../../ChronosProcessor.h"
#include "../Colours.h"

#include <algorithm>
#include <cmath>

namespace MarsDSP::GUI {
namespace {

constexpr int kDisplayTimerHz = 60;
constexpr int kVerticalGridLines = 12;

// Ease time constants in seconds.
constexpr float kTauMove = 0.060f;
constexpr float kTauGain = 0.050f;
constexpr float kTauDie  = 0.080f;
constexpr float kTauSpan = 0.120f;

// A dying tap below this gain is removed.
constexpr float kCullGain = 0.005f;

String formatTimeSpanLabel(const float seconds)
{
    if (seconds < 1.0f)
        return String(std::round(seconds * 1000.0f)) + " ms";

    return String(std::round(seconds * 100.0f) / 100.0f) + " s";
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

bool TapDisplay::paramsChanged_(const TapSim::Parameters& p) const
{
    const auto& a = lastParams_;
    return a.timeLSeconds != p.timeLSeconds
        || a.timeRSeconds != p.timeRSeconds
        || a.feedback != p.feedback
        || a.crossFeed != p.crossFeed
        || a.mix != p.mix
        || a.delaySync != p.delaySync
        || a.delayDivision != p.delayDivision
        || a.secondsPerBeat != p.secondsPerBeat;
}

void TapDisplay::runSimulation_(const TapSim::Parameters& p)
{
    const auto result = TapSim::Engine::simulate(p);
    targetTotalTime_ = std::max(0.25f, result.totalTimeSeconds);

    // The base delay is the first repeat time. Fall back to the parameter.
    const float baseL = (result.left.size() > 1) ? result.left[1].timeSeconds : p.timeLSeconds;
    const float baseR = (result.right.size() > 1) ? result.right[1].timeSeconds : p.timeRSeconds;

    matchChannel_(trackedL_, result.left, baseL);
    matchChannel_(trackedR_, result.right, baseR);
}

void TapDisplay::matchChannel_(std::vector<TrackedTap>& tracked,
                                const std::vector<TapSim::Tap>& simTaps,
                                const float baseTime)
{
    const float invBase = (baseTime > 1e-6f) ? (1.0f / baseTime) : 0.0f;

    // Mark every tracked tap as dying. Clear the flag when a sim tap matches.
    for (auto& t : tracked)
        t.dying = true;

    for (const auto& sim : simTaps)
    {
        if (sim.empty)
            continue;

        // Key by the repeat index. The dry tap keys to zero.
        const int key = (invBase > 0.0f && !sim.dry)
            ? static_cast<int>(std::round(sim.timeSeconds * invBase))
            : 0;

        TrackedTap* found = nullptr;
        for (auto& t : tracked)
        {
            if (t.key == key)
            {
                found = &t;
                break;
            }
        }

        if (found != nullptr)
        {
            found->dry = sim.dry;
            found->targetTime = sim.timeSeconds;
            found->targetGain = sim.gain;
            found->dying = false;
        }
        else
        {
            TrackedTap t;
            t.key = key;
            t.dry = sim.dry;
            t.targetTime = sim.timeSeconds;
            t.targetGain = sim.gain;
            // A new tap grows from gain zero at its own position.
            t.displayedTime = sim.timeSeconds;
            t.displayedGain = 0.0f;
            t.dying = false;
            tracked.push_back(std::move(t));
        }
    }
}

void TapDisplay::advanceEases_(const float deltaSeconds)
{
    if (deltaSeconds <= 0.0f)
        return;

    const float kMove = 1.0f - std::exp(-deltaSeconds / kTauMove);
    const float kGain = 1.0f - std::exp(-deltaSeconds / kTauGain);
    const float kDie  = 1.0f - std::exp(-deltaSeconds / kTauDie);
    const float kSpan = 1.0f - std::exp(-deltaSeconds / kTauSpan);

    const auto easeLane = [&](std::vector<TrackedTap>& lane)
    {
        for (auto it = lane.begin(); it != lane.end(); )
        {
            TrackedTap& t = *it;

            if (t.dying)
            {
                // A dying tap keeps its position. Only the gain decays.
                t.displayedGain += (0.0f - t.displayedGain) * kDie;

                if (std::fabs(t.displayedGain) < kCullGain)
                {
                    it = lane.erase(it);
                    continue;
                }
            }
            else
            {
                t.displayedTime += (t.targetTime - t.displayedTime) * kMove;
                t.displayedGain += (t.targetGain - t.displayedGain) * kGain;
            }

            ++it;
        }
    };

    easeLane(trackedL_);
    easeLane(trackedR_);

    displayedTotalTime_ += (targetTotalTime_ - displayedTotalTime_) * kSpan;
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

    // Drain the wet-level feed every tick. This is independent of the sim.
    TapFeedFrame frame{};
    float latestL = currentWetLevelL_;
    float latestR = currentWetLevelR_;
    bool hasFrame = false;
    while (processorRef_.getTapFifo().pop(frame))
    {
        latestL = frame.wetRmsL;
        latestR = frame.wetRmsR;
        hasFrame = true;
    }
    if (hasFrame)
    {
        currentWetLevelL_ = 0.65f * currentWetLevelL_ + 0.35f * latestL;
        currentWetLevelR_ = 0.65f * currentWetLevelR_ + 0.35f * latestR;
    }
    else
    {
        currentWetLevelL_ *= 0.90f;
        currentWetLevelR_ *= 0.90f;
    }

    // Run the sim once per tick, and only when a parameter changed.
    const auto params = buildParameters_();
    if (!hasState_ || paramsChanged_(params))
    {
        runSimulation_(params);
        lastParams_ = params;
        hasState_ = true;
    }

    advanceEases_(static_cast<float>(dt));
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

    const float totalTime = std::max(0.25f, displayedTotalTime_);

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
    const auto drawLane = [&](const std::vector<TrackedTap>& taps, bool isTopLane)
    {
        const float wetLevel = isTopLane ? currentWetLevelL_ : currentWetLevelR_;
        const float envIntensity = std::clamp(wetLevel * 3.5f, 0.0f, 1.0f);

        for (const auto& tap : taps)
        {
            const float timeNorm = std::clamp(tap.displayedTime / totalTime, 0.0f, 1.0f);
            const float x = displayBounds.getX() + timeNorm * displayBounds.getWidth();

            const float gain = std::clamp(std::fabs(tap.displayedGain), 0.0f, 1.0f);
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

            const float baseAlpha = tap.dry ? 0.45f : 0.85f;
            g.setColour(tapCol.withAlpha(baseAlpha));
            g.strokePath(barPath, PathStrokeType(2.5f, PathStrokeType::curved, PathStrokeType::rounded));

            // Head dot with envelope brightness modulation
            const float headY = isTopLane ? (centerY - barHeight) : (centerY + barHeight);
            const float headRadius = tap.dry ? 2.5f : (2.5f + envIntensity * 2.0f);
            const float glowAlpha = tap.dry ? 1.0f : std::clamp(0.8f + envIntensity * 0.2f, 0.0f, 1.0f);
            g.setColour(tapCol.withAlpha(glowAlpha));
            g.fillEllipse(x - headRadius, headY - headRadius, headRadius * 2.0f, headRadius * 2.0f);
        }
    };

    drawLane(trackedL_, true);
    drawLane(trackedR_, false);

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

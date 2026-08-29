#include "TapDisplay.h"
#include "../../ChronosProcessor.h"
#include "../Colours.h"
#include "utils/helpers/TempoSync.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace MarsDSP::GUI {
namespace {

constexpr int kDisplayTimerHz = 60;

// Ease time constants in seconds.
constexpr float kTauMove = 0.060f;
constexpr float kTauGain = 0.050f;
constexpr float kTauDie  = 0.080f;
constexpr float kTauSpan = 0.120f;

// A dying tap below this gain is removed.
constexpr float kCullGain = 0.005f;

// The reserved ruler lane height at the bottom of the display.
constexpr float kRulerLaneHeight = 18.0f;
// The snap radius in pixels for a tap head under the cursor.
constexpr float kSnapPx = 6.0f;

const String kMidDot = String::charToString(static_cast<juce_wchar>(0x00B7));

// Division names in parameter-layout order, index 0 to 19.
static const char* const kDivisionNames[] = {
    "1/64", "1/32T", "1/32", "1/16T", "1/32.", "1/16",
    "1/8T", "1/16.", "1/8", "1/4T", "1/8.", "1/4",
    "1/2T", "1/4.", "1/2", "1/1T", "1/2.", "1/1",
    "2/1", "4/1"
};

String divisionName(const int index)
{
    constexpr int n = static_cast<int>(sizeof(kDivisionNames) / sizeof(kDivisionNames[0]));
    if (index < 0 || index >= n)
        return {};
    return String(kDivisionNames[static_cast<std::size_t>(index)]);
}

// Format a time the same way as the time display in free mode.
String formatRulerLabel(const float seconds)
{
    const float ms = seconds * 1000.0f;
    if (ms < 1000.0f)
        return String(roundToInt(ms)) + " ms";
    return String(roundToInt(seconds * 100.0f) / 100.0) + " s";
}

// Format the cursor time: milliseconds, plus beats when synced.
String formatHoverTime(const float seconds, const float secondsPerBeat, const bool synced)
{
    String text = formatRulerLabel(seconds);
    if (synced && secondsPerBeat > 0.0f)
    {
        const float beats = seconds / secondsPerBeat;
        text += "  " + kMidDot + "  " + String(beats, 2) + " beats";
    }
    return text;
}

// Format a linear gain as decibels with one decimal and a sign.
String formatDb(float gain)
{
    const float g = std::fabs(gain);
    const float db = (g > 1e-6f) ? (20.0f * std::log10(g)) : -120.0f;
    String s = String(db, 1);
    if (db >= 0.0f && ! s.startsWithChar('-'))
        s = "+" + s;
    return s + " dB";
}

struct RulerTicks {
    std::vector<float> majors;
    std::vector<float> minors;
};

// Free-mode 1-2-5 major step: keep 4 to 8 majors over the span.
float chooseMajorStep(const float span)
{
    if (span <= 0.0f)
        return 1.0f;

    float bestStep = span;
    int bestDist = 1000;
    for (int e = -4; e <= 4; ++e)
    {
        const float p = std::pow(10.0f, static_cast<float>(e));
        const float steps[3] = { p, 2.0f * p, 5.0f * p };
        for (const float step : steps)
        {
            const int count = static_cast<int>(std::lround(span / step));
            if (count >= 4 && count <= 8)
            {
                const int dist = std::abs(count - 6);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestStep = step;
                }
            }
        }
    }
    return bestStep;
}

// Minor subdivisions per major, chosen for a neat minor step.
int minorPerMajor(const float step)
{
    const float p = std::pow(10.0f, std::floor(std::log10(step)));
    const float f = step / p;
    if (f < 1.5f)
        return 5;
    if (f < 3.5f)
        return 4;
    return 5;
}

RulerTicks computeFreeTicks(const float span)
{
    RulerTicks t;
    const float major = chooseMajorStep(span);
    const float minor = major / static_cast<float>(minorPerMajor(major));
    const float eps = minor * 1e-4f;

    for (float s = 0.0f; s <= span + eps; s += minor)
    {
        const float m = std::round(s / major) * major;
        if (std::fabs(s - m) < minor * 0.25f)
            t.majors.push_back(s);
        else
            t.minors.push_back(s);
    }
    return t;
}

// Sync-mode ticks: majors per beat, minors per current division.
RulerTicks computeSyncTicks(const float span, const float secondsPerBeat, const float divisionSeconds)
{
    RulerTicks t;
    if (secondsPerBeat <= 0.0f)
        return t;

    const float eps = secondsPerBeat * 1e-4f;
    for (float s = 0.0f; s <= span + eps; s += secondsPerBeat)
        t.majors.push_back(s);

    if (divisionSeconds > 0.0f && divisionSeconds < secondsPerBeat)
    {
        const float dep = divisionSeconds * 1e-4f;
        for (float s = 0.0f; s <= span + dep; s += divisionSeconds)
        {
            bool isMajor = false;
            for (const float m : t.majors)
            {
                if (std::fabs(s - m) < divisionSeconds * 0.25f)
                {
                    isMajor = true;
                    break;
                }
            }
            if (! isMajor)
                t.minors.push_back(s);
        }
    }
    return t;
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

    const auto plotBounds = displayBounds.withTrimmedBottom(kRulerLaneHeight);
    const auto rulerBounds = displayBounds.withTrimmedTop(displayBounds.getHeight() - kRulerLaneHeight);

    const float centerY = plotBounds.getCentreY();
    const float maxLaneHeight = plotBounds.getHeight() * 0.5f - 8.0f;
    if (maxLaneHeight <= 0.0f)
        return;

    const float totalTime = std::max(0.25f, displayedTotalTime_);

    const bool synced = processorRef_.getParameters().getRawDelaySync();
    const double bpm = processorRef_.getCachedBpm();
    const float secondsPerBeat = static_cast<float>((bpm > 0.0) ? (60.0 / bpm) : 0.5f);

    RulerTicks ticks;
    if (synced)
    {
        const int div = processorRef_.getParameters().getRawDelayDivision();
        const double divMs = Utils::Helpers::TempoSync::convertChoiceIndexToMilliseconds(div, bpm);
        const float divSec = static_cast<float>(std::clamp(divMs, 1.0, 5000.0) * 0.001);
        ticks = computeSyncTicks(totalTime, secondsPerBeat, divSec);
    }
    else
    {
        ticks = computeFreeTicks(totalTime);
    }

    const auto timeToX = [&](const float t) -> float
    {
        return plotBounds.getX() + std::clamp(t / totalTime, 0.0f, 1.0f) * plotBounds.getWidth();
    };

    // Minor grid lines
    g.setColour(accent.withAlpha(0.05f));
    for (const float t : ticks.minors)
        g.drawVerticalLine(static_cast<int>(timeToX(t)), plotBounds.getY(), plotBounds.getBottom());

    // Major grid lines
    g.setColour(accent.withAlpha(0.12f));
    for (const float t : ticks.majors)
        g.drawVerticalLine(static_cast<int>(timeToX(t)), plotBounds.getY(), plotBounds.getBottom());

    // Center divider
    g.setColour(accent.withAlpha(0.12f));
    g.drawHorizontalLine(static_cast<int>(centerY), plotBounds.getX(), plotBounds.getRight());

    // Draw taps
    const auto drawLane = [&](const std::vector<TrackedTap>& taps, const bool isTopLane)
    {
        const float wetLevel = isTopLane ? currentWetLevelL_ : currentWetLevelR_;
        const float envIntensity = std::clamp(wetLevel * 3.5f, 0.0f, 1.0f);

        for (const auto& tap : taps)
        {
            const float timeNorm = std::clamp(tap.displayedTime / totalTime, 0.0f, 1.0f);
            const float x = plotBounds.getX() + timeNorm * plotBounds.getWidth();

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

    // Hover half-highlight and channel affordance label
    if (isHovered_)
    {
        const bool timeLinked = processorRef_.getParameters().getRawTimeLink();
        const bool hoverUpper = (hoverPos_.y <= centerY);

        if (timeLinked)
        {
            g.setColour(accent.withAlpha(0.04f));
            g.fillRect(plotBounds);
            g.setColour(accent.withAlpha(0.6f));
            g.setFont(Font(FontOptions(10.0f)).boldened());
            g.drawText("L/R LINK", plotBounds.reduced(4.0f).toNearestInt(), Justification::topLeft, false);
        }
        else if (hoverUpper)
        {
            g.setColour(accent.withAlpha(0.05f));
            g.fillRect(plotBounds.withBottom(centerY));
            g.setColour(accent.withAlpha(0.6f));
            g.setFont(Font(FontOptions(10.0f)).boldened());
            g.drawText("LEFT TIME", plotBounds.reduced(4.0f).toNearestInt(), Justification::topLeft, false);
        }
        else
        {
            g.setColour(accent.withAlpha(0.05f));
            g.fillRect(plotBounds.withTop(centerY));
            g.setColour(accent.withAlpha(0.6f));
            g.setFont(Font(FontOptions(10.0f)).boldened());
            g.drawText("RIGHT TIME", plotBounds.reduced(4.0f).toNearestInt(), Justification::bottomLeft, false);
        }
    }

    // Ruler tick marks
    const float tickTop = plotBounds.getBottom();
    g.setColour(accent.withAlpha(0.25f));
    for (const float t : ticks.minors)
        g.drawVerticalLine(static_cast<int>(timeToX(t)), tickTop, tickTop + 4.0f);

    g.setColour(accent.withAlpha(0.5f));
    for (const float t : ticks.majors)
        g.drawVerticalLine(static_cast<int>(timeToX(t)), tickTop, tickTop + 6.0f);

    // Ruler labels
    g.setColour(Colours::rulerText);
    g.setFont(Font(FontOptions(10.0f)));
    if (synced)
    {
        const int div = processorRef_.getParameters().getRawDelayDivision();
        g.drawText(divisionName(div), rulerBounds.reduced(2.0f, 0.0f).toNearestInt(), Justification::centredLeft, true);

        for (const float t : ticks.majors)
        {
            const int beat = static_cast<int>(std::round(t / secondsPerBeat)) + 1;
            const float x = timeToX(t);
            const auto r = Rectangle<float>(x - 12.0f, rulerBounds.getY(), 24.0f, rulerBounds.getHeight()).toNearestInt();
            g.drawText(String(beat), r, Justification::centred, true);
        }
    }
    else
    {
        for (const float t : ticks.majors)
        {
            const float x = timeToX(t);
            const auto r = Rectangle<float>(x - 24.0f, rulerBounds.getY(), 48.0f, rulerBounds.getHeight()).toNearestInt();
            g.drawText(formatRulerLabel(t), r, Justification::centred, true);
        }
    }

    // Hover cursor and measurement readout
    if (isHovered_)
    {
        const bool hoverUpper = (hoverPos_.y <= centerY);
        const auto& lane = hoverUpper ? trackedL_ : trackedR_;

        float cursorX = hoverPos_.x;
        int snapKey = -1;
        float snapTime = 0.0f;
        float snapGain = 0.0f;

        for (const auto& tap : lane)
        {
            if (tap.dry)
                continue;
            const float tx = timeToX(tap.displayedTime);
            if (std::fabs(tx - hoverPos_.x) <= kSnapPx)
            {
                snapKey = tap.key;
                snapTime = tap.targetTime;
                snapGain = tap.targetGain;
                cursorX = tx;
                break;
            }
        }

        g.setColour(accent.withAlpha(0.6f));
        g.drawVerticalLine(static_cast<int>(cursorX), plotBounds.getY(), plotBounds.getBottom());

        String readout;
        if (snapKey >= 1)
            readout = String(snapKey) + " x " + formatRulerLabel(snapTime) + "  " + kMidDot + "  " + formatDb(snapGain);
        else
        {
            const float t = std::clamp((hoverPos_.x - plotBounds.getX()) / plotBounds.getWidth(), 0.0f, 1.0f) * totalTime;
            readout = formatHoverTime(t, secondsPerBeat, synced);
        }

        g.setColour(Colours::rulerText);
        g.setFont(Font(FontOptions(10.0f)).boldened());
        const float readoutW = 160.0f;
        float rx = cursorX + 6.0f;
        if (rx + readoutW > displayBounds.getRight())
            rx = cursorX - readoutW - 6.0f;
        const auto readoutRect = Rectangle<float>(rx, plotBounds.getY() + 2.0f, readoutW, 14.0f).toNearestInt();
        g.drawText(readout, readoutRect, Justification::centredLeft, true);
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

void TapDisplay::mouseDoubleClick(const MouseEvent& e)
{
    const bool isUpper = (e.position.y <= getHeight() * 0.5f);
    const bool timeLinked = processorRef_.getParameters().getRawTimeLink();

    // When linked, the left time drives both channels.
    auto& apvts = processorRef_.getAPVTS();
    auto* p = (timeLinked || isUpper) ? apvts.getParameter("delayTime")
                                      : apvts.getParameter("delayTimeR");
    if (p == nullptr)
        return;

    p->beginChangeGesture();
    p->setValueNotifyingHost(p->getDefaultValue());
    p->endChangeGesture();
}

void TapDisplay::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& wheel)
{
    const bool isUpper = (e.position.y <= getHeight() * 0.5f);
    const bool timeLinked = processorRef_.getParameters().getRawTimeLink();
    const bool synced = processorRef_.getParameters().getRawDelaySync();
    const bool fine = e.mods.isShiftDown();

    auto& apvts = processorRef_.getAPVTS();

    if (synced)
    {
        auto* pDiv = apvts.getParameter("delayDivision");
        if (pDiv == nullptr)
            return;

        const int curIdx = processorRef_.getParameters().getRawDelayDivision();
        const int dir = (wheel.deltaY > 0.0f) ? 1 : -1;
        const int newIdx = std::clamp(curIdx + dir, 0, 19);
        if (newIdx == curIdx)
            return;

        pDiv->beginChangeGesture();
        pDiv->setValueNotifyingHost(pDiv->getNormalisableRange().convertTo0to1(static_cast<float>(newIdx)));
        pDiv->endChangeGesture();
    }
    else
    {
        auto* p = (timeLinked || isUpper) ? apvts.getParameter("delayTime")
                                          : apvts.getParameter("delayTimeR");
        if (p == nullptr)
            return;

        // Nudge the normalized value. Shift makes the step fine.
        constexpr float kCoarse = 0.02f;
        constexpr float kFine = 0.004f;
        const float step = fine ? kFine : kCoarse;
        const float cur = p->getValue();
        const float next = std::clamp(cur + wheel.deltaY * step, 0.0f, 1.0f);
        if (next == cur)
            return;

        p->beginChangeGesture();
        p->setValueNotifyingHost(next);
        p->endChangeGesture();
    }
}

void TapDisplay::resized()
{
}

} // namespace MarsDSP::GUI

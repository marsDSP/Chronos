#include "TapDisplay.h"
#include "../../ChronosProcessor.h"
#include "../Colours.h"
#include "../Fonts.h"
#include "TickGenerator.h"
#include "utils/helpers/TempoSync.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace MarsDSP::GUI {
namespace {

// Ease time constants in seconds.
const float kTauMove = 0.060f;
const float kTauGain = 0.050f;
const float kTauDie  = 0.080f;
const float kTauSpan = 0.120f;

// A dying tap below this gain is removed.
const float kCullGain = 0.005f;

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

// Format a ruler label.
String formatRulerLabel(const float seconds, const float step = 0.0f)
{
    if (seconds < 1.0f)
        return String(roundToInt(seconds * 1000.0f)) + " ms";
    const int decimals = (step >= 1.0f) ? 0 : 1;
    return String(seconds, decimals) + " s";
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

// Collision rule: drop labels that would overlap.
std::vector<int> collisionFilteredIndices(const std::vector<float>& xs,
                                           const std::vector<String>& labels,
                                           const Font& font,
                                           const float minGap)
{
    std::vector<int> idx;
    for (int i = 0; i < static_cast<int>(xs.size()); ++i)
        idx.push_back(i);

    for (int pass = 0; pass < 2 && idx.size() > 1; ++pass)
    {
        bool collision = false;
        for (std::size_t i = 1; i < idx.size(); ++i)
        {
            const float halfW0 = Fonts::textWidth(font, labels[static_cast<std::size_t>(idx[i - 1])]) * 0.5f;
            const float halfW1 = Fonts::textWidth(font, labels[static_cast<std::size_t>(idx[i])]) * 0.5f;
            const float gap = xs[static_cast<std::size_t>(idx[i])] - xs[static_cast<std::size_t>(idx[i - 1])] - halfW0 - halfW1;
            if (gap < minGap)
                collision = true;
        }
        if (! collision)
            break;
        std::vector<int> kept;
        for (std::size_t i = 0; i < idx.size(); ++i)
            if (i % 2 == 0)
                kept.push_back(idx[i]);
        idx = std::move(kept);
    }
    return idx;
}

} // namespace
TapDisplay::TapDisplay(ChronosProcessor& processor)
    : processorRef_(processor)
{
}

TapDisplay::~TapDisplay()
{
    stopTimer();
}

void TapDisplay::visibilityChanged()
{
    updateTimerState_();
}

void TapDisplay::parentHierarchyChanged()
{
    updateTimerState_();
}

// Run the timer only while the component is on a visible peer.
void TapDisplay::updateTimerState_()
{
    if (isShowing())
    {
        if (! isTimerRunning())
        {
            lastTimeSecs_ = 0.0;
            startTimerHz(60);
        }
    }
    else
    {
        stopTimer();
    }
}

void TapDisplay::setMetrics(const Metrics& m)
{
    metrics_ = m;
    repaint();
}

void TapDisplay::setAccentColour(const Colour c)
{
    accentColour_ = c;
    repaint();
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
    bool simRan = false;
    const auto params = buildParameters_();
    if (!hasState_ || paramsChanged_(params))
    {
        runSimulation_(params);
        lastParams_ = params;
        hasState_ = true;
        simRan = true;
    }

    advanceEases_(static_cast<float>(dt));

    // Repaint only when something changed since the last tick.
    const bool hoverChanged = (isHovered_ != prevIsHovered_)
                         || (hoverPos_ != prevHoverPos_);
    const bool wetChanged = (std::fabs(currentWetLevelL_ - prevWetLevelL_) > 0.005f
                          || std::fabs(currentWetLevelR_ - prevWetLevelR_) > 0.005f);

    const auto laneConverging = [](const std::vector<TrackedTap>& lane) {
        for (const auto& t : lane)
            if (t.dying || std::fabs(t.displayedTime - t.targetTime) > 1e-4f
                || std::fabs(t.displayedGain - t.targetGain) > 1e-4f)
                return true;
        return false;
    };
    const bool tapsConverging = laneConverging(trackedL_) || laneConverging(trackedR_);
    const bool spanConverging = (std::fabs(displayedTotalTime_ - targetTotalTime_) > 1e-4f);

    prevIsHovered_ = isHovered_;
    prevHoverPos_ = hoverPos_;
    prevWetLevelL_ = currentWetLevelL_;
    prevWetLevelR_ = currentWetLevelR_;
    if (! simRan && ! tapsConverging && ! spanConverging && ! hoverChanged && ! wetChanged)
        return;

    repaint();
}

void TapDisplay::paint(Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();
    const Colour accent = accentColour_;
    const Colour plotFill    = tintInk(accent, kTintPlotFill);
    const Colour plotFillLit = tintInk(accent, kTintPlotFillLit);

    const float displayCorner = metrics_.pxf(Metrics::kCornerDisplay);
    const float displayStroke = metrics_.stroke(Metrics::kHairline);

    g.setColour(plotFill);
    g.fillRoundedRectangle(bounds, displayCorner);

    g.setColour(tintInk(accent, kTintDisplayBorder));
    g.drawRoundedRectangle(bounds.reduced(displayStroke / 2), displayCorner, displayStroke);
    const float padding = metrics_.pxf(Metrics::kPlotPad);
    const auto displayBounds = bounds.reduced(padding);

    const float rulerLaneH = metrics_.pxf(Metrics::kRulerLaneHeight);
    const auto plotBounds = displayBounds.withTrimmedBottom(rulerLaneH);
    const auto rulerBounds = displayBounds.withTrimmedTop(displayBounds.getHeight() - rulerLaneH);

    const float centerY = plotBounds.getCentreY();
    const float maxLaneHeight = plotBounds.getHeight() * 0.5f - metrics_.pxf(Metrics::kLaneHeadroom);
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
        ticks = computeFreeTicks(totalTime, plotBounds.getWidth(), metrics_.s);
    }

    const auto timeToX = [&](const float t) -> float
    {
        return plotBounds.getX() + std::clamp(t / totalTime, 0.0f, 1.0f) * plotBounds.getWidth();
    };

    // Minor grid lines
    g.setColour(tintInk(accent, kTintGridMinor));
    for (const float t : ticks.minors)
        g.drawVerticalLine(roundToInt(timeToX(t)), plotBounds.getY(), plotBounds.getBottom());

    // Major grid lines
    g.setColour(tintInk(accent, kTintGridMajor));
    for (const float t : ticks.majors)
        g.drawVerticalLine(roundToInt(timeToX(t)), plotBounds.getY(), plotBounds.getBottom());

    // Center divider
    g.setColour(tint(plotFillLit, accent, kTintCentreLine));
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
            g.strokePath(barPath, PathStrokeType(metrics_.pxf(Metrics::kTapBarStroke), PathStrokeType::curved, PathStrokeType::rounded));

            // Head dot with envelope brightness modulation
            const float headY = isTopLane ? (centerY - barHeight) : (centerY + barHeight);
            const float baseHead = metrics_.pxf(Metrics::kTapHeadRadius);
            const float headRadius = tap.dry ? baseHead : (baseHead + envIntensity * metrics_.pxf(Metrics::kTapHeadGrow));
            const float glowAlpha = tap.dry ? 1.0f : std::clamp(0.8f + envIntensity * 0.2f, 0.0f, 1.0f);
            g.setColour(tapCol.withAlpha(glowAlpha));
            g.fillEllipse(x - headRadius, headY - headRadius, headRadius * 2, headRadius * 2);
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
            g.setFont(Fonts::font(Fonts::Weight::Medium, metrics_.font(10.0f)));
            g.drawText("L/R LINK", plotBounds.reduced(4.0f).toNearestInt(), Justification::topLeft, false);
        }
        else if (hoverUpper)
        {
            g.setColour(accent.withAlpha(0.05f));
            g.fillRect(plotBounds.withBottom(centerY));
            g.setColour(accent.withAlpha(0.6f));
            g.setFont(Fonts::font(Fonts::Weight::Medium, metrics_.font(10.0f)));
            g.drawText("LEFT TIME", plotBounds.reduced(4.0f).toNearestInt(), Justification::topLeft, false);
        }
        else
        {
            g.setColour(accent.withAlpha(0.05f));
            g.fillRect(plotBounds.withTop(centerY));
            g.setColour(accent.withAlpha(0.6f));
            g.setFont(Fonts::font(Fonts::Weight::Medium, metrics_.font(10.0f)));
            g.drawText("RIGHT TIME", plotBounds.reduced(4.0f).toNearestInt(), Justification::bottomLeft, false);
        }
    }

    // Ruler tick marks
    const float tickTop = plotBounds.getBottom();
    const float majorTickLen = metrics_.pxf(Metrics::kMajorTick);
    const float minorTickLen = metrics_.pxf(Metrics::kMinorTick);
    g.setColour(tintInk(accent, kTintGridMajor));
    for (const float t : ticks.minors)
        g.drawVerticalLine(roundToInt(timeToX(t)), tickTop, tickTop + minorTickLen);
    g.setColour(tintInk(accent, kTintGridMajor));
    for (const float t : ticks.majors)
        g.drawVerticalLine(roundToInt(timeToX(t)), tickTop, tickTop + majorTickLen);

    // Ruler labels
    const Font rulerFont = Fonts::font(Fonts::Weight::Regular, metrics_.font(Metrics::kRulerFont));
    const float baselineY = tickTop + majorTickLen + metrics_.pxf(Metrics::kRulerLabelGap)
                        + Fonts::kCapHeightRatio * metrics_.font(Metrics::kRulerFont);
    const int baselineYi = roundToInt(baselineY);
    const float minLabelGap = metrics_.pxf(8.0f);
    g.setFont(rulerFont);
    g.setColour(Colours::rulerText);

    if (synced)
    {
        const int div = processorRef_.getParameters().getRawDelayDivision();
        g.drawText(divisionName(div), rulerBounds.reduced(2.0f, 0.0f).toNearestInt(), Justification::centredLeft, true);

        std::vector<float> majorXs;
        std::vector<String> majorLabels;
        for (const float t : ticks.majors)
        {
            const int beat = static_cast<int>(std::round(t / secondsPerBeat)) + 1;
            majorXs.push_back(timeToX(t));
            majorLabels.push_back(String(beat));
        }
        const auto drawIdx = collisionFilteredIndices(majorXs, majorLabels, rulerFont, minLabelGap);
        for (const int i : drawIdx)
            g.drawSingleLineText(majorLabels[static_cast<std::size_t>(i)],
                                   roundToInt(majorXs[static_cast<std::size_t>(i)]), baselineYi,
                                   Justification::horizontallyCentred);
    }
    else
    {
        std::vector<float> majorXs;
        std::vector<String> majorLabels;
        for (const float t : ticks.majors)
        {
            majorXs.push_back(timeToX(t));
            majorLabels.push_back(formatRulerLabel(t, ticks.majorStep));
        }
        const auto drawIdx = collisionFilteredIndices(majorXs, majorLabels, rulerFont, minLabelGap);
        for (const int i : drawIdx)
            g.drawSingleLineText(majorLabels[static_cast<std::size_t>(i)],
                                   roundToInt(majorXs[static_cast<std::size_t>(i)]), baselineYi,
                                   Justification::horizontallyCentred);
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
            if (std::fabs(tx - hoverPos_.x) <= metrics_.pxf(Metrics::kSnapRadius))
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

        g.setColour(accent);
        g.setFont(Fonts::font(Fonts::Weight::Medium, metrics_.font(10.0f)));
        const float readoutW = metrics_.pxf(Metrics::kHoverReadoutW);
        float rx = cursorX + 6.0f;
        if (rx + readoutW > displayBounds.getRight())
            rx = cursorX - readoutW - 6.0f;
        const auto readoutRect = Rectangle<float>(rx, plotBounds.getY() + 2.0f, readoutW, metrics_.pxf(Metrics::kHoverReadoutH)).toNearestInt();
        g.drawText(readout, readoutRect, Justification::centredLeft, true);
    }
}

void TapDisplay::mouseDown(const MouseEvent& e)
{
    dragStartX_ = e.position.x;
    dragStartY_ = e.position.y;
    dragging_ = true;

    auto& apvts = processorRef_.getAPVTS();
    auto* pDelayL = apvts.getParameter(delayTimeParamID.getParamID());
    auto* pDelayR = apvts.getParameter(delayTimeRParamID.getParamID());
    auto* pFb = apvts.getParameter(feedbackParamID.getParamID());
    auto* pDiv = apvts.getParameter(delayDivisionParamID.getParamID());

    startNormL_ = pDelayL ? pDelayL->getValue() : 0.0f;
    startNormR_ = pDelayR ? pDelayR->getValue() : 0.0f;
    startNormFb_ = pFb ? pFb->getValue() : 0.0f;
    startDiv_ = processorRef_.getParameters().getRawDelayDivision();

    const bool isUpper = (e.position.y <= getHeight() * 0.5f);

    // Store the mode and link state at mouse-down. The drag reads only this.
    dragSynced_ = processorRef_.getParameters().getRawDelaySync();
    dragLinked_ = processorRef_.getParameters().getRawTimeLink();
    dragIsUpper_ = isUpper;

    dragGestures_.clear();

    if (dragSynced_)
    {
        if (pDiv != nullptr)
        {
            pDiv->beginChangeGesture();
            dragGestures_.push_back(pDiv);
        }
    }
    else if (dragLinked_ || dragIsUpper_)
    {
        if (pDelayL != nullptr)
        {
            pDelayL->beginChangeGesture();
            dragGestures_.push_back(pDelayL);
        }
    }
    else if (pDelayR != nullptr)
    {
        pDelayR->beginChangeGesture();
        dragGestures_.push_back(pDelayR);
    }

    if (pFb != nullptr)
    {
        pFb->beginChangeGesture();
        dragGestures_.push_back(pFb);
    }
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

    // Write only to a parameter in the snapshot set.
    const auto inDragSet = [this](const RangedAudioParameter* p)
    {
        return std::find(dragGestures_.begin(), dragGestures_.end(), p) != dragGestures_.end();
    };

    if (dragSynced_)
    {
        auto* pDiv = apvts.getParameter(delayDivisionParamID.getParamID());
        if (pDiv != nullptr && inDragSet(pDiv))
        {
            const float pixelsPerDiv = 25.0f;
            const int step = static_cast<int>(std::round(dx / pixelsPerDiv));
            const int newDiv = std::clamp(startDiv_ + step, 0, 19);
            pDiv->setValueNotifyingHost(pDiv->getNormalisableRange().convertTo0to1(static_cast<float>(newDiv)));
        }
    }
    else
    {
        const float dNorm = (dx / w) * 1.2f;
        auto* pDelayL = apvts.getParameter(delayTimeParamID.getParamID());
        auto* pDelayR = apvts.getParameter(delayTimeRParamID.getParamID());

        if (dragLinked_ || dragIsUpper_)
        {
            if (pDelayL != nullptr && inDragSet(pDelayL))
                pDelayL->setValueNotifyingHost(std::clamp(startNormL_ + dNorm, 0.0f, 1.0f));
        }
        else if (pDelayR != nullptr && inDragSet(pDelayR))
        {
            pDelayR->setValueNotifyingHost(std::clamp(startNormR_ + dNorm, 0.0f, 1.0f));
        }
    }

    // Vertical drag modulates feedback.
    auto* pFb = apvts.getParameter(feedbackParamID.getParamID());
    if (pFb != nullptr && inDragSet(pFb))
    {
        const float dNormFb = -dy / h;
        pFb->setValueNotifyingHost(std::clamp(startNormFb_ + dNormFb, 0.0f, 1.0f));
    }
}

void TapDisplay::mouseUp(const MouseEvent&)
{
    if (!dragging_)
        return;

    // Close every gesture from the snapshot. Do not re-read the mode.
    for (auto* p : dragGestures_)
        p->endChangeGesture();
    dragGestures_.clear();

    dragging_ = false;
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
    auto* p = (timeLinked || isUpper) ? apvts.getParameter(delayTimeParamID.getParamID())
                                      : apvts.getParameter(delayTimeRParamID.getParamID());
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
        auto* pDiv = apvts.getParameter(delayDivisionParamID.getParamID());
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
        auto* p = (timeLinked || isUpper) ? apvts.getParameter(delayTimeParamID.getParamID())
                                          : apvts.getParameter(delayTimeRParamID.getParamID());
        if (p == nullptr)
            return;

        // Nudge the normalized value. Shift makes the step fine.
        const float kCoarse = 0.02f;
        const float kFine = 0.004f;
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

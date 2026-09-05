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

class MonotoneSpline
{
public:
    void addKnot(const float x, const float y)
    {
        xs_.push_back(x);
        ys_.push_back(y);
    }

    [[nodiscard]] int numKnots() const noexcept { return static_cast<int>(xs_.size()); }
    [[nodiscard]] float x(const int i) const noexcept { return xs_[static_cast<std::size_t>(i)]; }
    [[nodiscard]] float y(const int i) const noexcept { return ys_[static_cast<std::size_t>(i)]; }
    [[nodiscard]] float tangent(const int i) const noexcept { return ms_[static_cast<std::size_t>(i)]; }

    // Precompute the per-knot tangents. Call once after the last addKnot.
    void build()
    {
        const int n = numKnots();
        ms_.assign(static_cast<std::size_t>(std::max(n, 0)), 0.0f);
        if (n < 2)
            return;

        std::vector<float> d(static_cast<std::size_t>(n - 1));
        for (int k = 0; k < n - 1; ++k)
        {
            const float dx = x(k + 1) - x(k);
            d[static_cast<std::size_t>(k)] = (dx > 1e-6f) ? (y(k + 1) - y(k)) / dx : 0.0f;
        }

        ms_[0] = d[0];
        ms_[static_cast<std::size_t>(n - 1)] = d[static_cast<std::size_t>(n - 2)];
        for (int k = 1; k < n - 1; ++k)
        {
            const float dPrev = d[static_cast<std::size_t>(k - 1)];
            const float dNext = d[static_cast<std::size_t>(k)];
            ms_[static_cast<std::size_t>(k)] = (dPrev * dNext <= 0.0f) ? 0.0f : (dPrev + dNext) * 0.5f;
        }

        // The Fritsch-Carlson clamp: rescale each segment's tangent pair so
        // the curve stays within [min(yk, yk+1), max(yk, yk+1)] — no overshoot.
        for (int k = 0; k < n - 1; ++k)
        {
            const float dk = d[static_cast<std::size_t>(k)];
            if (std::fabs(dk) < 1e-9f)
            {
                ms_[static_cast<std::size_t>(k)] = 0.0f;
                ms_[static_cast<std::size_t>(k + 1)] = 0.0f;
                continue;
            }
            const float a = ms_[static_cast<std::size_t>(k)] / dk;
            const float b = ms_[static_cast<std::size_t>(k + 1)] / dk;
            const float mag = a * a + b * b;
            if (mag > 9.0f)
            {
                const float tau = 3.0f / std::sqrt(mag);
                ms_[static_cast<std::size_t>(k)] = tau * a * dk;
                ms_[static_cast<std::size_t>(k + 1)] = tau * b * dk;
            }
        }
    }

    // Evaluate at px. Clamps to the first/last knot outside the domain.
    [[nodiscard]] float evaluate(const float px) const noexcept
    {
        const int n = numKnots();
        if (n == 0)
            return 0.0f;
        if (n == 1 || px <= x(0))
            return y(0);
        if (px >= x(n - 1))
            return y(n - 1);

        int k = 0;
        while (k < n - 2 && px > x(k + 1))
            ++k;

        const float x0 = x(k);
        const float x1 = x(k + 1);
        const float dx = std::max(1e-6f, x1 - x0);
        const float t = (px - x0) / dx;
        const float t2 = t * t;
        const float t3 = t2 * t;
        const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        const float h10 = t3 - 2.0f * t2 + t;
        const float h01 = -2.0f * t3 + 3.0f * t2;
        const float h11 = t3 - t2;
        return h00 * y(k) + h10 * dx * tangent(k) + h01 * y(k + 1) + h11 * dx * tangent(k + 1);
    }

private:
    std::vector<float> xs_, ys_, ms_;
};

} // namespace

TapDisplay::TapDisplay(ChronosProcessor& processor)
    : processorRef_(processor)
{
    // Arrow keys step the delay time and the division.
    setWantsKeyboardFocus(true);

    // Build the diffusion model once.
    if (const double sr = processorRef_.getSampleRate(); sr > 0.0)
        diffusionModel_ = std::make_unique<DiffusionModel>(sr);


    const bool diffuserOn = processorRef_.getParameters().getRawEnableDiffuser();
    // source/target are the fixed off/on endpoints; only target(bool) ever
    // flips direction afterward, so a reversal mid-fade stays continuous
    // instead of snapping (see the parameterChanged comment below).
    haloFadeAnim_.setSourceValue(0.0f);
    haloFadeAnim_.setTargetValue(1.0f);
    haloFadeAnim_.target(diffuserOn, true);
    haloFade_ = haloFadeAnim_.value();

    pendingDiffusion_.store(processorRef_.getParameters().getRawDiffusion(),
                           std::memory_order_relaxed);
    pendingDiffuserSize_.store(processorRef_.getParameters().getRawDiffuserSize(),
                              std::memory_order_relaxed);
    lastDiffusion_ = pendingDiffusion_.load(std::memory_order_relaxed);
    lastDiffuserSize_ = pendingDiffuserSize_.load(std::memory_order_relaxed);

    processorRef_.getAPVTS().addParameterListener(enableDiffuserParamID.getParamID(), this);
    processorRef_.getAPVTS().addParameterListener(diffusionParamID.getParamID(), this);
    processorRef_.getAPVTS().addParameterListener(diffuserSizeParamID.getParamID(), this);
}

TapDisplay::~TapDisplay()
{
    stopTimer();
    processorRef_.getAPVTS().removeParameterListener(enableDiffuserParamID.getParamID(), this);
    processorRef_.getAPVTS().removeParameterListener(diffusionParamID.getParamID(), this);
    processorRef_.getAPVTS().removeParameterListener(diffuserSizeParamID.getParamID(), this);
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

void TapDisplay::parameterChanged(const String& parameterID, const float newValue)
{
    if (parameterID == enableDiffuserParamID.getParamID())
    {
        haloFadeAnim_.target(newValue > 0.5f);
    }
    else if (parameterID == diffusionParamID.getParamID())
        pendingDiffusion_.store(newValue, std::memory_order_relaxed);
    else if (parameterID == diffuserSizeParamID.getParamID())
        pendingDiffuserSize_.store(newValue, std::memory_order_relaxed);

    triggerAsyncUpdate();
}

void TapDisplay::handleAsyncUpdate()
{
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

    // The window shows the first three repeats and caps at the largest level.
    const float maxTime = std::max(p.timeLSeconds, p.timeRSeconds);
    p.maxWindowSeconds = std::clamp(3.0f * maxTime, 2.0f, 16.0f);

    // The modulation fields carry the depth and rate for the wobble.
    // The simulation itself ignores them.
    p.modDepthCents = params.getRawDelayModDepth();
    p.modRateHz = params.getRawDelayModRateHz();

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
        || a.secondsPerBeat != p.secondsPerBeat
        || a.maxWindowSeconds != p.maxWindowSeconds;
}

void TapDisplay::runSimulation_(const TapSim::Parameters& p)
{
    const auto result = TapSim::Engine::simulate(p);
    tracker_.retarget(result, p);
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

    // Drain the input-level feed every tick. This is independent of the sim.
    TapFeedFrame frame{};
    float latestL = currentInputLevelL_;
    float latestR = currentInputLevelR_;
    bool hasFrame = false;
    while (processorRef_.getTapFifo().pop(frame))
    {
        latestL = std::max(latestL, frame.rmsL);
        latestR = std::max(latestR, frame.rmsR);
        hasFrame = true;
    }
    if (hasFrame)
    {
        currentInputLevelL_ = latestL;
        currentInputLevelR_ = latestR;
    }
    else
    {
        currentInputLevelL_ *= TapTracker::kEnvHoldDecay;
        currentInputLevelR_ *= TapTracker::kEnvHoldDecay;
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

    // Push the input envelope to the tracker.
    tracker_.pushEnvelope(true, currentInputLevelL_, static_cast<float>(dt));
    tracker_.pushEnvelope(false, currentInputLevelR_, static_cast<float>(dt));

    tracker_.advance(static_cast<float>(dt));


    haloFade_ = haloFadeAnim_.update();


    const float diffusion = pendingDiffusion_.load(std::memory_order_relaxed);
    const float diffuserSize = pendingDiffuserSize_.load(std::memory_order_relaxed);
    const bool haloGeometryChanged = std::fabs(diffusion - lastDiffusion_) > 0.001f
                                || std::fabs(diffuserSize - lastDiffuserSize_) > 0.001f;
    lastDiffusion_ = diffusion;
    lastDiffuserSize_ = diffuserSize;

    // Repaint only when something changed since the last tick.
    const bool hoverChanged = (isHovered_ != prevIsHovered_)
                         || (hoverPos_ != prevHoverPos_);
    const bool inputChanged = (std::fabs(currentInputLevelL_ - prevInputLevelL_) > 0.005f
                          || std::fabs(currentInputLevelR_ - prevInputLevelR_) > 0.005f);

    prevIsHovered_ = isHovered_;
    prevHoverPos_ = hoverPos_;
    prevInputLevelL_ = currentInputLevelL_;
    prevInputLevelR_ = currentInputLevelR_;
    if (! simRan && ! tracker_.converging() && ! hoverChanged && ! inputChanged
        && ! tracker_.wobbling() && ! haloFadeAnim_.isAnimating() && ! haloGeometryChanged)
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

    const float totalTime = std::max(0.25f, tracker_.displayedSpan());

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

    // Minor grid lines.
    g.setColour(tintInk(accent, kTintGridMinor));
    for (const float t : ticks.minors)
        g.drawVerticalLine(roundToInt(timeToX(t)), plotBounds.getY(), plotBounds.getBottom());

    // Major grid lines.
    g.setColour(tintInk(accent, kTintGridMajor));
    for (const float t : ticks.majors)
        g.drawVerticalLine(roundToInt(timeToX(t)), plotBounds.getY(), plotBounds.getBottom());

    // Center divider.
    g.setColour(tint(plotFillLit, accent, kTintCentreLine));
    g.drawHorizontalLine(static_cast<int>(centerY), plotBounds.getX(), plotBounds.getRight());


    const float diffusion = processorRef_.getParameters().getRawDiffusion();
    const float diffuserSize = processorRef_.getParameters().getRawDiffuserSize();
    const float pxPerSecond = plotBounds.getWidth() / totalTime;
    const bool haloOn = diffusionModel_ != nullptr
                         && haloFade_ > 0.001f && diffusion > 0.001f;

    if (haloOn)
    {
        const float sigma1 = diffusionModel_->sigma1Seconds(diffusion, diffuserSize);
        const float baseAlpha = Metrics::kHaloAlpha * haloFade_ * diffusion;

        struct HaloKnot { float x; float gain; float reachPx; };
        constexpr float kMinDx = 0.01f;


        const auto buildLaneWash = [&](const std::vector<TapTracker::TrackedTap>& taps, const bool leftLane,
                                       Path& wash, ColourGradient& shade) -> bool
        {
            std::vector<HaloKnot> knots;
            for (const auto& tap : taps)
            {
                if (tap.dry || tap.key < 1)
                    continue;
                const float gain = std::clamp(std::fabs(tap.displayedGain), 0.0f, 1.0f);
                if (gain <= 0.001f)
                    continue;
                const float timeNorm = std::clamp(tap.displayedTime / totalTime, 0.0f, 1.0f);
                const float modOffsetPx = metrics_.pxf(tracker_.modOffset(leftLane, tap.key));
                const float reachPx = Metrics::kHaloSigmas * sigma1
                                     * std::sqrt(static_cast<float>(tap.key)) * pxPerSecond;
                knots.push_back({ plotBounds.getX() + timeNorm * plotBounds.getWidth() + modOffsetPx,
                                  gain, reachPx });
            }
            if (knots.empty())
                return false;
            std::sort(knots.begin(), knots.end(),
                      [](const HaloKnot& a, const HaloKnot& b) { return a.x < b.x; });

            // Guarantee strictly increasing x: the mod-jitter offset can
            // otherwise push two adjacent repeats to (near-)equal x.
            for (std::size_t i = 1; i < knots.size(); ++i)
                knots[i].x = std::max(knots[i].x, knots[i - 1].x + kMinDx);

            // Taper to zero before the first repeat and after the last, so
            // the curve rises from and falls to silence instead of
            // starting or ending on a hard edge. The taper distance is the
            // same physically-modelled spread the halo has always used.
            MonotoneSpline spline;
            spline.addKnot(knots.front().x - std::max(knots.front().reachPx, kMinDx), 0.0f);
            for (const auto& k : knots)
                spline.addKnot(k.x, k.gain);
            spline.addKnot(knots.back().x + std::max(knots.back().reachPx, kMinDx), 0.0f);
            spline.build();

            // The visible column range, clamped to the plot.
            const float x0 = std::max(plotBounds.getX(), spline.x(0));
            const float x1 = std::min(plotBounds.getRight(), spline.x(spline.numKnots() - 1));
            const int xi0 = static_cast<int>(std::floor(x0));
            const int xi1 = static_cast<int>(std::ceil(x1));
            if (xi1 - xi0 < 1)
                return false;

            // The top edge: one cubic Bezier per spline segment, an exact
            // conversion from the Hermite tangents, so the curve is never
            // approximated by short straight lines.
            const float dir = leftLane ? -1.0f : 1.0f;
            const auto toY = [&](const float gain) { return centerY + dir * gain * maxLaneHeight; };

            wash.startNewSubPath(spline.x(0), toY(spline.y(0)));
            for (int k = 0; k < spline.numKnots() - 1; ++k)
            {
                const float sx0 = spline.x(k);
                const float sx1 = spline.x(k + 1);
                const float segDx = sx1 - sx0;
                const float c1y = spline.y(k) + spline.tangent(k) * segDx / 3.0f;
                const float c2y = spline.y(k + 1) - spline.tangent(k + 1) * segDx / 3.0f;
                wash.cubicTo(sx0 + segDx / 3.0f, toY(c1y), sx1 - segDx / 3.0f, toY(c2y),
                            sx1, toY(spline.y(k + 1)));
            }
            wash.closeSubPath();

            // The column alpha follows the sqrt of the same spline (sqrt,
            // not linear, so the late quiet repeats keep their visibility
            // instead of vanishing) and rides a horizontal gradient with
            // one stop per column, so no shading step is ever visible.
            const auto alphaAt = [&](const float x)
            {
                return std::clamp(baseAlpha * std::sqrt(std::max(0.0f, spline.evaluate(x))), 0.0f, 1.0f);
            };
            shade = ColourGradient(accent.withAlpha(alphaAt(static_cast<float>(xi0))),
                                   Point<float>(static_cast<float>(xi0), centerY),
                                   accent.withAlpha(alphaAt(static_cast<float>(xi1))),
                                   Point<float>(static_cast<float>(xi1), centerY), false);
            const int cols = xi1 - xi0 + 1;
            const auto invSpan = 1.0 / static_cast<double>(std::max(1, cols - 1));
            for (int i = 1; i < cols - 1; ++i)
                shade.addColour(static_cast<double>(i) * invSpan,
                                accent.withAlpha(alphaAt(static_cast<float>(xi0 + i))));
            return true;
        };

        Path topWash, bottomWash;
        ColourGradient topShade, bottomShade;
        const bool hasTop = buildLaneWash(tracker_.lane(true), true, topWash, topShade);
        const bool hasBottom = buildLaneWash(tracker_.lane(false), false, bottomWash, bottomShade);

        if (hasTop || hasBottom)
        {
            const float blurPx = metrics_.pxf(Metrics::kHaloBlurRadius);
            const float pad = blurPx * 2.0f + 4.0f;
            const auto imgArea = plotBounds.expanded(pad);
            double scaleFactor = Component::getApproximateScaleFactorForComponent(this);
            scaleFactor = std::clamp(scaleFactor, 1.0, 4.0);
            const float imgScale = static_cast<float>(scaleFactor);
            const int imgW = std::max(1, roundToInt(imgArea.getWidth() * imgScale));
            const int imgH = std::max(1, roundToInt(imgArea.getHeight() * imgScale));

            Image haloImg(Image::ARGB, imgW, imgH, true);
            {
                Graphics haloG(haloImg);
                haloG.addTransform(AffineTransform::translation(-imgArea.getX(), -imgArea.getY())
                                       .scaled(imgScale));
                if (hasTop)
                {
                    haloG.setGradientFill(topShade);
                    haloG.fillPath(topWash);
                }
                if (hasBottom)
                {
                    haloG.setGradientFill(bottomShade);
                    haloG.fillPath(bottomWash);
                }
            }
            if (auto pixelData = haloImg.getPixelData())
                pixelData->applyGaussianBlurEffect(blurPx * imgScale);

            Graphics::ScopedSaveState ss(g);
            g.reduceClipRegion(plotBounds.toNearestInt());
            g.drawImageTransformed(haloImg,
                AffineTransform::scale(1.0f / imgScale).translated(imgArea.getX(), imgArea.getY()));
        }
    }

    // Draw taps. The head scales with the displayed gain and the activity.
    const auto drawLane = [&](const std::vector<TapTracker::TrackedTap>& taps, const bool isTopLane)
    {
        const bool leftLane = isTopLane;
        for (const auto& tap : taps)
        {
            const float timeNorm = std::clamp(tap.displayedTime / totalTime, 0.0f, 1.0f);
            const float modOffsetPx = metrics_.pxf(tracker_.modOffset(leftLane, tap.key));
            const float x = plotBounds.getX() + timeNorm * plotBounds.getWidth() + modOffsetPx;

            const float gain = std::clamp(std::fabs(tap.displayedGain), 0.0f, 1.0f);
            if (gain <= 0.001f)
                continue;

            // The activity lights the tap as the audio passes through.
            const float act = tracker_.activity(leftLane, tap.dry ? 0.0f : tap.targetTime);


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

            // The halo softens the bar as the diffusion rises.
            const float baseAlpha = tap.dry ? kTapDryAlpha : kTapBarAlpha;
            const float barFade = tap.dry ? 1.0f : (1.0f - Metrics::kHaloBarFade * haloFade_ * diffusion);
            const float barAlpha = tap.dry ? baseAlpha : baseAlpha * (0.70f + 0.30f * act) * barFade;
            g.setColour(tapCol.withAlpha(barAlpha));
            g.strokePath(barPath, PathStrokeType(metrics_.pxf(Metrics::kTapBarStroke), PathStrokeType::curved, PathStrokeType::rounded));

            // Head dot. The head scales with the displayed gain and the activity.
            const float headY = isTopLane ? (centerY - barHeight) : (centerY + barHeight);
            const float baseHead = metrics_.pxf(Metrics::kTapHeadRadius);
            const float headScale = std::clamp(tap.displayedGain / TapTracker::kHeadFullGain, 0.0f, 1.0f);
            const float headRadius = baseHead * headScale + act * metrics_.pxf(Metrics::kTapHeadGrow);
            if (headRadius > 0.25f)
            {
                const float headAlpha = (0.55f + 0.45f * act) * headScale;
                g.setColour(tapCol.withAlpha(headAlpha));
                g.fillEllipse(x - headRadius, headY - headRadius, headRadius * 2, headRadius * 2);
            }
        }
    };
    drawLane(tracker_.lane(true), true);
    drawLane(tracker_.lane(false), false);

    // Hover half-highlight and channel affordance label.
    if (isHovered_)
    {
        const bool timeLinked = processorRef_.getParameters().getRawTimeLink();
        const bool hoverUpper = (hoverPos_.y <= centerY);

        if (timeLinked)
        {
            g.setColour(accent.withAlpha(kTapHoverLinkAlpha));
            g.fillRect(plotBounds);
            g.setColour(accent.withAlpha(kTapCursorAlpha));
            g.setFont(Fonts::font(Fonts::Weight::Medium, metrics_.font(Metrics::kTapLabelFont)));
            g.drawText("L/R LINK", plotBounds.reduced(metrics_.pxf(Metrics::kTapLabelInset)).toNearestInt(), Justification::topLeft, false);
        }
        else if (hoverUpper)
        {
            g.setColour(accent.withAlpha(kTapHoverFillAlpha));
            g.fillRect(plotBounds.withBottom(centerY));
            g.setColour(accent.withAlpha(kTapCursorAlpha));
            g.setFont(Fonts::font(Fonts::Weight::Medium, metrics_.font(Metrics::kTapLabelFont)));
            g.drawText("LEFT TIME", plotBounds.reduced(metrics_.pxf(Metrics::kTapLabelInset)).toNearestInt(), Justification::topLeft, false);
        }
        else
        {
            g.setColour(accent.withAlpha(kTapHoverFillAlpha));
            g.fillRect(plotBounds.withTop(centerY));
            g.setColour(accent.withAlpha(kTapCursorAlpha));
            g.setFont(Fonts::font(Fonts::Weight::Medium, metrics_.font(Metrics::kTapLabelFont)));
            g.drawText("RIGHT TIME", plotBounds.reduced(metrics_.pxf(Metrics::kTapLabelInset)).toNearestInt(), Justification::bottomLeft, false);
        }
    }

    // Ruler tick marks.
    const float tickTop = plotBounds.getBottom();
    const float majorTickLen = metrics_.pxf(Metrics::kMajorTick);
    const float minorTickLen = metrics_.pxf(Metrics::kMinorTick);
    g.setColour(tintInk(accent, kTintGridMajor));
    for (const float t : ticks.minors)
        g.drawVerticalLine(roundToInt(timeToX(t)), tickTop, tickTop + minorTickLen);
    g.setColour(tintInk(accent, kTintGridMajor));
    for (const float t : ticks.majors)
        g.drawVerticalLine(roundToInt(timeToX(t)), tickTop, tickTop + majorTickLen);

    // Ruler labels. The display face carries the numeric readouts.
    const Font rulerFont = Fonts::display(metrics_.displayFont(Metrics::kRulerFont));
    const float baselineY = tickTop + majorTickLen + metrics_.pxf(Metrics::kRulerLabelGap)
                        + rulerFont.getAscent();
    const int baselineYi = roundToInt(baselineY);
    const float minLabelGap = metrics_.pxf(Metrics::kTapLabelGapMin);
    g.setFont(rulerFont);
    g.setColour(Colours::rulerText);

    // Clamp the label into the plot. A centred label that would cross
    // the edge sticks to the edge, so the first label clears the border.
    const float labelEdgeLeft = plotBounds.getX() + metrics_.pxf(Metrics::kRulerLabelInset);
    const float labelEdgeRight = plotBounds.getRight() - metrics_.pxf(Metrics::kRulerLabelInset);
    const auto drawRulerLabel = [&](const String& text, const float cx)
    {
        const float w = Fonts::textWidth(rulerFont, text);
        const float left = cx - w * 0.5f;
        const float right = cx + w * 0.5f;
        if (left < labelEdgeLeft)
            g.drawSingleLineText(text, roundToInt(labelEdgeLeft), baselineYi, Justification::left);
        else if (right > labelEdgeRight)
            g.drawSingleLineText(text, roundToInt(labelEdgeRight), baselineYi, Justification::right);
        else
            g.drawSingleLineText(text, roundToInt(cx), baselineYi, Justification::horizontallyCentred);
    };

    if (synced)
    {
        const int div = processorRef_.getParameters().getRawDelayDivision();
        g.drawText(divisionName(div), rulerBounds.reduced(metrics_.pxf(Metrics::kRulerLabelInset), 0.0f).toNearestInt(), Justification::centredLeft, true);

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
            drawRulerLabel(majorLabels[static_cast<std::size_t>(i)],
                           majorXs[static_cast<std::size_t>(i)]);
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
            drawRulerLabel(majorLabels[static_cast<std::size_t>(i)],
                           majorXs[static_cast<std::size_t>(i)]);
    }

    // Hover cursor and measurement readout.
    if (isHovered_)
    {
        const bool hoverUpper = (hoverPos_.y <= centerY);
        const auto& lane = hoverUpper ? tracker_.lane(true) : tracker_.lane(false);

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

        g.setColour(accent.withAlpha(kTapCursorAlpha));
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
        g.setFont(Fonts::display(metrics_.displayFont(Metrics::kTapReadoutFont)));
        const float readoutW = metrics_.pxf(Metrics::kHoverReadoutW);
        const float readoutGap = metrics_.pxf(Metrics::kHoverReadoutGap);
        float rx = cursorX + readoutGap;
        if (rx + readoutW > displayBounds.getRight())
            rx = cursorX - readoutW - readoutGap;
        const auto readoutRect = Rectangle<float>(rx, plotBounds.getY() + metrics_.pxf(Metrics::kHoverReadoutTop), readoutW, metrics_.pxf(Metrics::kHoverReadoutH)).toNearestInt();
        g.drawText(readout, readoutRect, Justification::centredLeft, true);
    }
}

void TapDisplay::mouseDown(const MouseEvent& e)
{
    // An inert display takes no drag.
    if (! isEnabled())
        return;

    dragStartX_ = e.position.x;
    dragStartY_ = e.position.y;
    dragging_ = true;
    dragAxis_ = 0;

    auto& apvts = processorRef_.getAPVTS();
    auto* pDelayL = apvts.getParameter(delayTimeParamID.getParamID());
    auto* pDelayR = apvts.getParameter(delayTimeRParamID.getParamID());
    auto* pFb = apvts.getParameter(feedbackParamID.getParamID());
    auto* pDiv = apvts.getParameter(delayDivisionParamID.getParamID());

    startNormL_ = pDelayL ? pDelayL->getValue() : 0.0f;
    startNormR_ = pDelayR ? pDelayR->getValue() : 0.0f;
    startNormFb_ = pFb ? pFb->getValue() : 0.0f;
    startDiv_ = processorRef_.getParameters().getRawDelayDivision();

    // Store the mode and link state at mouse-down. The drag reads only this.
    dragSynced_ = processorRef_.getParameters().getRawDelaySync();
    dragLinked_ = processorRef_.getParameters().getRawTimeLink();
    dragIsUpper_ = (e.position.y <= getHeight() * 0.5f);

    // Open no gesture until the dead zone clears and the axis latches.
    dragGestures_.clear();
}

void TapDisplay::mouseDrag(const MouseEvent& e)
{
    if (!dragging_)
        return;

    const float dx = e.position.x - dragStartX_;
    const float dy = e.position.y - dragStartY_;
    const float w = static_cast<float>(std::max(1, getWidth()));
    const float h = static_cast<float>(std::max(1, getHeight()));

    // The dead zone. Nothing is written until it clears.
    const float deadZone = metrics_.pxf(Metrics::kDragDeadZone);
    if (dragAxis_ == 0)
    {
        if (std::fabs(dx) < deadZone && std::fabs(dy) < deadZone)
            return;

        // Latch to the dominant axis.
        dragAxis_ = (std::fabs(dx) >= std::fabs(dy)) ? 1 : 2;
        const auto cursor = (dragAxis_ == 1) ? MouseCursor::LeftRightResizeCursor
                                           : MouseCursor::UpDownResizeCursor;
        setMouseCursor(cursor);
    }

    auto& apvts = processorRef_.getAPVTS();

    // Horizontal writes time (or the division under sync).
    if (dragAxis_ == 1)
    {
        if (dragSynced_)
        {
            auto* pDiv = apvts.getParameter(delayDivisionParamID.getParamID());
            if (pDiv != nullptr && dragGestures_.empty())
            {
                pDiv->beginChangeGesture();
                dragGestures_.push_back(pDiv);
            }
            if (std::find(dragGestures_.begin(), dragGestures_.end(), pDiv) != dragGestures_.end())
            {
                const float pixelsPerDiv = metrics_.pxf(Metrics::kDragPixelsPerDivision);
                const int step = static_cast<int>(std::round(dx / pixelsPerDiv));
                const int newDiv = std::clamp(startDiv_ + step, 0, 19);
                pDiv->setValueNotifyingHost(pDiv->getNormalisableRange().convertTo0to1(static_cast<float>(newDiv)));
            }
        }
        else
        {
            const float dNorm = (dx / w) * Metrics::kDragTimeGain;
            auto* pDelayL = apvts.getParameter(delayTimeParamID.getParamID());
            auto* pDelayR = apvts.getParameter(delayTimeRParamID.getParamID());

            // Open the gesture the existing link/upper rule selects.
            if (dragGestures_.empty())
            {
                if (dragLinked_ || dragIsUpper_)
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
            }

            const auto inDragSet = [this](const RangedAudioParameter* p)
            {
                return std::find(dragGestures_.begin(), dragGestures_.end(), p) != dragGestures_.end();
            };

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
    }
    else // dragAxis_ == 2: vertical writes feedback.
    {
        auto* pFb = apvts.getParameter(feedbackParamID.getParamID());
        if (pFb != nullptr && dragGestures_.empty())
        {
            pFb->beginChangeGesture();
            dragGestures_.push_back(pFb);
        }
        if (std::find(dragGestures_.begin(), dragGestures_.end(), pFb) != dragGestures_.end())
        {
            const float dNormFb = -dy / h;
            pFb->setValueNotifyingHost(std::clamp(startNormFb_ + dNormFb, 0.0f, 1.0f));
        }
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
    dragAxis_ = 0;
    setMouseCursor(MouseCursor::NormalCursor);
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
    // An inert display takes no double-click.
    if (! isEnabled())
        return;

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
    // An inert display takes no wheel input.
    if (! isEnabled())
        return;

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

// Step the same parameters the wheel acts on. Up and right increase.
// One bracket pair per press.
bool TapDisplay::keyPressed(const KeyPress& key)
{
    if (! isEnabled())
        return false;

    double dir = 0.0;
    if (key == KeyPress::rightKey || key == KeyPress::upKey)        dir = 1.0;
    else if (key == KeyPress::leftKey || key == KeyPress::downKey)  dir = -1.0;
    if (dir == 0.0)
        return false;

    const bool fine = key.getModifiers().isShiftDown();
    auto& apvts = processorRef_.getAPVTS();

    if (processorRef_.getParameters().getRawDelaySync())
    {
        auto* pDiv = apvts.getParameter(delayDivisionParamID.getParamID());
        if (pDiv == nullptr)
            return false;
        const int cur = processorRef_.getParameters().getRawDelayDivision();
        const int newIdx = std::clamp(cur + ((dir > 0.0) ? 1 : -1), 0, 19);
        if (newIdx == cur)
            return true;
        pDiv->beginChangeGesture();
        pDiv->setValueNotifyingHost(
            pDiv->getNormalisableRange().convertTo0to1(static_cast<float>(newIdx)));
        pDiv->endChangeGesture();
        repaint();
        return true;
    }

    // The linked keyboard writes the left channel. The unlinked
    // keyboard writes the right channel.
    auto* p = processorRef_.getParameters().getRawTimeLink()
        ? apvts.getParameter(delayTimeParamID.getParamID())
        : apvts.getParameter(delayTimeRParamID.getParamID());
    if (p == nullptr)
        return false;

    const double step = dir * (fine ? Metrics::kWheelStepFine : Metrics::kWheelStepCoarse);
    const double prop = std::clamp(p->getValue() + step, 0.0, 1.0);
    p->beginChangeGesture();
    p->setValueNotifyingHost(static_cast<float>(prop));
    p->endChangeGesture();
    repaint();
    return true;
}

void TapDisplay::resized()
{
}

} // namespace MarsDSP::GUI

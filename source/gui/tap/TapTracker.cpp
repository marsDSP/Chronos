#include "TapTracker.h"

#include <algorithm>
#include <cmath>

// The modulation jitter display constants live in Metrics.h (section 4.7).
// Repeat them here so this translation unit stays JUCE-free.
namespace { constexpr float kModJitterDU = 4.0f; constexpr float kModJitterMaxDU = 16.0f; }

namespace MarsDSP::GUI {

TapTracker::TapTracker()
    : envL_(static_cast<std::size_t>(kEnvBuckets), 0.0f),
      envR_(static_cast<std::size_t>(kEnvBuckets), 0.0f)
{
    // Seed the two RNG streams with distinct indices.
    rngL_.seed(0x9E3779B97F4A7C15uLL, 1u);
    rngR_.seed(0x9E3779B97F4A7C15uLL, 2u);
}

void TapTracker::retarget(const TapSim::SimulationResult& result, const TapSim::Parameters& params)
{
    // Read the modulation fields the simulation ignores.
    modDepth_ = params.modDepthCents;
    modRateHz_ = params.modRateHz;

    // The base delay is the first repeat time. Fall back to the parameter.
    const float baseL = (result.left.size() > 1) ? result.left[1].timeSeconds : params.timeLSeconds;
    const float baseR = (result.right.size() > 1) ? result.right[1].timeSeconds : params.timeRSeconds;

    const auto matchLane = [&](std::vector<TrackedTap>& tracked,
                              const std::vector<TapSim::Tap>& simTaps,
                              float baseTime)
    {
        const float invBase = (baseTime > 1e-6f) ? (1.0f / baseTime) : 0.0f;

        // Track which keys the sim still emits. Clear the flag on a match.
        for (auto& t : tracked)
            t.matched = false;

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
                // A returning key eases from its current displayed gain.
                found->dry = sim.dry;
                found->targetTime = sim.timeSeconds;
                found->targetGain = sim.gain;
                found->matched = true;
            }
            else
            {
                // A new key starts at gain zero at its own position.
                TrackedTap t;
                t.key = key;
                t.dry = sim.dry;
                t.targetTime = sim.timeSeconds;
                t.targetGain = sim.gain;
                t.displayedTime = sim.timeSeconds;
                t.displayedGain = 0.0f;
                t.matched = true;
                tracked.push_back(std::move(t));
            }
        }

        // A tap the sim no longer emits fades toward zero. It keeps its position.
        for (auto& t : tracked)
            if (! t.matched)
                t.targetGain = 0.0f;
    };

    matchLane(left_, result.left, baseL);
    matchLane(right_, result.right, baseR);

    baseL_ = baseL;
    baseR_ = baseR;

    // The span follows the last tap time.
    const float lastL = result.left.empty() ? 0.0f : result.left.back().timeSeconds;
    const float lastR = result.right.empty() ? 0.0f : result.right.back().timeSeconds;
    updateSpanLevel_(std::max(lastL, lastR));
}

void TapTracker::updateSpanLevel_(const float lastTapSeconds)
{
    // The target level is the smallest level whose fill holds the last tap.
    // When no level holds it, the largest level.
    int newLevel = static_cast<int>(kSpanLevels.size()) - 1;
    for (int i = 0; i < static_cast<int>(kSpanLevels.size()); ++i)
    {
        if (lastTapSeconds <= kSpanLevels[static_cast<std::size_t>(i)] * kSpanFill)
        {
            newLevel = i;
            break;
        }
    }

    // Hysteresis: step down only when the fill holds at the lower level.
    if (newLevel < spanLevel_)
    {
        if (lastTapSeconds <= kSpanLevels[static_cast<std::size_t>(newLevel)] * kSpanDownFill)
            spanLevel_ = newLevel;
    }
    else
    {
        spanLevel_ = newLevel;
    }

    targetSpan_ = kSpanLevels[static_cast<std::size_t>(spanLevel_)];
}

void TapTracker::pushEnvelope(const bool left, const float value, const float deltaSeconds)
{
    auto& env      = left ? envL_      : envR_;
    auto& writeIdx = left ? envWriteIdxL_ : envWriteIdxR_;
    auto& writeTime = left ? envWriteTimeL_ : envWriteTimeR_;
    auto& envRef   = left ? envRefL_   : envRefR_;

    // The reference follower makes the animation level-independent.
    envRef = std::max(value, envRef * std::exp(-deltaSeconds / kEnvRefRelease));
    if (envRef < kEnvFloor)
        envRef = kEnvFloor;

    // Fill every bucket between the previous write time and now.
    const float prevTime = writeTime;
    writeTime += deltaSeconds;
    const int prevIdx = writeIdx;
    const int bucketsDelta = static_cast<int>(deltaSeconds * kEnvRateHz);
    const int newIdx = (prevIdx + std::max(1, bucketsDelta)) % kEnvBuckets;

    // Write the value into every bucket from prevIdx to newIdx.
    int idx = prevIdx;
    for (int i = 0; i < std::max(1, bucketsDelta); ++i)
    {
        idx = (idx + 1) % kEnvBuckets;
        env[static_cast<std::size_t>(idx)] = value;
    }
    if (newIdx != prevIdx)
        env[static_cast<std::size_t>(newIdx)] = value;
    writeIdx = newIdx;
}

float TapTracker::envAt_(const bool left, const float t) const noexcept
{
    const auto& env      = left ? envL_      : envR_;
    const auto writeIdx = left ? envWriteIdxL_ : envWriteIdxR_;
    const auto writeTime = left ? envWriteTimeL_ : envWriteTimeR_;

    if (t <= 0.0f)
        return env[static_cast<std::size_t>(writeIdx)];

    // The bucket t seconds back.
    const float bucketF = t * kEnvRateHz;
    const int bucket = static_cast<int>(bucketF);
    const int readIdx = ((writeIdx - bucket) % kEnvBuckets + kEnvBuckets) % kEnvBuckets;
    const int nextIdx = (readIdx + 1) % kEnvBuckets;
    const float frac = bucketF - static_cast<float>(bucket);
    return env[static_cast<std::size_t>(readIdx)] * (1.0f - frac)
         + env[static_cast<std::size_t>(nextIdx)] * frac;
}

float TapTracker::activity(const bool left, const float targetTime) const noexcept
{
    const float e = envAt_(left, targetTime);
    const float ref = left ? envRefL_ : envRefR_;
    return std::sqrt(std::clamp(e / ref, 0.0f, 1.0f));
}

float TapTracker::modOffset(const bool left, const int n) const noexcept
{
    if (n <= 0 || modDepth_ <= 0.0f)
        return 0.0f;

    // The wobble grows with the square root of the repeat index, capped at 3.
    const float scale = std::min(std::sqrt(static_cast<float>(n)), 3.0f);
    const float depthNorm = modDepth_ / 50.0f;
    const float x = left ? ouL_.state() : ouR_.state();
    float offset = kModJitterDU * depthNorm * scale * static_cast<float>(x);
    return std::clamp(offset, -kModJitterMaxDU, kModJitterMaxDU);
}

void TapTracker::advance(const float deltaSeconds)
{
    if (deltaSeconds <= 0.0f)
        return;

    const float kMove = 1.0f - std::exp(-deltaSeconds / kTauMove);
    const float kGain = 1.0f - std::exp(-deltaSeconds / kTauGain);
    const float kFade = 1.0f - std::exp(-deltaSeconds / kTauFade);
    const float kSpan = 1.0f - std::exp(-deltaSeconds / kTauSpan);

    // Step the modulation wobble once per tick.
    if (modDepth_ > 0.0f)
    {
        ouL_.setRate(1.0f / deltaSeconds, modRateHz_);
        ouR_.setRate(1.0f / deltaSeconds, modRateHz_);
        ouL_.next(rngL_);
        ouR_.next(rngR_);
    }

    const auto easeLane = [&](std::vector<TrackedTap>& lane, float base)
    {
        for (auto it = lane.begin(); it != lane.end(); )
        {
            TrackedTap& t = *it;

            if (t.targetGain <= 0.0f)
            {
                // An unmatched tap fades toward zero and is culled.
                t.displayedGain += (0.0f - t.displayedGain) * kFade;

                if (std::fabs(t.displayedGain) < kCullGain)
                {
                    it = lane.erase(it);
                    continue;
                }

                // A fading tap keeps easing toward its key position.
                const float keyTime = t.dry ? 0.0f : static_cast<float>(t.key) * base;
                t.displayedTime += (keyTime - t.displayedTime) * kMove;
            }
            else
            {
                t.displayedTime += (t.targetTime - t.displayedTime) * kMove;
                t.displayedGain += (t.targetGain - t.displayedGain) * kGain;
            }

            ++it;
        }
    };

    easeLane(left_, baseL_);
    easeLane(right_, baseR_);

    displayedSpan_ += (targetSpan_ - displayedSpan_) * kSpan;
}

bool TapTracker::converging() const noexcept
{
    const auto laneConverging = [](const std::vector<TrackedTap>& lane) {
        for (const auto& t : lane)
        {
            if (t.targetGain <= 0.0f)
                return true;
            if (std::fabs(t.displayedTime - t.targetTime) > 1e-4f
                || std::fabs(t.displayedGain - t.targetGain) > 1e-4f)
                return true;
        }
        return false;
    };

    if (laneConverging(left_) || laneConverging(right_))
        return true;

    return std::fabs(displayedSpan_ - targetSpan_) > 1e-4f;
}

} // namespace MarsDSP::GUI

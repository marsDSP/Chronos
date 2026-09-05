#include "TapTracker.h"

#include <algorithm>
#include <cmath>

namespace MarsDSP::GUI {

TapTracker::TapTracker() = default;

void TapTracker::retarget(const TapSim::SimulationResult& result, const TapSim::Parameters& params)
{
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

void TapTracker::advance(const float deltaSeconds)
{
    if (deltaSeconds <= 0.0f)
        return;

    const float kMove = 1.0f - std::exp(-deltaSeconds / kTauMove);
    const float kGain = 1.0f - std::exp(-deltaSeconds / kTauGain);
    const float kFade = 1.0f - std::exp(-deltaSeconds / kTauFade);
    const float kSpan = 1.0f - std::exp(-deltaSeconds / kTauSpan);

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

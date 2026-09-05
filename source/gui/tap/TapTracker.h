#pragma once

#ifndef CHRONOS_TAP_TRACKER_H
#define CHRONOS_TAP_TRACKER_H

#include "TapSimulation.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace MarsDSP::GUI {

// Owns the two lanes of tracked taps, the span, and the eases.
// JUCE-free. The display drives this from its timer tick.
class TapTracker {
public:
    // One tap tracked by identity across frames.
    struct TrackedTap {
        int key = 0;
        bool dry = false;
        float targetTime = 0.0f;
        float targetGain = 0.0f;
        float displayedTime = 0.0f;
        float displayedGain = 0.0f;
        bool matched = false;
    };

    // Span quantisation levels in seconds (section 4.6).
    static constexpr std::array<float, 12> kSpanLevels = {
        0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f, 16.0f
    };
    static constexpr float kSpanFill      = 0.92f;
    static constexpr float kSpanDownFill  = 0.78f;

    // Ease time constants in seconds (section 4.6).
    static constexpr float kTauMove = 0.060f;
    static constexpr float kTauGain = 0.050f;
    static constexpr float kTauSpan = 0.060f;
    static constexpr float kTauFade = 0.035f;

    // A tap below this gain is culled (section 4.6).
    static constexpr float kCullGain = 0.02f;
    // The gain at which the head reaches full size (section 4.6).
    static constexpr float kHeadFullGain = 0.12f;

    TapTracker();

    // Reset the tracked taps from a fresh simulation result.
    void retarget(const TapSim::SimulationResult& result, const TapSim::Parameters& params);

    // Advance every ease by deltaSeconds. Cull taps below the gain floor.
    void advance(float deltaSeconds);

    // The two lanes.
    const std::vector<TrackedTap>& lane(bool left) const noexcept { return left ? left_ : right_; }

    // The eased span the plot maps time to.
    float displayedSpan() const noexcept { return displayedSpan_; }

    // The target span level the eased span follows.
    float targetSpan() const noexcept { return targetSpan_; }

    // True when any tap or the span is still converging.
    bool converging() const noexcept;

private:
    // Pick the span level for the last tap time, with hysteresis.
    void updateSpanLevel_(float lastTapSeconds);

    std::vector<TrackedTap> left_;
    std::vector<TrackedTap> right_;
    float displayedSpan_ = 0.5f;
    float targetSpan_ = 0.5f;
    int spanLevel_ = 1;
    // The base delay per lane. A fading tap eases toward key * baseTime.
    float baseL_ = 0.375f;
    float baseR_ = 0.375f;
};

} // namespace MarsDSP::GUI

#endif

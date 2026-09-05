#pragma once

#ifndef CHRONOS_TAP_TRACKER_H
#define CHRONOS_TAP_TRACKER_H

#include "TapSimulation.h"
#include "../../dsp/Modulation.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace MarsDSP::GUI {

// Owns the two lanes of tracked taps, the span, the eases, the input
// envelope history, and the modulation wobble. JUCE-free. The display
// drives this from its timer tick.
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

    // Envelope history (section 4.7). One lane of buckets per second.
    static constexpr float kEnvRateHz    = 200.0f;
    static constexpr float kEnvSeconds    = 16.0f;
    static constexpr int   kEnvBuckets  = static_cast<int>(kEnvRateHz * kEnvSeconds);
    static constexpr float kEnvHoldDecay = 0.90f;
    static constexpr float kEnvRefRelease = 1.5f;
    static constexpr float kEnvFloor      = 1e-3f;

    TapTracker();

    // Reset the tracked taps from a fresh simulation result.
    void retarget(const TapSim::SimulationResult& result, const TapSim::Parameters& params);

    // Advance every ease by deltaSeconds. Step the modulation wobble.
    void advance(float deltaSeconds);

    // Push an input envelope value for the lane. Fill the buckets
    // between the previous write time and now.
    void pushEnvelope(bool left, float value, float deltaSeconds);

    // The activity for a tap at targetTime seconds in the lane.
    // The dry tap reads t = 0.
    [[nodiscard]] float activity(bool left, float targetTime) const noexcept;

    // The modulation display offset for repeat n in the lane, design units.
    // n = 0 (dry) gives 0.
    [[nodiscard]] float modOffset(bool left, int n) const noexcept;

    // The two lanes.
    const std::vector<TrackedTap>& lane(bool left) const noexcept { return left ? left_ : right_; }

    // The eased span the plot maps time to.
    float displayedSpan() const noexcept { return displayedSpan_; }

    // The target span level the eased span follows.
    float targetSpan() const noexcept { return targetSpan_; }

    // True when any tap, the span, or any activity is still converging.
    bool converging() const noexcept;

    // True when the modulation depth is above zero (the wobble moves).
    bool wobbling() const noexcept { return modDepth_ > 0.0f; }

private:
    // Pick the span level for the last tap time, with hysteresis.
    void updateSpanLevel_(float lastTapSeconds);

    // Read the envelope history t seconds back with linear interpolation.
    [[nodiscard]] float envAt_(bool left, float t) const noexcept;

    std::vector<TrackedTap> left_;
    std::vector<TrackedTap> right_;
    float displayedSpan_ = 0.5f;
    float targetSpan_ = 0.5f;
    int spanLevel_ = 1;
    // The base delay per lane. A fading tap eases toward key * baseTime.
    float baseL_ = 0.375f;
    float baseR_ = 0.375f;

    // The input envelope history, one per lane.
    std::vector<float> envL_;
    std::vector<float> envR_;
    // The write cursor in buckets and the time of the last write.
    int envWriteIdxL_ = 0;
    int envWriteIdxR_ = 0;
    float envWriteTimeL_ = 0.0f;
    float envWriteTimeR_ = 0.0f;
    // The reference follower per lane.
    float envRefL_ = 0.0f;
    float envRefR_ = 0.0f;

    // The modulation wobble. One OU and one RNG per lane.
    Mod::OrnsteinUhlenbeck ouL_;
    Mod::OrnsteinUhlenbeck ouR_;
    Mod::Pcg32 rngL_;
    Mod::Pcg32 rngR_;
    float modDepth_ = 0.0f;
    float modRateHz_ = 0.35f;
};

} // namespace MarsDSP::GUI

#endif

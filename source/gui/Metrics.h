#pragma once

#ifndef CHRONOS_METRICS_H
#define CHRONOS_METRICS_H

#include <JuceHeader.h>
#include <algorithm>

namespace MarsDSP::GUI
{
    // Scale metrics for the editor.
    // One scale factor s derives every dimension from the 1000 x 640 design box.
    // Components read dimensions through px, pxf, font, and stroke.
    struct Metrics
    {
        // Design box (section 4.1).
        static constexpr int kDesignWidth = 1000;
        static constexpr int kDesignHeight = 640;
        static constexpr double kDesignAspect = 1.5625;

        // Scale clamp range (section 4.1).
        static constexpr float kScaleMin = 0.64f;
        static constexpr float kScaleMax = 1.60f;

        // Window envelope (section 4.2), design units.
        static constexpr int kDefaultWidth = 760;
        static constexpr int kDefaultHeight = 486;
        static constexpr int kMinWidth = 640;
        static constexpr int kMaxWidth = 1600;
        static constexpr int kMinHeight = 410;
        static constexpr int kMaxHeight = 1024;

        // Band geometry (section 4.3), design units.
        static constexpr int kTopPad = 2;
        static constexpr int kHeaderH = 44;
        static constexpr int kGapHeader = 4;
        static constexpr int kTapH = 250;
        static constexpr int kGapTap = 8;
        static constexpr int kCardRowH = 292;
        static constexpr int kGapCards = 4;
        static constexpr int kFooterH = 34;
        static constexpr int kBottomPad = 2;
        static constexpr int kSideMargin = 14;

        // Card geometry (section 4.4), design units.
        static constexpr int kCardGutter = 8;
        static constexpr int kCardBorderStroke = 1;
        static constexpr int kCardCornerRadius = 6;
        static constexpr int kCardHPad = 12;
        static constexpr int kCardBottomPad = 12;
        static constexpr int kSubTabStripH = 26;
        static constexpr int kSubTabGap = 8;

        // Knob grid (section 4.4), design units.
        static constexpr int kKnobGutter = 10;
        static constexpr int kKnobLabelGap = 5;
        static constexpr int kLabelBandH = 13;
        static constexpr int kLabelReadoutGap = 5;
        static constexpr int kReadoutBandH = 20;
        static constexpr int kInterRowGap = 12;
        static constexpr int kKnobMin = 24;
        static constexpr int kKnobMax = 58;
        static constexpr int kHeroKnobMax = 72;

        // The scale factor, clamped to [kScaleMin, kScaleMax].
        float s = 1.0f;

        // Build metrics from the editor width.
        static Metrics fromWidth(const int editorWidth) noexcept
        {
            Metrics m;
            m.s = std::clamp(static_cast<float>(editorWidth) / static_cast<float>(kDesignWidth), kScaleMin, kScaleMax);
            return m;
        }

        // Round a design value to a pixel count.
        [[nodiscard]] int px(const float v) const noexcept { return roundToInt(v * s); }

        // Keep a design value as a float pixel value.
        [[nodiscard]] float pxf(const float v) const noexcept { return v * s; }

        // Scale a font height. Never round.
        [[nodiscard]] float font(const float v) const noexcept { return v * s; }

    // Scale a stroke width. Hold a hairline above zero.
    float stroke(float v) const noexcept { return std::max(1.0f, v * s); }
};

// The current editor metrics. The editor sets this once per resized().
// Components read it to scale dimensions and fonts.
inline Metrics& metricsMutable() noexcept
{
    static Metrics m{};
    return m;
}

inline const Metrics& currentMetrics() noexcept
{
    return metricsMutable();
}

inline void setCurrentMetrics(Metrics m) noexcept { metricsMutable() = m; }

} // namespace MarsDSP::GUI
#endif

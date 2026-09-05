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

        // Scale clamp range (section 4.1). The minimum derives from the
        // font floor: the smallest declared font lands on the floor at
        // the smallest scale.
        static constexpr float kScaleMin = 0.80f;
        static constexpr float kScaleMax = 1.60f;

        // Window envelope (section 4.2), design units.
        static constexpr int kDefaultWidth = 760;
        static constexpr int kDefaultHeight = 486;
        static constexpr int kMinWidth = 800;
        static constexpr int kMaxWidth = 1600;
        static constexpr int kMinHeight = 512;
        static constexpr int kMaxHeight = 1024;

        // Band geometry (section 4.3), design units.
        static constexpr int kTopPad = 2;
        static constexpr int kHeaderH = 52;
        static constexpr int kGapHeader = 4;
        static constexpr int kTapH = 250;
        static constexpr int kGapTap = 8;
        static constexpr int kCardRowH = 284;
        static constexpr int kGapCards = 4;
        static constexpr int kFooterH = 34;
        static constexpr int kBottomPad = 2;
        static constexpr int kSideMargin = 14;

        // Preset bar geometry (section 4.7), design units.
        static constexpr int kPresetBarW = 320;
        static constexpr int kPresetBarH = 26;
        static constexpr int kPresetBarArrow = 20;
        static constexpr int kPresetBarMenu = 22;
        static constexpr float kPresetBarFont = 12.0f;

        // The header bypass button (section 4.6), design units.
        static constexpr float kHeaderBypassSize = 22.0f;

        // Font heights (section 4.7), design units. Every one stays at
        // or above kFontMinDU, so the floor is a backstop and not a
        // layout hazard.
        static constexpr float kWordmarkFont    = 15.0f;
        static constexpr float kKnobLabelFont   = 11.0f;
        static constexpr float kFooterFont      = 10.0f;
        static constexpr float kTapLabelFont    = 10.0f;
        static constexpr float kTapReadoutFont  = 10.0f;
        static constexpr float kLabelFont       = 13.0f;
        static constexpr float kComboFont       = 12.0f;
        static constexpr float kMenuFont        = 14.0f;
        static constexpr float kTooltipFont     = 11.0f;
        static constexpr float kSegmentFont     = 10.0f;
        static constexpr float kSubTabFont      = 10.0f;
        static constexpr float kReadoutFont     = 13.0f;

        // Element dimensions (section 4.7), design units.
        static constexpr float kHeaderSideMargin   = 14.0f;
        static constexpr float kWordmarkGap        = 8.0f;
        static constexpr float kKnobLabelInset     = 2.0f;
        static constexpr float kFooterSideMargin   = 12.0f;
        static constexpr float kTapLabelGapMin     = 8.0f;
        static constexpr float kMenuHighlightCorner = 3.0f;
        static constexpr float kSubTabButtonW      = 90.0f;
        static constexpr float kSubTabButtonGap    = 6.0f;
        static constexpr float kSubTabButtonH      = 24.0f;
        static constexpr float kToggleSize         = 24.0f;
        static constexpr float kToggleGap          = 4.0f;
        static constexpr float kSelectorNudge      = 1.0f;
        static constexpr float kEnableGap          = 8.0f;

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

        static constexpr float kCornerSmall        = 4.0f;
        static constexpr float kCornerDisplay      = 5.0f;
        static constexpr float kHairline           = 1.0f;
        static constexpr float kGroupStroke        = 2.0f;
        static constexpr float kSegmentGap         = 4.0f;
        static constexpr float kPlotPad            = 10.0f;
        static constexpr float kTapBarStroke       = 2.5f;
        static constexpr float kTapHeadRadius      = 2.5f;
        static constexpr float kTapHeadGrow        = 2.0f;
        static constexpr float kLaneHeadroom       = 8.0f;
        static constexpr float kSnapRadius         = 6.0f;
        static constexpr float kHoverReadoutW      = 160.0f;
        static constexpr float kHoverReadoutH      = 14.0f;
        static constexpr float kComboArrowInset    = 20.0f;
        static constexpr float kMenuItemInset      = 12.0f;
        static constexpr float kMenuArrowBox       = 16.0f;
        static constexpr float kMenuSeparatorInset = 14.0f;
        static constexpr float kIconStroke         = 1.5f;
        static constexpr float kLockStroke         = 1.8f;

        // Ruler and selector geometry, design units at s = 1.
        static constexpr float kRulerLaneHeight    = 28.0f;
        static constexpr float kRulerLabelGap      = 7.0f;
        static constexpr float kMajorTick          = 6.0f;
        static constexpr float kMinorTick          = 4.0f;
        static constexpr float kRulerFont           = 10.0f;
        static constexpr float kSelectorRowH       = 24.0f;
        static constexpr float kEnableRowH         = 22.0f;

        // The legibility floor (section 4.7). No glyph renders below the
        // floor at any scale, and no declared font drops below the
        // minimum design height.
        static constexpr float kFontFloorPx = 8.0f;
        static constexpr float kFontMinDU   = 10.0f;

        // Interaction constants (section 4.6).
        // The wheel and the arrow keys step in proportion space.
        static constexpr double kWheelStepCoarse = 0.02;
        static constexpr double kWheelStepFine = 0.004;
        // The idle time that closes a wheel gesture burst.
        static constexpr int kWheelGestureMs = 250;
        // The smallest interactive dimension, design units.
        static constexpr int kHitTargetMin = 16;

        // Every interactive dimension clears the hit target.
        static_assert(kPresetBarArrow >= kHitTargetMin);
        static_assert(kPresetBarMenu >= kHitTargetMin);
        static_assert(kHeaderBypassSize >= static_cast<float>(kHitTargetMin));
        static_assert(kKnobMin >= kHitTargetMin);
        static_assert(kSelectorRowH >= static_cast<float>(kHitTargetMin));
        static_assert(kEnableRowH >= static_cast<float>(kHitTargetMin));
        static_assert(kReadoutBandH >= kHitTargetMin);

        // The three legibility constants cannot drift apart.
        static_assert(kFontMinDU * kScaleMin >= kFontFloorPx);

        // No declared font drops below the minimum design height.
        static_assert(kWordmarkFont   >= kFontMinDU);
        static_assert(kKnobLabelFont  >= kFontMinDU);
        static_assert(kFooterFont     >= kFontMinDU);
        static_assert(kTapLabelFont   >= kFontMinDU);
        static_assert(kTapReadoutFont >= kFontMinDU);
        static_assert(kLabelFont      >= kFontMinDU);
        static_assert(kComboFont      >= kFontMinDU);
        static_assert(kMenuFont       >= kFontMinDU);
        static_assert(kTooltipFont    >= kFontMinDU);
        static_assert(kSegmentFont    >= kFontMinDU);
        static_assert(kSubTabFont     >= kFontMinDU);
        static_assert(kReadoutFont    >= kFontMinDU);
        static_assert(kRulerFont      >= kFontMinDU);
        static_assert(kPresetBarFont  >= kFontMinDU);

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

        // Scale a font height. Never round. Hold every glyph at or above
        // the floor, so no text renders illegible at any scale.
        [[nodiscard]] float font(const float v) const noexcept { return std::max(kFontFloorPx, v * s); }

    // Scale a stroke width. Hold a hairline above zero.
    float stroke(float v) const noexcept { return std::max(1.0f, v * s); }
};

} // namespace MarsDSP::GUI
#endif

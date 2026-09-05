#pragma once

#ifndef CHRONOS_METRICS_H
#define CHRONOS_METRICS_H

#include <JuceHeader.h>
#include <algorithm>

namespace MarsDSP::GUI
{
    // Scale metrics for the editor.
    // One scale factor s derives every dimension from the 520 x 932 design box.
    // Components read dimensions through px, pxf, font, displayFont, and stroke.
    struct Metrics
    {
        // Design box (section 4.1). The editor is 520 design units wide.
        static constexpr int kDesignWidth = 520;
        static constexpr int kDesignHeight = 932;
        static constexpr double kDesignAspect = 520.0 / 932.0;

        // Scale clamp range (section 4.1). The minimum derives from the
        // font floor: the smallest declared font lands on the floor at
        // the smallest scale.
        static constexpr float kScaleMin = 0.80f;
        static constexpr float kScaleMax = 1.60f;

        // Window envelope (section 4.1), design units. The heights derive
        // from the widths through the aspect, with no literal.
        static constexpr int kMinWidth = 416;
        static constexpr int kDefaultWidth = 440;
        static constexpr int kMaxWidth = 832;
        static constexpr int kMinHeight = (kMinWidth * kDesignHeight + kDesignWidth / 2) / kDesignWidth;
        static constexpr int kDefaultHeight = (kDefaultWidth * kDesignHeight + kDesignWidth / 2) / kDesignWidth;
        static constexpr int kMaxHeight = (kMaxWidth * kDesignHeight + kDesignWidth / 2) / kDesignWidth;

        // Band geometry (section 4.1), design units. The nine bands sum to 932.
        static constexpr int kTopPad = 2;
        static constexpr int kHeaderH = 40;
        static constexpr int kGapHeader = 4;
        static constexpr int kTapH = 170;
        static constexpr int kGapTap = 8;
        static constexpr int kCardAreaH = 676;
        static constexpr int kGapCards = 4;
        static constexpr int kFooterH = 26;
        static constexpr int kBottomPad = 2;
        static constexpr int kSideMargin = 12;

        // The two grid rows (section 4.1), design units.
        static constexpr int kRow1H = 270;
        static constexpr int kRow2H = 398;
        static constexpr int kCardGutter = 8;

        // Preset bar geometry (section 4.1), design units.
        static constexpr int kPresetBarW = 240;
        static constexpr int kPresetBarH = 24;
        static constexpr int kPresetBarArrow = 20;
        static constexpr int kPresetBarMenu = 22;

        // The header cluster (section 4.1), design units.
        static constexpr float kHeaderSideMargin   = 12.0f;
        static constexpr float kHeaderBypassSize   = 22.0f;
        static constexpr float kHeaderClusterGap   = 6.0f;
        static constexpr float kHistoryButtonSize  = 22.0f;
        static constexpr float kHistoryButtonGap   = 4.0f;

        // Wordmark geometry (section 4.1), design units.
        static constexpr int kWordmarkReserve = 70;
        static constexpr float kWordmarkGap   = 8.0f;

        // The header reserves do not overlap the centred preset bar.
        // The left reserve and the right cluster each fit in the half
        // of the width the preset bar leaves clear.
        static constexpr int kHeaderHalfClear = (kDesignWidth - kPresetBarW) / 2;
        static_assert(static_cast<int>(kHeaderSideMargin) + kWordmarkReserve
                          + static_cast<int>(kWordmarkGap)
                      <= kHeaderHalfClear);
        static_assert(static_cast<int>(kHeaderSideMargin) + static_cast<int>(kHeaderBypassSize)
                          + static_cast<int>(kHeaderClusterGap) + static_cast<int>(kHistoryButtonSize)
                          + static_cast<int>(kHistoryButtonGap) + static_cast<int>(kHistoryButtonSize)
                      <= kHeaderHalfClear);

        // Font heights (section 4.1), design units. Every one stays at
        // or above kFontMinDU, so the floor is a backstop. The display
        // constants render in the display face and stay at or above
        // kDisplayFontMinDU.
        static constexpr float kWordmarkFont    = 15.0f;
        static constexpr float kKnobLabelFont   = 11.0f;
        static constexpr float kCardTitleFont   = 11.0f;
        static constexpr float kFooterFont      = 10.0f;
        static constexpr float kTapLabelFont    = 10.0f;
        static constexpr float kTapReadoutFont  = 13.0f;
        static constexpr float kLabelFont       = 13.0f;
        static constexpr float kComboFont       = 12.0f;
        static constexpr float kMenuFont        = 14.0f;
        static constexpr float kTooltipFont     = 11.0f;
        static constexpr float kSegmentFont     = 10.0f;
        static constexpr float kReadoutFont     = 14.0f;
        static constexpr float kPadReadoutFont  = 13.0f;
        static constexpr float kPresetBarFont   = 12.0f;

        // The card title letter tracking (section 4.2).
        static constexpr float kTitleTracking   = 0.08f;

        // Element dimensions (section 4.1), design units.
        static constexpr float kKnobLabelInset     = 2.0f;
        static constexpr float kFooterSideMargin   = 12.0f;
        static constexpr float kTapLabelGapMin     = 8.0f;
        static constexpr float kMenuHighlightCorner = 3.0f;
        static constexpr float kSelectorNudge      = 1.0f;

        // Card frame (section 4.1), design units. The frame overhead is
        // 2 * border + title + title gap + bottom pad = 40.
        static constexpr int kCardBorderStroke  = 1;
        static constexpr int kCardCornerRadius  = 6;
        static constexpr int kCardHPad           = 10;
        static constexpr int kCardTitleH         = 22;
        static constexpr int kCardTitleGap       = 6;
        static constexpr int kCardBottomPad      = 10;

        // Row primitives (section 4.1), design units.
        static constexpr int kKnobMax       = 58;
        static constexpr int kKnobMin       = 24;
        static constexpr int kLabelBandH    = 13;
        static constexpr int kKnobLabelGap  = 5;
        static constexpr int kKnobRowH      = 76;
        static constexpr int kKnobGutter    = 10;
        static constexpr int kInterRowGap   = 10;
        static constexpr int kSelectorRowH  = 24;
        static constexpr int kReadoutBandH  = 20;
        static constexpr int kLabelReadoutGap = 5;
        static constexpr int kEnableRowH    = 24;
        static constexpr int kEnableGap     = 8;
        static constexpr int kToggleSize    = 24;
        static constexpr int kToggleGap     = 4;
        static constexpr int kPadH          = 240;
        static constexpr int kPadInset      = 8;
        static constexpr float kPadHandleR  = 6.0f;
        static constexpr float kDragDeadZone = 4.0f;

        // The value arc geometry (section 4.4), design units.
        static constexpr float kKnobArcStroke = 2.5f;
        static constexpr float kKnobArcGap    = 2.0f;

        // The tap drag sensitivity (section 4.6), design units.
        static constexpr float kDragTimeGain       = 1.2f;
        static constexpr float kDragPixelsPerDivision = 25.0f;

        // The modulation wobble display offset (section 4.7), design units.
        // The pixel scale is perceptual, not physical.
        static constexpr float kModJitterDU    = 4.0f;
        static constexpr float kModJitterMaxDU = 16.0f;

        // The most slack a shorter card may leave in a grid row (section 4.1).
        static constexpr int kRowSlackMaxDU = 12;

        // Declared card heights (section 4.1). Each equals its breakdown sum.
        static constexpr int kFrameOverhead  = 2 * kCardBorderStroke + kCardTitleH + kCardTitleGap + kCardBottomPad;
        static constexpr int kTimeCardH      = kFrameOverhead
                                               + kKnobRowH + kLabelReadoutGap + kReadoutBandH
                                               + kInterRowGap + kSelectorRowH + kInterRowGap + kKnobRowH;
        static constexpr int kRepeatsCardH   = kFrameOverhead
                                               + kSelectorRowH + kInterRowGap + kKnobRowH
                                               + kInterRowGap + kKnobRowH + kInterRowGap + kSelectorRowH;
        static constexpr int kDriveCardH     = kFrameOverhead + kKnobRowH;
        static constexpr int kFilterCardH    = kFrameOverhead + kSelectorRowH + kInterRowGap + kKnobRowH;
        static constexpr int kLevelCardH     = kFrameOverhead + kKnobRowH;
        static constexpr int kDiffuserCardH  = kFrameOverhead
                                               + kEnableRowH + kEnableGap + kPadH
                                               + kInterRowGap + kKnobRowH;

        // The right column stacks to the diffuser card height.
        static_assert(kDriveCardH + kCardGutter + kFilterCardH + kCardGutter + kLevelCardH
                      == kDiffuserCardH);
        // Row 1 is the taller of the two cards in it.
        static_assert(kRow1H == std::max(kTimeCardH, kRepeatsCardH));
        // Row 2 is the diffuser card.
        static_assert(kRow2H == kDiffuserCardH);
        // The card area is the two rows plus the gutter between them.
        static_assert(kRow1H + kCardGutter + kRow2H == kCardAreaH);
        // The only slack in row 1 sits under the time card.
        static_assert(kRow1H - kTimeCardH <= kRowSlackMaxDU);

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
        static constexpr float kRulerFont           = 13.0f;

        // Tap display label and readout offsets (section 4.3, 4.6), design units.
        static constexpr float kTapLabelInset      = 4.0f;
        static constexpr float kRulerLabelInset    = 2.0f;
        static constexpr float kHoverReadoutGap    = 6.0f;
        static constexpr float kHoverReadoutTop    = 2.0f;

        // The legibility floor (section 4.1). No glyph renders below the
        // floor at any scale, and no declared font drops below the
        // minimum design height.
        static constexpr float kFontFloorPx = 8.0f;
        static constexpr float kFontMinDU   = 10.0f;

        // The display-face legibility floor (section 4.3). No display
        // glyph renders below this floor at any scale, and no declared
        // display font drops below the minimum display design height.
        static constexpr float kDisplayFontFloorPx = 10.0f;
        static constexpr float kDisplayFontMinDU   = 13.0f;

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
        static_assert(kHistoryButtonSize >= static_cast<float>(kHitTargetMin));
        static_assert(kKnobMin >= kHitTargetMin);
        static_assert(kSelectorRowH >= kHitTargetMin);
        static_assert(kEnableRowH >= kHitTargetMin);
        static_assert(kReadoutBandH >= kHitTargetMin);
        static_assert(kToggleSize >= kHitTargetMin);

        // The three legibility constants cannot drift apart.
        static_assert(kFontMinDU * kScaleMin >= kFontFloorPx);

        // The display legibility constants cannot drift apart.
        static_assert(kDisplayFontMinDU * kScaleMin >= kDisplayFontFloorPx);

        // No declared font drops below the minimum design height.
        static_assert(kWordmarkFont   >= kFontMinDU);
        static_assert(kKnobLabelFont  >= kFontMinDU);
        static_assert(kCardTitleFont  >= kFontMinDU);
        static_assert(kFooterFont     >= kFontMinDU);
        static_assert(kTapLabelFont   >= kFontMinDU);
        static_assert(kTapReadoutFont >= kFontMinDU);
        static_assert(kLabelFont      >= kFontMinDU);
        static_assert(kComboFont      >= kFontMinDU);
        static_assert(kMenuFont       >= kFontMinDU);
        static_assert(kTooltipFont    >= kFontMinDU);
        static_assert(kSegmentFont    >= kFontMinDU);
        static_assert(kReadoutFont    >= kFontMinDU);
        static_assert(kRulerFont      >= kFontMinDU);
        static_assert(kPresetBarFont  >= kFontMinDU);

        // The four display constants stay at or above the display minimum.
        static_assert(kReadoutFont    >= kDisplayFontMinDU);
        static_assert(kRulerFont      >= kDisplayFontMinDU);
        static_assert(kTapReadoutFont >= kDisplayFontMinDU);
        static_assert(kPadReadoutFont >= kDisplayFontMinDU);

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

        // Scale a display font height. Hold every display glyph at or
        // above the display floor, so no readout renders illegible at any scale.
        [[nodiscard]] float displayFont(const float v) const noexcept { return std::max(kDisplayFontFloorPx, v * s); }

    // Scale a stroke width. Hold a hairline above zero.
    float stroke(float v) const noexcept { return std::max(1.0f, v * s); }
};

} // namespace MarsDSP::GUI
#endif

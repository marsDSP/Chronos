/**
 * Metrics, tint, tick, and band harness for rev G7 (spec appendix B).
 * Host-free: links Colours.h, Metrics.h, and the tick generator only.
 * It does not open an editor.
 */

#include "gui/Colours.h"
#include "gui/Metrics.h"
#include "gui/tap/TickGenerator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <print>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Linearise one sRGB channel. The WCAG 2.x transfer function.
float lineariseChannel(const float c)
{
    return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// The WCAG relative luminance of an sRGB colour.
float luminance(juce::Colour c)
{
    return 0.2126f * lineariseChannel(c.getFloatRed())
         + 0.7152f * lineariseChannel(c.getFloatGreen())
         + 0.0722f * lineariseChannel(c.getFloatBlue());
}

// The WCAG contrast ratio between two colours.
float contrast(juce::Colour a, juce::Colour b)
{
    const float la = luminance(a);
    const float lb = luminance(b);
    const float hi = std::max(la, lb);
    const float lo = std::min(la, lb);
    return (hi + 0.05f) / (lo + 0.05f);
}

// The declared font constants, for the floor assertion.
struct FontConst { const char* name; float du; };
const FontConst kFontConsts[] = {
    { "wordmark",   MarsDSP::GUI::Metrics::kWordmarkFont },
    { "knobLabel",  MarsDSP::GUI::Metrics::kKnobLabelFont },
    { "cardTitle",  MarsDSP::GUI::Metrics::kCardTitleFont },
    { "footer",     MarsDSP::GUI::Metrics::kFooterFont },
    { "tapLabel",   MarsDSP::GUI::Metrics::kTapLabelFont },
    { "tapReadout", MarsDSP::GUI::Metrics::kTapReadoutFont },
    { "label",      MarsDSP::GUI::Metrics::kLabelFont },
    { "combo",      MarsDSP::GUI::Metrics::kComboFont },
    { "menu",       MarsDSP::GUI::Metrics::kMenuFont },
    { "tooltip",    MarsDSP::GUI::Metrics::kTooltipFont },
    { "segment",    MarsDSP::GUI::Metrics::kSegmentFont },
    { "readout",    MarsDSP::GUI::Metrics::kReadoutFont },
    { "padReadout", MarsDSP::GUI::Metrics::kPadReadoutFont },
    { "ruler",      MarsDSP::GUI::Metrics::kRulerFont },
    { "presetBar",  MarsDSP::GUI::Metrics::kPresetBarFont },
};
constexpr int kNumFontConsts = static_cast<int>(std::size(kFontConsts));

// The display font constants, for the display floor assertion.
const FontConst kDisplayFontConsts[] = {
    { "readout",    MarsDSP::GUI::Metrics::kReadoutFont },
    { "ruler",      MarsDSP::GUI::Metrics::kRulerFont },
    { "tapReadout", MarsDSP::GUI::Metrics::kTapReadoutFont },
    { "padReadout", MarsDSP::GUI::Metrics::kPadReadoutFont },
};
constexpr int kNumDisplayFontConsts = static_cast<int>(std::size(kDisplayFontConsts));

// The declared text and icon colour pairs, for the contrast assertion.
struct ColourPair { const char* fgName; juce::Colour fg; const char* bgName; juce::Colour bg; float minRatio; };
const ColourPair kTextPairs[] = {
    { "textPrimary", MarsDSP::GUI::Colours::textPrimary, "background",       MarsDSP::GUI::Colours::background,       4.5f },
    { "textBright",  MarsDSP::GUI::Colours::textBright,  "background",       MarsDSP::GUI::Colours::background,       4.5f },
    { "textMuted",   MarsDSP::GUI::Colours::textMuted,   "background",       MarsDSP::GUI::Colours::background,       4.5f },
    { "textMuted",   MarsDSP::GUI::Colours::textMuted,   "panelBackground",  MarsDSP::GUI::Colours::panelBackground,  4.5f },
    { "textMuted",   MarsDSP::GUI::Colours::textMuted,   "headerBackground", MarsDSP::GUI::Colours::headerBackground, 4.5f },
    { "textMuted",   MarsDSP::GUI::Colours::textMuted,   "footerBackground", MarsDSP::GUI::Colours::footerBackground, 4.5f },
    { "textBright",  MarsDSP::GUI::Colours::textBright,  "panelBackground",  MarsDSP::GUI::Colours::panelBackground,  4.5f },
    { "textPrimary", MarsDSP::GUI::Colours::textPrimary, "headerBackground", MarsDSP::GUI::Colours::headerBackground, 4.5f },
    { "textBright",  MarsDSP::GUI::Colours::textBright,  "headerBackground", MarsDSP::GUI::Colours::headerBackground, 4.5f },
    { "textPrimary", MarsDSP::GUI::Colours::textPrimary, "footerBackground", MarsDSP::GUI::Colours::footerBackground, 4.5f },
    { "rulerText",   MarsDSP::GUI::Colours::rulerText,   "background",       MarsDSP::GUI::Colours::background,       4.5f },
};
constexpr int kNumTextPairs = static_cast<int>(std::size(kTextPairs));

const ColourPair kIconPairs[] = {
    { "textMuted",   MarsDSP::GUI::Colours::textMuted,   "panelBackground",  MarsDSP::GUI::Colours::panelBackground,  3.0f },
    { "textMuted",   MarsDSP::GUI::Colours::textMuted,   "headerBackground", MarsDSP::GUI::Colours::headerBackground, 3.0f },
    { "textPrimary", MarsDSP::GUI::Colours::textPrimary, "background",       MarsDSP::GUI::Colours::background,       3.0f },
};
constexpr int kNumIconPairs = static_cast<int>(std::size(kIconPairs));

// The nine tint coefficients in spec order.
struct CoeffSet { const char* name; float coeff; };
const CoeffSet kCoeffSets[] = {
    { "plotFill",      MarsDSP::GUI::kTintPlotFill },
    { "plotFillLit",   MarsDSP::GUI::kTintPlotFillLit },
    { "gridMinor",     MarsDSP::GUI::kTintGridMinor },
    { "gridMajor",     MarsDSP::GUI::kTintGridMajor },
    { "displayBorder",  MarsDSP::GUI::kTintDisplayBorder },
    { "readoutFill",   MarsDSP::GUI::kTintReadoutFill },
    { "readoutBorder",  MarsDSP::GUI::kTintReadoutBorder },
    { "cardBorder",    MarsDSP::GUI::kTintCardBorder },
    { "centreLine",    MarsDSP::GUI::kTintCentreLine },
};
constexpr int kNumCoeffs = static_cast<int>(std::size(kCoeffSets));

// Knob diameter derivation (mirrors ChronosEditor.cpp section 4.4).
float knobDiameterPx(const MarsDSP::GUI::Metrics& m,
                     const float contentW, const float rowH,
                     const int n, const bool hasReadout,
                     const float dMaxDU)
{
    const float g = m.pxf(static_cast<float>(MarsDSP::GUI::Metrics::kKnobGutter));
    const float cellW = (contentW - static_cast<float>(n - 1) * g) / static_cast<float>(n);
    const float cellH = rowH
        - static_cast<float>(m.px(static_cast<float>(MarsDSP::GUI::Metrics::kLabelBandH)))
        - static_cast<float>(m.px(static_cast<float>(MarsDSP::GUI::Metrics::kKnobLabelGap)))
        - (hasReadout
               ? static_cast<float>(m.px(static_cast<float>(MarsDSP::GUI::Metrics::kReadoutBandH)))
                 + static_cast<float>(m.px(static_cast<float>(MarsDSP::GUI::Metrics::kLabelReadoutGap)))
               : 0.0f);
    return std::clamp(std::min(cellW, cellH),
                      m.pxf(static_cast<float>(MarsDSP::GUI::Metrics::kKnobMin)),
                      m.pxf(dMaxDU));
}

// The nine band heights of section 4.1, in order.
struct BandSet { const char* name; int du; };
const BandSet kBands[] = {
    { "topPad",    MarsDSP::GUI::Metrics::kTopPad },
    { "header",    MarsDSP::GUI::Metrics::kHeaderH },
    { "gapHeader", MarsDSP::GUI::Metrics::kGapHeader },
    { "tap",       MarsDSP::GUI::Metrics::kTapH },
    { "gapTap",    MarsDSP::GUI::Metrics::kGapTap },
    { "cardArea",  MarsDSP::GUI::Metrics::kCardAreaH },
    { "gapCards",  MarsDSP::GUI::Metrics::kGapCards },
    { "footer",    MarsDSP::GUI::Metrics::kFooterH },
    { "bottomPad", MarsDSP::GUI::Metrics::kBottomPad },
};
constexpr int kNumBands = static_cast<int>(std::size(kBands));

// The six declared card heights (section 4.1).
const BandSet kCardHeights[] = {
    { "time",     MarsDSP::GUI::Metrics::kTimeCardH },
    { "repeats",  MarsDSP::GUI::Metrics::kRepeatsCardH },
    { "drive",    MarsDSP::GUI::Metrics::kDriveCardH },
    { "filter",   MarsDSP::GUI::Metrics::kFilterCardH },
    { "level",    MarsDSP::GUI::Metrics::kLevelCardH },
    { "diffuser", MarsDSP::GUI::Metrics::kDiffuserCardH },
};
constexpr int kNumCardHeights = static_cast<int>(std::size(kCardHeights));

// Design constants swept in the px non-decreasing check.
const float kDesignConsts[] = {
    static_cast<float>(MarsDSP::GUI::Metrics::kTopPad),
    static_cast<float>(MarsDSP::GUI::Metrics::kHeaderH),
    static_cast<float>(MarsDSP::GUI::Metrics::kGapHeader),
    static_cast<float>(MarsDSP::GUI::Metrics::kTapH),
    static_cast<float>(MarsDSP::GUI::Metrics::kGapTap),
    static_cast<float>(MarsDSP::GUI::Metrics::kCardAreaH),
    static_cast<float>(MarsDSP::GUI::Metrics::kGapCards),
    static_cast<float>(MarsDSP::GUI::Metrics::kFooterH),
    static_cast<float>(MarsDSP::GUI::Metrics::kBottomPad),
    static_cast<float>(MarsDSP::GUI::Metrics::kSideMargin),
    static_cast<float>(MarsDSP::GUI::Metrics::kCardGutter),
    static_cast<float>(MarsDSP::GUI::Metrics::kKnobGutter),
    static_cast<float>(MarsDSP::GUI::Metrics::kKnobMin),
    static_cast<float>(MarsDSP::GUI::Metrics::kKnobMax),
    static_cast<float>(MarsDSP::GUI::Metrics::kPresetBarW),
    static_cast<float>(MarsDSP::GUI::Metrics::kPresetBarH),
    static_cast<float>(MarsDSP::GUI::Metrics::kPresetBarArrow),
    static_cast<float>(MarsDSP::GUI::Metrics::kPresetBarMenu),
    static_cast<float>(MarsDSP::GUI::Metrics::kPresetBarFont),
};
constexpr int kNumDesignConsts = static_cast<int>(std::size(kDesignConsts));

int runAll()
{
    // ----------------------------------------------------------------
    // 1. Tint anchor: accent = 0xFFFF8557, two regression anchors.
    // ----------------------------------------------------------------
    g_section = "tint_anchor";
    {
        const juce::Colour accent { 0xFFFF8557 };
        const juce::Colour expected052 { 0xFF1A1410 };
        const juce::Colour expected093 { 0xFF231913 };

        const auto r052 = MarsDSP::GUI::tintInk(accent, MarsDSP::GUI::kTintPlotFill);
        const auto r093 = MarsDSP::GUI::tintInk(accent, MarsDSP::GUI::kTintPlotFillLit);

        const auto check1 = [](const juce::Colour& got, const juce::Colour& exp) {
            for (int ch = 0; ch < 3; ++ch)
            {
                const int g = (ch == 0) ? got.getRed()   : (ch == 1) ? got.getGreen() : got.getBlue();
                const int e = (ch == 0) ? exp.getRed()   : (ch == 1) ? exp.getGreen() : exp.getBlue();
                if (std::abs(g - e) > 1)
                    FAIL("tint anchor channel {} differs by {} (got {} exp {})", ch, g - e, g, e);
            }
        };
        check1(r052, expected052);
        check1(r093, expected093);
        std::println("tint anchor (0.052, 0.093): PASS");
    }

    // ----------------------------------------------------------------
    // 2. Tint totality: 64 accents, gamut + luminance order.
    // ----------------------------------------------------------------
    g_section = "tint_totality";
    {
        for (int h = 0; h < 64; ++h)
        {
            const float hue = static_cast<float>(h) * (360.0f / 64.0f);
            const juce::Colour accent = juce::Colour::fromHSV(hue, 1.0f, 1.0f, 1.0f);

            float prevLum = -1.0f;
            for (int i = 0; i < kNumCoeffs; ++i)
            {
                const auto& cs = kCoeffSets[i];
                const juce::Colour c =
                    (i < 5)
                        ? MarsDSP::GUI::tintInk(accent, cs.coeff)
                        : (std::strcmp(cs.name, "cardBorder") == 0)
                            ? MarsDSP::GUI::tint(MarsDSP::GUI::Colours::panelBackground, accent, cs.coeff)
                            : MarsDSP::GUI::tint(MarsDSP::GUI::tintInk(accent, MarsDSP::GUI::kTintPlotFillLit),
                                                   accent, cs.coeff);

                for (int ch = 0; ch < 3; ++ch)
                {
                    const int v = (ch == 0) ? c.getRed() : (ch == 1) ? c.getGreen() : c.getBlue();
                    if (v < 0 || v > 255)
                        FAIL("accent {} coeff {} channel {} = {} out of gamut", h, cs.name, ch, v);
                }

                if (i < 5)
                {
                    const float lum = luminance(c);
                    if (lum <= prevLum)
                        FAIL("accent {} luminance not increasing at {} ({} <= {})",
                             h, cs.name, lum, prevLum);
                    prevLum = lum;
                }
            }
        }
        std::println("tint totality (64 accents, gamut + order): PASS");
    }

    // ----------------------------------------------------------------
    // 3. Tick non-emptiness: T log sweep, W set, >= 3 majors, every major in [0, T].
    // ----------------------------------------------------------------
    g_section = "tick_non_emptiness";
    {
        const float Ws[] = { 200.0f, 340.0f, 496.0f, 640.0f, 1024.0f };
        constexpr int kNumW = static_cast<int>(std::size(Ws));
        bool anyFail = false;
        for (int wi = 0; wi < kNumW; ++wi)
        {
            const float W = Ws[wi];
            for (int step = 0; step < 2000; ++step)
            {
                const float T = 0.001f * std::pow(10.0f, 3.0f * static_cast<float>(step) / 1999.0f);
                const auto t = MarsDSP::GUI::computeFreeTicks(T, W, 1.0f);
                if (t.majors.size() < 3)
                {
                    FAIL("W={} T={}: only {} majors (need >= 3)", W, T, t.majors.size());
                    anyFail = true;
                }
                for (const float m : t.majors)
                    if (m < -1e-9f || m > T + 1e-4f)
                    {
                        FAIL("W={} T={}: major {} out of [0, T] (got {})", W, T, m, m);
                        anyFail = true;
                    }
            }
        }
        if (! anyFail)
            std::println("tick non-emptiness (2000 x {} W): PASS", kNumW);
    }

    // ----------------------------------------------------------------
    // 4. Tick cleanliness: major step in {1,2,5} x 10^n, minor divides major.
    // ----------------------------------------------------------------
    g_section = "tick_cleanliness";
    {
        const float W = 496.0f;
        for (int step = 0; step < 2000; ++step)
        {
            const float T = 0.001f * std::pow(10.0f, 3.0f * static_cast<float>(step) / 1999.0f);
            const auto t = MarsDSP::GUI::computeFreeTicks(T, W, 1.0f);
            if (t.majorStep <= 0.0f)
                FAIL("T={}: majorStep {} non-positive", T, t.majorStep);

            const float mant = t.majorStep;
            const float decade = std::pow(10.0f, std::floor(std::log10(mant)));
            const float norm = mant / decade;
            if (! (std::fabs(norm - 1.0f) < 1e-6f
                  || std::fabs(norm - 2.0f) < 1e-6f
                  || std::fabs(norm - 5.0f) < 1e-6f))
                FAIL("T={}: majorStep {} not in {{1,2,5}} x 10^n (norm {})", T, t.majorStep, norm);

            if (t.minors.size() >= 2)
            {
                const float minorStep = t.minors[1] - t.minors[0];
                const float ratio = t.majorStep / minorStep;
                const int r = static_cast<int>(std::round(ratio));
                if (std::fabs(ratio - static_cast<float>(r)) > 1e-3f)
                    FAIL("T={}: minor step {} does not divide major {} (ratio {})",
                         T, minorStep, t.majorStep, ratio);
            }
        }
        std::println("tick cleanliness (2000 steps): PASS");
    }

    // ----------------------------------------------------------------
    // 5. Scale monotonicity: s sweep, px non-decreasing, knob bounds.
    // ----------------------------------------------------------------
    g_section = "scale_monotonicity";
    {
        bool anyFail = false;
        int prevPx[kNumDesignConsts] {};
        for (int si = 0; si <= 96; ++si)
        {
            const float s = 0.64f + static_cast<float>(si) * 0.01f;
            const MarsDSP::GUI::Metrics m =
                MarsDSP::GUI::Metrics::fromWidth(
                    static_cast<int>(static_cast<float>(MarsDSP::GUI::Metrics::kDesignWidth) * s));

            for (int ci = 0; ci < kNumDesignConsts; ++ci)
            {
                const int v = m.px(kDesignConsts[ci]);
                if (si > 0 && v < prevPx[ci])
                {
                    FAIL("s={:.2f} px({}) decreased from {} to {}", s, kDesignConsts[ci], prevPx[ci], v);
                    anyFail = true;
                }
                prevPx[ci] = v;
            }

            // Knob derivation bounds: standard in [24s, 58s].
            const float fs = m.s;
            const float contentW = 200.0f * fs;
            const float rowH = 80.0f * fs;
            const float dStd = knobDiameterPx(m, contentW, rowH, 2, false,
                                            static_cast<float>(MarsDSP::GUI::Metrics::kKnobMax));
            if (dStd < 24.0f * fs - 1e-3f || dStd > 58.0f * fs + 1e-3f)
            {
                FAIL("s={:.2f} standard knob {} out of [24s, 58s]=[{}, {}]", s, dStd,
                     24.0f * fs, 58.0f * fs);
                anyFail = true;
            }
        }
        if (! anyFail)
            std::println("scale monotonicity (97 steps, px + knob bounds): PASS");
    }

    // ----------------------------------------------------------------
    // 6. Band closure: 9 bands sum to px(932), card sums, row/col, slack.
    // ----------------------------------------------------------------
    g_section = "band_closure";
    {
        bool anyFail = false;

        // The design-unit band heights must sum to 932 exactly.
        int sumDU = 0;
        for (int bi = 0; bi < kNumBands; ++bi)
            sumDU += kBands[bi].du;
        if (sumDU != MarsDSP::GUI::Metrics::kDesignHeight)
        {
            FAIL("band design-unit sum {} != 932 (regression)", sumDU);
            anyFail = true;
        }

        for (int si = 0; si <= 96; ++si)
        {
            const float s = 0.64f + static_cast<float>(si) * 0.01f;
            const MarsDSP::GUI::Metrics m =
                MarsDSP::GUI::Metrics::fromWidth(
                    static_cast<int>(static_cast<float>(MarsDSP::GUI::Metrics::kDesignWidth) * s));

            // The px sum must match px(932) within the px rounding tolerance.
            int sum = 0;
            for (int bi = 0; bi < kNumBands; ++bi)
                sum += m.px(kBands[bi].du);
            const int target = m.px(static_cast<float>(MarsDSP::GUI::Metrics::kDesignHeight));
            if (std::abs(sum - target) > 2)
            {
                FAIL("s={:.2f} band px sum {} != px(932)={} (diff {})", s, sum, target, sum - target);
                anyFail = true;
            }
        }

        // The six card heights.
        for (int ci = 0; ci < kNumCardHeights; ++ci)
            CHECK(kCardHeights[ci].du > 0);

        // Row 1 is the taller of the two cards in it.
        CHECK(MarsDSP::GUI::Metrics::kRow1H
              == std::max(MarsDSP::GUI::Metrics::kTimeCardH, MarsDSP::GUI::Metrics::kRepeatsCardH));
        // Row 2 is the diffuser card.
        CHECK(MarsDSP::GUI::Metrics::kRow2H == MarsDSP::GUI::Metrics::kDiffuserCardH);
        // The card area is the two rows plus the gutter.
        CHECK(MarsDSP::GUI::Metrics::kRow1H + MarsDSP::GUI::Metrics::kCardGutter
                  + MarsDSP::GUI::Metrics::kRow2H
              == MarsDSP::GUI::Metrics::kCardAreaH);
        // The right column stacks to the diffuser card.
        CHECK(MarsDSP::GUI::Metrics::kDriveCardH + MarsDSP::GUI::Metrics::kCardGutter
                  + MarsDSP::GUI::Metrics::kFilterCardH + MarsDSP::GUI::Metrics::kCardGutter
                  + MarsDSP::GUI::Metrics::kLevelCardH
              == MarsDSP::GUI::Metrics::kDiffuserCardH);
        // The slack in row 1 sits under the time card.
        CHECK(MarsDSP::GUI::Metrics::kRow1H - MarsDSP::GUI::Metrics::kTimeCardH
              <= MarsDSP::GUI::Metrics::kRowSlackMaxDU);

        // The header reserves fit in the half the bar leaves clear.
        CHECK(static_cast<int>(MarsDSP::GUI::Metrics::kHeaderSideMargin)
                  + MarsDSP::GUI::Metrics::kWordmarkReserve
                  + static_cast<int>(MarsDSP::GUI::Metrics::kWordmarkGap)
              <= MarsDSP::GUI::Metrics::kHeaderHalfClear);
        CHECK(static_cast<int>(MarsDSP::GUI::Metrics::kHeaderSideMargin)
                  + static_cast<int>(MarsDSP::GUI::Metrics::kHeaderBypassSize)
                  + static_cast<int>(MarsDSP::GUI::Metrics::kHeaderClusterGap)
                  + static_cast<int>(MarsDSP::GUI::Metrics::kHistoryButtonSize)
                  + static_cast<int>(MarsDSP::GUI::Metrics::kHistoryButtonGap)
                  + static_cast<int>(MarsDSP::GUI::Metrics::kHistoryButtonSize)
              <= MarsDSP::GUI::Metrics::kHeaderHalfClear);

        if (! anyFail)
            std::println("band closure (9 bands to 932, 6 cards, rows, slack, header): PASS");
    }

    // ----------------------------------------------------------------
    // 7. Font floor: every declared font constant renders at or above
    //    kFontFloorPx at kScaleMin, 1.0, and kScaleMax.
    // ----------------------------------------------------------------
    g_section = "font_floor";
    {
        const float scales[] = {
            MarsDSP::GUI::Metrics::kScaleMin,
            1.0f,
            MarsDSP::GUI::Metrics::kScaleMax
        };

        for (const float s : scales)
        {
            MarsDSP::GUI::Metrics m;
            m.s = s;
            for (int ci = 0; ci < kNumFontConsts; ++ci)
            {
                const float h = m.font(kFontConsts[ci].du);
                if (h < MarsDSP::GUI::Metrics::kFontFloorPx)
                    FAIL("s={:.2f} font {} = {} below the floor {}",
                         s, kFontConsts[ci].name, h, MarsDSP::GUI::Metrics::kFontFloorPx);
            }
        }
        std::println("font floor ({} constants x 3 scales): PASS", kNumFontConsts);
    }

    // ----------------------------------------------------------------
    // 7b. Display font floor: every display font constant renders at
    //     or above kDisplayFontFloorPx at kScaleMin, 1.0, and kScaleMax.
    // ----------------------------------------------------------------
    g_section = "display_font_floor";
    {
        const float scales[] = {
            MarsDSP::GUI::Metrics::kScaleMin,
            1.0f,
            MarsDSP::GUI::Metrics::kScaleMax
        };

        for (const float s : scales)
        {
            MarsDSP::GUI::Metrics m;
            m.s = s;
            for (int ci = 0; ci < kNumDisplayFontConsts; ++ci)
            {
                const float h = m.displayFont(kDisplayFontConsts[ci].du);
                if (h < MarsDSP::GUI::Metrics::kDisplayFontFloorPx)
                    FAIL("s={:.2f} display font {} = {} below the display floor {}",
                         s, kDisplayFontConsts[ci].name, h, MarsDSP::GUI::Metrics::kDisplayFontFloorPx);
            }
        }
        std::println("display font floor ({} constants x 3 scales): PASS", kNumDisplayFontConsts);
    }

    // ----------------------------------------------------------------
    // 8. Contrast: every declared text pair clears 4.5 to 1 and every
    //    declared icon pair clears 3.0 to 1.
    // ----------------------------------------------------------------
    g_section = "contrast";
    {
        for (int i = 0; i < kNumTextPairs; ++i)
        {
            const auto& p = kTextPairs[i];
            const float ratio = contrast(p.fg, p.bg);
            if (ratio < p.minRatio)
                FAIL("text {} on {}: ratio {} below {}",
                     p.fgName, p.bgName, ratio, p.minRatio);
        }
        for (int i = 0; i < kNumIconPairs; ++i)
        {
            const auto& p = kIconPairs[i];
            const float ratio = contrast(p.fg, p.bg);
            if (ratio < p.minRatio)
                FAIL("icon {} on {}: ratio {} below {}",
                     p.fgName, p.bgName, ratio, p.minRatio);
        }
        std::println("contrast ({} text pairs, {} icon pairs): PASS",
                     kNumTextPairs, kNumIconPairs);
    }

    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos gui_metrics_check ===");
    std::println();

    const int r = runAll();

    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

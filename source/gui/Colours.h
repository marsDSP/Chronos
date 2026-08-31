#pragma once

#ifndef CHRONOS_COLOURS_H
#define CHRONOS_COLOURS_H

#include <JuceHeader.h>

namespace MarsDSP::GUI {

// Colour constants for the user interface.
struct Colours {
    // Surface colours
    static inline const Colour background       { 0xFF1A1A1D };
    static inline const Colour panelBackground  { 0xFF171818 };
    static inline const Colour panelBorder      { 0xFF2A2A2D };
    static inline const Colour headerBackground { 0xFF222225 };
    static inline const Colour footerBackground { 0xFF161618 };
    static inline const Colour rulerText        { 0xFF9A9A9A };

    // Text colours
    static inline const Colour textPrimary { 0xFFABABAB };
    static inline const Colour textBright  { 0xFFD0D0D0 };
    static inline const Colour textDim     { 0xFF666666 };

    // Delay core accent colours. One is live at a time, set by delayMode.
    static inline const Colour accentDelayDigital { 0xFFFF8557 };
    static inline const Colour accentDelayBBD     { 0xFF14EEA4 };

    // Knob colours
    static inline const Colour knobTrack       { 0x14ABABAB };
    static inline const Colour knobFill        { 0xFF303030 };
    static inline const Colour knobGradientTop { 0xFF464646 };
    static inline const Colour knobGradientBot { 0xFF101010 };

    // Dropdown menu colours
    static inline const Colour dropdownBg     { 0xFF232326 };
    static inline const Colour dropdownBorder { 0xFF3A3A3D };
};

// The tint law (section 4.6). One ink base, one accent, one coefficient.
constexpr juce::uint8 kTintInk = 13;

// Blend a base colour toward an accent by the fraction t.
inline Colour tint (Colour base, Colour accent, float t) noexcept
{
    return base.interpolatedWith (accent, t);
}

// Blend the ink base toward an accent by the fraction t.
inline Colour tintInk (Colour accent, float t) noexcept
{
    return tint (Colour (kTintInk, kTintInk, kTintInk), accent, t);
}

// Tint coefficients (section 4.6), solved from the shipped baseline (section 3.7).
constexpr float kTintPlotFill       = 0.052f;
constexpr float kTintPlotFillLit    = 0.093f;
constexpr float kTintGridMinor      = 0.136f;
constexpr float kTintGridMajor      = 0.199f;
constexpr float kTintDisplayBorder  = 0.294f;
constexpr float kTintReadoutFill    = 0.052f;
constexpr float kTintReadoutBorder  = 0.242f;
constexpr float kTintCardBorder     = 0.352f;
constexpr float kTintCentreLine     = 0.473f;

// The alpha of the bypass scrim drawn over the tap and card bands.
constexpr float kBypassScrimAlpha   = 0.38f;

} // namespace MarsDSP::GUI

#endif

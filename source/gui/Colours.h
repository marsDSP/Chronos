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
    static inline const Colour cardBackground   { 0xFF1C1C1F };
    static inline const Colour cardBorder       { 0xFF333337 };
    static inline const Colour burntOrange      { 0xFF1A1410 };

    // Text colours
    static inline const Colour textPrimary { 0xFFABABAB };
    static inline const Colour textBright  { 0xFFD0D0D0 };
    static inline const Colour textDim     { 0xFF666666 };

    // Accent colours
    static inline const Colour accentRed    { 0xFFE0115F };
    static inline const Colour accentGreen  { 0xFF14EEA4 };
    static inline const Colour accentBlue   { 0xFF4FC3F7 };
    static inline const Colour accentPurple { 0xFF7E6AFF };
    static inline const Colour accentPink   { 0xFFFF6B9D };
    static inline const Colour accentOrange { 0xFFFF8A50 };
    static inline const Colour accentYellow { 0xFFE8D44D };

    // Delay core accent colours
    static inline const Colour accentDelayDigital { 0xFFFF8A50 };
    static inline const Colour accentDelayBBD     { 0xFF14EEA4 };

    // Knob colours
    static inline const Colour knobTrack       { 0x14ABABAB };
    static inline const Colour knobArc         { 0xFFABABAB };
    static inline const Colour knobFill        { 0xFF303030 };
    static inline const Colour knobGradientTop { 0xFF464646 };
    static inline const Colour knobGradientBot { 0xFF101010 };

    // Dropdown menu colours
    static inline const Colour dropdownBg     { 0xFF232326 };
    static inline const Colour dropdownBorder { 0xFF3A3A3D };
};

} // namespace MarsDSP::GUI

#endif

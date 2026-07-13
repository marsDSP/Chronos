#pragma once

#ifndef CHRONOS_COLOURS_H
#define CHRONOS_COLOURS_H

#include <JuceHeader.h>

namespace MarsDSP::GUI
{
    struct Colours
    {
        // Surfaces
        static inline const Colour background       { 0xFF1A1A1D };
        static inline const Colour panelBackground  { 0xFF171818 };
        static inline const Colour panelBorder      { 0xFF2A2A2D };
        static inline const Colour headerBackground { 0xFF222225 };

        // Text
        static inline const Colour textPrimary { 0xFFABABAB };
        static inline const Colour textBright  { 0xFFD0D0D0 };
        static inline const Colour textDim     { 0xFF666666 };

        // Accents
        static inline const Colour accentRed    { 0xFFE0115F };
        static inline const Colour accentGreen  { 0xFF14EEA4 };
        static inline const Colour accentBlue   { 0xFF4FC3F7 };
        static inline const Colour accentPurple { 0xFF7E6AFF };
        static inline const Colour accentOrange { 0xFFFF8A50 };
        static inline const Colour accentYellow { 0xFFE8D44D };

        // Knob look
        static inline const Colour knobTrack { 0x14ABABAB };

        // Dropdown menu
        static inline const Colour dropdownBg     { 0xFF232326 };
        static inline const Colour dropdownBorder { 0xFF3A3A3D };
    };
}
#endif
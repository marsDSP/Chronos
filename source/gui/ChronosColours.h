#pragma once

#ifndef CHRONOS_CHRONOSCOLOURS_H
#define CHRONOS_CHRONOSCOLOURS_H

#include <JuceHeader.h>

namespace Chronos
{
    // Plugin-wide palette. Each section of the editor picks an accent
    // colour from this struct so the GUI is self-consistent and easy to
    // re-skin from a single point.
    struct ChronosColours
    {
        // Surfaces
        static inline const juce::Colour background       { 0xFF1A1A1D };
        static inline const juce::Colour panelBackground  { 0xFF171818 };
        static inline const juce::Colour panelBorder      { 0xFF2A2A2D };
        static inline const juce::Colour headerBackground { 0xFF222225 };

        // Text
        static inline const juce::Colour textPrimary { 0xFFABABAB };
        static inline const juce::Colour textBright  { 0xFFD0D0D0 };
        static inline const juce::Colour textDim     { 0xFF666666 };

        // Accents - one per section
        static inline const juce::Colour accentRed    { 0xFFE0115F };
        static inline const juce::Colour accentGreen  { 0xFF14EEA4 };
        static inline const juce::Colour accentBlue   { 0xFF4FC3F7 };
        static inline const juce::Colour accentPurple { 0xFF7E6AFF };
        static inline const juce::Colour accentOrange { 0xFFFF8A50 };
        static inline const juce::Colour accentYellow { 0xFFE8D44D };

        // Knob look
        static inline const juce::Colour knobTrack { 0x14ABABAB };

        // Dropdown menu
        static inline const juce::Colour dropdownBg     { 0xFF232326 };
        static inline const juce::Colour dropdownBorder { 0xFF3A3A3D };
    };
} // namespace Chronos

#endif

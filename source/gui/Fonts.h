#pragma once

#ifndef CHRONOS_FONTS_H
#define CHRONOS_FONTS_H

#include <JuceHeader.h>

namespace MarsDSP::GUI::Fonts {

// Label weights for the interface typeface.
enum class Weight { Regular, Medium, Semibold };

// Clash Grotesk cap height as a fraction of the em height.
constexpr float kCapHeightRatio = 0.71f;

// Return the typeface for a weight. Return the default face when the font resource is absent.
Typeface::Ptr typefaceFor(Weight weight);

// Return a font of the given weight and pixel height.
// Fall back to the default face when the resource is absent.
Font font(Weight weight, float height);

// Return the short form of a label, or the input when no short form exists.
String shortLabel(const String& full);

// Return the fixed advance width for digit glyphs at this font height.
// Cache the result so a changing value does not shift the string.
float digitAdvance(const Font& font);

// Draw text with a fixed advance for digit glyphs so a value does not shift.
// Center the result in the given area.
void drawFixedAdvanceText(Graphics& g, const Font& font, const String& text,
                          const Rectangle<float>& area, Colour colour);

} // namespace MarsDSP::GUI::Fonts

#endif

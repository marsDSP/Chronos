#pragma once

#ifndef CHRONOS_HALO_IMAGE_H
#define CHRONOS_HALO_IMAGE_H

#include <JuceHeader.h>
#include "../Metrics.h"

#include <cmath>

namespace MarsDSP::GUI {

// A 256 x 2 ARGB image whose alpha across x follows exp(-z^2/2)
// for z in [-kHaloSigmas, +kHaloSigmas], in the accent colour.
// One drawImageTransformed per halo, under g.setOpacity, draws it at
// any width and height with no allocation. Build in a constructor or
// setAccentColour, never in paint (invariant 18).
inline Image makeHaloImage(Colour accent)
{
    constexpr int kWidth = 256;
    constexpr int kHeight = 2;
    Image img(Image::ARGB, kWidth, kHeight, true);

    Graphics g(img);
    for (int x = 0; x < kWidth; ++x)
    {
        // Map x to z in [-kHaloSigmas, +kHaloSigmas].
        const float z = (static_cast<float>(x) / static_cast<float>(kWidth - 1) - 0.5f)
                       * (2.0f * Metrics::kHaloSigmas);
        const float a = std::exp(-0.5f * z * z);
        const Colour c = accent.withMultipliedAlpha(a);
        g.setColour(c);
        g.fillRect(x, 0, 1, kHeight);
    }
    return img;
}

} // namespace MarsDSP::GUI

#endif

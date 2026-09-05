#pragma once

#ifndef CHRONOS_HALO_IMAGE_H
#define CHRONOS_HALO_IMAGE_H

#include <JuceHeader.h>
#include "../Metrics.h"

#include <cmath>

namespace MarsDSP::GUI {


inline Image makeHaloImage(Colour accent)
{
    constexpr int kWidth = 512;
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

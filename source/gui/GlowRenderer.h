#pragma once

#ifndef CHRONOS_GLOW_RENDERER_H
#define CHRONOS_GLOW_RENDERER_H

#include <JuceHeader.h>

#include <algorithm>

namespace MarsDSP::GUI {

template <typename FillFn>
void drawBlurredGlow(Graphics& dest, Component& component, Rectangle<float> clipBounds,
                     float blurPx, FillFn&& fillFn)
{
    if (clipBounds.getWidth() <= 0.0f || clipBounds.getHeight() <= 0.0f)
        return;

    const float pad = blurPx * 2.0f + 4.0f;
    const auto imgArea = clipBounds.expanded(pad);

    double scaleFactor = Component::getApproximateScaleFactorForComponent(&component);
    scaleFactor = std::clamp(scaleFactor, 1.0, 4.0);
    const float imgScale = static_cast<float>(scaleFactor);
    const int imgW = std::max(1, roundToInt(imgArea.getWidth() * imgScale));
    const int imgH = std::max(1, roundToInt(imgArea.getHeight() * imgScale));

    Image img(Image::ARGB, imgW, imgH, true);
    {
        Graphics g(img);
        g.addTransform(AffineTransform::translation(-imgArea.getX(), -imgArea.getY()).scaled(imgScale));
        fillFn(g);
    }

    if (auto pixelData = img.getPixelData())
        pixelData->applyGaussianBlurEffect(blurPx * imgScale);

    Graphics::ScopedSaveState ss(dest);
    dest.reduceClipRegion(clipBounds.toNearestInt());
    dest.drawImageTransformed(img, AffineTransform::scale(1.0f / imgScale)
                                       .translated(imgArea.getX(), imgArea.getY()));
}

} // namespace MarsDSP::GUI

#endif

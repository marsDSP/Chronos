#pragma once

#ifndef CHRONOS_SATURATOR_MAKEUP_H
#define CHRONOS_SATURATOR_MAKEUP_H

#include <array>
#include <algorithm>
#include <cmath>

// Saturator makeup gain table.
// Gives the RMS ratio of tanh(k * x) to x for a 0.5-amplitude sine.
// Gives the output makeup and the loop trim from the ratio.
// The table covers k from 1 to 16. Block-rate callers interpolate.
// scripts/python/gen_makeup_table.py derives the values.

namespace MarsDSP::Math
{
    inline constexpr int kMakeupTableSize = 65;
    inline constexpr float kMakeupLog2Step = 0.0625f; // (4.0 - 0.0) / (65 - 1)

    // rms(tanh(k * 0.5 * sin)) / rms(0.5 * sin) on a 65-point log2(k) grid.
    inline constexpr std::array<float, kMakeupTableSize> kRmsRatioTable{ {
        0.942466974f, 0.979221046f, 1.01699519f,  1.05576766f,  1.09551024f,
        1.13618743f,  1.17775595f,  1.22016466f,  1.26335371f,  1.30725491f,
        1.3517915f,   1.39687788f,  1.44242001f,  1.48831582f,  1.5344559f,
        1.580724f,    1.62699795f,  1.67315125f,  1.7190541f,   1.76457489f,
        1.809582f,    1.85394585f,  1.89753985f,  1.94024348f,  1.98194277f,
        2.02253294f,  2.06191969f,  2.10002017f,  2.13676429f,  2.17209506f,
        2.20597005f,  2.2383604f,   2.26925111f,  2.29864049f,  2.32653928f,
        2.35296988f,  2.37796497f,  2.40156555f,  2.42382121f,  2.44478679f,
        2.46452188f,  2.48308921f,  2.50055337f,  2.51697922f,  2.53243184f,
        2.54697442f,  2.56066799f,  2.57357121f,  2.58573961f,  2.59722471f,
        2.60807538f,  2.61833596f,  2.62804794f,  2.63724899f,  2.64597368f,
        2.65425324f,  2.662117f,    2.66959071f,  2.67669845f,  2.6834619f,
        2.68990135f,  2.69603539f,  2.70188046f,  2.70745277f,  2.71276641f,
    } };

    // pow(rmsRatio, -0.7) at each grid point. The output makeup gain.
    inline constexpr std::array<float, kMakeupTableSize> kOutputMakeupTable{ {
        1.04235029f, 1.01480711f, 0.988272667f, 0.962724805f, 0.938141763f,
        0.91450274f, 0.891787291f, 0.869975686f, 0.849048555f, 0.828987122f,
        0.809772849f, 0.79138732f, 0.773812592f, 0.757030606f, 0.741023362f,
        0.725772917f, 0.711261153f, 0.697469711f, 0.684380054f, 0.671973169f,
        0.660229921f, 0.649130583f, 0.638655066f, 0.628782809f, 0.619492769f,
        0.61076349f, 0.602573156f, 0.594899476f, 0.587719858f, 0.581011593f,
        0.574751675f, 0.568917096f, 0.563484788f, 0.558431923f, 0.553735971f,
        0.549374521f, 0.545325994f, 0.541569114f, 0.538083375f, 0.534849107f,
        0.531847477f, 0.529060543f, 0.526471317f, 0.524063885f, 0.521823406f,
        0.519735992f, 0.517788887f, 0.51597023f,  0.514269352f, 0.512676418f,
        0.511182427f, 0.509779334f, 0.508459926f, 0.507217467f, 0.506046176f,
        0.504940629f, 0.503896117f, 0.50290823f,  0.501973033f, 0.50108707f,
        0.500247061f, 0.499450088f, 0.498693496f, 0.497974813f, 0.497291803f,
    } };

    // pow(rmsRatio, -0.5) at each grid point. The loop output trim.
    inline constexpr std::array<float, kMakeupTableSize> kLoopTrimTable{ {
        1.03007042f, 1.01055419f, 0.991609216f, 0.973230779f, 0.955414355f,
        0.93815589f, 0.921451211f, 0.905296385f, 0.889687598f, 0.874620914f,
        0.860092461f, 0.846098244f, 0.832633972f, 0.819695294f, 0.807277381f,
        0.795375109f, 0.783982754f, 0.773094177f, 0.762702584f, 0.752800584f,
        0.74338001f,  0.734431803f, 0.725946367f, 0.717913091f, 0.710320652f,
        0.703156829f, 0.69640857f,  0.690062225f, 0.68410331f,  0.678516746f,
        0.673286915f, 0.668397784f, 0.663832843f, 0.659575462f, 0.655608833f,
        0.651916265f, 0.648481011f, 0.645286798f, 0.642317414f, 0.639557362f,
        0.636991501f, 0.634605527f, 0.632385552f, 0.630318701f, 0.628392696f,
        0.626596153f, 0.624918461f, 0.623349905f, 0.621881485f, 0.620504916f,
        0.619212806f, 0.617998362f, 0.616855383f, 0.615778387f, 0.614762306f,
        0.613802731f, 0.612895489f, 0.612037003f, 0.611223817f, 0.610453069f,
        0.609721899f, 0.609027922f, 0.608368814f, 0.607742429f, 0.607146919f,
    } };

    // Catmull-Rom interpolation over a 65-point table indexed by log2(k).
    // Clamp k below 1 and above 16 to the table ends.
    inline float makeupInterpolate(const std::array<float, kMakeupTableSize>& t,
                                   float k) noexcept
    {
        const float u = std::log2(std::max(k, 1.0f));
        if (u <= 0.0f) return t[0];
        if (u >= 4.0f) return t[kMakeupTableSize - 1];
        const float fi = u / kMakeupLog2Step;
        const int i = static_cast<int>(fi);
        const float f = fi - static_cast<float>(i);
        const auto idx = static_cast<std::size_t>(i);
        // Linear extrapolation at the boundaries preserves the slope.
        // This is more accurate than repeating the endpoint on the concave
        // saturating curve near k = 16.
        const float p0 = (i > 0) ? t[idx - 1] : 2.0f * t[0] - t[1];
        const float p1 = t[idx];
        const float p2 = t[idx + 1];
        const float p3 = (i + 2 < kMakeupTableSize) ? t[idx + 2] : 2.0f * t[idx + 1] - t[idx];
        const float f2 = f * f;
        return 0.5f * ((2.0f * p1)
                       + (-p0 + p2) * f
                       + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * f2
                       + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * f2 * f);
    }

    // Return the RMS gain of tanh at the given drive.
    inline float rmsRatio(float k) noexcept
    {
        return makeupInterpolate(kRmsRatioTable, k);
    }

    // The inverse of the makeup at k = 1. Multiply by this to normalise
    // the makeup to unity at zero drive. This keeps the output unchanged
    // when the drive is 0 dB and preserves the 2.75 dB rise across the sweep.
    inline constexpr float kOutputMakeupUnity = 1.0f / kOutputMakeupTable[0];

    // Return the makeup gain for the output saturator.
    inline float outputMakeup(float k) noexcept
    {
        return makeupInterpolate(kOutputMakeupTable, k);
    }

    // Return the trim gain for the loop output.
    inline float loopTrim(float k) noexcept
    {
        return makeupInterpolate(kLoopTrimTable, k);
    }
}

#endif

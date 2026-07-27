#pragma once

#ifndef CHRONOS_WAVESHAPER_H
#define CHRONOS_WAVESHAPER_H

#include <cmath>
#include <algorithm>

namespace MarsDSP::DSP::Primitives
{
    enum class WaveshaperType
    {
        Soft = 0,
        Hard,
        Asymmetric,
        Sine,
        Digital,
        FwRectify,
        FuzzSoft,
        Asinh,
        NUM_TYPES
    };

    inline const char *waveshaperName(WaveshaperType t)
    {
        switch (t)
        {
            case WaveshaperType::Soft: return "Soft";
            case WaveshaperType::Asymmetric: return "Asymmetric";
            case WaveshaperType::Sine: return "Sine";
            default: return "Unknown";
        }
    }

    inline float waveshape(WaveshaperType type, float x)
    {
        switch (type)
        {
            case WaveshaperType::Soft:
            {
                // tanh soft clip
                x = std::clamp(x, -3.f, 3.f);
                return std::tanh(x);
            }
            case WaveshaperType::Asymmetric:
            {
                // Asymmetric soft clip: positive half is tanh, negative half is weaker
                if (x >= 0.f) return std::tanh(x);

                return std::tanh(x * 0.5f);
            }
            case WaveshaperType::Sine:
            {
                // Sinusoidal waveshaper: sin(pi/2 * x) for |x| <= 1, clipped outside
                if (std::fabs(x) > 1.f) return x > 0.f ? 1.f : -1.f;

                return std::sin(static_cast<float>(M_PI) * 0.5f * x);
            }
            default:
                return x;
        }
    }
    static constexpr int kNumWaveshaperTypes = static_cast<int>(WaveshaperType::NUM_TYPES);
}
#endif

#pragma once

#ifndef CHRONOS_WAVESHAPER_H
#define CHRONOS_WAVESHAPER_H

#include <cmath>
#include <algorithm>
#include <utility>

namespace MarsDSP::DSP::Primitives {
    enum class WaveshaperType
    {
        Soft = 0,
        Hard,
        Asymmetric,
        Sine,
        NUM_TYPES
    };

    inline const char *waveshaperName(WaveshaperType t) {
        switch (t) {
            case WaveshaperType::Soft: return "Soft";
            case WaveshaperType::Asymmetric: return "Asymmetric";
            case WaveshaperType::Sine: return "Sine";
            default: return "Unknown";
        }
    }

    inline float waveshape(WaveshaperType type, float x) {
        switch (type) {
            case WaveshaperType::Soft: {
                x = std::clamp(x, -3.f, 3.f);
                return std::tanh(x);
            }
            case WaveshaperType::Asymmetric: {
                if (x >= 0.f) return std::tanh(x);
                return std::tanh(x * 0.5f);
            }
            case WaveshaperType::Sine: {
                if (std::fabs(x) > 1.f) return x > 0.f ? 1.f : -1.f;
                return std::sin(static_cast<float>(M_PI) * 0.5f * x);
            }
            default:
                return x;
        }
    }
    static constexpr int kNumWaveshaperTypes = std::to_underlying(WaveshaperType::NUM_TYPES);
}
#endif

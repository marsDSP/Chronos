#pragma once

#ifndef CHRONOS_TICK_GENERATOR_H
#define CHRONOS_TICK_GENERATOR_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace MarsDSP::GUI {

// A set of ruler ticks: major positions, minor positions, and the major step.
struct RulerTicks {
    std::vector<float> majors;
    std::vector<float> minors;
    float majorStep = 0.0f;
};

// Free-mode tick generator.
// T is the eased visible span in seconds. W is the plot width in pixels. s is the scale.
inline RulerTicks computeFreeTicks(const float T, const float W, const float s)
{
    RulerTicks t;
    if (T <= 0.0f || W <= 0.0f || s <= 0.0f)
        return t;

    const int N = std::clamp(static_cast<int>(std::floor(W / (108.0f * s))), 5, 10);
    const float q = T / static_cast<float>(N);
    const float d = std::pow(10.0f, std::floor(std::log10(q)));
    const float u = q / d;

    static constexpr float kMantissas[] = { 1.0f, 2.0f, 5.0f, 10.0f };
    float m = 10.0f;
    for (const float cand : kMantissas)
        if (cand >= u) { m = cand; break; }

    const float step = m * d;
    const int k = (m == 2.0f) ? 4 : 5;
    const float minor = step / static_cast<float>(k);
    const float eps = minor * 1e-4f;

    t.majorStep = step;

    for (int i = 0; ; ++i)
    {
        const float mt = static_cast<float>(i) * step;
        if (mt > T + eps) break;
        t.majors.push_back(mt);
    }

    for (int j = 0; ; ++j)
    {
        const float nt = static_cast<float>(j) * minor;
        if (nt > T + eps) break;
        if (j % k != 0)
            t.minors.push_back(nt);
    }

    return t;
}

// Sync-mode tick generator: majors per beat, minors per current division.
inline RulerTicks computeSyncTicks(const float span, const float secondsPerBeat,
                                    const float divisionSeconds)
{
    RulerTicks t;
    if (secondsPerBeat <= 0.0f)
        return t;

    const float eps = secondsPerBeat * 1e-4f;
    for (float s = 0.0f; s <= span + eps; s += secondsPerBeat)
        t.majors.push_back(s);

    if (divisionSeconds > 0.0f && divisionSeconds < secondsPerBeat)
    {
        const float dep = divisionSeconds * 1e-4f;
        for (float s = 0.0f; s <= span + dep; s += divisionSeconds)
        {
            bool isMajor = false;
            for (const float m : t.majors)
            {
                if (std::fabs(s - m) < divisionSeconds * 0.25f)
                {
                    isMajor = true;
                    break;
                }
            }
            if (! isMajor)
                t.minors.push_back(s);
        }
    }

    return t;
}

} // namespace MarsDSP::GUI

#endif

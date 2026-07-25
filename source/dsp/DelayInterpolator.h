#pragma once

#ifndef CHRONOS_DELAY_INTERPOLATOR_H
#define CHRONOS_DELAY_INTERPOLATOR_H

#include <cassert>
#include <cmath>

namespace MarsDSP::Delays {
    enum class Interpolation : int
    {
        Linear      = 0,
        Lagrange3rd = 1,
        Lagrange5th = 2
    };

    struct Coeffs6
    {
        float c[6] = {};
    };

    inline Coeffs6 makeCoeffs(Interpolation mode, float f) noexcept
    {
        f = f - std::floor(f); // reduce to [0, 1)
        assert(f >= 0.0f && f < 1.0f);
        const float e = 3.0f - f;

        const auto basis = [](const int* S, int nS, int j, float ev) noexcept {
            float num = 1.0f;
            float den = 1.0f;
            for (int idx = 0; idx < nS; ++idx)
            {
                const int k = S[idx];
                if (k == j)
                    continue;
                num *= (ev - static_cast<float>(k));
                den *= static_cast<float>(j - k);
            }
            return num / den;
        };

        Coeffs6 out;

        switch (mode)
        {
            case Interpolation::Linear:
            {
                constexpr int S[] = {2, 3};
                for (int j : S)
                    out.c[j] = basis(S, 2, j, e);
                break;
            }
            case Interpolation::Lagrange3rd:
            {
                constexpr int S[] = {1, 2, 3, 4};
                for (int j : S)
                    out.c[j] = basis(S, 4, j, e);
                break;
            }
            case Interpolation::Lagrange5th:
            {
                constexpr int S[] = {0, 1, 2, 3, 4, 5};
                for (int j : S)
                    out.c[j] = basis(S, 6, j, e);
                break;
            }
        }
        return out;
    }

}
#endif

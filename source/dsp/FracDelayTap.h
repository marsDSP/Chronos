#pragma once

#ifndef CHRONOS_FRAC_DELAY_TAP_H
#define CHRONOS_FRAC_DELAY_TAP_H

#include "DelayInterpolator.h"
#include "Pow2RingBuffer.h"
#include "simd/Config.h"

#include <cassert>
#include <cmath>

namespace MarsDSP::Delays {

    struct FracDelayTap
    {
        struct Coeffs4 { float c1, c2, c3, c4; };

        [[nodiscard]] static Coeffs4 lagrange3(float f) noexcept
        {
            assert(f >= 0.0f && f < 1.0f);
            constexpr float kInv6 = 1.0f / 6.0f;
            const float e  = 3.0f - f;
            const float e1 = e - 1.0f;   //  2 - f
            const float e2 = e - 2.0f;   //  1 - f
            const float e3 = e - 3.0f;   //     -f
            const float e4 = e - 4.0f;   // -1 - f
            const float e34  = e3 * e4;
            const float e12  = e1 * e2;
            return { e2 * e34 * -kInv6,
                     e1 * e34 *  0.5f,
                     e12 * e4 * -0.5f,
                     e12 * e3 *  kInv6 };
        }

        [[nodiscard]] static float read(const Pow2RingBuffer& rb,
                                        int writeIdx,
                                        float delaySamples) noexcept
        {
            assert(delaySamples >= 3.0f);
            assert(delaySamples <=
                   static_cast<float>(rb.getCapacity() - Pow2RingBuffer::kTail - 2));

            const auto  i = static_cast<int>(delaySamples); // delay >= 0: trunc == floor
            const float f = delaySamples - static_cast<float>(i);
            const int base = (writeIdx - i - 3) & rb.mask();

            const Coeffs4 k = lagrange3(f);

            const float* w = rb.windowPtr(base, 6);
            float scratch[6];
            if (w == nullptr)
            {
                rb.readWindow(scratch, base, 6);
                w = scratch;
            }

            const M128 taps = MM(loadu_ps)(w + 1);
            const M128 cf   = MM(set_ps)(k.c4, k.c3, k.c2, k.c1);
            const M128 prod = MM(mul_ps)(taps, cf);
            const M128 sh1  = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
            const M128 sh2  = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
            return MM(cvtss_f32)(sh2);
        }

        [[nodiscard]] static float readRef(const Pow2RingBuffer& rb, int writeIdx, float delaySamples) noexcept
        {
            const auto  i = static_cast<int>(std::floor(delaySamples));
            const float f = delaySamples - static_cast<float>(i);
            const int base = (writeIdx - i - 3) & rb.mask();

            const Coeffs6 c = makeCoeffs(Interpolation::Lagrange3rd, f);
            float scratch[6];
            rb.readWindow(scratch, base, 6);
            float acc = 0.0f;
            for (int t = 0; t < 6; ++t)
                acc += scratch[t] * c.c[static_cast<std::size_t>(t)];
            return acc;
        }
    };
}
#endif

#pragma once

#ifndef CHRONOS_BLOCK_TAP_READER_H
#define CHRONOS_BLOCK_TAP_READER_H

#include "DelayInterpolator.h"
#include "Pow2RingBuffer.h"
#include "simd/Config.h"

#include <array>

namespace MarsDSP::Delays {
    struct BlockTapReader {
        struct Window
        {
            const float* ptr;
            bool aligned;
        };

        [[nodiscard]] static Window acquireWindow(const Pow2RingBuffer& rb,
                                                  int base, int winLen,
                                                  float* scratch) noexcept
        {
            const float* p = rb.windowPtr(base, winLen);
            if (p == nullptr)
            {
                rb.readWindow(scratch, base, winLen);
                p = scratch;
            }
            return { p, p == scratch };
        }

        [[nodiscard]] static float dot6(const Coeffs6& c, const float* w) noexcept
        {
            return c.c[0] * w[0] + c.c[1] * w[1] + c.c[2] * w[2]
                 + c.c[3] * w[3] + c.c[4] * w[4] + c.c[5] * w[5];
        }

        static void eval4(const float* oldWin, bool oldAligned,
                          const float* newWin, bool newAligned,
                          const std::array<M128, 6>& cbOld,
                          const std::array<M128, 6>& cbNew,
                          M128 vAlpha,
                          float* dst) noexcept
        {
            M128 vOld = MM(setzero_ps)();
            M128 vNew = MM(setzero_ps)();

            for (int t = 0; t < 6; ++t)
            {
                const M128 wOld = (t == 0 && oldAligned)
                                      ? MM(load_ps)(oldWin)
                                      : MM(loadu_ps)(oldWin + t);
                vOld = FMADD(wOld, cbOld[static_cast<std::size_t>(t)], vOld);

                const M128 wNew = (t == 0 && newAligned)
                                      ? MM(load_ps)(newWin)
                                      : MM(loadu_ps)(newWin + t);
                vNew = FMADD(wNew, cbNew[static_cast<std::size_t>(t)], vNew);
            }

            const M128 vDelta = MM(sub_ps)(vNew, vOld);
            const M128 vOut = FMADD(vAlpha, vDelta, vOld);
            MM(storeu_ps)(dst, vOut);
        }
    };
}
#endif

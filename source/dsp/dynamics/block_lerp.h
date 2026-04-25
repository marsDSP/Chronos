#pragma once

#ifndef CHRONOS_BLOCK_LERP_H
#define CHRONOS_BLOCK_LERP_H

// ============================================================================
//  block_lerp.h
// ----------------------------------------------------------------------------
//  Block-rate linear interpolator for control-rate parameters that need to
//  cross a host block without zipper noise. The pattern is:
//
//      lerp.setTarget (newTarget);   // call once per block
//      for (int i = 0; i < N; ++i)
//          out[i] = in[i] * lerp.next();   // per-sample ramp
//
//  Two flavours:
//
//   * BlockLerp<T>           : scalar one-pole ramp (current → target over N
//                              samples), suitable for any control parameter.
//
//   * BlockLerpSIMD          : pre-computes a per-quad line of four ramped
//                              values so the audio inner loop can multiply
//                              a SIMD_M128 of input by a SIMD_M128 of gain
//                              without any per-sample bookkeeping. The line
//                              array is kept aligned to 16 bytes for native
//                              SSE2 alignment.
//
//  Both classes assume blockSize is a multiple of 4.
// ============================================================================

#include <cassert>
#include <cstddef>
#include "dsp/math/simd/simd_config.h"

namespace MarsDSP::DSP::Dynamics
{
    template <typename T = float>
    class BlockLerp
    {
    public:
        void setBlockSize (int n) noexcept
        {
            assert (n > 0);
            blockSize    = n;
            blockSizeInv = static_cast<T>(1) / static_cast<T>(n);
        }

        void setTarget (T newTarget) noexcept
        {
            current  = target;
            target   = newTarget;
            if (firstRun)
            {
                current  = newTarget;
                firstRun = false;
            }
            increment = (target - current) * blockSizeInv;
            v         = current;
        }

        void instantize (T value) noexcept
        {
            target    = value;
            current   = value;
            v         = value;
            increment = static_cast<T>(0);
            firstRun  = false;
        }

        // Advance one sample and return the new value.
        inline T next() noexcept
        {
            const T r = v;
            v += increment;
            return r;
        }

        [[nodiscard]] T getCurrent() const noexcept { return v; }
        [[nodiscard]] T getTarget()  const noexcept { return target; }

    private:
        int  blockSize    { 64 };
        T    blockSizeInv { static_cast<T>(1.0 / 64.0) };
        T    target       { static_cast<T>(0) };
        T    current      { static_cast<T>(0) };
        T    increment    { static_cast<T>(0) };
        T    v            { static_cast<T>(0) };
        bool firstRun     { true };
    };

    // ------------------------------------------------------------------ //
    //  SIMD variant. blockSize must be a power of two and ≥ 4.
    // ------------------------------------------------------------------ //
    template <int MaxBlockSize>
    class alignas(16) BlockLerpSIMD
    {
        static_assert ((MaxBlockSize & (MaxBlockSize - 1)) == 0,
                       "BlockLerpSIMD: MaxBlockSize must be a power of two.");
        static_assert (MaxBlockSize >= 4,
                       "BlockLerpSIMD: MaxBlockSize must be at least 4.");

    public:
        static constexpr int kMaxQuads = MaxBlockSize / 4;

        BlockLerpSIMD()
        {
            const float zbq alignas(16) [4] = { 0.25f, 0.5f, 0.75f, 1.0f };
            zeroToOne = SIMD_MM(load_ps)(zbq);
            setBlockSize (MaxBlockSize);
        }

        void setBlockSize (int bs) noexcept
        {
            assert (bs > 0 && (bs & (bs - 1)) == 0);
            assert (bs <= MaxBlockSize);
            blockSize       = bs;
            numQuads        = bs / 4;
            quadCountInv    = 1.0f / static_cast<float>(numQuads);
        }

        void setTarget (float newTarget) noexcept
        {
            current = target;
            target  = newTarget;
            if (firstRun)
            {
                current  = newTarget;
                firstRun = false;
            }
            updateLine();
        }

        void instantize (float value) noexcept
        {
            target  = value;
            current = value;
            firstRun = false;
            updateLine();
        }

        // Read access to the precomputed per-quad ramp values; line[i] holds
        // the ramped values for the i-th quad (lanes are sample 0..3 of the
        // quad).
        [[nodiscard]] SIMD_M128 quad (int i) const noexcept
        {
            assert (i >= 0 && i < numQuads);
            return line[i];
        }

        // Multiply N samples in src by the ramped gain in place.
        void multiplyInPlace (float* __restrict src) const noexcept
        {
            for (int i = 0; i < numQuads; ++i)
            {
                const auto v = SIMD_MM(loadu_ps)(src + (i << 2));
                SIMD_MM(storeu_ps)(src + (i << 2), SIMD_MM(mul_ps)(v, line[i]));
            }
        }

        [[nodiscard]] int   getBlockSize() const noexcept { return blockSize; }
        [[nodiscard]] int   getNumQuads()  const noexcept { return numQuads; }
        [[nodiscard]] float getTarget()    const noexcept { return target; }
        [[nodiscard]] float getCurrent()   const noexcept { return current; }

    private:
        void updateLine() noexcept
        {
            const auto cs   = SIMD_MM(set1_ps)(current);
            const auto step = SIMD_MM(set1_ps)((target - current) * quadCountInv);
            auto       dy   = SIMD_MM(mul_ps)(step, zeroToOne);
            for (int i = 0; i < numQuads; ++i)
            {
                line[i] = SIMD_MM(add_ps)(cs, dy);
                dy      = SIMD_MM(add_ps)(dy, step);
            }
            current = target;
        }

        SIMD_M128 line[kMaxQuads];
        SIMD_M128 zeroToOne;

        int   blockSize    { MaxBlockSize };
        int   numQuads     { MaxBlockSize / 4 };
        float quadCountInv { 1.0f / static_cast<float>(MaxBlockSize / 4) };
        float target       { 0.0f };
        float current      { 0.0f };
        bool  firstRun     { true };
    };
}
#endif

#pragma once

#ifndef CHRONOS_NESTED_ALLPASS_H
#define CHRONOS_NESTED_ALLPASS_H

#if defined(_MSC_VER)
    #pragma fp_contract(off)
#else
    #pragma STDC FP_CONTRACT OFF
#endif

#include "BlockTapReader.h"
#include "FracDelayTap.h"
#include "Pow2RingBuffer.h"
#include "simd/Config.h"
#include "utils/memory/BumpArena.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace MarsDSP::Diffusion
{
    // Nested allpass filter section.
    // The delay element contains an inner allpass filter.
    class NestedAllpass
    {
    public:
        static constexpr int kChunk = 16;

        // Return the storage size in floats for the arena.
        [[nodiscard]] static std::size_t ringStorageFloats(int maxDOut, int maxDIn) noexcept
        {
            const int minCapOut = maxDOut + Delays::Pow2RingBuffer::kTail + 8;
            const int minCapIn = maxDIn + Delays::Pow2RingBuffer::kTail + 8;
            return Delays::Pow2RingBuffer::arenaFloatsFor(minCapOut)
                 + Delays::Pow2RingBuffer::arenaFloatsFor(minCapIn);
        }

        void prepare(int dOut, int dIn) noexcept
        {
            prepareImpl_(dOut, dIn, nullptr);
        }

        void prepare(int dOut, int dIn, Memory::BumpArena &arena) noexcept
        {
            prepareImpl_(dOut, dIn, &arena);
        }

        void reset() noexcept
        {
            ringOut_.clear();
            ringIn_.clear();
            wOut_ = 0;
            wIn_ = 0;
        }

        void setCoefficients(float gOut, float gIn) noexcept
        {
            gOut_ = gOut;
            gIn_ = gIn;
        }

        void setDelays(float dOut, float dIn) noexcept
        {
            assert(dOut > static_cast<float>(kChunk));
            assert(dIn > static_cast<float>(kChunk));
            dOut_ = dOut;
            dIn_ = dIn;
        }

        [[nodiscard]] float centroidSamples() const noexcept
        {
            return dOut_ + dIn_;
        }

        [[nodiscard]] float getDelayOut() const noexcept { return dOut_; }
        [[nodiscard]] float getDelayIn() const noexcept { return dIn_; }
        [[nodiscard]] float getGainOut() const noexcept { return gOut_; }
        [[nodiscard]] float getGainIn() const noexcept { return gIn_; }

        // Process samples with the scalar reference path.
        void processRef(float *io, int n) noexcept
        {
            assert(io != nullptr);
            const int maskOut = ringOut_.mask();
            const int maskIn = ringIn_.mask();

            for (int s = 0; s < n; ++s)
            {
                const float x = io[s];
                const float d = Delays::FracDelayTap::read(ringOut_, wOut_, dOut_);
                float v = x - gOut_ * d;
                if (!std::isfinite(v)) v = 0.0f;

                const float dIn = Delays::FracDelayTap::read(ringIn_, wIn_, dIn_);
                float vIn = v - gIn_ * dIn;
                if (!std::isfinite(vIn)) vIn = 0.0f;

                const float w = dIn + gIn_ * vIn;

                ringIn_.writeBlock(&vIn, wIn_, 1);
                ringIn_.refreshMirror(wIn_, 1);
                wIn_ = (wIn_ + 1) & maskIn;

                ringOut_.writeBlock(&w, wOut_, 1);
                ringOut_.refreshMirror(wOut_, 1);
                wOut_ = (wOut_ + 1) & maskOut;

                const float y = d + gOut_ * v;
                io[s] = y;
            }
        }

        // Process samples in four-wide vector chunks.
        void processBlock(float *io, int n) noexcept
        {
            assert(io != nullptr);
            const int maskOut = ringOut_.mask();
            const int maskIn = ringIn_.mask();

            const auto iIntOut = static_cast<int>(dOut_);
            const float fOut = dOut_ - static_cast<float>(iIntOut);
            const auto kOut = Delays::FracDelayTap::lagrange3(fOut);
            const M128 cfOut = MM(set_ps)(kOut.c4, kOut.c3, kOut.c2, kOut.c1);

            const auto iIntIn = static_cast<int>(dIn_);
            const float fIn = dIn_ - static_cast<float>(iIntIn);
            const auto kIn = Delays::FracDelayTap::lagrange3(fIn);
            const M128 cfIn = MM(set_ps)(kIn.c4, kIn.c3, kIn.c2, kIn.c1);

            const M128 vGOut = MM(set1_ps)(gOut_);
            const M128 vGIn = MM(set1_ps)(gIn_);

            for (int off = 0; off < n; off += kChunk)
            {
                const int m = std::min(kChunk, n - off);

                // Read the outer delay taps.
                const int baseOut = (wOut_ - iIntOut - 3) & maskOut;
                const int winLenOut = m + 6;
                const auto wPtrOut = Delays::BlockTapReader::acquireWindow(ringOut_, baseOut, winLenOut, tapWinOut_.data());
                const float *winOut = wPtrOut.ptr;

                alignas(16) float dArr[kChunk];
                for (int i = 0; i < m; ++i)
                {
                    const M128 taps = MM(loadu_ps)(winOut + i + 1);
                    const M128 prod = MM(mul_ps)(taps, cfOut);
                    const M128 sh1 = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
                    const M128 sh2 = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
                    dArr[i] = MM(cvtss_f32)(sh2);
                }

                // Compute v = x - g_out * d.
                alignas(16) float vArr[kChunk];
                const int m4 = m & ~3;
                for (int i = 0; i < m4; i += 4)
                {
                    const M128 xV = MM(loadu_ps)(io + off + i);
                    const M128 dV = MM(load_ps)(dArr + i);
                    const M128 vV = MM(sub_ps)(xV, MM(mul_ps)(vGOut, dV));
                    MM(store_ps)(vArr + i, vV);
                }
                for (int i = m4; i < m; ++i)
                {
                    vArr[i] = io[off + i] - gOut_ * dArr[i];
                }
                for (int i = 0; i < m; ++i)
                {
                    if (!std::isfinite(vArr[i])) vArr[i] = 0.0f;
                }

                // Read the inner delay taps.
                const int baseIn = (wIn_ - iIntIn - 3) & maskIn;
                const int winLenIn = m + 6;
                const auto wPtrIn = Delays::BlockTapReader::acquireWindow(ringIn_, baseIn, winLenIn, tapWinIn_.data());
                const float *winIn = wPtrIn.ptr;

                alignas(16) float dInArr[kChunk];
                for (int i = 0; i < m; ++i)
                {
                    const M128 taps = MM(loadu_ps)(winIn + i + 1);
                    const M128 prod = MM(mul_ps)(taps, cfIn);
                    const M128 sh1 = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
                    const M128 sh2 = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
                    dInArr[i] = MM(cvtss_f32)(sh2);
                }

                // Compute v_in and w.
                alignas(16) float vInArr[kChunk];
                alignas(16) float wArr[kChunk];
                for (int i = 0; i < m4; i += 4)
                {
                    const M128 vV = MM(load_ps)(vArr + i);
                    const M128 dInV = MM(load_ps)(dInArr + i);
                    const M128 vInV = MM(sub_ps)(vV, MM(mul_ps)(vGIn, dInV));
                    MM(store_ps)(vInArr + i, vInV);
                }
                for (int i = m4; i < m; ++i)
                {
                    vInArr[i] = vArr[i] - gIn_ * dInArr[i];
                }
                for (int i = 0; i < m; ++i)
                {
                    if (!std::isfinite(vInArr[i])) vInArr[i] = 0.0f;
                }

                for (int i = 0; i < m4; i += 4)
                {
                    const M128 dInV = MM(load_ps)(dInArr + i);
                    const M128 vInV = MM(load_ps)(vInArr + i);
                    const M128 wV = MM(add_ps)(dInV, MM(mul_ps)(vGIn, vInV));
                    MM(store_ps)(wArr + i, wV);
                }
                for (int i = m4; i < m; ++i)
                {
                    wArr[i] = dInArr[i] + gIn_ * vInArr[i];
                }

                // Write the inner ring.
                ringIn_.writeBlock(vInArr, wIn_, m);
                ringIn_.refreshMirror(wIn_, m);
                wIn_ = (wIn_ + m) & maskIn;

                // Write the outer ring.
                ringOut_.writeBlock(wArr, wOut_, m);
                ringOut_.refreshMirror(wOut_, m);
                wOut_ = (wOut_ + m) & maskOut;

                // Compute y = d + g_out * v.
                for (int i = 0; i < m4; i += 4)
                {
                    const M128 dV = MM(load_ps)(dArr + i);
                    const M128 vV = MM(load_ps)(vArr + i);
                    const M128 yV = MM(add_ps)(dV, MM(mul_ps)(vGOut, vV));
                    MM(storeu_ps)(io + off + i, yV);
                }
                for (int i = m4; i < m; ++i)
                {
                    io[off + i] = dArr[i] + gOut_ * vArr[i];
                }
            }
        }

    private:
        void prepareImpl_(int dOut, int dIn, Memory::BumpArena *arena) noexcept
        {
            assert(dOut > kChunk);
            assert(dIn > kChunk);
            dOut_ = static_cast<float>(dOut);
            dIn_ = static_cast<float>(dIn);

            const int minCapOut = dOut + Delays::Pow2RingBuffer::kTail + 8;
            const int minCapIn = dIn + Delays::Pow2RingBuffer::kTail + 8;

            if (arena != nullptr)
            {
                ringOut_.prepare(minCapOut, *arena);
                ringIn_.prepare(minCapIn, *arena);
            }
            else
            {
                ringOut_.prepare(minCapOut);
                ringIn_.prepare(minCapIn);
            }
            reset();
        }

        Delays::Pow2RingBuffer ringOut_;
        Delays::Pow2RingBuffer ringIn_;
        int wOut_ = 0;
        int wIn_ = 0;
        float dOut_ = 0.0f;
        float dIn_ = 0.0f;
        float gOut_ = 0.0f;
        float gIn_ = 0.0f;

        alignas(16) std::array<float, static_cast<std::size_t>(kChunk) + Delays::Pow2RingBuffer::kTail> tapWinOut_{};
        alignas(16) std::array<float, static_cast<std::size_t>(kChunk) + Delays::Pow2RingBuffer::kTail> tapWinIn_{};
    };
}

#endif

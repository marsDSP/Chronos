#pragma once

#ifndef CHRONOS_SIMD_DELAY_LINE_H
#define CHRONOS_SIMD_DELAY_LINE_H

#include "DelayInterpolator.h"
#include "OnePoleSmoother.h"
#include "Pow2RingBuffer.h"
#include "simd/Config.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>

namespace MarsDSP::Delays
{
    class SimdDelayLine
    {
    public:
        static constexpr int kSubBlock = 16;
        static constexpr int kTail = Pow2RingBuffer::kTail; // 8
        static constexpr int kScratchLen = kSubBlock + kTail; // 24
        static constexpr int kGuard = 4;

        static constexpr double kDelaySmoothMs = 20.0;

        void prepare(double sampleRate, int maxBlockSize, float maxDelayMs) noexcept
        {
            assert(sampleRate > 0.0);
            assert(maxBlockSize > 0);
            assert(maxDelayMs > 0.0f);

            const auto fs = sampleRate > 0.0 ? sampleRate : 48000.0;
            const auto maxDelaySamples = static_cast<int>(std::ceil(static_cast<double>(maxDelayMs) * fs / 1000.0));
            const int blk = std::max(maxBlockSize, 1);

            const int raw = maxDelaySamples + blk + kTail + kGuard;
            const int capacityReq = std::max(raw, kScratchLen);
            const auto capacity = static_cast<int>(std::bit_ceil(static_cast<unsigned int>(capacityReq)));
            assert(capacity >= kScratchLen);

            bufL_.prepare(capacity);
            bufR_.prepare(capacity);
            maxBlockSize_ = blk;
            writeIdx_ = 0;

            posSmoother_.reset(sampleRate, kDelaySmoothMs * 0.001, kSubBlock);
            firstBlock_ = true;
        }

        void reset() noexcept
        {
            bufL_.clear();
            bufR_.clear();
            writeIdx_ = 0;
            firstBlock_ = true;
        }

        void setInterpolation(Interpolation mode) noexcept { mode_ = mode; }
        [[nodiscard]] Interpolation getInterpolation() const noexcept { return mode_; }
        [[nodiscard]] int getCapacity() const noexcept { return bufL_.getCapacity(); }
        [[nodiscard]] int getWriteIndex() const noexcept { return writeIdx_; }

        void process(const float *inL, const float *inR,
                     float *wetL, float *wetR,
                     int n,
                     float delayStartSamples, float delayEndSamples) noexcept
        {
            processImpl<kUseSimd>(inL, inR, wetL, wetR, n, delayStartSamples, delayEndSamples);
        }

        void processScalar(const float *inL, const float *inR,
                           float *wetL, float *wetR,
                           int n,
                           float delayStartSamples, float delayEndSamples) noexcept
        {
            processImpl<false>(inL, inR, wetL, wetR, n, delayStartSamples, delayEndSamples);
        }

    private:
        static constexpr bool kUseSimd = true;

        template<bool UseSimd>
        void processImpl(const float *inL, const float *inR,
                         float *wetL, float *wetR,
                         int n,
                         float delayStartSamples, float delayEndSamples) noexcept
        {
            assert(inL != nullptr);
            assert(wetL != nullptr);

            if (n <= 0) return;

            const int cap = bufL_.getCapacity();
            assert(n <= cap);
            const int mask = bufL_.mask();

            // ---- write-before-read ----
            const int blockStart = writeIdx_;
            const float *srcR = (inR != nullptr) ? inR : inL;
            bufL_.writeBlock(inL, blockStart, n);
            bufR_.writeBlock(srcR, blockStart, n);
            bufL_.refreshMirror(blockStart, n);
            bufR_.refreshMirror(blockStart, n);

            const auto  maxDelay = static_cast<float>(cap - maxBlockSize_ - kSubBlock - kTail - kGuard);
            const float dStart = std::clamp(delayStartSamples, 0.0f, maxDelay);
            const float dEnd = std::clamp(delayEndSamples, 0.0f, maxDelay);

            // ---- delay-position smoother setup ----
            if (firstBlock_)
            {
                posSmoother_.setCurrentAndTargetValue(dStart);
                firstBlock_ = false;
            }
            posSmoother_.setTargetValue(dEnd);

            const bool hasR = (wetR != nullptr);

            // ---- sub-block read loop ----
            int sampleOffset = 0;
            while (sampleOffset < n)
            {
                const int subN = std::min(kSubBlock, n - sampleOffset);

                const float posOld = posSmoother_.getCurrentValue();
                posSmoother_.processN(subN);
                const float posNew = posSmoother_.getCurrentValue();

                const auto iOld = static_cast<int>(std::floor(posOld));
                const float fOld = posOld - static_cast<float>(iOld);
                const auto iNew = static_cast<int>(std::floor(posNew));
                const float fNew = posNew - static_cast<float>(iNew);

                const Coeffs6 cOld = makeCoeffs(mode_, fOld);
                const Coeffs6 cNew = makeCoeffs(mode_, fNew);

                const int bOld = (blockStart + sampleOffset - iOld - 3) & mask;
                const int bNew = (blockStart + sampleOffset - iNew - 3) & mask;

                const int winLen = subN + kTail;
                bufL_.readWindow(scratchOldL_.data(), bOld, winLen);
                bufL_.readWindow(scratchNewL_.data(), bNew, winLen);

                if (hasR)
                {
                    bufR_.readWindow(scratchOldR_.data(), bOld, winLen);
                    bufR_.readWindow(scratchNewR_.data(), bNew, winLen);
                }

                const float invSubN = 1.0f / static_cast<float>(subN);

                if constexpr (UseSimd)
                {
                    const std::array<M128, 6> cbOld{{
                        MM(set1_ps)(cOld.c[0]), MM(set1_ps)(cOld.c[1]), MM(set1_ps)(cOld.c[2]),
                        MM(set1_ps)(cOld.c[3]), MM(set1_ps)(cOld.c[4]), MM(set1_ps)(cOld.c[5])
                    }};
                    const std::array<M128, 6> cbNew{{
                        MM(set1_ps)(cNew.c[0]), MM(set1_ps)(cNew.c[1]), MM(set1_ps)(cNew.c[2]),
                        MM(set1_ps)(cNew.c[3]), MM(set1_ps)(cNew.c[4]), MM(set1_ps)(cNew.c[5])
                    }};
                    const M128 laneOff = MM(mul_ps)(MM(set_ps)(3.0f, 2.0f, 1.0f, 0.0f), MM(set1_ps)(invSubN));

                    // Evaluate 4 consecutive output samples starting at j0 into dst.
                    auto eval4 = [&](float const *scratchOld, float const *scratchNew, float *dst, int j0)
                    {
                        const M128 vAlpha = MM(add_ps)(laneOff, MM(set1_ps)(static_cast<float>(j0) * invSubN));

                        M128 vOld = MM(setzero_ps)();
                        M128 vNew = MM(setzero_ps)();

                        for (int t = 0; t < 6; ++t)
                        {
                            const M128 wOld = (t == 0)
                                                  ? MM(load_ps)(scratchOld + j0)
                                                  : MM(loadu_ps)(scratchOld + j0 + t);
                            vOld = FMADD(wOld, cbOld[t], vOld);

                            const M128 wNew = (t == 0)
                                                  ? MM(load_ps)(scratchNew + j0)
                                                  : MM(loadu_ps)(scratchNew + j0 + t);
                            vNew = FMADD(wNew, cbNew[t], vNew);
                        }

                        const M128 vDelta = MM(sub_ps)(vNew, vOld);
                        const M128 vOut = FMADD(vAlpha, vDelta, vOld);
                        MM(storeu_ps)(dst, vOut);
                    };

                    const int jFull = subN & ~3; // largest multiple of 4 ≤ subN
                    for (int j0 = 0; j0 + 4 <= subN; j0 += 4)
                    {
                        eval4(scratchOldL_.data(), scratchNewL_.data(), wetL + sampleOffset + j0, j0);
                        if (hasR) eval4(scratchOldR_.data(), scratchNewR_.data(), wetR + sampleOffset + j0, j0);
                    }
                    // ---- scalar tail for the 0..3 remaining samples ----
                    for (int j = jFull; j < subN; ++j)
                    {
                        const float alpha = static_cast<float>(j) * invSubN;
                        const float yOldL = dot6(cOld, &scratchOldL_[j]);
                        const float yNewL = dot6(cNew, &scratchNewL_[j]);
                        wetL[sampleOffset + j] = (1.0f - alpha) * yOldL + alpha * yNewL;
                        if (hasR)
                        {
                            const float yOldR = dot6(cOld, &scratchOldR_[j]);
                            const float yNewR = dot6(cNew, &scratchNewR_[j]);
                            wetR[sampleOffset + j] = (1.0f - alpha) * yOldR + alpha * yNewR;
                        }
                    }
                } else
                {
                    for (int j = 0; j < subN; ++j)
                    {
                        const float alpha = static_cast<float>(j) * invSubN;
                        const float yOldL = dot6(cOld, &scratchOldL_[j]);
                        const float yNewL = dot6(cNew, &scratchNewL_[j]);
                        wetL[sampleOffset + j] = (1.0f - alpha) * yOldL + alpha * yNewL;
                        if (hasR)
                        {
                            const float yOldR = dot6(cOld, &scratchOldR_[j]);
                            const float yNewR = dot6(cNew, &scratchNewR_[j]);
                            wetR[sampleOffset + j] = (1.0f - alpha) * yOldR + alpha * yNewR;
                        }
                    }
                }
                sampleOffset += subN;
            }
            // ---- advance the write index ----
            writeIdx_ = (blockStart + n) & mask;
        }

        static float dot6(const Coeffs6 &c, const float *w) noexcept
        {
            return c.c[0] * w[0] + c.c[1] * w[1] + c.c[2] * w[2]
                 + c.c[3] * w[3] + c.c[4] * w[4] + c.c[5] * w[5];
        }

        Pow2RingBuffer bufL_;
        Pow2RingBuffer bufR_;
        int writeIdx_ = 0;
        int maxBlockSize_ = 0;
        Interpolation mode_ = Interpolation::Lagrange5th;
        Smoothers::OnePoleSmoother<float> posSmoother_;
        bool firstBlock_ = true;

        alignas(16) std::array<float, kScratchLen> scratchOldL_{};
        alignas(16) std::array<float, kScratchLen> scratchNewL_{};
        alignas(16) std::array<float, kScratchLen> scratchOldR_{};
        alignas(16) std::array<float, kScratchLen> scratchNewR_{};
    };
}
#endif

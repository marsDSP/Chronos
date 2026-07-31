#pragma once

#ifndef CHRONOS_SIMD_DELAY_LINE_H
#define CHRONOS_SIMD_DELAY_LINE_H

#include "BlockTapReader.h"
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
            maxDelaySamples_ = maxDelaySamples;
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
        [[nodiscard]] int getMaxDelaySamples() const noexcept { return maxDelaySamples_; }

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

                const auto wOldL = BlockTapReader::acquireWindow(bufL_, bOld, winLen, scratchOldL_.data());
                const auto wNewL = BlockTapReader::acquireWindow(bufL_, bNew, winLen, scratchNewL_.data());
                const float* oldL = wOldL.ptr;
                const float* newL = wNewL.ptr;
                const bool  oldLAligned = wOldL.aligned;
                const bool  newLAligned = wNewL.aligned;

                const float* oldR = nullptr;
                const float* newR = nullptr;
                bool oldRAligned = false;
                bool newRAligned = false;
                if (hasR)
                {
                    const auto wOldR = BlockTapReader::acquireWindow(bufR_, bOld, winLen, scratchOldR_.data());
                    const auto wNewR = BlockTapReader::acquireWindow(bufR_, bNew, winLen, scratchNewR_.data());
                    oldR = wOldR.ptr;  newR = wNewR.ptr;
                    oldRAligned = wOldR.aligned; newRAligned = wNewR.aligned;
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

                    const int jFull = subN & ~3; // largest multiple of 4 ≤ subN
                    for (int j0 = 0; j0 + 4 <= subN; j0 += 4)
                    {
                        const M128 vAlpha = MM(add_ps)(laneOff, MM(set1_ps)(static_cast<float>(j0) * invSubN));
                        BlockTapReader::eval4(oldL + j0, oldLAligned, newL + j0, newLAligned,
                                              cbOld, cbNew, vAlpha, wetL + sampleOffset + j0);
                        if (hasR)
                            BlockTapReader::eval4(oldR + j0, oldRAligned, newR + j0, newRAligned,
                                                  cbOld, cbNew, vAlpha, wetR + sampleOffset + j0);
                    }
                    // ---- scalar tail for the 0..3 remaining samples ----
                    for (int j = jFull; j < subN; ++j)
                    {
                        const float alpha = static_cast<float>(j) * invSubN;
                        const float yOldL = BlockTapReader::dot6(cOld, oldL + j);
                        const float yNewL = BlockTapReader::dot6(cNew, newL + j);
                        wetL[sampleOffset + j] = (1.0f - alpha) * yOldL + alpha * yNewL;
                        if (hasR)
                        {
                            const float yOldR = BlockTapReader::dot6(cOld, oldR + j);
                            const float yNewR = BlockTapReader::dot6(cNew, newR + j);
                            wetR[sampleOffset + j] = (1.0f - alpha) * yOldR + alpha * yNewR;
                        }
                    }
                } else
                {
                    for (int j = 0; j < subN; ++j)
                    {
                        const float alpha = static_cast<float>(j) * invSubN;
                        const float yOldL = BlockTapReader::dot6(cOld, oldL + j);
                        const float yNewL = BlockTapReader::dot6(cNew, newL + j);
                        wetL[sampleOffset + j] = (1.0f - alpha) * yOldL + alpha * yNewL;
                        if (hasR)
                        {
                            const float yOldR = BlockTapReader::dot6(cOld, oldR + j);
                            const float yNewR = BlockTapReader::dot6(cNew, newR + j);
                            wetR[sampleOffset + j] = (1.0f - alpha) * yOldR + alpha * yNewR;
                        }
                    }
                }
                sampleOffset += subN;
            }
            // ---- advance the write index ----
            writeIdx_ = (blockStart + n) & mask;
        }

        Pow2RingBuffer bufL_;
        Pow2RingBuffer bufR_;
        int writeIdx_ = 0;
        int maxBlockSize_ = 0;
        int maxDelaySamples_ = 0;
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

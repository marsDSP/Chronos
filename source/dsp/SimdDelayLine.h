#pragma once

#ifndef CHRONOS_SIMD_DELAY_LINE_H
#define CHRONOS_SIMD_DELAY_LINE_H

#include "DelayInterpolator.h"
#include "OnePoleSmoother.h"
#include "Pow2RingBuffer.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>

namespace MarsDSP::Delays {
    class SimdDelayLine {
    public:
        static constexpr int kSubBlock   = 16;
        static constexpr int kTail       = Pow2RingBuffer::kTail;   // 8
        static constexpr int kScratchLen = kSubBlock + kTail;        // 24
        static constexpr int kGuard      = 4;

        // smoothing time constant. Matches the 20 ms ramp used by other smoothers
        // the one-pole gives an exponential (asymptotic) trajectory
        static constexpr double kDelaySmoothMs = 20.0;

        void prepare(double sampleRate, int maxBlockSize, float maxDelayMs) noexcept
        {
            assert(sampleRate > 0.0);
            assert(maxBlockSize > 0);
            assert(maxDelayMs > 0.0f);

            const auto fs = sampleRate > 0.0 ? sampleRate : 48000.0;
            const int maxDelaySamples = static_cast<int>(std::ceil(static_cast<double>(maxDelayMs) * fs / 1000.0));
            const int blk = std::max(maxBlockSize, 1);

            const int raw = maxDelaySamples + blk + kTail + kGuard;
            const int capacityReq = std::max(raw, kScratchLen);
            const auto capacity = static_cast<int>(std::bit_ceil(static_cast<unsigned int>(capacityReq)));
            assert(capacity >= kScratchLen);

            bufL_.prepare(capacity);
            bufR_.prepare(capacity);
            maxBlockSize_ = blk;
            writeIdx_ = 0;

            // Prepare the delay-position smoother: 20 ms one-pole, with the
            // (1-alpha)^kSubBlock cache so the common sub-block advance via
            // processN(kSubBlock) costs no std::pow.
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

            const float maxDelay = static_cast<float>(cap - maxBlockSize_ - kSubBlock - kTail - kGuard);
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

                const int   iOld = static_cast<int>(std::floor(posOld));
                const float fOld = posOld - static_cast<float>(iOld);
                const int   iNew = static_cast<int>(std::floor(posNew));
                const float fNew = posNew - static_cast<float>(iNew);

                const Coeffs6 cOld = makeCoeffs(mode_, fOld);
                const Coeffs6 cNew = makeCoeffs(mode_, fNew);

                const int bOld = (blockStart + sampleOffset - iOld - 3) & mask;
                const int bNew = (blockStart + sampleOffset - iNew - 3) & mask;

                const int winLen = subN + kTail;
                bufL_.readWindow(scratchOldL_, bOld, winLen);
                bufL_.readWindow(scratchNewL_, bNew, winLen);

                if (hasR)
                {
                    bufR_.readWindow(scratchOldR_, bOld, winLen);
                    bufR_.readWindow(scratchNewR_, bNew, winLen);
                }

                const float invSubN = 1.0f / static_cast<float>(subN);
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
                sampleOffset += subN;
            }
            // ---- advance the write index ----
            writeIdx_ = (blockStart + n) & mask;
        }

    private:
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

        alignas(16) float scratchOldL_[kScratchLen] = {};
        alignas(16) float scratchNewL_[kScratchLen] = {};
        alignas(16) float scratchOldR_[kScratchLen] = {};
        alignas(16) float scratchNewR_[kScratchLen] = {};
    };
}
#endif

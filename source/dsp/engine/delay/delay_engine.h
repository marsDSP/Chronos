#pragma once

#ifndef CHRONOS_DELAY_ENGINE_H
#define CHRONOS_DELAY_ENGINE_H

#include <cassert>
#include <JuceHeader.h>
#include "dsp/math/fastermath.h"

namespace MarsDSP::DSP {
    template<typename SampleType, int N_BLOCK = 4112>
    class DelayEngine {
    public:
        DelayEngine() = default;

        void AllocBuffer() noexcept
        {
            if (bufferL.size() < static_cast<size_t>(kBufSize + kTail))
            {
                bufferL.assign(static_cast<size_t>(kBufSize + kTail), SampleType(0));
                bufferR.assign(static_cast<size_t>(kBufSize + kTail), SampleType(0));
            }
        }

        struct LagrangeCoeffs
        {
            SampleType c[6];
            SampleType frac;
        };

        static SampleType readInterpolated(const SampleType *buf, int readIdx, const LagrangeCoeffs& coeffs) noexcept
        {
            return buf[readIdx] * coeffs.c[0] + coeffs.frac * (buf[readIdx + 1] * coeffs.c[1] +
                                                               buf[readIdx + 2] * coeffs.c[2] +
                                                               buf[readIdx + 3] * coeffs.c[3] +
                                                               buf[readIdx + 4] * coeffs.c[4] +
                                                               buf[readIdx + 5] * coeffs.c[5]);
        }

        void updateDuckGain(SampleType modspeed) noexcept
        {
            constexpr auto duckFloor = static_cast<SampleType>(0.08);

            constexpr auto duckSens  = static_cast<SampleType>(0.20);

            const SampleType desired = std::clamp(static_cast<SampleType>(1) /
                                                 (static_cast<SampleType>(1) + duckSens * modspeed),
                                                 duckFloor, static_cast<SampleType>(1));

            const SampleType coeff = desired < duckGain ? duckAtkCoeff : duckRelCoeff;
            duckGain += coeff * (desired - duckGain);
        }

        void reset() noexcept
        {
            writeIdxL = writeIdxR = 0;
            prevPos   = SampleType(0);
            duckGain  = SampleType(1);
            if (!bufferL.empty()) std::fill(bufferL.begin(), bufferL.end(), SampleType(0));
            if (!bufferR.empty()) std::fill(bufferR.begin(), bufferR.end(), SampleType(0));
        }

        void prepare(const dsp::ProcessSpec &spec) noexcept
        {
            sampleRate = spec.sampleRate;
            AllocBuffer();

            // 10ms attack, 100ms release for ducking response
            duckAtkCoeff = static_cast<SampleType>(1.0 - std::exp(-1.0 / (0.010 * sampleRate)));
            duckRelCoeff = static_cast<SampleType>(1.0 - std::exp(-1.0 / (0.100 * sampleRate)));

            reset();
        }

        void process(const dsp::AudioBlock<SampleType> &block, const int numSamples) noexcept
        {
            if (bypassed)
                return;

            const size_t numCh = block.getNumChannels();
            auto *ch0 = numCh > 0 ? block.getChannelPointer(0) : nullptr;
            auto *ch1 = numCh > 1 ? block.getChannelPointer(1) : nullptr;

            const auto delayMsParam = static_cast<SampleType>(std::clamp(delayTime, minDelayTime, maxDelayTime));
            const auto delayMsToSamples = static_cast<SampleType>(sampleRate * (delayMsParam * 0.001f));

            const SampleType mixParam = static_cast<SampleType>(std::clamp(mix, 0.0f, 1.0f));
            const SampleType fbLParam = static_cast<SampleType>(std::clamp(feedbackL, 0.0f, 0.99f));
            const SampleType fbRParam = static_cast<SampleType>(std::clamp(feedbackR, 0.0f, 0.99f));

            const SampleType currentPos = delayMsToSamples;
            const size_t numSamplesSize = static_cast<size_t>(numSamples);
            assert(numSamplesSize <= N_BLOCK - 8);

            const SampleType pos = std::max(currentPos, static_cast<SampleType>(numSamplesSize + 1));
            const int delayInt = static_cast<int>(std::floor(static_cast<double>(pos)));
            const SampleType delayFrac = pos - static_cast<SampleType>(delayInt);
            int offset = delayInt;

            // Pre-read from circular buffer into linear scratch (tL/tR) using memcpys
            const int total = static_cast<int>(numSamplesSize) + kTail;
            
            const int rposL = (writeIdxL - offset) & kBufMask;
            const int firstL = std::min(total, kBufSize + kTail - rposL);
            std::memcpy(tL, &bufferL[rposL], firstL * sizeof(SampleType));
            if (firstL < total)
                std::memcpy(tL + firstL, &bufferL[kTail], (total - firstL) * sizeof(SampleType));

            if (!isMono() && ch1 != nullptr)
            {
                const int rposR = (writeIdxR - offset) & kBufMask;
                const int firstR = std::min(total, kBufSize + kTail - rposR);
                std::memcpy(tR, &bufferR[rposR], firstR * sizeof(SampleType));
                if (firstR < total)
                    std::memcpy(tR + firstR, &bufferR[kTail], (total - firstR) * sizeof(SampleType));
            }

            LagrangeCoeffs coeffs;
            coeffs.frac = delayFrac;
            {
                const float d1 = delayFrac - 1.0f;
                const float d2 = delayFrac - 2.0f;
                const float d3 = delayFrac - 3.0f;
                const float d4 = delayFrac - 4.0f;
                const float d5 = delayFrac - 5.0f;
                coeffs.c[0] = -d1 * d2 * d3 * d4 * d5 / 120.0f;
                coeffs.c[1] = d2 * d3 * d4 * d5 / 24.0f;
                coeffs.c[2] = -d1 * d3 * d4 * d5 / 12.0f;
                coeffs.c[3] = d1 * d2 * d4 * d5 / 12.0f;
                coeffs.c[4] = -d1 * d2 * d3 * d5 / 24.0f;
                coeffs.c[5] = d1 * d2 * d3 * d4 / 120.0f;
            }

            const auto vC1 = SIMD_MM(set1_ps)(coeffs.c[0]);
            const auto vC2 = SIMD_MM(set1_ps)(coeffs.c[1]);
            const auto vC3 = SIMD_MM(set1_ps)(coeffs.c[2]);
            const auto vC4 = SIMD_MM(set1_ps)(coeffs.c[3]);
            const auto vC5 = SIMD_MM(set1_ps)(coeffs.c[4]);
            const auto vC6 = SIMD_MM(set1_ps)(coeffs.c[5]);
            const auto vFrac = SIMD_MM(set1_ps)(delayFrac);

            const SampleType modspeed = std::abs(currentPos - prevPos);
            prevPos = currentPos;
            updateDuckGain(modspeed);
            const auto vDuckGain = SIMD_MM(set1_ps)(duckGain);

            if (isMono()) // mono
            {
                const SampleType feedbackParam = fbLParam;
                const SampleType oneMinusMix = static_cast<SampleType>(1) - mixParam;

                size_t n = 0;
                // vectorized block processing
                for (; n + 3 < numSamplesSize; n += 4)
                {
                    auto vMonoSum = SIMD_MM(setzero_ps)();
                    if (ch0 != nullptr && ch1 != nullptr)
                    {
                        vMonoSum = SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(0.5f), SIMD_MM(add_ps)(SIMD_MM(loadu_ps)(ch0 + n),
                                                                                           SIMD_MM(loadu_ps)(ch1 + n)));
                    }
                    else if (ch0 != nullptr)
                    {
                        vMonoSum = SIMD_MM(loadu_ps)(ch0 + n);
                    }
                    else if (ch1 != nullptr)
                    {
                        vMonoSum = SIMD_MM(loadu_ps)(ch1 + n);
                    }

                    // Vectorized interpolation for mono
                    SIMD_M128 vDelayedOut;
                    
                    // No branch needed!
                    auto v0 = SIMD_MM(load_ps)(tL + n);
                    auto v1 = SIMD_MM(loadu_ps)(tL + n + 1);
                    auto v2 = SIMD_MM(loadu_ps)(tL + n + 2);
                    auto v3 = SIMD_MM(loadu_ps)(tL + n + 3);
                    auto v4 = SIMD_MM(loadu_ps)(tL + n + 4);
                    auto v5 = SIMD_MM(loadu_ps)(tL + n + 5);

                    auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2),
                                SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3),
                                SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4),
                                SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5),
                                                SIMD_MM(mul_ps)(v5, vC6)))));

                    vDelayedOut = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1),
                                                  SIMD_MM(mul_ps)(vFrac, vSum));
                    
                    auto vDuckedOut = SIMD_MM(mul_ps)(vDelayedOut, vDuckGain);

                    // feedback-mix in SIMD
                    auto vWriteVal = fasterTanhBounded(SIMD_MM(add_ps)(vMonoSum, SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(feedbackParam), vDuckedOut)));
                    SIMD_MM(storeu_ps)(&wL[n], vWriteVal);

                    auto vOut = fasterTanhBounded(SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vDuckedOut, SIMD_MM(set1_ps)(mixParam)),
                                                                                       SIMD_MM(mul_ps)(vMonoSum, SIMD_MM(set1_ps)(oneMinusMix))));

                    if (ch0 != nullptr) SIMD_MM(storeu_ps)(ch0 + n, vOut);
                    if (ch1 != nullptr) SIMD_MM(storeu_ps)(ch1 + n, vOut);
                }

                // remainder loop
                for (; n < numSamplesSize; ++n)
                {
                    SampleType monoSum = ch0 != nullptr ? ch0[n] : static_cast<SampleType>(0);
                    if (ch1 != nullptr)
                        monoSum = static_cast<SampleType>(0.5) * (monoSum + ch1[n]);

                    const SampleType delayedOut = readInterpolated(tL, (int)n, coeffs);
                    const SampleType duckedOut = delayedOut * duckGain;

                    wL[n] = softClip(monoSum + feedbackParam * duckedOut);

                    const SampleType out = softClip(duckedOut * mixParam + monoSum * oneMinusMix);

                    if (ch0 != nullptr) ch0[n] = out;
                    if (ch1 != nullptr) ch1[n] = out;
                }

                // Block write & Mirror
                const bool wrapped = (writeIdxL + (int)numSamplesSize) > kBufSize;
                if (wrapped)
                {
                    for (size_t k = 0; k < numSamplesSize; ++k)
                    {
                        const int idx = (writeIdxL + (int)k) & kBufMask;
                        bufferL[idx] = wL[k];
                        bufferR[idx] = wL[k];
                    }
                }
                else
                {
                    std::memcpy(&bufferL[writeIdxL], wL, numSamplesSize * sizeof(SampleType));
                    std::memcpy(&bufferR[writeIdxL], wL, numSamplesSize * sizeof(SampleType));
                }

                const int oldIdx = writeIdxL;
                writeIdxL = (oldIdx + static_cast<int>(numSamplesSize)) & kBufMask;
                writeIdxR = writeIdxL;
                if (wrapped || oldIdx < kTail)
                {
                    for (int k = 0; k < kTail; ++k)
                    {
                        bufferL[kBufSize + k] = bufferL[k];
                        bufferR[kBufSize + k] = bufferR[k];
                    }
                }
            }
            else // stereo
            {
                const SampleType oneMinusMix = static_cast<SampleType>(1) - mixParam;

                size_t n = 0;
                for (; n + 3 < numSamplesSize; n += 4)
                {
                    // Left channel
                    if (ch0 != nullptr)
                    {
                        auto vXL = SIMD_MM(loadu_ps)(ch0 + n);
                        
                        auto v0 = SIMD_MM(load_ps)(tL + n);
                        auto v1 = SIMD_MM(loadu_ps)(tL + n + 1);
                        auto v2 = SIMD_MM(loadu_ps)(tL + n + 2);
                        auto v3 = SIMD_MM(loadu_ps)(tL + n + 3);
                        auto v4 = SIMD_MM(loadu_ps)(tL + n + 4);
                        auto v5 = SIMD_MM(loadu_ps)(tL + n + 5);

                        auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5),
                                                    SIMD_MM(mul_ps)(v5, vC6)))));

                        auto vYL = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1),
                                                   SIMD_MM(mul_ps)(vFrac, vSum));
                        
                        auto vYL_ducked = SIMD_MM(mul_ps)(vYL, vDuckGain);
                        auto vWriteValL = fasterTanhBounded(SIMD_MM(add_ps)(vXL, SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(fbLParam), vYL_ducked)));
                        SIMD_MM(storeu_ps)(&wL[n], vWriteValL);

                        auto vOutL = fasterTanhBounded(SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vYL_ducked, SIMD_MM(set1_ps)(mixParam)),
                                                                                           SIMD_MM(mul_ps)(vXL, SIMD_MM(set1_ps)(oneMinusMix))));
                        SIMD_MM(storeu_ps)(ch0 + n, vOutL);
                    }

                    // Right channel
                    if (ch1 != nullptr)
                    {
                        auto vXR = SIMD_MM(loadu_ps)(ch1 + n);
                        
                        auto v0 = SIMD_MM(load_ps)(tR + n);
                        auto v1 = SIMD_MM(loadu_ps)(tR + n + 1);
                        auto v2 = SIMD_MM(loadu_ps)(tR + n + 2);
                        auto v3 = SIMD_MM(loadu_ps)(tR + n + 3);
                        auto v4 = SIMD_MM(loadu_ps)(tR + n + 4);
                        auto v5 = SIMD_MM(loadu_ps)(tR + n + 5);

                        auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5),
                                                    SIMD_MM(mul_ps)(v5, vC6)))));

                        auto vYR = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1),
                                                   SIMD_MM(mul_ps)(vFrac, vSum));
                        
                        auto vYR_ducked = SIMD_MM(mul_ps)(vYR, vDuckGain);
                        auto vWriteValR = fasterTanhBounded(SIMD_MM(add_ps)(vXR, SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(fbRParam), vYR_ducked)));
                        SIMD_MM(storeu_ps)(&wR[n], vWriteValR);

                        auto vOutR = fasterTanhBounded(SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vYR_ducked, SIMD_MM(set1_ps)(mixParam)),
                                                                                           SIMD_MM(mul_ps)(vXR, SIMD_MM(set1_ps)(oneMinusMix))));
                        SIMD_MM(storeu_ps)(ch1 + n, vOutR);
                    }
                }

                // remainder loop
                for (; n < numSamplesSize; ++n)
                {
                    if (ch0 != nullptr)
                    {
                        const SampleType xL = ch0[n];
                        const SampleType yL_ducked = readInterpolated(tL, (int)n, coeffs) * duckGain;
                        wL[n] = softClip(xL + fbLParam * yL_ducked);
                        ch0[n] = softClip(yL_ducked * mixParam + xL * oneMinusMix);
                    }

                    if (ch1 != nullptr)
                    {
                        const SampleType xR = ch1[n];
                        const SampleType yR_ducked = readInterpolated(tR, (int)n, coeffs) * duckGain;
                        wR[n] = softClip(xR + fbRParam * yR_ducked);
                        ch1[n] = softClip(yR_ducked * mixParam + xR * oneMinusMix);
                    }
                }

                // Block write & Mirror L
                if (ch0 != nullptr) {
                    const int oldIdxL = writeIdxL;
                    const bool wrappedL = (oldIdxL + (int)numSamplesSize) > kBufSize;
                    if (wrappedL) {
                        for (size_t k = 0; k < numSamplesSize; ++k)
                            bufferL[(oldIdxL + (int)k) & kBufMask] = wL[k];
                    } else {
                        std::memcpy(&bufferL[oldIdxL], wL, numSamplesSize * sizeof(SampleType));
                    }
                    writeIdxL = (oldIdxL + (int)numSamplesSize) & kBufMask;
                    if (wrappedL || oldIdxL < kTail) {
                        for (int k = 0; k < kTail; ++k)
                            bufferL[kBufSize + k] = bufferL[k];
                    }
                }

                // Block write & Mirror R
                if (ch1 != nullptr) {
                    const int oldIdxR = writeIdxR;
                    const bool wrappedR = (oldIdxR + (int)numSamplesSize) > kBufSize;
                    if (wrappedR) {
                        for (size_t k = 0; k < numSamplesSize; ++k)
                            bufferR[(oldIdxR + (int)k) & kBufMask] = wR[k];
                    } else {
                        std::memcpy(&bufferR[oldIdxR], wR, numSamplesSize * sizeof(SampleType));
                    }
                    writeIdxR = (oldIdxR + (int)numSamplesSize) & kBufMask;
                    if (wrappedR || oldIdxR < kTail) {
                        for (int k = 0; k < kTail; ++k)
                            bufferR[kBufSize + k] = bufferR[k];
                    }
                }
            }
        }

        void setDelayTimeParam(const float milliseconds) noexcept
        {
            delayTime = milliseconds;
        }

        void setMixParam(const float value) noexcept
        {
            mix = std::clamp(value, 0.0f, 1.0f);
        }

        void setMixPercentage(const float value) noexcept
        {
            mix = std::clamp(value * 0.01f, 0.0f, 1.0f);
        }

        void setFeedbackParam(const float value) noexcept
        {
            const float fb = std::clamp(value, 0.0f, 0.99f);
            feedbackL = fb;
            feedbackR = fb;
        }

        void setBypassed(const bool shouldBypass) noexcept
        {
            bypassed = shouldBypass;
        }

        [[nodiscard]] bool isBypassed() const noexcept
        {
            return bypassed;
        }

        void setMono(const bool shouldBeMono) noexcept
        {
            mono = shouldBeMono;
        }

        [[nodiscard]] bool isMono() const noexcept
        {
            return mono;
        }

    private:
        SampleType softClip(SampleType x) noexcept
        {
            return fasterTanhBounded(x);
        }

        std::vector<SampleType> bufferL, bufferR;
        // scratch buffers are thread_local to avoid per-instance allocation while remaining thread-safe.
        // Assumes no re-entrant process() on the same thread (e.g. sidechain feedback loops).
        alignas(16) static thread_local inline float tL[N_BLOCK];
        alignas(16) static thread_local inline float tR[N_BLOCK];
        alignas(16) static thread_local inline float wL[N_BLOCK];
        alignas(16) static thread_local inline float wR[N_BLOCK];

        double sampleRate = 44100.0;

        SampleType prevPos = static_cast<SampleType>(0);
        SampleType duckGain = static_cast<SampleType>(1);
        SampleType duckAtkCoeff = static_cast<SampleType>(1);
        SampleType duckRelCoeff = static_cast<SampleType>(1);

        static constexpr float minDelayTime = 5.0f;
        static constexpr float maxDelayTime = 5000.0f;

        float mix = 1.0f;
        float feedbackL = 0.0f;
        float feedbackR = 0.0f;
        float delayTime = 50.0f;

        // allocates a fixed 262,144-sample buffer, saves the clock cycles from '%' and '/'
        // 1 << 18 = 262,144 samples, which at 44.1 kHz gives ~5.9 seconds of delay
        // (1 << 18) - 1 = 0x3FFFF = 0b0011'1111'1111'1111'1111
        // at sizeof(float) that's ~1 MB per channel give or take
        // clamp time param to (maxDelaySamples - 1) to avoid outside buffer reads
        static constexpr int kBufSize = 1 << 18;
        static constexpr int kBufMask = kBufSize - 1;
        static constexpr int kTail    = 8;                  // for 5th-order Lagrange window

        int writeIdxL = 0;
        int writeIdxR = 0;

        bool mono = false;
        bool bypassed = false;
    };
}
#endif
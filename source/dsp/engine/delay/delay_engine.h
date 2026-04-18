#pragma once

#ifndef CHRONOS_DELAY_ENGINE_H
#define CHRONOS_DELAY_ENGINE_H

#include <cassert>
#include <JuceHeader.h>
#include "dsp/math/fastermath.h"
#include "delay_interpolator.h"

namespace MarsDSP::DSP {
    template<typename SampleType>
    class DelayEngine {
    public:
        DelayEngine() = default;

        // should probably template this so it can allocate for 3rd & 5th polynomial interp
        void AllocBuffer(int maxLengthInSamples) noexcept
        {
            assert(maxLengthInSamples > 0);

            // adding the padding here for 5th order polynomial interp prevents me
            // from using bitwise operations to save cycles on the modulo wrap around.
            // should probably look into how to make the optimization work with this.
            int paddedLength = maxLengthInSamples + 8;
            if (bufferLength < paddedLength)
            {
                bufferLength = paddedLength;
                bufferL.assign(static_cast<size_t>(bufferLength), SampleType(0));
                bufferR.assign(static_cast<size_t>(bufferLength), SampleType(0));
            }
        }

        void writeSample(SampleType *buf, int &idx, SampleType input) noexcept
        {
            assert(bufferLength > 0);
            buf[static_cast<size_t>(idx)] = input;
            idx += 1;
            if (idx >= bufferLength)
                idx = 0;
        }

        SampleType readInterpolated(const SampleType *buf, const int idx, SampleType delaySamples, Lagrange5th &interp) const noexcept
        {
            if (bufferLength <= 8)
                return static_cast<SampleType>(0);

            // clamp to valid range
            const auto maxDelay = static_cast<SampleType>(bufferLength - 8);
            SampleType delayMaxSamples = std::clamp(delaySamples, static_cast<SampleType>(0), maxDelay);

            int delayInt = static_cast<int>(std::floor(static_cast<double>(delayMaxSamples)));
            SampleType delayFrac = delayMaxSamples - static_cast<SampleType>(delayInt);

            // adjust for 5th-order lagrange internal alignment
            int offset = delayInt;
            interp.write(offset, delayFrac);

            // starting index for 6smp wrap around window
            int start = idx - offset;
            while (start < 0)
                start += bufferLength;

            if (start >= bufferLength)
                start %= bufferLength;

            // copy 6 consecutive samples into a small local buffer
            SampleType window[6];
            for (int k = 0; k < 6; ++k)
            {
                int pos = start + k;
                if (pos >= bufferLength)
                    pos -= bufferLength;

                window[k] = buf[static_cast<size_t>(pos)];
            }
            // call interpolator with local window
            // delayInt is zero because window starts at the base index.
            return interp.read<SampleType, SampleType, SampleType>(window, 0, delayFrac);
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
            writeIdxL = 0;
            writeIdxR = 0;

            if (!bufferL.empty())
            {
                std::fill(bufferL.begin(), bufferL.end(), static_cast<SampleType>(0));
            }

            if (!bufferR.empty())
            {
                std::fill(bufferR.begin(), bufferR.end(), static_cast<SampleType>(0));
            }
        }

        void prepare(const dsp::ProcessSpec &spec) noexcept
        {
            constexpr double maxSamples = maxDelaySamples - 1;
            const int maxCapacity = static_cast<int>(std::ceil(maxSamples));
            AllocBuffer(maxCapacity);
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
            // Document SIMD-safety: block size is 4, so delay must be at least 4 samples to avoid read/write overlap in the same vectorized block.
            assert(currentPos >= 4.0f);

            // Precompute Lagrange 5th order coefficients once for the entire block
            int delayInt = static_cast<int>(std::floor(static_cast<double>(currentPos)));
            SampleType delayFrac = currentPos - static_cast<SampleType>(delayInt);
            int offset = delayInt;
            lagrange5thL.write(offset, delayFrac); // Adjust offset/frac for 5th order alignment

            const float d1 = delayFrac - 1.0f;
            const float d2 = delayFrac - 2.0f;
            const float d3 = delayFrac - 3.0f;
            const float d4 = delayFrac - 4.0f;
            const float d5 = delayFrac - 5.0f;

            const auto vC1 = SIMD_MM(set1_ps)(-d1 * d2 * d3 * d4 * d5 / 120.0f);
            const auto vC2 = SIMD_MM(set1_ps)(d2 * d3 * d4 * d5 / 24.0f);
            const auto vC3 = SIMD_MM(set1_ps)(-d1 * d3 * d4 * d5 / 12.0f);
            const auto vC4 = SIMD_MM(set1_ps)(d1 * d2 * d4 * d5 / 12.0f);
            const auto vC5 = SIMD_MM(set1_ps)(-d1 * d2 * d3 * d5 / 24.0f);
            const auto vC6 = SIMD_MM(set1_ps)(d1 * d2 * d3 * d4 / 120.0f);
            const auto vFrac = SIMD_MM(set1_ps)(delayFrac);

            if (isMono()) // mono
            {
                const SampleType feedbackParam = fbLParam;
                const SampleType oneMinusMix = static_cast<SampleType>(1) - mixParam;

                size_t n = 0;
                const size_t numSamplesSize = static_cast<size_t>(numSamples);

                // vectorized block processing
                for (; n + 3 < numSamplesSize; n += 4)
                {
                    float duckGains[4];
                    for (int i = 0; i < 4; ++i)
                    {
                        const SampleType modspeed = std::abs(currentPos - prevPos);
                        prevPos = currentPos;
                        updateDuckGain(modspeed);
                        duckGains[i] = duckGain;
                    }

                    auto vMonoSum = SIMD_MM(setzero_ps)();
                    if (ch0 != nullptr && ch1 != nullptr)
                    {
                        vMonoSum = SIMD_MM(mul_ps)(SIMD_MM(set_ps1)(0.5f), SIMD_MM(add_ps)(SIMD_MM(loadu_ps)(ch0 + n),
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
                    int start0 = (writeIdxL - offset + bufferLength) % bufferLength;
                    SIMD_M128 vDelayedOut;
                    if (start0 + 8 < bufferLength)
                    {
                        auto v0 = SIMD_MM(loadu_ps)(bufferL.data() + start0);
                        auto v1 = SIMD_MM(loadu_ps)(bufferL.data() + start0 + 1);
                        auto v2 = SIMD_MM(loadu_ps)(bufferL.data() + start0 + 2);
                        auto v3 = SIMD_MM(loadu_ps)(bufferL.data() + start0 + 3);
                        auto v4 = SIMD_MM(loadu_ps)(bufferL.data() + start0 + 4);
                        auto v5 = SIMD_MM(loadu_ps)(bufferL.data() + start0 + 5);

                        auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4),
                                    SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5),
                                                    SIMD_MM(mul_ps)(v5, vC6)))));

                        vDelayedOut = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1),
                                                      SIMD_MM(mul_ps)(vFrac, vSum));
                    }
                    else
                    {
                        float delayedOuts[4];
                        for (int i = 0; i < 4; ++i)
                        {
                            const int rIdx = (writeIdxL + i + bufferLength) % bufferLength;
                            delayedOuts[i] = readInterpolated(bufferL.data(), rIdx, currentPos, lagrange5thL);
                        }
                        vDelayedOut = SIMD_MM(loadu_ps)(delayedOuts);
                    }
                    auto vDuckGain = SIMD_MM(loadu_ps)(duckGains);
                    auto vDuckedOut = SIMD_MM(mul_ps)(vDelayedOut, vDuckGain);

                    // feedback-mix in SIMD
                    auto vWriteVal = fasterTanhBounded(SIMD_MM(add_ps)(vMonoSum, SIMD_MM(mul_ps)(SIMD_MM(set_ps1)(feedbackParam), vDuckedOut)));

                    float writeVals[4];
                    SIMD_MM(storeu_ps)(writeVals, vWriteVal);

                    for (int i = 0; i < 4; ++i)
                        writeSample(bufferL.data(), writeIdxL, writeVals[i]);

                    auto vOut = fasterTanhBounded(SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vDuckedOut, SIMD_MM(set_ps1)(mixParam)),
                                                           SIMD_MM(mul_ps)(vMonoSum, SIMD_MM(set_ps1)(oneMinusMix))));

                    if (ch0 != nullptr) SIMD_MM(storeu_ps)(ch0 + n, vOut);
                    if (ch1 != nullptr) SIMD_MM(storeu_ps)(ch1 + n, vOut);
                }

                // remainder loop
                for (; n < numSamplesSize; ++n)
                {
                    const SampleType modspeed = std::abs(currentPos - prevPos);
                    prevPos = currentPos;
                    updateDuckGain(modspeed);

                    SampleType monoSum = ch0 != nullptr ? ch0[n] : static_cast<SampleType>(0);
                    if (ch1 != nullptr)
                        monoSum = static_cast<SampleType>(0.5) * (monoSum + ch1[n]);

                    const int readIdxL = (writeIdxL + bufferLength) % bufferLength;
                    const SampleType delayedOut = readInterpolated(bufferL.data(), readIdxL, currentPos, lagrange5thL);
                    const SampleType duckedOut = delayedOut * duckGain;

                    const SampleType writeVal = softClip(monoSum + feedbackParam * duckedOut);
                    writeSample(bufferL.data(), writeIdxL, writeVal);

                    const SampleType out = softClip(duckedOut * mixParam + monoSum * oneMinusMix);

                    if (ch0 != nullptr) ch0[n] = out;
                    if (ch1 != nullptr) ch1[n] = out;
                }
            }
            else // stereo
            {
                const SampleType currentPos = delayMsToSamples;
                const SampleType oneMinusMix = static_cast<SampleType>(1) - mixParam;

                size_t n = 0;
                const size_t numSamplesSize = static_cast<size_t>(numSamples);

                for (; n + 3 < numSamplesSize; n += 4)
                {
                    float duckGains[4];
                    for (int i = 0; i < 4; ++i)
                    {
                        const SampleType modspeed = std::abs(currentPos - prevPos);
                        prevPos = currentPos;
                        updateDuckGain(modspeed);
                        duckGains[i] = duckGain;
                    }

                    auto vDuckGain = SIMD_MM(loadu_ps)(duckGains);

                    if (ch0 != nullptr)
                    {
                        auto vXL = SIMD_MM(loadu_ps)(ch0 + n);
                        // Vectorized interpolation for stereo L
                        int start0L = (writeIdxL - offset + bufferLength) % bufferLength;
                        SIMD_M128 vYL;
                        if (start0L + 8 < bufferLength)
                        {
                            auto v0 = SIMD_MM(loadu_ps)(bufferL.data() + start0L);
                            auto v1 = SIMD_MM(loadu_ps)(bufferL.data() + start0L + 1);
                            auto v2 = SIMD_MM(loadu_ps)(bufferL.data() + start0L + 2);
                            auto v3 = SIMD_MM(loadu_ps)(bufferL.data() + start0L + 3);
                            auto v4 = SIMD_MM(loadu_ps)(bufferL.data() + start0L + 4);
                            auto v5 = SIMD_MM(loadu_ps)(bufferL.data() + start0L + 5);

                            auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5),
                                                        SIMD_MM(mul_ps)(v5, vC6)))));

                            vYL = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1),
                                                  SIMD_MM(mul_ps)(vFrac, vSum));
                        }
                        else
                        {
                            float delayedOutsL[4];
                            for (int i = 0; i < 4; ++i)
                            {
                                const int rIdx = (writeIdxL + i + bufferLength) % bufferLength;
                                delayedOutsL[i] = readInterpolated(bufferL.data(), rIdx, currentPos, lagrange5thL);
                            }
                            vYL = SIMD_MM(loadu_ps)(delayedOutsL);
                        }
                        auto vYL_ducked = SIMD_MM(mul_ps)(vYL, vDuckGain);

                        auto vWriteValL = fasterTanhBounded(SIMD_MM(add_ps)(vXL, SIMD_MM(mul_ps)(SIMD_MM(set_ps1)(fbLParam), vYL_ducked)));

                        float writeValsL[4];
                        SIMD_MM(storeu_ps)(writeValsL, vWriteValL);

                        for (int i = 0; i < 4; ++i)
                            writeSample(bufferL.data(), writeIdxL, writeValsL[i]);

                        auto vOutL = fasterTanhBounded(SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vYL_ducked, SIMD_MM(set_ps1)(mixParam)),
                                                                       SIMD_MM(mul_ps)(vXL, SIMD_MM(set_ps1)(oneMinusMix))));
                        SIMD_MM(storeu_ps)(ch0 + n, vOutL);
                    }

                    if (ch1 != nullptr)
                    {
                        auto vXR = SIMD_MM(loadu_ps)(ch1 + n);
                        // Vectorized interpolation for stereo R
                        int start0R = (writeIdxR - offset + bufferLength) % bufferLength;
                        SIMD_M128 vYR;
                        if (start0R + 8 < bufferLength)
                        {
                            auto v0 = SIMD_MM(loadu_ps)(bufferR.data() + start0R);
                            auto v1 = SIMD_MM(loadu_ps)(bufferR.data() + start0R + 1);
                            auto v2 = SIMD_MM(loadu_ps)(bufferR.data() + start0R + 2);
                            auto v3 = SIMD_MM(loadu_ps)(bufferR.data() + start0R + 3);
                            auto v4 = SIMD_MM(loadu_ps)(bufferR.data() + start0R + 4);
                            auto v5 = SIMD_MM(loadu_ps)(bufferR.data() + start0R + 5);

                            auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5),
                                                        SIMD_MM(mul_ps)(v5, vC6)))));

                            vYR = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1),
                                                  SIMD_MM(mul_ps)(vFrac, vSum));
                        }
                        else
                        {
                            float delayedOutsR[4];
                            for (int i = 0; i < 4; ++i)
                            {
                                const int rIdx = (writeIdxR + i + bufferLength) % bufferLength;
                                delayedOutsR[i] = readInterpolated(bufferR.data(), rIdx, currentPos, lagrange5thR);
                            }
                            vYR = SIMD_MM(loadu_ps)(delayedOutsR);
                        }
                        auto vYR_ducked = SIMD_MM(mul_ps)(vYR, vDuckGain);

                        auto vWriteValR = fasterTanhBounded(SIMD_MM(add_ps)(vXR, SIMD_MM(mul_ps)(SIMD_MM(set_ps1)(fbRParam), vYR_ducked)));

                        float writeValsR[4];
                        SIMD_MM(storeu_ps)(writeValsR, vWriteValR);

                        for (int i = 0; i < 4; ++i)
                            writeSample(bufferR.data(), writeIdxR, writeValsR[i]);

                        auto vOutR = fasterTanhBounded(SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vYR_ducked, SIMD_MM(set_ps1)(mixParam)),
                                                                       SIMD_MM(mul_ps)(vXR, SIMD_MM(set_ps1)(oneMinusMix))));
                        SIMD_MM(storeu_ps)(ch1 + n, vOutR);
                    }
                }

                for (; n < numSamplesSize; ++n)
                {
                    const SampleType modspeed = std::abs(currentPos - prevPos);
                    prevPos = currentPos;
                    updateDuckGain(modspeed);

                    if (ch0 != nullptr)
                    {
                        const SampleType xL = ch0[n];
                        const int rIdx = (writeIdxL + bufferLength) % bufferLength;
                        const SampleType yL_ducked = readInterpolated (bufferL.data(), rIdx, currentPos, lagrange5thL) * duckGain;
                        writeSample (bufferL.data(), writeIdxL, softClip (xL + fbLParam * yL_ducked));
                        ch0[n] = softClip(yL_ducked * mixParam + xL * oneMinusMix);
                    }

                    if (ch1 != nullptr)
                    {
                        const SampleType xR = ch1[n];
                        const int rIdx = (writeIdxR + bufferLength) % bufferLength;
                        const SampleType yR_ducked = readInterpolated (bufferR.data(), rIdx, currentPos, lagrange5thR) * duckGain;
                        writeSample (bufferR.data(), writeIdxR, softClip (xR + fbRParam * yR_ducked));
                        ch1[n] = softClip(yR_ducked * mixParam + xR * oneMinusMix);
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
            mix = (value > 1.0f ? value * 0.01f : value);
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
        static constexpr int maxDelaySamples = 1 << 18;

        int bufferLength = 0;
        int writeIdxL = 0;
        int writeIdxR = 0;

        Lagrange5th lagrange5thL, lagrange5thR;

        bool mono = false;
        bool bypassed = false;
    };
}
#endif
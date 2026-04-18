#pragma once

#ifndef CHRONOS_DELAY_ENGINE_H
#define CHRONOS_DELAY_ENGINE_H

#include <cassert>
#include <JuceHeader.h>
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

            if (isMono()) // mono
            {
                for (auto n {0uz}; n < static_cast<size_t>(numSamples); ++n)
                {
                    const SampleType currentPos = delayMsToSamples;
                    const SampleType feedbackParam = fbLParam;

                    const SampleType modspeed = std::abs(currentPos - prevPos);
                    prevPos = currentPos;
                    updateDuckGain(modspeed);

                    SampleType monoSum = ch0 != nullptr ? ch0[n] : static_cast<SampleType>(0);
                    if (ch1 != nullptr)
                        monoSum = static_cast<SampleType>(0.5) * (monoSum + ch1[n]);


                    // prevent negative indices
                    const int readIdxL = ((writeIdxL - 1) + bufferLength) % bufferLength;

                    // calc read pos
                    const SampleType delayedOut = readInterpolated(bufferL.data(), readIdxL, currentPos, lagrange5thL);
                    const SampleType duckedOut = delayedOut * duckGain;

                    // write (input + fb) -> softclip sweeps
                    const SampleType writeVal = softClip(monoSum + feedbackParam * duckedOut);
                    writeSample(bufferL.data(), writeIdxL, writeVal);

                    const SampleType out = softClip(duckedOut * mixParam + monoSum * (static_cast<SampleType>(1) - mixParam));

                    if (ch0 != nullptr) ch0[n] = out;
                    if (ch1 != nullptr) ch1[n] = out;
                }
            }
            else // stereo
            {
                for (auto n {0uz}; n < static_cast<size_t>(numSamples); ++n)
                {
                    const SampleType currentPos2 = delayMsToSamples;
                    const SampleType modspeed = std::abs(currentPos2 - prevPos);
                    prevPos = currentPos2;
                    updateDuckGain(modspeed);

                    SampleType xL = static_cast<SampleType> (0);
                    SampleType xR = static_cast<SampleType> (0);
                    SampleType yL_ducked = static_cast<SampleType> (0);
                    SampleType yR_ducked = static_cast<SampleType> (0);

                    if (ch0 != nullptr)
                    {
                        xL = ch0[n];
                        const int lastIndexL = (writeIdxL - 1 + bufferLength) % bufferLength;
                        const SampleType yL = readInterpolated (bufferL.data(), lastIndexL, currentPos2, lagrange5thL);
                        yL_ducked = yL * duckGain;
                    }

                    if (ch1 != nullptr)
                    {
                        xR = ch1[n];
                        const int lastIndexR = (writeIdxR - 1 + bufferLength) % bufferLength;
                        const SampleType yR = readInterpolated (bufferR.data(), lastIndexR, currentPos2, lagrange5thR);
                        yR_ducked = yR * duckGain;
                    }

                    SampleType fbInL = yL_ducked;
                    SampleType fbInR = yR_ducked;
                    SampleType wetOutL = yL_ducked;
                    SampleType wetOutR = yR_ducked;

                    if (ch0 != nullptr)
                    {
                        const SampleType writeValueL = softClip (xL + fbLParam * fbInL);
                        writeSample (bufferL.data(), writeIdxL, writeValueL);
                        ch0[n] = softClip(wetOutL * mixParam + xL * (static_cast<SampleType> (1) - mixParam));
                    }

                    if (ch1 != nullptr)
                    {
                        const SampleType writeValueR = softClip (xR + fbRParam * fbInR);
                        writeSample (bufferR.data(), writeIdxR, writeValueR);
                        ch1[n] = softClip(wetOutR * mixParam + xR * (static_cast<SampleType> (1) - mixParam));
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
            return std::tanh(x);
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
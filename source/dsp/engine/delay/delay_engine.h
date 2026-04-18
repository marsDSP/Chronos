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

            // snap smoothers so the first block after reset doesn't ramp from 0.
            lMix.instantize(std::clamp(mix,       0.0f, 1.0f));
            lFbL.instantize(std::clamp(feedbackL, 0.0f, 0.99f));
            lFbR.instantize(std::clamp(feedbackR, 0.0f, 0.99f));
            lCrossfeed.instantize(std::clamp(crossfeed, 0.0f, 1.0f));
            lagDelayMs.newValue(std::clamp(delayTime, minDelayTime, maxDelayTime));
            lagDelayMs.instantize();

            // Clear biquad state on reset.
            fbLP_L.reset(); fbLP_R.reset();
            fbHP_L.reset(); fbHP_R.reset();
        }

        void prepare(const dsp::ProcessSpec &spec) noexcept
        {
            sampleRate = spec.sampleRate;
            AllocBuffer();

            // 10ms attack, 100ms release for ducking response
            duckAtkCoeff = static_cast<SampleType>(1.0 - std::exp(-1.0 / (0.010 * sampleRate)));
            duckRelCoeff = static_cast<SampleType>(1.0 - std::exp(-1.0 / (0.100 * sampleRate)));

            lagDelayMs.setRateInMilliseconds(150.0, sampleRate, 1.0);

            // Initial filter coefficients.
            updateFilterCoeffs();
            lastLowCutHz  = lowCutHz;
            lastHighCutHz = highCutHz;

            reset();
        }

        void process(const dsp::AudioBlock<SampleType> &block, const int numSamples) noexcept
        {
            if (bypassed)
                return;

            const size_t numCh = block.getNumChannels();
            auto *ch0 = numCh > 0 ? block.getChannelPointer(0) : nullptr;
            auto *ch1 = numCh > 1 ? block.getChannelPointer(1) : nullptr;

            // push targets into block-rate linear ramp smoothers
            // SIMD inner loops consume per-quad vectors via .quad(q).
            lMix.setTarget(std::clamp(mix,       0.0f, 1.0f),  numSamples);
            lFbL.setTarget(std::clamp(feedbackL, 0.0f, 0.99f), numSamples);
            lFbR.setTarget(std::clamp(feedbackR, 0.0f, 0.99f), numSamples);
            lCrossfeed.setTarget(std::clamp(crossfeed, 0.0f, 1.0f), numSamples);

            // Recompute feedback-path filter coefficients only when cutoffs change.
            if (lowCutHz != lastLowCutHz || highCutHz != lastHighCutHz)
            {
                updateFilterCoeffs();
                lastLowCutHz  = lowCutHz;
                lastHighCutHz = highCutHz;
            }

            const float delayMsOld = lagDelayMs.getValue();
            lagDelayMs.newValue(std::clamp(delayTime, minDelayTime, maxDelayTime));
            lagDelayMs.processN(numSamples);
            const float delayMsNew = lagDelayMs.getValue();

            const size_t numSamplesSize = static_cast<size_t>(numSamples);
            assert(numSamplesSize <= N_BLOCK - 8);

            auto msToPos = [&](float ms) {
                const auto s = static_cast<SampleType>(sampleRate * (ms * 0.001));
                return std::max(s, static_cast<SampleType>(numSamplesSize + 1));
            };
            const SampleType posOld = msToPos(delayMsOld);
            const SampleType posNew = msToPos(delayMsNew);

            const int offsetOld = static_cast<int>(std::floor(static_cast<double>(posOld)));
            const int offsetNew = static_cast<int>(std::floor(static_cast<double>(posNew)));
            const SampleType fracOld = posOld - static_cast<SampleType>(offsetOld);
            const SampleType fracNew = posNew - static_cast<SampleType>(offsetNew);

            // Dual scratch pre-read (memcpy with tail-mirror trick from earlier design).
            const int total = static_cast<int>(numSamplesSize) + kTail;
            auto readScratch = [&](const std::vector<SampleType>& src,
                                   SampleType* dst, int rpos) {
                const int first = std::min(total, kBufSize + kTail - rpos);
                std::memcpy(dst, &src[rpos], first * sizeof(SampleType));
                if (first < total)
                    std::memcpy(dst + first, &src[kTail], (total - first) * sizeof(SampleType));
            };

            readScratch(bufferL, tL,  (writeIdxL - offsetNew) & kBufMask);
            readScratch(bufferL, tL2, (writeIdxL - offsetOld) & kBufMask);

            if (!isMono() && ch1 != nullptr)
            {
                readScratch(bufferR, tR,  (writeIdxR - offsetNew) & kBufMask);
                readScratch(bufferR, tR2, (writeIdxR - offsetOld) & kBufMask);
            }

            // Dual Lagrange coefficient set: N for new frac, O for old frac.
            auto computeCoeffs = [](SampleType frac, LagrangeCoeffs& c) {
                const float d1 = frac - 1.0f;
                const float d2 = frac - 2.0f;
                const float d3 = frac - 3.0f;
                const float d4 = frac - 4.0f;
                const float d5 = frac - 5.0f;
                c.c[0] = -d1 * d2 * d3 * d4 * d5 / 120.0f;
                c.c[1] =  d2 * d3 * d4 * d5       /  24.0f;
                c.c[2] = -d1 * d3 * d4 * d5       /  12.0f;
                c.c[3] =  d1 * d2 * d4 * d5       /  12.0f;
                c.c[4] = -d1 * d2 * d3 * d5       /  24.0f;
                c.c[5] =  d1 * d2 * d3 * d4       / 120.0f;
                c.frac = frac;
            };

            LagrangeCoeffs coeffsN, coeffsO;
            computeCoeffs(fracNew, coeffsN);
            computeCoeffs(fracOld, coeffsO);

            // SIMD broadcast of both coefficient sets (N = new, O = old).
            const auto vC1N = SIMD_MM(set1_ps)(coeffsN.c[0]);
            const auto vC2N = SIMD_MM(set1_ps)(coeffsN.c[1]);
            const auto vC3N = SIMD_MM(set1_ps)(coeffsN.c[2]);
            const auto vC4N = SIMD_MM(set1_ps)(coeffsN.c[3]);
            const auto vC5N = SIMD_MM(set1_ps)(coeffsN.c[4]);
            const auto vC6N = SIMD_MM(set1_ps)(coeffsN.c[5]);
            const auto vFracN = SIMD_MM(set1_ps)(fracNew);

            const auto vC1O = SIMD_MM(set1_ps)(coeffsO.c[0]);
            const auto vC2O = SIMD_MM(set1_ps)(coeffsO.c[1]);
            const auto vC3O = SIMD_MM(set1_ps)(coeffsO.c[2]);
            const auto vC4O = SIMD_MM(set1_ps)(coeffsO.c[3]);
            const auto vC5O = SIMD_MM(set1_ps)(coeffsO.c[4]);
            const auto vC6O = SIMD_MM(set1_ps)(coeffsO.c[5]);
            const auto vFracO = SIMD_MM(set1_ps)(fracOld);

            // Alpha ramp: 0 at block start, 1 at block end. Applied per SIMD lane.
            const float invN = (numSamplesSize > 0)
                               ? 1.0f / static_cast<float>(numSamplesSize) : 0.0f;
            const auto vInvN    = SIMD_MM(set1_ps)(invN);
            const auto vLaneIdx = SIMD_MM(setr_ps)(0.0f, 1.0f, 2.0f, 3.0f);

            // duck gain update uses the end-of-block position as "current".
            const SampleType modspeed = std::abs(posNew - prevPos);
            prevPos = posNew;
            updateDuckGain(modspeed);
            const auto vDuckGain = SIMD_MM(set1_ps)(duckGain);

            if (isMono()) // mono
            {
                // ---------------- PASS 1: SIMD Lagrange blend → dsL[] ----------------
                {
                    size_t n = 0;
                    for (; n + 3 < numSamplesSize; n += 4)
                    {
                        const auto vNf    = SIMD_MM(set1_ps)(static_cast<float>(n));
                        const auto vAlpha = SIMD_MM(mul_ps)(vInvN, SIMD_MM(add_ps)(vNf, vLaneIdx));

                        SIMD_M128 vYN;
                        {
                            auto v0 = SIMD_MM(load_ps) (tL + n);
                            auto v1 = SIMD_MM(loadu_ps)(tL + n + 1);
                            auto v2 = SIMD_MM(loadu_ps)(tL + n + 2);
                            auto v3 = SIMD_MM(loadu_ps)(tL + n + 3);
                            auto v4 = SIMD_MM(loadu_ps)(tL + n + 4);
                            auto v5 = SIMD_MM(loadu_ps)(tL + n + 5);
                            auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5N),
                                                        SIMD_MM(mul_ps)(v5, vC6N)))));
                            vYN = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1N),
                                                  SIMD_MM(mul_ps)(vFracN, vSum));
                        }

                        SIMD_M128 vYO;
                        {
                            auto v0 = SIMD_MM(load_ps) (tL2 + n);
                            auto v1 = SIMD_MM(loadu_ps)(tL2 + n + 1);
                            auto v2 = SIMD_MM(loadu_ps)(tL2 + n + 2);
                            auto v3 = SIMD_MM(loadu_ps)(tL2 + n + 3);
                            auto v4 = SIMD_MM(loadu_ps)(tL2 + n + 4);
                            auto v5 = SIMD_MM(loadu_ps)(tL2 + n + 5);
                            auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5O),
                                                        SIMD_MM(mul_ps)(v5, vC6O)))));
                            vYO = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1O),
                                                  SIMD_MM(mul_ps)(vFracO, vSum));
                        }

                        const auto vDelayedOut = SIMD_MM(add_ps)(vYO,
                                                    SIMD_MM(mul_ps)(vAlpha,
                                                                    SIMD_MM(sub_ps)(vYN, vYO)));
                        SIMD_MM(store_ps)(&dsL[n], vDelayedOut);
                    }
                    for (; n < numSamplesSize; ++n)
                    {
                        const SampleType alpha = static_cast<SampleType>(static_cast<float>(n) * invN);
                        const SampleType yN = readInterpolated(tL,  static_cast<int>(n), coeffsN);
                        const SampleType yO = readInterpolated(tL2, static_cast<int>(n), coeffsO);
                        dsL[n] = static_cast<float>(yO + alpha * (yN - yO));
                    }
                }

                // ---------------- PASS 2: scalar HP → LP on dsL[] -------------------
                // Biquads are stateful so this pass is intrinsically scalar, but it's
                // a tight sequential loop over ~4KB in L1 so it's cheap.
                for (size_t k = 0; k < numSamplesSize; ++k)
                    dsL[k] = fbLP_L.processSample(fbHP_L.processSample(dsL[k]));

                // ---------------- PASS 3: SIMD feedback MAC + dry/wet mix -----------
                {
                    size_t n = 0;
                    for (; n + 3 < numSamplesSize; n += 4)
                    {
                        const int q = static_cast<int>(n) >> 2;
                        const auto vMix         = lMix.quad(q);
                        const auto vOneMinusMix = SIMD_MM(sub_ps)(SIMD_MM(set1_ps)(1.0f), vMix);
                        const auto vFb          = lFbL.quad(q);

                        auto vMonoSum = SIMD_MM(setzero_ps)();
                        if (ch0 != nullptr && ch1 != nullptr)
                            vMonoSum = SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(0.5f),
                                                       SIMD_MM(add_ps)(SIMD_MM(loadu_ps)(ch0 + n),
                                                                       SIMD_MM(loadu_ps)(ch1 + n)));
                        else if (ch0 != nullptr) vMonoSum = SIMD_MM(loadu_ps)(ch0 + n);
                        else if (ch1 != nullptr) vMonoSum = SIMD_MM(loadu_ps)(ch1 + n);

                        const auto vFiltered  = SIMD_MM(load_ps)(&dsL[n]);
                        const auto vDuckedOut = SIMD_MM(mul_ps)(vFiltered, vDuckGain);

                        auto vWriteVal = fasterTanhBounded(
                            SIMD_MM(add_ps)(vMonoSum, SIMD_MM(mul_ps)(vFb, vDuckedOut)));
                        SIMD_MM(storeu_ps)(&wL[n], vWriteVal);

                        auto vOut = fasterTanhBounded(
                            SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vDuckedOut, vMix),
                                            SIMD_MM(mul_ps)(vMonoSum,   vOneMinusMix)));

                        if (ch0 != nullptr) SIMD_MM(storeu_ps)(ch0 + n, vOut);
                        if (ch1 != nullptr) SIMD_MM(storeu_ps)(ch1 + n, vOut);
                    }
                    for (; n < numSamplesSize; ++n)
                    {
                        const SampleType mixP       = static_cast<SampleType>(lMix.at(static_cast<int>(n)));
                        const SampleType oneMinusMx = static_cast<SampleType>(1) - mixP;
                        const SampleType fbP        = static_cast<SampleType>(lFbL.at(static_cast<int>(n)));

                        SampleType monoSum = ch0 != nullptr ? ch0[n] : static_cast<SampleType>(0);
                        if (ch1 != nullptr)
                            monoSum = static_cast<SampleType>(0.5) * (monoSum + ch1[n]);

                        const SampleType duckedOut = static_cast<SampleType>(dsL[n]) * duckGain;

                        wL[n] = softClip(monoSum + fbP * duckedOut);
                        const SampleType out = softClip(duckedOut * mixP + monoSum * oneMinusMx);

                        if (ch0 != nullptr) ch0[n] = out;
                        if (ch1 != nullptr) ch1[n] = out;
                    }
                }

                // Block write & Mirror
                const bool wrapped = (writeIdxL + static_cast<int>(numSamplesSize)) > kBufSize;
                if (wrapped)
                {
                    for (size_t k = 0; k < numSamplesSize; ++k)
                    {
                        const int idx = (writeIdxL + static_cast<int>(k)) & kBufMask;
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
                // ---------------- PASS 1: SIMD fill dsL[] and dsR[] ----------------
                {
                    size_t n = 0;
                    for (; n + 3 < numSamplesSize; n += 4)
                    {
                        const auto vNf    = SIMD_MM(set1_ps)(static_cast<float>(n));
                        const auto vAlpha = SIMD_MM(mul_ps)(vInvN, SIMD_MM(add_ps)(vNf, vLaneIdx));

                        // L: Lagrange N + O, blend → dsL[n..n+3]
                        SIMD_M128 vYL_N;
                        {
                            auto v0 = SIMD_MM(load_ps) (tL + n);
                            auto v1 = SIMD_MM(loadu_ps)(tL + n + 1);
                            auto v2 = SIMD_MM(loadu_ps)(tL + n + 2);
                            auto v3 = SIMD_MM(loadu_ps)(tL + n + 3);
                            auto v4 = SIMD_MM(loadu_ps)(tL + n + 4);
                            auto v5 = SIMD_MM(loadu_ps)(tL + n + 5);
                            auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5N),
                                                        SIMD_MM(mul_ps)(v5, vC6N)))));
                            vYL_N = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1N),
                                                    SIMD_MM(mul_ps)(vFracN, vSum));
                        }
                        SIMD_M128 vYL_O;
                        {
                            auto v0 = SIMD_MM(load_ps) (tL2 + n);
                            auto v1 = SIMD_MM(loadu_ps)(tL2 + n + 1);
                            auto v2 = SIMD_MM(loadu_ps)(tL2 + n + 2);
                            auto v3 = SIMD_MM(loadu_ps)(tL2 + n + 3);
                            auto v4 = SIMD_MM(loadu_ps)(tL2 + n + 4);
                            auto v5 = SIMD_MM(loadu_ps)(tL2 + n + 5);
                            auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5O),
                                                        SIMD_MM(mul_ps)(v5, vC6O)))));
                            vYL_O = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1O),
                                                    SIMD_MM(mul_ps)(vFracO, vSum));
                        }
                        const auto vDelL = SIMD_MM(add_ps)(vYL_O,
                                              SIMD_MM(mul_ps)(vAlpha,
                                                              SIMD_MM(sub_ps)(vYL_N, vYL_O)));
                        SIMD_MM(store_ps)(&dsL[n], vDelL);

                        // R: Lagrange N + O, blend → dsR[n..n+3]
                        SIMD_M128 vYR_N;
                        {
                            auto v0 = SIMD_MM(load_ps) (tR + n);
                            auto v1 = SIMD_MM(loadu_ps)(tR + n + 1);
                            auto v2 = SIMD_MM(loadu_ps)(tR + n + 2);
                            auto v3 = SIMD_MM(loadu_ps)(tR + n + 3);
                            auto v4 = SIMD_MM(loadu_ps)(tR + n + 4);
                            auto v5 = SIMD_MM(loadu_ps)(tR + n + 5);
                            auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4N),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5N),
                                                        SIMD_MM(mul_ps)(v5, vC6N)))));
                            vYR_N = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1N),
                                                    SIMD_MM(mul_ps)(vFracN, vSum));
                        }
                        SIMD_M128 vYR_O;
                        {
                            auto v0 = SIMD_MM(load_ps) (tR2 + n);
                            auto v1 = SIMD_MM(loadu_ps)(tR2 + n + 1);
                            auto v2 = SIMD_MM(loadu_ps)(tR2 + n + 2);
                            auto v3 = SIMD_MM(loadu_ps)(tR2 + n + 3);
                            auto v4 = SIMD_MM(loadu_ps)(tR2 + n + 4);
                            auto v5 = SIMD_MM(loadu_ps)(tR2 + n + 5);
                            auto vSum = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v1, vC2O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v2, vC3O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v3, vC4O),
                                        SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v4, vC5O),
                                                        SIMD_MM(mul_ps)(v5, vC6O)))));
                            vYR_O = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1O),
                                                    SIMD_MM(mul_ps)(vFracO, vSum));
                        }
                        const auto vDelR = SIMD_MM(add_ps)(vYR_O,
                                              SIMD_MM(mul_ps)(vAlpha,
                                                              SIMD_MM(sub_ps)(vYR_N, vYR_O)));
                        SIMD_MM(store_ps)(&dsR[n], vDelR);
                    }
                    for (; n < numSamplesSize; ++n)
                    {
                        const SampleType alpha = static_cast<SampleType>(static_cast<float>(n) * invN);
                        const SampleType yLN = readInterpolated(tL,  static_cast<int>(n), coeffsN);
                        const SampleType yLO = readInterpolated(tL2, static_cast<int>(n), coeffsO);
                        const SampleType yRN = readInterpolated(tR,  static_cast<int>(n), coeffsN);
                        const SampleType yRO = readInterpolated(tR2, static_cast<int>(n), coeffsO);
                        dsL[n] = static_cast<float>(yLO + alpha * (yLN - yLO));
                        dsR[n] = static_cast<float>(yRO + alpha * (yRN - yRO));
                    }
                }

                // ---------------- PASS 2: scalar filter + crossfeed blend ----------
                // Each sample: HP → LP per channel, then blend the two filtered
                // signals with the smoothed crossfeed amount to form the feedback
                // input. This is the ping-pong path.
                for (size_t k = 0; k < numSamplesSize; ++k)
                {
                    const float filtL = fbLP_L.processSample(fbHP_L.processSample(dsL[k]));
                    const float filtR = fbLP_R.processSample(fbHP_R.processSample(dsR[k]));
                    const float cf    = lCrossfeed.at(static_cast<int>(k));
                    const float cfInv = 1.0f - cf;
                    dsL[k] = cfInv * filtL + cf * filtR;
                    dsR[k] = cfInv * filtR + cf * filtL;
                }

                // ---------------- PASS 3: SIMD feedback MAC + dry/wet mix ---------
                {
                    size_t n = 0;
                    for (; n + 3 < numSamplesSize; n += 4)
                    {
                        const int q = static_cast<int>(n) >> 2;
                        const auto vMix         = lMix.quad(q);
                        const auto vOneMinusMix = SIMD_MM(sub_ps)(SIMD_MM(set1_ps)(1.0f), vMix);

                        if (ch0 != nullptr)
                        {
                            const auto vFbL = lFbL.quad(q);
                            auto vXL = SIMD_MM(loadu_ps)(ch0 + n);
                            auto vYL_ducked = SIMD_MM(mul_ps)(SIMD_MM(load_ps)(&dsL[n]), vDuckGain);

                            auto vWriteValL = fasterTanhBounded(
                                SIMD_MM(add_ps)(vXL, SIMD_MM(mul_ps)(vFbL, vYL_ducked)));
                            SIMD_MM(storeu_ps)(&wL[n], vWriteValL);

                            auto vOutL = fasterTanhBounded(
                                SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vYL_ducked, vMix),
                                                SIMD_MM(mul_ps)(vXL,        vOneMinusMix)));
                            SIMD_MM(storeu_ps)(ch0 + n, vOutL);
                        }

                        if (ch1 != nullptr)
                        {
                            const auto vFbR = lFbR.quad(q);
                            auto vXR = SIMD_MM(loadu_ps)(ch1 + n);
                            auto vYR_ducked = SIMD_MM(mul_ps)(SIMD_MM(load_ps)(&dsR[n]), vDuckGain);

                            auto vWriteValR = fasterTanhBounded(
                                SIMD_MM(add_ps)(vXR, SIMD_MM(mul_ps)(vFbR, vYR_ducked)));
                            SIMD_MM(storeu_ps)(&wR[n], vWriteValR);

                            auto vOutR = fasterTanhBounded(
                                SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vYR_ducked, vMix),
                                                SIMD_MM(mul_ps)(vXR,        vOneMinusMix)));
                            SIMD_MM(storeu_ps)(ch1 + n, vOutR);
                        }
                    }
                    for (; n < numSamplesSize; ++n)
                    {
                        const SampleType mixP       = static_cast<SampleType>(lMix.at(static_cast<int>(n)));
                        const SampleType oneMinusMx = static_cast<SampleType>(1) - mixP;

                        if (ch0 != nullptr)
                        {
                            const SampleType fbLP = static_cast<SampleType>(lFbL.at(static_cast<int>(n)));
                            const SampleType xL   = ch0[n];
                            const SampleType yL_ducked = static_cast<SampleType>(dsL[n]) * duckGain;
                            wL[n]  = softClip(xL + fbLP * yL_ducked);
                            ch0[n] = softClip(yL_ducked * mixP + xL * oneMinusMx);
                        }
                        if (ch1 != nullptr)
                        {
                            const SampleType fbRP = static_cast<SampleType>(lFbR.at(static_cast<int>(n)));
                            const SampleType xR   = ch1[n];
                            const SampleType yR_ducked = static_cast<SampleType>(dsR[n]) * duckGain;
                            wR[n]  = softClip(xR + fbRP * yR_ducked);
                            ch1[n] = softClip(yR_ducked * mixP + xR * oneMinusMx);
                        }
                    }
                }

                // Block write & Mirror L
                if (ch0 != nullptr) {
                    const int oldIdxL = writeIdxL;
                    const bool wrappedL = (oldIdxL + static_cast<int>(numSamplesSize)) > kBufSize;
                    if (wrappedL) {
                        for (size_t k = 0; k < numSamplesSize; ++k)
                            bufferL[(oldIdxL + static_cast<int>(k)) & kBufMask] = wL[k];
                    } else {
                        std::memcpy(&bufferL[oldIdxL], wL, numSamplesSize * sizeof(SampleType));
                    }
                    writeIdxL = (oldIdxL + static_cast<int>(numSamplesSize)) & kBufMask;
                    if (wrappedL || oldIdxL < kTail) {
                        for (int k = 0; k < kTail; ++k)
                            bufferL[kBufSize + k] = bufferL[k];
                    }
                }

                // Block write & Mirror R
                if (ch1 != nullptr) {
                    const int oldIdxR = writeIdxR;
                    const bool wrappedR = (oldIdxR + static_cast<int>(numSamplesSize)) > kBufSize;
                    if (wrappedR) {
                        for (size_t k = 0; k < numSamplesSize; ++k)
                            bufferR[(oldIdxR + static_cast<int>(k)) & kBufMask] = wR[k];
                    } else {
                        std::memcpy(&bufferR[oldIdxR], wR, numSamplesSize * sizeof(SampleType));
                    }
                    writeIdxR = (oldIdxR + static_cast<int>(numSamplesSize)) & kBufMask;
                    if (wrappedR || oldIdxR < kTail) {
                        for (int k = 0; k < kTail; ++k)
                            bufferR[kBufSize + k] = bufferR[k];
                    }
                }
            }

            // advance block-rate ramps so next block starts from this block's target
            lMix.advanceBlock();
            lFbL.advanceBlock();
            lFbR.advanceBlock();
            lCrossfeed.advanceBlock();
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

        // Feedback-path low-cut (highpass) corner in Hz.
        void setLowCutParam(const float hz) noexcept
        {
            lowCutHz = std::clamp(hz, 20.0f, 20000.0f);
        }

        // Feedback-path high-cut (lowpass) corner in Hz.
        void setHighCutParam(const float hz) noexcept
        {
            highCutHz = std::clamp(hz, 20.0f, 20000.0f);
        }

        // Stereo crossfeed / ping-pong amount (0..1). 0 = no crossfeed, 1 = full swap.
        void setCrossfeedParam(const float value) noexcept
        {
            crossfeed = std::clamp(value, 0.0f, 1.0f);
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

        // ------------------------------------------------------------------
        // Tail prediction
        // ------------------------------------------------------------------
        // Returns the number of samples after which the feedback tail can be
        // considered decayed below -60 dB (the threshold most hosts treat as
        // "silent"). Hosts can use this to know when to free / suspend the
        // plugin after an input transition to silence.
        //
        // Calculation:
        //   repeats until |fb|^n < 0.001, i.e. n = ceil(log(0.001) / log(fb))
        // plus a safety margin for biquad ring-down + parameter smoothing.
        // Clamped to a sane maximum so pathological feedback values (~0.99)
        // don't yield absurd tail lengths.
        [[nodiscard]] int ringoutSamples() const noexcept
        {
            if (bypassed) return 0;

            const float delayMs     = std::clamp(delayTime, minDelayTime, maxDelayTime);
            const float delaySamples = static_cast<float>(sampleRate * delayMs * 0.001);
            const float fb           = std::clamp(std::max(feedbackL, feedbackR),
                                                  0.0f, 0.9999f);

            constexpr float silenceDb  = 0.001f;  // -60 dB
            constexpr int   kMargin    = 2048;    // biquad ring-down + smoothers
            constexpr int   kMaxTail   = 1 << 20; // clamp (~21.8 s @ 48 kHz)

            if (fb < 1.0e-4f)
                return std::min(static_cast<int>(delaySamples) + kMargin, kMaxTail);

            const int repeats = static_cast<int>(std::ceil(
                std::log(silenceDb) / std::log(fb)));
            const long long tail = static_cast<long long>(delaySamples)
                                 * static_cast<long long>(std::max(repeats, 1))
                                 + kMargin;
            return static_cast<int>(std::min<long long>(tail, kMaxTail));
        }

        // ------------------------------------------------------------------
        // Streaming-version contract
        // ------------------------------------------------------------------
        // Bumped whenever the meaning of any parameter value stream-in changes
        // in a way that can't be represented by the host's normal state
        // serialization. Increment and add a migration branch in
        // remapParametersForStreamingVersion whenever:
        //   - a parameter is renamed or removed
        //   - a parameter's units / range / skew change meaning
        //   - the engine's internal interpretation of a value changes
        //
        // Adding a new parameter with a sane default does NOT require bumping.
        static constexpr int16_t streamingVersion{1};

        // Given a flat parameter array streamed from an older version,
        // migrate its values in place to the current schema. Called by the
        // plugin layer before pushing values into the engine. The input array
        // must be large enough to hold every parameter for the *newer* of the
        // two schemas.
        static void remapParametersForStreamingVersion(
            int16_t streamedFrom, float* const /*parameters*/) noexcept
        {
            assert(streamedFrom <= streamingVersion);
            // v1 is the initial schema; nothing to migrate yet.
            (void)streamedFrom;
        }

    private:
        struct LipolSIMD
        {
            float current = 0.0f;
            float target  = 0.0f;
            float delta   = 0.0f;

            void setTarget(float t, int blockSize) noexcept
            {
                target = t;
                delta  = (blockSize > 0) ? (t - current) / static_cast<float>(blockSize) : 0.0f;
            }
            void instantize(float v) noexcept { current = target = v; delta = 0.0f; }
            void advanceBlock()       noexcept { current = target;    delta = 0.0f; }

            // 4-lane vector for samples [4q, 4q+1, 4q+2, 4q+3] within the block
            SIMD_M128 quad(int q) const noexcept
            {
                const float base = current + delta * static_cast<float>(4 * q);
                return SIMD_MM(add_ps)(SIMD_MM(set1_ps)(base),
                                       SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(delta),
                                                       SIMD_MM(setr_ps)(0.0f, 1.0f, 2.0f, 3.0f)));
            }
            // scalar value at sample offset i within the block
            float at(int i) const noexcept
            {
                return current + delta * static_cast<float>(i);
            }
        };

        template <class T, bool first_run_checks = true>
        struct OnePoleLag
        {
            OnePoleLag() { setRate(static_cast<T>(0.004)); }
            explicit OnePoleLag(T rate) { setRate(rate); }

            void setRate(T lp_)
            {
                lp    = lp_;
                lpinv = static_cast<T>(1) - lp_;
            }

            void setRateInMilliseconds(double miliSeconds, double sampleRate, double blockSizeInv)
            {
                setRate(static_cast<T>(
                    1.0 - std::exp(-2.0 * M_PI / (miliSeconds * 0.001 * sampleRate * blockSizeInv))));
            }

            void setTarget(T f)
            {
                target_v = f;
                if (first_run_checks && first_run)
                {
                    v = target_v;
                    first_run = false;
                }
            }
            void snapTo(T f)
            {
                target_v = f;
                v = f;
                first_run = false;
            }

            void snapToTarget()      { snapTo(target_v); }
            T    getTargetValue() const { return target_v; }
            T    getValue()       const { return v; }

            void process() { v = v * lpinv + target_v * lp; }

            void processN(int n)
            {
                if (n <= 0) return;
                const T decay = std::pow(lpinv, static_cast<T>(n));
                v = target_v + (v - target_v) * decay;
            }

            T v{0};
            T target_v{0};
            bool first_run{true};
          protected:
            T lp{0}, lpinv{0};
        };

        // Ported from sst-basic-blocks dsp/Lag.h
        // SurgeLag: legacy-named alias with newValue/instantize aliases.
        template <typename T, bool first_run_checks = true>
        struct SurgeLag : OnePoleLag<T, first_run_checks>
        {
            SurgeLag()                : OnePoleLag<T, first_run_checks>() {}
            explicit SurgeLag(T lp_)  : OnePoleLag<T, first_run_checks>(lp_) {}

            void newValue(T f)   { this->setTarget(f); }
            void startValue(T f) { this->snapTo(f); }
            void instantize()    { this->snapToTarget(); }
        };

        // RBJ biquad (Direct Form II Transposed). Zero heap allocation.
        // Used on the feedback path to shape the delayed signal spectrum
        // before it's mixed back into the write buffer.
        struct Biquad
        {
            float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            float z1 = 0.0f, z2 = 0.0f;

            void reset() noexcept { z1 = z2 = 0.0f; }

            float processSample(float x) noexcept
            {
                const float y = b0 * x + z1;
                z1 = b1 * x - a1 * y + z2;
                z2 = b2 * x - a2 * y;
                return y;
            }

            void setLowPass(double fs, double fc, double Q) noexcept
            {
                const double fcc = std::clamp(fc, 20.0, 0.49 * fs);
                const double w = 2.0 * M_PI * fcc / fs;
                const double cw = std::cos(w), sw = std::sin(w);
                const double alpha = sw / (2.0 * Q);
                const double a0 = 1.0 + alpha;
                b0 = static_cast<float>((1.0 - cw) * 0.5 / a0);
                b1 = static_cast<float>((1.0 - cw)       / a0);
                b2 = b0;
                a1 = static_cast<float>(-2.0 * cw / a0);
                a2 = static_cast<float>((1.0 - alpha) / a0);
            }

            void setHighPass(double fs, double fc, double Q) noexcept
            {
                const double fcc = std::clamp(fc, 20.0, 0.49 * fs);
                const double w = 2.0 * M_PI * fcc / fs;
                const double cw = std::cos(w), sw = std::sin(w);
                const double alpha = sw / (2.0 * Q);
                const double a0 = 1.0 + alpha;
                b0 = static_cast<float>((1.0 + cw) * 0.5 / a0);
                b1 = static_cast<float>(-(1.0 + cw)      / a0);
                b2 = b0;
                a1 = static_cast<float>(-2.0 * cw / a0);
                a2 = static_cast<float>((1.0 - alpha) / a0);
            }
        };

        void updateFilterCoeffs() noexcept
        {
            constexpr double Q = 0.707;
            fbLP_L.setLowPass (sampleRate, highCutHz, Q);
            fbLP_R.setLowPass (sampleRate, highCutHz, Q);
            fbHP_L.setHighPass(sampleRate, lowCutHz,  Q);
            fbHP_R.setHighPass(sampleRate, lowCutHz,  Q);
        }

        SampleType softClip(SampleType x) noexcept
        {
            return fasterTanhBounded(x);
        }

        std::vector<SampleType> bufferL, bufferR;
        // scratch buffers are thread_local to avoid per-instance allocation while remaining thread-safe.
        // Assumes no re-entrant process() on the same thread (e.g. sidechain feedback loops).
        alignas(16) static thread_local inline float tL [N_BLOCK];  // scratch: NEW-offset L read
        alignas(16) static thread_local inline float tR [N_BLOCK];  // scratch: NEW-offset R read
        alignas(16) static thread_local inline float tL2[N_BLOCK];  // scratch: OLD-offset L read
        alignas(16) static thread_local inline float tR2[N_BLOCK];  // scratch: OLD-offset R read
        alignas(16) static thread_local inline float dsL[N_BLOCK];  // scratch: filtered/crossfed L feedback signal
        alignas(16) static thread_local inline float dsR[N_BLOCK];  // scratch: filtered/crossfed R feedback signal
        alignas(16) static thread_local inline float wL [N_BLOCK];  // scratch: write-back L
        alignas(16) static thread_local inline float wR [N_BLOCK];  // scratch: write-back R

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

        LipolSIMD          lMix, lFbL, lFbR, lCrossfeed;
        SurgeLag<float>    lagDelayMs;

        // Feedback-path filters (per channel): highpass before lowpass.
        Biquad fbLP_L, fbLP_R, fbHP_L, fbHP_R;

        // Filter + crossfeed parameter targets. lowCutHz / highCutHz trigger
        // coefficient recomputation at the top of process() when they change.
        float lowCutHz       = 20.0f;
        float highCutHz      = 20000.0f;
        float lastLowCutHz   = -1.0f;
        float lastHighCutHz  = -1.0f;
        float crossfeed      = 0.0f;

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
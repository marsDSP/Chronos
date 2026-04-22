#pragma once

#ifndef CHRONOS_DELAY_ENGINE_H
#define CHRONOS_DELAY_ENGINE_H

#include <cassert>
#include <JuceHeader.h>
#include "dsp/math/fastermath.h"
#include "dsp/engine/delay/waveshaper.h"
#include "dsp/filter/splane_curvefit_highpass.h"
#include "dsp/filter/splane_curvefit_lowpass.h"
#include "dsp/buffers/aligned_simd_buffer.h"
#include "dsp/buffers/aligned_simd_buffer_view.h"
#include "dsp/diffusion/diffusion_processor.h"
#include "dsp/diffusion/fdn_stereo_processor.h"
#include "dsp/bypass/crossfade_bypass_engine.h"

namespace MarsDSP::DSP {
    template<typename SampleType, int N_BLOCK = 4112>
    class DelayEngine {
    public:
        DelayEngine() = default;

        void AllocBuffer() noexcept
        {
            // The circular buffer holds stereo samples in an xsimd-aligned
            // backing store. setMaxSize() is idempotent if the requested
            // size is already <= the allocated size, but we re-check
            // explicitly so resizing only happens on first prepare() call.
            if (circularDelayBuffer.getNumChannels() < 2
                || circularDelayBuffer.getNumSamples() < (kBufSize + kTail))
            {
                circularDelayBuffer.setMaxSize(2, kBufSize + kTail);
                circularDelayBuffer.setCurrentSize(2, kBufSize + kTail);
                circularDelayBuffer.clearAllSamples();
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
            circularDelayBuffer.clearAllSamples();

            // snap smoothers so the first block after reset doesn't ramp from 0.
            lMix.instantize(std::clamp(mix,             0.0f, 1.0f));
            lFbL.instantize(std::clamp(feedbackL,       0.0f, 0.99f));
            lFbR.instantize(std::clamp(feedbackR,       0.0f, 0.99f));
            lCrossfeed.instantize(std::clamp(crossfeed, 0.0f, 1.0f));
            lagDelayMs.newValue(std::clamp(delayTime,   minDelayTime, maxDelayTime));
            lagDelayMs.instantize();

            // Clear s-plane feedback-path filter state on reset.
            fbLP_L.reset(); fbLP_R.reset();
            fbHP_L.reset(); fbHP_R.reset();

            // Flush the diffusion chain's delay state.
            stereoDiffusionProcessor.reset();

            // Flush the FDN's delay state + shelf filters.
            stereoFdnProcessor.reset();

            // Latch current on/off states in the bypass engines so the
            // first block after reset doesn't trigger a spurious crossfade.
            masterPluginBypassEngine.reset(!bypassed);
            fdnReverbSendBypassEngine.reset(stereoFdnProcessor.getFdnAmount() > static_cast<SampleType>(0));

            // Clear ADAA carry state so first samples start from x=0, F=fastTanhIntegral(0)=0.
            adaaFbL .reset(); adaaFbR .reset();
            adaaOutL.reset(); adaaOutR.reset();
        }

        void prepare(const dsp::ProcessSpec &spec) noexcept
        {
            sampleRate = spec.sampleRate;
            AllocBuffer();

            // 10ms attack, 50ms release for ducking response
            duckAtkCoeff = static_cast<SampleType>(1.0 - std::exp(-1.0 / (0.010 * sampleRate)));
            duckRelCoeff = static_cast<SampleType>(1.0 - std::exp(-1.0 / (0.050 * sampleRate)));

            lagDelayMs.setRateInMilliseconds(150.0, sampleRate, 1.0);

            // Push the sample rate into each s-plane feedback-path filter
            // before computing cutoff-dependent coefficients.
            fbLP_L.prepare(sampleRate); fbLP_R.prepare(sampleRate);
            fbHP_L.prepare(sampleRate); fbHP_R.prepare(sampleRate);

            // Prepare the Dattorro-style stereo diffusion processor.
            // The processor bypasses itself when diffusionAmount is 0, so
            // this is zero-cost until the user dials the diffusion in.
            stereoDiffusionProcessor.prepare(sampleRate);

            // Prepare the Householder-mixed FDN processor. Also bypass-when-amount=0.
            stereoFdnProcessor.prepare(sampleRate);

            // Prepare the crossfade bypass engines. masterPluginBypassEngine
            // wraps the whole engine for the plugin's Bypass parameter.
            // fdnReverbSendBypassEngine wraps the post-delay FDN loop so
            // moving the FDN amount across zero doesn't click.
            const int preparedMaxBlockSize =
                std::max(static_cast<int>(spec.maximumBlockSize), 1);
            const int preparedNumberOfChannels =
                std::max(static_cast<int>(spec.numChannels), 2);
            masterPluginBypassEngine.prepare(
                preparedMaxBlockSize, preparedNumberOfChannels, !bypassed);
            fdnReverbSendBypassEngine.prepare(
                preparedMaxBlockSize, preparedNumberOfChannels,
                stereoFdnProcessor.getFdnAmount()
                    > static_cast<SampleType>(0));

            // Initial filter coefficients.
            updateFilterCoeffs();
            lastLowCutHz  = lowCutHz;
            lastHighCutHz = highCutHz;

            reset();
        }

        void process(const AlignedBuffers::AlignedSIMDBufferView<SampleType> &block,
                     const int numSamples) noexcept
        {
            // Master crossfade bypass: captures the dry input on the block
            // where the Bypass parameter toggles and tells us whether to
            // run the engine at all. A matching
            // crossfadeWithCapturedDryInputIfTransitioning call at the end
            // of process() performs a one-block fade so Bypass never clicks.
            const bool masterEngineShouldRunThisBlock =
                masterPluginBypassEngine
                    .captureDryInputAndDecideWhetherToProcess(block, !bypassed);

            if (!masterEngineShouldRunThisBlock)
            {
                // Steady-state bypass: output stays equal to dry input,
                // skip all processing. No transition in flight so the
                // crossfade would be a no-op either way; return early.
                return;
            }

            const int numCh = block.getNumChannels();
            auto *ch0 = numCh > 0 ? block.getChannelPointer(0) : nullptr;
            auto *ch1 = numCh > 1 ? block.getChannelPointer(1) : nullptr;

            // Pull the circular buffer's raw per-channel pointers once per
            // block so the hot loops below still see the same plain
            // SampleType* they had before the std::vector -> AlignedSIMDBuffer
            // migration.
            auto* const bufferL = circularDelayBuffer.getWritePointer(0);
            auto* const bufferR = circularDelayBuffer.getWritePointer(1);

            // push targets into block-rate linear ramp smoothers
            // SIMD inner loops consume per-quad vectors via .quad(q).
            lMix.setTarget(std::clamp(mix,             0.0f, 1.0f),  numSamples);
            lFbL.setTarget(std::clamp(feedbackL,       0.0f, 0.99f), numSamples);
            lFbR.setTarget(std::clamp(feedbackR,       0.0f, 0.99f), numSamples);
            lCrossfeed.setTarget(std::clamp(crossfeed, 0.0f, 1.0f),  numSamples);

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

            const auto numSamplesSize = static_cast<size_t>(numSamples);
            assert(numSamplesSize <= N_BLOCK - 8);

            auto msToPos = [&](float ms)
            {
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
            auto readScratch = [&](const SampleType* src, SampleType* dst, int rpos)
            {
                const int first = std::min(total, kBufSize + kTail - rpos);
                std::memcpy(dst, src + rpos, first * sizeof(SampleType));

                if (first < total)
                    std::memcpy(dst + first, src + kTail, (total - first) * sizeof(SampleType));
            };

            readScratch(bufferL, tL,  (writeIdxL - offsetNew) & kBufMask);
            readScratch(bufferL, tL2, (writeIdxL - offsetOld) & kBufMask);

            if (!isMono() && ch1 != nullptr)
            {
                readScratch(bufferR, tR,  (writeIdxR - offsetNew) & kBufMask);
                readScratch(bufferR, tR2, (writeIdxR - offsetOld) & kBufMask);
            }

            // Dual Lagrange coefficient set: N for new frac, O for old frac.
            auto computeCoeffs = [](SampleType frac, LagrangeCoeffs& c)
            {
                const float d1 = frac - 1.0f;
                const float d2 = frac - 2.0f;
                const float d3 = frac - 3.0f;
                const float d4 = frac - 4.0f;
                const float d5 = frac - 5.0f;
                c.c[0] = -d1 * d2 * d3 * d4 * d5 / 120.0f;
                c.c[1] =  d2 * d3 * d4 * d5      /  24.0f;
                c.c[2] = -d1 * d3 * d4 * d5      /  12.0f;
                c.c[3] =  d1 * d2 * d4 * d5      /  12.0f;
                c.c[4] = -d1 * d2 * d3 * d5      /  24.0f;
                c.c[5] =  d1 * d2 * d3 * d4      / 120.0f;
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
            const float invN = (numSamplesSize > 0) ? 1.0f / static_cast<float>(numSamplesSize) : 0.0f;
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

                            vYN = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1N), SIMD_MM(mul_ps)(vFracN, vSum));
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

                            vYO = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1O), SIMD_MM(mul_ps)(vFracO, vSum));
                        }

                        const auto vDelayedOut = SIMD_MM(add_ps)(vYO, SIMD_MM(mul_ps)(vAlpha, SIMD_MM(sub_ps)(vYN, vYO)));

                        SIMD_MM(store_ps)(&dsL[n], vDelayedOut);
                    }
                    for (; n < numSamplesSize; ++n)
                    {
                        const auto alpha = static_cast<SampleType>(static_cast<float>(n) * invN);
                        const SampleType yN = readInterpolated(tL,  static_cast<int>(n), coeffsN);
                        const SampleType yO = readInterpolated(tL2, static_cast<int>(n), coeffsO);
                        dsL[n] = static_cast<float>(yO + alpha * (yN - yO));
                    }
                }

                // ---------------- PASS 2: scalar HP → LP on dsL[] -------------------
                // The s-plane filter sections are stateful so this pass is intrinsically
                // scalar, but it's a tight sequential loop over ~4KB in L1 so it's cheap.
                for (size_t k = 0; k < numSamplesSize; ++k)
                    dsL[k] = fbLP_L.processSample(fbHP_L.processSample(dsL[k]));

                // ---------------- PASS 3: SIMD feedback MAC + dry/wet mix -----------
                {
                    // Pull carries into locals so ADAA can keep them in registers.
                    float cxFb  = adaaFbL .getCarryX(), cfFb  = adaaFbL .getCarryF();
                    float cxOut = adaaOutL.getCarryX(), cfOut = adaaOutL.getCarryF();

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

                        // ADAA softclip on the write-feedback branch.
                        const auto vFbArg = SIMD_MM(add_ps)(vMonoSum, SIMD_MM(mul_ps)(vFb, vDuckedOut));
                        const auto vWriteVal = fasterTanhADAA(vFbArg, cxFb, cfFb);
                        SIMD_MM(storeu_ps)(&wL[n], vWriteVal);

                        // ADAA softclip on the dry/wet output branch.
                        const auto vOutArg = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vDuckedOut, vMix), SIMD_MM(mul_ps)(vMonoSum, vOneMinusMix));
                        const auto vOut = fasterTanhADAA(vOutArg, cxOut, cfOut);

                        if (ch0 != nullptr) SIMD_MM(storeu_ps)(ch0 + n, vOut);
                        if (ch1 != nullptr) SIMD_MM(storeu_ps)(ch1 + n, vOut);
                    }
                    for (; n < numSamplesSize; ++n)
                    {
                        const auto mixP       = static_cast<SampleType>(lMix.at(static_cast<int>(n)));
                        const SampleType oneMinusMx = static_cast<SampleType>(1) - mixP;
                        const auto fbP        = static_cast<SampleType>(lFbL.at(static_cast<int>(n)));

                        SampleType monoSum = ch0 != nullptr ? ch0[n] : static_cast<SampleType>(0);
                        if (ch1 != nullptr)
                            monoSum = static_cast<SampleType>(0.5) * (monoSum + ch1[n]);

                        const SampleType duckedOut = static_cast<SampleType>(dsL[n]) * duckGain;

                        wL[n] = static_cast<SampleType>(fasterTanhADAA(static_cast<float>(monoSum + fbP * duckedOut), cxFb, cfFb));
                        const auto out = static_cast<SampleType>(fasterTanhADAA(static_cast<float>(duckedOut * mixP + monoSum * oneMinusMx), cxOut, cfOut));

                        if (ch0 != nullptr) ch0[n] = out;
                        if (ch1 != nullptr) ch1[n] = out;
                    }

                    adaaFbL .setCarry(cxFb,  cfFb);
                    adaaOutL.setCarry(cxOut, cfOut);
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

                            vYL_N = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1N), SIMD_MM(mul_ps)(vFracN, vSum));
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

                            vYL_O = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1O), SIMD_MM(mul_ps)(vFracO, vSum));
                        }
                        const auto vDelL = SIMD_MM(add_ps)(vYL_O, SIMD_MM(mul_ps)(vAlpha, SIMD_MM(sub_ps)(vYL_N, vYL_O)));
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

                            vYR_N = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1N), SIMD_MM(mul_ps)(vFracN, vSum));
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

                            vYR_O = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(v0, vC1O), SIMD_MM(mul_ps)(vFracO, vSum));
                        }
                        const auto vDelR = SIMD_MM(add_ps)(vYR_O, SIMD_MM(mul_ps)(vAlpha, SIMD_MM(sub_ps)(vYR_N, vYR_O)));
                        SIMD_MM(store_ps)(&dsR[n], vDelR);
                    }
                    for (; n < numSamplesSize; ++n)
                    {
                        const auto alpha = static_cast<SampleType>(static_cast<float>(n) * invN);

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
                // input. This is the ping-pong path. After the crossfeed mix,
                // each sample is driven through the Dattorro-style stereo
                // diffusion processor; when diffusionAmount is 0 this call is
                // branch-free pass-through so existing behaviour is preserved.
                for (size_t k = 0; k < numSamplesSize; ++k)
                {
                    const float filtL = fbLP_L.processSample(fbHP_L.processSample(dsL[k]));
                    const float filtR = fbLP_R.processSample(fbHP_R.processSample(dsR[k]));
                    const float cf    = lCrossfeed.at(static_cast<int>(k));
                    const float cfInv = 1.0f - cf;

                    float feedbackPathLeft  = cfInv * filtL + cf * filtR;
                    float feedbackPathRight = cfInv * filtR + cf * filtL;

                    // Diffusion lives in the feedback loop (safe: it is
                    // built from orthogonal mixers + passive delays and is
                    // always bounded by |H| <= 1 per stage). The whole
                    // diffusion stage is gated by the Reverb Bypass flag.
                    if (!reverbChainBypassed)
                        stereoDiffusionProcessor.processSingleStereoPairInPlace(
                            feedbackPathLeft, feedbackPathRight);

                    // NOTE: the FDN is NOT applied here. Putting an FDN
                    // inside another feedback loop produces compound loop
                    // gains > 1 at non-DC frequencies for long T60 and
                    // feeds back into infinity. The FDN is applied as a
                    // post-delay reverb send on the final output buffer
                    // instead (see the block after the circular-buffer
                    // writes below).

                    dsL[k] = feedbackPathLeft;
                    dsR[k] = feedbackPathRight;
                }

                // ---------------- PASS 3: SIMD feedback MAC + dry/wet mix ---------
                {
                    // carry locals for each of the four ADAA softclip streams.
                    float cxFbL  = adaaFbL .getCarryX(), cfFbL  = adaaFbL .getCarryF();
                    float cxOutL = adaaOutL.getCarryX(), cfOutL = adaaOutL.getCarryF();
                    float cxFbR  = adaaFbR .getCarryX(), cfFbR  = adaaFbR .getCarryF();
                    float cxOutR = adaaOutR.getCarryX(), cfOutR = adaaOutR.getCarryF();

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

                            const auto vFbArgL = SIMD_MM(add_ps)(vXL,SIMD_MM(mul_ps)(vFbL, vYL_ducked));
                            const auto vWriteValL = fasterTanhADAA(vFbArgL, cxFbL, cfFbL);
                            SIMD_MM(storeu_ps)(&wL[n], vWriteValL);

                            const auto vOutArgL = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vYL_ducked, vMix), SIMD_MM(mul_ps)(vXL, vOneMinusMix));
                            const auto vOutL = fasterTanhADAA(vOutArgL, cxOutL, cfOutL);
                            SIMD_MM(storeu_ps)(ch0 + n, vOutL);
                        }

                        if (ch1 != nullptr)
                        {
                            const auto vFbR = lFbR.quad(q);
                            auto vXR = SIMD_MM(loadu_ps)(ch1 + n);
                            auto vYR_ducked = SIMD_MM(mul_ps)(SIMD_MM(load_ps)(&dsR[n]), vDuckGain);

                            const auto vFbArgR = SIMD_MM(add_ps)(vXR, SIMD_MM(mul_ps)(vFbR, vYR_ducked));
                            const auto vWriteValR = fasterTanhADAA(vFbArgR, cxFbR, cfFbR);
                            SIMD_MM(storeu_ps)(&wR[n], vWriteValR);

                            const auto vOutArgR = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vYR_ducked, vMix), SIMD_MM(mul_ps)(vXR, vOneMinusMix));
                            const auto vOutR = fasterTanhADAA(vOutArgR, cxOutR, cfOutR);
                            SIMD_MM(storeu_ps)(ch1 + n, vOutR);
                        }
                    }
                    for (; n < numSamplesSize; ++n)
                    {
                        const auto mixP             = static_cast<SampleType>(lMix.at(static_cast<int>(n)));
                        const SampleType oneMinusMx = static_cast<SampleType>(1) - mixP;

                        if (ch0 != nullptr)
                        {
                            const auto fbLP            = static_cast<SampleType>(lFbL.at(static_cast<int>(n)));
                            const SampleType xL        = ch0[n];
                            const SampleType yL_ducked = static_cast<SampleType>(dsL[n]) * duckGain;

                            wL[n]  = static_cast<SampleType>(fasterTanhADAA(static_cast<float>(xL + fbLP * yL_ducked), cxFbL, cfFbL));
                            ch0[n] = static_cast<SampleType>(fasterTanhADAA(static_cast<float>(yL_ducked * mixP + xL * oneMinusMx), cxOutL, cfOutL));
                        }
                        if (ch1 != nullptr)
                        {
                            const auto fbRP            = static_cast<SampleType>(lFbR.at(static_cast<int>(n)));
                            const SampleType xR        = ch1[n];
                            const SampleType yR_ducked = static_cast<SampleType>(dsR[n]) * duckGain;

                            wR[n]  = static_cast<SampleType>(fasterTanhADAA(static_cast<float>(xR + fbRP * yR_ducked), cxFbR, cfFbR));
                            ch1[n] = static_cast<SampleType>(fasterTanhADAA(static_cast<float>(yR_ducked * mixP + xR * oneMinusMx), cxOutR, cfOutR));
                        }
                    }

                    adaaFbL .setCarry(cxFbL,  cfFbL);
                    adaaOutL.setCarry(cxOutL, cfOutL);
                    adaaFbR .setCarry(cxFbR,  cfFbR);
                    adaaOutR.setCarry(cxOutR, cfOutR);
                }

                // Block write & Mirror L
                if (ch0 != nullptr)
                {
                    const int oldIdxL = writeIdxL;
                    const bool wrappedL = (oldIdxL + static_cast<int>(numSamplesSize)) > kBufSize;
                    if (wrappedL)
                    {
                        for (size_t k = 0; k < numSamplesSize; ++k)
                            bufferL[(oldIdxL + static_cast<int>(k)) & kBufMask] = wL[k];
                    }
                    else
                    {
                        std::memcpy(&bufferL[oldIdxL], wL, numSamplesSize * sizeof(SampleType));
                    }
                    writeIdxL = (oldIdxL + static_cast<int>(numSamplesSize)) & kBufMask;

                    if (wrappedL || oldIdxL < kTail)
                    {
                        for (int k = 0; k < kTail; ++k)
                            bufferL[kBufSize + k] = bufferL[k];
                    }
                }

                // Block write & Mirror R
                if (ch1 != nullptr)
                {
                    const int oldIdxR = writeIdxR;
                    const bool wrappedR = (oldIdxR + static_cast<int>(numSamplesSize)) > kBufSize;
                    if (wrappedR)
                    {
                        for (size_t k = 0; k < numSamplesSize; ++k)
                            bufferR[(oldIdxR + static_cast<int>(k)) & kBufMask] = wR[k];
                    }
                    else
                    {
                        std::memcpy(&bufferR[oldIdxR], wR, numSamplesSize * sizeof(SampleType));
                    }
                    writeIdxR = (oldIdxR + static_cast<int>(numSamplesSize)) & kBufMask;

                    if (wrappedR || oldIdxR < kTail)
                    {
                        for (int k = 0; k < kTail; ++k)
                            bufferR[kBufSize + k] = bufferR[k];
                    }
                }

                // ---------------- POST-DELAY FDN REVERB SEND ----------------
                // Apply the Householder-mixed FDN strictly downstream of the
                // delay's feedback loop, so its internal loop gain never
                // compounds with the delay's feedback gain. Safe because the
                // FDN's output only reaches the listener - it is NEVER
                // written back to the delay's circular buffer. Wrapped in
                // its own CrossfadeBypassEngine instance so moving the FDN
                // amount across zero produces a click-free transition.
                if (ch0 != nullptr && ch1 != nullptr)
                {
                    // Single reverb-bypass gate: when the user flips the
                    // Reverb Bypass button we want the FDN to fade out
                    // click-free via its bypass engine, so fold the flag
                    // into the 'is active' condition here rather than
                    // branching around the bypass engine.
                    const bool fdnReverbIsActiveThisBlock =
                        !reverbChainBypassed
                        && stereoFdnProcessor.getFdnAmount()
                            > static_cast<SampleType>(0);

                    const bool fdnReverbShouldRunThisBlock =
                        fdnReverbSendBypassEngine
                            .captureDryInputAndDecideWhetherToProcess(
                                block, fdnReverbIsActiveThisBlock);

                    if (fdnReverbShouldRunThisBlock)
                    {
                        for (size_t k = 0; k < numSamplesSize; ++k)
                        {
                            float reverbSendLeft  = ch0[k];
                            float reverbSendRight = ch1[k];
                            stereoFdnProcessor.processSingleStereoPairInPlace(
                                reverbSendLeft, reverbSendRight);
                            ch0[k] = reverbSendLeft;
                            ch1[k] = reverbSendRight;
                        }
                    }

                    fdnReverbSendBypassEngine
                        .crossfadeWithCapturedDryInputIfTransitioning(
                            block, fdnReverbIsActiveThisBlock);
                }
            }

            // advance block-rate ramps so next block starts from this block's target
            lMix.advanceBlock();
            lFbL.advanceBlock();
            lFbR.advanceBlock();
            lCrossfeed.advanceBlock();

            // Close out the master bypass crossfade. When the user toggles
            // Bypass this block, this is where the one-block linear fade
            // between the captured dry input and the freshly-processed
            // output is applied in place to the block.
            masterPluginBypassEngine
                .crossfadeWithCapturedDryInputIfTransitioning(block, !bypassed);
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

        // Diffusion amount: 0 = bypassed (default), 1 = full wet through
        // the Dattorro-style diffusion chain. Applied inside the feedback
        // loop so it shapes the repeats only.
        void setDiffusionAmountParam(const float normalisedAmount) noexcept
        {
            stereoDiffusionProcessor.setDiffusionAmount(
                std::clamp(normalisedAmount, 0.0f, 1.0f));
        }

        // Diffusion size in milliseconds: the overall diffusion window
        // that the chain spreads its per-stage delays across.
        void setDiffusionSizeMsParam(const float milliseconds) noexcept
        {
            // Clamp high enough to be audible but well below the ring size
            // of the diffusion delay bank (kDefaultMaxDiffusionDelaySamplesPerChannel).
            stereoDiffusionProcessor.setDiffusionSizeInMilliseconds(
                std::clamp(milliseconds, 1.0f, 200.0f));
        }

        // FDN amount: 0 = bypass, 1 = full Lexicon-style wet.
        void setFdnAmountParam(const float normalisedAmount) noexcept
        {
            stereoFdnProcessor.setFdnAmount(
                std::clamp(normalisedAmount, 0.0f, 1.0f));
        }

        // FDN size = longest per-channel delay in the network (ms).
        void setFdnSizeMsParam(const float milliseconds) noexcept
        {
            stereoFdnProcessor.setFdnSizeInMilliseconds(
                std::clamp(milliseconds, 10.0f, 200.0f));
        }

        // FDN decay (T60 low-frequency) in ms. High-frequency T60 is
        // derived internally as 0.6 * this value.
        void setFdnDecayMsParam(const float milliseconds) noexcept
        {
            stereoFdnProcessor.setFdnDecayInMilliseconds(
                std::clamp(milliseconds, 100.0f, 15000.0f));
        }

        // FDN damping shelf crossover frequency in Hz.
        void setFdnDampingHzParam(const float crossoverHz) noexcept
        {
            stereoFdnProcessor.setFdnDampingCrossoverInHz(
                std::clamp(crossoverHz, 200.0f, 18000.0f));
        }

        // Master bypass for the whole reverb chain (diffusion + FDN). When
        // true both processors are taken out of the signal path; the main
        // delay keeps running normally. Transitions are handled click-free
        // by the diffusion / FDN bypass engines.
        void setReverbBypassedParam(const bool shouldBypassReverb) noexcept
        {
            reverbChainBypassed = shouldBypassReverb;
        }

        [[nodiscard]] bool isReverbBypassed() const noexcept
        {
            return reverbChainBypassed;
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
        // plus a safety margin for s-plane filter ring-down + parameter smoothing.
        // Clamped to a sane maximum so pathological feedback values (~0.99)
        // don't yield absurd tail lengths.
        [[nodiscard]] int ringoutSamples() const noexcept
        {
            if (bypassed) return 0;

            const float delayMs      = std::clamp(delayTime, minDelayTime, maxDelayTime);
            const auto delaySamples  = static_cast<float>(sampleRate * delayMs * 0.001);
            const float fb           = std::clamp(std::max(feedbackL, feedbackR), 0.0f, 0.9999f);

            constexpr float silenceDb  = 0.001f;  // -60 dB
            constexpr int   kMargin    = 2048;    // s-plane filter ring-down + smoothers
            constexpr int   kMaxTail   = 1 << 20; // clamp (~21.8 s @ 48 kHz)

            if (fb < 1.0e-4f) return std::min(static_cast<int>(delaySamples) + kMargin, kMaxTail);

            const int repeats = static_cast<int>(std::ceil(std::log(silenceDb) / std::log(fb)));
            const long long tail = static_cast<long long>(delaySamples) * static_cast<long long>(std::max(repeats, 1)) + kMargin;

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
        static void remapParametersForStreamingVersion(int16_t streamedFrom, float* const /*parameters*/) noexcept
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

            void instantize(float v)  noexcept { current = target = v; delta = 0.0f; }
            void advanceBlock()       noexcept { current = target;     delta = 0.0f; }

            // 4-lane vector for samples [4q, 4q+1, 4q+2, 4q+3] within the block
            [[nodiscard]] SIMD_M128 quad(int q) const noexcept
            {
                const float base = current + delta * static_cast<float>(4 * q);
                return SIMD_MM(add_ps)(SIMD_MM(set1_ps)(base), SIMD_MM(mul_ps)(SIMD_MM(set1_ps)(delta), SIMD_MM(setr_ps)(0.0f, 1.0f, 2.0f, 3.0f)));
            }
            // scalar value at sample offset i within the block
            [[nodiscard]] float at(int i) const noexcept
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
                setRate(static_cast<T>(1.0 - std::exp(-2.0 * M_PI / (miliSeconds * 0.001 * sampleRate * blockSizeInv))));
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

            void snapToTarget()         { snapTo(target_v); }
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
            SurgeLag()                : OnePoleLag<T, first_run_checks>()    {}
            explicit SurgeLag(T lp_)  : OnePoleLag<T, first_run_checks>(lp_) {}

            void newValue(T f)   { this->setTarget(f); }
            void startValue(T f) { this->snapTo(f); }
            void instantize()    { this->snapToTarget(); }
        };

        // Push the current low-cut / high-cut corners into the s-plane
        // feedback-path filters. The prototype (Butterworth s-plane poles +
        // residues) is baked into each filter at construction time; only
        // the cutoff is rescaled per call, so this is cheap.
        void updateFilterCoeffs() noexcept
        {
            fbLP_L.setCutoffFrequencyInHz(highCutHz);
            fbLP_R.setCutoffFrequencyInHz(highCutHz);
            fbHP_L.setCutoffFrequencyInHz(lowCutHz);
            fbHP_R.setCutoffFrequencyInHz(lowCutHz);
        }

        SampleType softClip(SampleType x) noexcept
        {
            return fasterTanhBounded(x);
        }

        // Circular delay buffer. 2 channels x (kBufSize + kTail) samples,
        // heap-allocated through xsimd's aligned allocator so every channel
        // base pointer satisfies the SIMD batch alignment. Replaces the
        // previous std::vector<SampleType> pair.
        AlignedBuffers::AlignedSIMDBuffer<SampleType> circularDelayBuffer {};

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

        SampleType prevPos      = static_cast<SampleType>(0);
        SampleType duckGain     = static_cast<SampleType>(1);
        SampleType duckAtkCoeff = static_cast<SampleType>(1);
        SampleType duckRelCoeff = static_cast<SampleType>(1);

        static constexpr float minDelayTime = 5.0f;
        static constexpr float maxDelayTime = 5000.0f;

        float mix       = 1.0f;
        float feedbackL = 0.0f;
        float feedbackR = 0.0f;
        float delayTime = 50.0f;

        LipolSIMD       lMix, lFbL, lFbR, lCrossfeed;
        SurgeLag<float> lagDelayMs;

        Waveshapers::ADAATanh adaaFbL,  adaaFbR;
        Waveshapers::ADAATanh adaaOutL, adaaOutR;

        SPlaneCurveFit::SPlaneCurveFitLowpassFilter  fbLP_L, fbLP_R;
        SPlaneCurveFit::SPlaneCurveFitHighpassFilter fbHP_L, fbHP_R;

        Diffusion::DiffusionProcessor<SampleType> stereoDiffusionProcessor{};
        Diffusion::FdnStereoProcessor<SampleType> stereoFdnProcessor{};

        Bypass::CrossfadeBypassEngine<SampleType> masterPluginBypassEngine{};
        Bypass::CrossfadeBypassEngine<SampleType> fdnReverbSendBypassEngine{};

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
        static constexpr int kTail    = 8; // for the 5th-order Lagrange window

        int writeIdxL = 0;
        int writeIdxR = 0;

        bool mono = false;
        bool bypassed = false;

        // Master bypass for the reverb chain (diffusion + FDN). When true,
        // the diffusion call in Pass-2 is skipped and the FDN bypass
        // engine transitions the FDN out. The delay itself is unaffected.
        bool reverbChainBypassed = false;
    };
}
#endif

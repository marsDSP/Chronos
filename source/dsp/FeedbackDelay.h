#pragma once

#ifndef CHRONOS_FEEDBACK_DELAY_H
#define CHRONOS_FEEDBACK_DELAY_H

#include "BlockTapReader.h"
#include "Diffuser.h"
#include "FracDelayTap.h"
#include "LinearSmoother.h"
#include "Modulation.h"
#include "Pow2RingBuffer.h"
#include "bbd/BrigadeLine.h"
#include "bbd/ClockModel.h"
#include "bbd/CompanderCell.h"
#include "math/SaturatorMakeup.h"
#include "math/Trigonometry.h"
#include "nonlinear/ADAA1.h"
#include "nonlinear/ADAA2.h"
#include "nonlinear/Nonlinearities.h"
#include "utils/memory/BumpArena.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace MarsDSP::Delays
{
    class FeedbackDelay
    {
    public:
        static constexpr float kMaxFeedback = 1.2f;
        static constexpr float kMinLoopDelay = 4.0f; // > FracDelayTap's 3.0 contract
        static constexpr float kMaxGlideStep = 4.0f; // maximum delay glide, in samples per sample

        static constexpr int kMaxChunk = 64; // max sub-chunk length (ramp-array footprint)
        static constexpr int kChunkGuard = 6; // interpolator window (base = wIdx - i - 3, len 6 ≤ kTail)
        static constexpr std::uint64_t kModSeed = 0xC47051D5uLL; // modulation RNG seed constant

        struct Params
        {
            float delaySamples = 4800.0f;
            float feedback = 0.0f; // 0..kMaxFeedback; > 1 self-oscillates, bounded
            float dampHz = 6000.0f; // one-pole lowpass in the loop
            float loopCutHz = 40.0f; // one-pole highpass in the loop
            float crossFeed = 0.0f; // 0 straight, 1 full ping-pong
            float loopDrive = 1.0f; // how hard repeats lean on the tanh ceiling
            int satOrder = 2; // 0 hard, 1 ADAA1, 2 ADAA2
            float diffusion = 0.7f; // 0..1 -> allpass coeff 0..0.92
            float diffuserSize = 0.5f; // 0..1 (1 = full path length)
            float diffModDepth = 0.30f; // milliseconds, 0..1.5
            float diffModRateHz = 0.5f; // 0..8
            bool enableDiffuser = false; // off by default
            float delayModDepth = 0.0f; // cents, 0..50
            float delayModRateHz = 0.35f; // Hz, 0.01..10
            int delayMode = 0; // 0: Digital, 1: BBD
        };

        void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples) noexcept
        {
            prepareImpl_(sampleRate, maxBlockSize, maxDelaySamples, nullptr);
        }

        void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples, Memory::BumpArena &arena) noexcept
        {
            prepareImpl_(sampleRate, maxBlockSize, maxDelaySamples, &arena);
        }

        static std::size_t ringStorageFloats(double sampleRate, int maxBlockSize, int maxDelaySamples) noexcept
        {
            const int minCap = maxDelaySamples + maxBlockSize + Pow2RingBuffer::kTail + 8;
            return 2 * Pow2RingBuffer::arenaFloatsFor(minCap)
                 + Diffusion::Diffuser::ringStorageFloats(sampleRate)
                 + BBD::BrigadeLine::bbdStorageFloats(2);
        }

        void reset() noexcept
        {
            ringL_.clear();
            ringR_.clear();
            writeIdx_ = 0;
            bbdL_.reset();
            bbdR_.reset();
            compL_.reset();
            compR_.reset();
            expL_.reset();
            expR_.reset();
            adaa1L_.reset();
            adaa1R_.reset();
            adaa2L_.reset();
            adaa2R_.reset();
            dampL_ = 0.0f;
            dampR_ = 0.0f;
            cutLpL_ = 0.0f;
            cutLpR_ = 0.0f;
            dcXL_ = 0.0f;
            dcXR_ = 0.0f;
            dcYL_ = 0.0f;
            dcYR_ = 0.0f;
            diffuser_.reset();
            enableDiffuser_ = false;
            diffState_ = DiffuserState::Off;
            diffFade_ = 0.0f;
            crossCos_ = 1.0f;
            crossSin_ = 0.0f;
            ouL_.reset();
            ouR_.reset();
            rngL_.seed(kModSeed, 1);
            rngR_.seed(kModSeed, 2);
            firstBlock_ = true;
            lastDelayMode_ = 0;
        }

        void resetParams(const Params &p) noexcept
        {
            applyBlockRate_(p);
            delaySm_.reset(sampleRate_, 0.020);
            delaySm_.setCurrentAndTargetValue(clampDelay_(p.delaySamples));
            lastGlideRampTime_ = 0.020;
            fbSm_.setCurrentAndTargetValue(std::clamp(p.feedback, 0.0f, kMaxFeedback));
            crossSm_.setCurrentAndTargetValue(std::clamp(p.crossFeed, 0.0f, 1.0f));
            driveSm_.setCurrentAndTargetValue(std::clamp(p.loopDrive, 0.501f, 15.849f));
            dampGSm_.setCurrentAndTargetValue(dampGSm_.getTargetValue());
            dampG_ = dampGSm_.getCurrentValue();
            cutGSm_.setCurrentAndTargetValue(cutGSm_.getTargetValue());
            cutG_ = cutGSm_.getCurrentValue();
            satLatencySm_.setCurrentAndTargetValue(satLatencySm_.getTargetValue());
            satLatency_ = satLatencySm_.getCurrentValue();
            modKSm_.setCurrentAndTargetValue(modKSm_.getTargetValue());
            applyDiffuserParams_(p);
            diffuser_.prime();
            enableDiffuser_ = p.enableDiffuser;
            delayMode_ = p.delayMode;
            diffState_ = enableDiffuser_ ? DiffuserState::On : DiffuserState::Off;
            diffFade_ = enableDiffuser_ ? 1.0f : 0.0f;
            firstBlock_ = false;
            lastDelayMode_ = p.delayMode;
        }

        void setEnvelopeFreeze(bool freeze) noexcept
        {
            compL_.setEnvelopeFreeze(freeze);
            compR_.setEnvelopeFreeze(freeze);
            expL_.setEnvelopeFreeze(freeze);
            expR_.setEnvelopeFreeze(freeze);
        }

        void setParams(const Params &p) noexcept
        {
            if (firstBlock_)
            {
                resetParams(p);
                return;
            }
            applyBlockRate_(p);
            retargetDelayGlide_(p.delaySamples);
            fbSm_.setTargetValue(std::clamp(p.feedback, 0.0f, kMaxFeedback));
            crossSm_.setTargetValue(std::clamp(p.crossFeed, 0.0f, 1.0f));
            driveSm_.setTargetValue(std::clamp(p.loopDrive, 0.501f, 15.849f));
            applyDiffuserParams_(p);
            enableDiffuser_ = p.enableDiffuser;
            // On the digital-to-bbd edge, prime the bucket register
            // from the ring. The first bbd repeats continue the audio.
            if (lastDelayMode_ == 0 && p.delayMode == 1)
                primeBbdFromRing_();
            delayMode_ = p.delayMode;
            lastDelayMode_ = p.delayMode;
        }

        void process(const float *inL, const float *inR, float *wetL, float *wetR, int n) noexcept
        {
            assert(inL != nullptr && wetL != nullptr);
            const bool hasR = (inR != nullptr && wetR != nullptr);
            const int mask = ringL_.mask();

            updateCrossRotation_(); // block-rate equal-power cross-feed coefficients
            diffuserTransition_(); // block-rate enable edge (primes on rising)
            const float modMix = std::clamp(crossCos_ * crossCos_ - crossSin_ * crossSin_, 0.0f, 1.0f);

            int s = 0;
            while (s < n)
            {
                const int remaining = n - s;

                const float baseT = (diffState_ != DiffuserState::Off)
                                        ? diffuser_.transportSamples()
                                        : 0.0f;

                if (delayMode_ == 1)
                {
                    const int Lc = std::min(kMaxChunk, remaining);
                    const bool runDiff = (diffState_ != DiffuserState::Off);
                    const float gdBank = static_cast<float>(BBD::BrigadeLine::getBankGroupDelayAtDC(sampleRate_))
                                          + BBD::BrigadeLine::kSplitStepOffset;

                    for (int i = 0; i < Lc; ++i)
                    {
                        const float d = delaySm_.getNextValue();
                        const float g = fbSm_.getNextValue();
                        crossSm_.skip();
                        const float drive = driveSm_.getNextValue();
                        const float fade = fadeStep_();
                        dampG_ = dampGSm_.getNextValue();
                        cutG_ = cutGSm_.getNextValue();
                        satLatency_ = satLatencySm_.getNextValue();
                        const float modK = modKSm_.getNextValue();
                        const float modL = modK * ouL_.next(rngL_);
                        const float modR = hasR ? modK * ouR_.next(rngR_) : 0.0f;

                        const float dBase = d - satLatency_ - fade * baseT - gdBank;
                        float tapL;
                        float tapR;
                        if (hasR)
                        {
                            const float modMean = 0.5f * (modL + modR);
                            const float dEffL = dBase + modMix * (modL - modMean) + modMean;
                            const float dEffR = dBase + modMix * (modR - modMean) + modMean;
                            bbdL_.setClockHz(BBD::ClockModel::clockFor(dEffL, sampleRate_));
                            bbdR_.setClockHz(BBD::ClockModel::clockFor(dEffR, sampleRate_));
                            tapL = expL_.processSample(bbdL_.readTap());
                            tapR = expR_.processSample(bbdR_.readTap());
                        }
                        else
                        {
                            const float dEffL = d + modL - satLatency_ - fade * baseT - gdBank;
                            bbdL_.setClockHz(BBD::ClockModel::clockFor(dEffL, sampleRate_));
                            tapL = expL_.processSample(bbdL_.readTap());
                            tapR = tapL;
                        }

                        if (runDiff)
                        {
                            float diffL = tapL;
                            float diffR = tapR;
                            diffuser_.processBlockRef(&diffL, hasR ? &diffR : nullptr, 1);
                            tapL = tapL * (1.0f - fade) + diffL * fade;
                            if (hasR)
                                tapR = tapR * (1.0f - fade) + diffR * fade;
                            else
                                tapR = tapL;
                        }

                        const float mixL = hasR ? (crossCos_ * tapL + crossSin_ * tapR) : tapL;
                        const float mixR = hasR ? (crossCos_ * tapR + crossSin_ * tapL) : tapL;
                        const float vL = g * mixL;
                        const float vR = g * mixR;

                        const float makeup = 1.0f / drive;
                        const float sL = saturate_(adaa1L_, adaa2L_, drive * vL) * makeup;
                        const float sR = hasR ? saturate_(adaa1R_, adaa2R_, drive * vR) * makeup : sL;

                        dampL_ += dampG_ * (sL - dampL_);
                        dampR_ += dampG_ * (sR - dampR_);

                        cutLpL_ += cutG_ * (dampL_ - cutLpL_);
                        cutLpR_ += cutG_ * (dampR_ - cutLpR_);
                        const float cutL = dampL_ - cutLpL_;
                        const float cutR = dampR_ - cutLpR_;

                        const float hL = cutL - dcXL_ + dcR_ * dcYL_;
                        dcXL_ = cutL;
                        dcYL_ = hL;
                        const float hR = cutR - dcXR_ + dcR_ * dcYR_;
                        dcXR_ = cutR;
                        dcYR_ = hR;

                        float wL = inL[s + i] + hL;
                        if (!std::isfinite(wL)) wL = 0.0f;
                        const float wCompL = compL_.processSample(wL);
                        bbdL_.writeSample(wCompL);
                        ringL_.writeBlock(&wL, writeIdx_, 1);
                        ringL_.refreshMirror(writeIdx_, 1);

                        if (hasR)
                        {
                            float wR = inR[s + i] + hR;
                            if (!std::isfinite(wR)) wR = 0.0f;
                            const float wCompR = compR_.processSample(wR);
                            bbdR_.writeSample(wCompR);
                            ringR_.writeBlock(&wR, writeIdx_, 1);
                            ringR_.refreshMirror(writeIdx_, 1);
                        }
                        writeIdx_ = (writeIdx_ + 1) & mask;

                        wetL[s + i] = tapL * loopTrim_;
                        if (hasR) wetR[s + i] = tapR * loopTrim_;
                    }
                    s += Lc;
                    continue;
                }

                const float dCur = delaySm_.getCurrentValue();
                const float dTgt = delaySm_.getTargetValue();
                const float satLatMax = std::max(satLatencySm_.getCurrentValue(), satLatencySm_.getTargetValue());
                // The OU state stays inside kClamp sigmas. The guard covers
                // the largest deviation the modulation can apply.
                const float modGuard = static_cast<float>(Mod::OrnsteinUhlenbeck::kClamp)
                                       * std::max(modKSm_.getCurrentValue(), modKSm_.getTargetValue());

                const float dMin = std::max(kMinLoopDelay, std::min(dCur, dTgt) - satLatMax - baseT - modGuard);

                int Lc = static_cast<int>(std::floor(dMin)) - kChunkGuard;
                Lc = std::clamp(Lc, 1, std::min(kMaxChunk, remaining));

                if (Lc < 4)
                {
                    // Per-sample scalar path (same code as processRef's body).
                    const bool runDiff = (diffState_ != DiffuserState::Off);
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float d = delaySm_.getNextValue();
                        const float g = fbSm_.getNextValue();
                        crossSm_.skip();
                        const float drive = driveSm_.getNextValue();
                        const float fade = fadeStep_();
                        dampG_ = dampGSm_.getNextValue();
                        cutG_ = cutGSm_.getNextValue();
                        satLatency_ = satLatencySm_.getNextValue();
                        const float modK = modKSm_.getNextValue();
                        const float modL = modK * ouL_.next(rngL_);
                        const float modR = hasR ? modK * ouR_.next(rngR_) : 0.0f;
                        processSampleScalar_(inL + s + i, hasR ? inR + s + i : nullptr,
                                             wetL + s + i, hasR ? wetR + s + i : nullptr,
                                             d, g, drive, hasR, mask,
                                             fade, runDiff ? baseT : 0.0f, modL, modR, modMix);
                    }
                    s += Lc;
                    continue;
                }

                alignas(16) std::array<float, kMaxChunk> dR{};
                alignas(16) std::array<float, kMaxChunk> gR{};
                alignas(16) std::array<float, kMaxChunk> crossR{};
                alignas(16) std::array<float, kMaxChunk> driveR{};
                alignas(16) std::array<float, kMaxChunk> fadeR{};
                alignas(16) std::array<float, kMaxChunk> dampGR{};
                alignas(16) std::array<float, kMaxChunk> satLatR{};
                alignas(16) std::array<float, kMaxChunk> cutGR{};
                alignas(16) std::array<float, kMaxChunk> modLR{};
                alignas(16) std::array<float, kMaxChunk> modRR{};
                const bool wasRunning = (diffState_ != DiffuserState::Off);
                for (int i = 0; i < Lc; ++i)
                {
                    dR[i] = delaySm_.getNextValue();
                    gR[i] = fbSm_.getNextValue();
                    crossR[i] = crossSm_.getNextValue();
                    driveR[i] = driveSm_.getNextValue();
                    fadeR[i] = fadeStep_();
                    dampGR[i] = dampGSm_.getNextValue();
                    cutGR[i] = cutGSm_.getNextValue();
                    satLatR[i] = satLatencySm_.getNextValue();
                    const float modK = modKSm_.getNextValue();
                    modLR[i] = modK * ouL_.next(rngL_);
                    modRR[i] = hasR ? modK * ouR_.next(rngR_) : 0.0f;
                }
                const bool runDiff = wasRunning || (diffState_ != DiffuserState::Off);

                alignas(16) std::array<float, kMaxChunk> tapL{};
                alignas(16) std::array<float, kMaxChunk> tapR{};
                // The settled bulk read needs a constant tap. It stays off
                // until the modulation depth reaches zero and stays there.
                const bool modOff = (modKSm_.getCurrentValue() == 0.0f
                                     && modKSm_.getTargetValue() == 0.0f);
                const bool settled = (dR[0] == dR[Lc - 1])
                                     && (fadeR[0] == fadeR[Lc - 1])
                                     && (satLatR[0] == satLatR[Lc - 1])
                                     && modOff;

                if (settled)
                {
                    const float readDelay = std::max(kMinLoopDelay, dR[0] - satLatR[0] - fadeR[0] * baseT);
                    const auto iInt = static_cast<int>(readDelay);
                    const float f = readDelay - static_cast<float>(iInt);
                    const FracDelayTap::Coeffs4 k = FracDelayTap::lagrange3(f);
                    const int base = (writeIdx_ - iInt - 3) & mask;
                    const int winLen = Lc + 6;
                    const M128 cf = MM(set_ps)(k.c4, k.c3, k.c2, k.c1);

                    const auto wL = BlockTapReader::acquireWindow(ringL_, base, winLen, tapWinL_.data());
                    const float *winL = wL.ptr;
                    for (int i = 0; i < Lc; ++i)
                    {
                        const M128 taps = MM(loadu_ps)(winL + i + 1);
                        const M128 prod = MM(mul_ps)(taps, cf);
                        const M128 sh1 = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
                        const M128 sh2 = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
                        tapL[i] = MM(cvtss_f32)(sh2);
                    }

                    if (hasR)
                    {
                        const auto wR = BlockTapReader::acquireWindow(ringR_, base, winLen, tapWinR_.data());
                        const float *winR = wR.ptr;
                        for (int i = 0; i < Lc; ++i)
                        {
                            const M128 taps = MM(loadu_ps)(winR + i + 1);
                            const M128 prod = MM(mul_ps)(taps, cf);
                            const M128 sh1 = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
                            const M128 sh2 = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
                            tapR[i] = MM(cvtss_f32)(sh2);
                        }
                    }
                } else
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float readDelayL = std::max(kMinLoopDelay,
                                                          dR[i] + modLR[i] - satLatR[i] - fadeR[i] * baseT);
                        tapL[i] = FracDelayTap::read(ringL_, writeIdx_ + i, readDelayL);
                        if (hasR)
                        {
                            const float readDelayR = std::max(kMinLoopDelay,
                                                              dR[i] + modRR[i] - satLatR[i] - fadeR[i] * baseT);
                            tapR[i] = FracDelayTap::read(ringR_, writeIdx_ + i, readDelayR);
                        }
                    }
                }

                if (!hasR)
                    for (int i = 0; i < Lc; ++i) tapR[i] = tapL[i];

                if (runDiff)
                {
                    alignas(16) std::array<float, kMaxChunk> rawL{};
                    alignas(16) std::array<float, kMaxChunk> rawR{};
                    std::memcpy(rawL.data(), tapL.data(), static_cast<std::size_t>(Lc) * sizeof(float));
                    std::memcpy(rawR.data(), tapR.data(), static_cast<std::size_t>(Lc) * sizeof(float));
                    diffuser_.processBlock(tapL.data(), hasR ? tapR.data() : nullptr, Lc);
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float a = fadeR[i];
                        tapL[i] = rawL[i] * (1.0f - a) + tapL[i] * a;
                        if (hasR)
                            tapR[i] = rawR[i] * (1.0f - a) + tapR[i] * a;
                        else
                            tapR[i] = tapL[i]; // mono: mirror the blended L
                    }
                }

                alignas(16) std::array<float, kMaxChunk> vL{};
                alignas(16) std::array<float, kMaxChunk> vR{};
                for (int i = 0; i < Lc; ++i)
                {
                    const float g = gR[i];
                    if (hasR)
                    {
                        const float mixL = crossCos_ * tapL[i] + crossSin_ * tapR[i];
                        const float mixR = crossCos_ * tapR[i] + crossSin_ * tapL[i];
                        vL[i] = g * mixL;
                        vR[i] = g * mixR;
                    } else
                    {
                        vL[i] = g * tapL[i];
                        vR[i] = vL[i];
                    }
                }

                alignas(16) std::array<float, kMaxChunk> wL{};
                alignas(16) std::array<float, kMaxChunk> wR{};
                if (satOrder_ == 0)
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / driveR[i];
                        vL[i] = std::clamp(driveR[i] * vL[i], -1.0f, 1.0f) * makeup;
                        if (hasR) vR[i] = std::clamp(driveR[i] * vR[i], -1.0f, 1.0f) * makeup;
                        else vR[i] = vL[i];
                    }
                } else if (satOrder_ == 1)
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / driveR[i];
                        vL[i] = static_cast<float>(adaa1L_.process(driveR[i] * vL[i])) * makeup;
                        if (hasR) vR[i] = static_cast<float>(adaa1R_.process(driveR[i] * vR[i])) * makeup;
                        else vR[i] = vL[i];
                    }
                } else
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / driveR[i];
                        vL[i] = static_cast<float>(adaa2L_.process(driveR[i] * vL[i])) * makeup;
                        if (hasR) vR[i] = static_cast<float>(adaa2R_.process(driveR[i] * vR[i])) * makeup;
                        else vR[i] = vL[i];
                    }
                }

                for (int i = 0; i < Lc; ++i)
                {
                    dampL_ += dampGR[i] * (vL[i] - dampL_);
                    dampR_ += dampGR[i] * (vR[i] - dampR_);

                    cutLpL_ += cutGR[i] * (dampL_ - cutLpL_);
                    cutLpR_ += cutGR[i] * (dampR_ - cutLpR_);
                    const float cutL = dampL_ - cutLpL_;
                    const float cutR = dampR_ - cutLpR_;

                    const float hL = cutL - dcXL_ + dcR_ * dcYL_;
                    dcXL_ = cutL;
                    dcYL_ = hL;
                    const float hR = cutR - dcXR_ + dcR_ * dcYR_;
                    dcXR_ = cutR;
                    dcYR_ = hR;

                    wL[i] = inL[s + i] + hL;
                    if (!std::isfinite(wL[i])) wL[i] = 0.0f;
                    if (hasR)
                    {
                        wR[i] = inR[s + i] + hR;
                        if (!std::isfinite(wR[i])) wR[i] = 0.0f;
                    }
                }

                ringL_.writeBlock(wL.data(), writeIdx_, Lc);
                ringL_.refreshMirror(writeIdx_, Lc);
                if (hasR)
                {
                    ringR_.writeBlock(wR.data(), writeIdx_, Lc);
                    ringR_.refreshMirror(writeIdx_, Lc);
                }
                writeIdx_ = (writeIdx_ + Lc) & mask;

                for (int i = 0; i < Lc; ++i)
                {
                    wetL[s + i] = tapL[i] * loopTrim_;
                    if (hasR) wetR[s + i] = tapR[i] * loopTrim_;
                }
                s += Lc;
            }
        }

        // reference only -- do not optimize, do not delete.
        void processRef(const float *inL, const float *inR, float *wetL, float *wetR, int n) noexcept
        {
            assert(inL != nullptr && wetL != nullptr);
            const bool hasR = (inR != nullptr && wetR != nullptr);
            const int mask = ringL_.mask();

            updateCrossRotation_(); // block-rate equal-power cross-feed coefficients
            diffuserTransition_();
            const float modMix = std::clamp(crossCos_ * crossCos_ - crossSin_ * crossSin_, 0.0f, 1.0f);

            for (int s = 0; s < n; ++s)
            {
                const float baseT = (diffState_ != DiffuserState::Off) ? diffuser_.transportSamples() : 0.0f;
                const float d = delaySm_.getNextValue();
                const float g = fbSm_.getNextValue();
                crossSm_.skip();
                const float drive = driveSm_.getNextValue();
                const float fade = fadeStep_();
                dampG_ = dampGSm_.getNextValue();
                cutG_ = cutGSm_.getNextValue();
                satLatency_ = satLatencySm_.getNextValue();
                const float modK = modKSm_.getNextValue();
                const float modL = modK * ouL_.next(rngL_);
                const float modR = hasR ? modK * ouR_.next(rngR_) : 0.0f;
                processSampleScalar_(inL + s, hasR ? inR + s : nullptr,
                                     wetL + s, hasR ? wetR + s : nullptr,
                                     d, g, drive, hasR, mask, fade, baseT,
                                     modL, modR, modMix);
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept { return 0; }
        [[nodiscard]] float getMaxDelay() const noexcept { return maxDelay_; }
        [[nodiscard]] float ouStateMaxSigma() const noexcept { return diffuser_.ouStateMaxSigma(); }
        [[nodiscard]] float currentDelaySamples() const noexcept { return delaySm_.getCurrentValue(); }

        // RMS ratio of tanh(k * x) to x for a 0.5-amplitude sine reference.
        // The loop output trim is pow(rmsRatio, -0.5). Computed by fixed
        // quadrature over one sine period.
        static float rmsRatioForDrive_(float k) noexcept
        {
            constexpr int N = 128;
            constexpr double kPi = 3.14159265358979323846;
            const double kd = k;
            double sum = 0.0;
            for (int i = 0; i < N; ++i)
            {
                const double t = kPi * (static_cast<double>(i) + 0.5) / static_cast<double>(N);
                const double x = kd * 0.5 * std::sin(2.0 * t);
                const double y = std::tanh(x);
                sum += y * y;
            }
            const double rmsTanh = std::sqrt(sum / static_cast<double>(N));
            constexpr double rmsRef = 0.5 / 1.41421356237309504880;
            return static_cast<float>(rmsTanh / rmsRef);
        }

    private:
        enum class DiffuserState { Off, FadingIn, On, FadingOut };

        static constexpr int kDiffuserFadeSamples = 480; // ~10 ms @48 kHz
        static constexpr float kDiffuserFadeInc = 1.0f / static_cast<float>(kDiffuserFadeSamples);

        void diffuserTransition_() noexcept
        {
            const bool wantOn = enableDiffuser_;
            if (wantOn && diffState_ == DiffuserState::Off)
            {
                diffuser_.prime(); // no stale audio replays
                diffState_ = DiffuserState::FadingIn;
                diffFade_ = 0.0f;
            } else if (!wantOn && diffState_ == DiffuserState::On)
                diffState_ = DiffuserState::FadingOut;
            else if (wantOn && diffState_ == DiffuserState::FadingOut)
                diffState_ = DiffuserState::FadingIn; // reverse: rings warm
            else if (!wantOn && diffState_ == DiffuserState::FadingIn)
                diffState_ = DiffuserState::FadingOut; // reverse
        }

        // Copy the most recent ring audio into the bucket register.
        // The ring write runs in both modes, so the reverse edge needs
        // no work. See docs/dsp-notes.md, "BBD mode-flip priming".
        void primeBbdFromRing_() noexcept
        {
            const int ringFill = std::min (writeIdx_, BBD::BrigadeLine::kStages);
            if (ringFill <= 0)
            {
                bbdL_.primeFrom (nullptr, 0);
                bbdR_.primeFrom (nullptr, 0);
                return;
            }
            const int mask = ringL_.mask();
            const int startL = (writeIdx_ - ringFill + mask + 1) & mask;
            bbdL_.primeFrom (ringL_.windowPtr (startL, ringFill), ringFill);
            if (ringR_.getCapacity() > 0)
            {
                const int startR = (writeIdx_ - ringFill + mask + 1) & mask;
                bbdR_.primeFrom (ringR_.windowPtr (startR, ringFill), ringFill);
            }
        }

        float fadeStep_() noexcept
        {
            const float a = diffFade_;
            if (diffState_ == DiffuserState::FadingIn)
            {
                diffFade_ += kDiffuserFadeInc;
                if (diffFade_ >= 1.0f)
                {
                    diffFade_ = 1.0f;
                    diffState_ = DiffuserState::On;
                }
            } else if (diffState_ == DiffuserState::FadingOut)
            {
                diffFade_ -= kDiffuserFadeInc;
                if (diffFade_ <= 0.0f)
                {
                    diffFade_ = 0.0f;
                    diffState_ = DiffuserState::Off;
                }
            }
            return a;
        }

        void applyDiffuserParams_(const Params &p) noexcept
        {
            diffuser_.setDiffusion(p.diffusion);
            diffuser_.setSize(p.diffuserSize);
            const float diffModSamples = p.diffModDepth * 0.001f * static_cast<float>(sampleRate_);
            diffuser_.setModDepthSamples(diffModSamples);
            diffuser_.setModRateHz(p.diffModRateHz);
        }

        void prepareImpl_(double sampleRate, int maxBlockSize, int maxDelaySamples,
                          Memory::BumpArena *arena) noexcept
        {
            assert(sampleRate > 0.0);
            assert(maxBlockSize > 0);
            assert(maxDelaySamples > static_cast<int>(kMinLoopDelay));

            sampleRate_ = sampleRate;
            const int minCap = maxDelaySamples + maxBlockSize
                               + Pow2RingBuffer::kTail + 8;
            constexpr std::size_t perChan = (static_cast<std::size_t>(BBD::BrigadeLine::kStages + 1) + 15u) & ~static_cast<std::size_t>(15u);
            if (arena != nullptr)
            {
                ringL_.prepare(minCap, *arena);
                ringR_.prepare(minCap, *arena);
                diffuser_.prepare(sampleRate, *arena);
                float *bbdMemL = arena->allocate<float>(perChan, Memory::BumpArena::kBaseAlignment);
                float *bbdMemR = arena->allocate<float>(perChan, Memory::BumpArena::kBaseAlignment);
                bbdL_.prepare(sampleRate, bbdMemL);
                bbdR_.prepare(sampleRate, bbdMemR);
            } else
            {
                ringL_.prepare(minCap);
                ringR_.prepare(minCap);
                diffuser_.prepare(sampleRate);
                bbdHeapStorage_.resize(BBD::BrigadeLine::bbdStorageFloats(2), 0.0f);
                bbdL_.prepare(sampleRate, bbdHeapStorage_.data());
                bbdR_.prepare(sampleRate, bbdHeapStorage_.data() + perChan);
            }
            compL_.prepare(sampleRate);
            compR_.prepare(sampleRate);
            expL_.prepare(sampleRate);
            expR_.prepare(sampleRate);
            maxDelay_ = static_cast<float>(
                ringL_.getCapacity() - Pow2RingBuffer::kTail - 2);

            delaySm_.reset(sampleRate, 0.020); // 20 ms glide ramp floor
            lastGlideRampTime_ = 0.020;
            fbSm_.reset(sampleRate, 0.020);
            crossSm_.reset(sampleRate, 0.020);
            driveSm_.reset(sampleRate, 0.020);
            dampGSm_.reset(sampleRate, 0.020);
            cutGSm_.reset(sampleRate, 0.020);
            satLatencySm_.reset(sampleRate, 0.010);
            modKSm_.reset(sampleRate, 0.020);
            reset();
        }

        [[nodiscard]] float clampDelay_(float d) const noexcept
        {
            return std::clamp(d, kMinLoopDelay + 1.5f, maxDelay_);
        }

        // Limit the delay glide to kMaxGlideStep samples per sample.
        // Reset the smoother only when the ramp time changes by more than one percent.
        // This avoids a restart each block.
        void retargetDelayGlide_(float targetSamples) noexcept
        {
            const float target = clampDelay_(targetSamples);
            const float current = delaySm_.getCurrentValue();
            const double dist = std::fabs(static_cast<double>(target) - static_cast<double>(current));
            const double rampTime = std::max(0.020, dist / (static_cast<double>(kMaxGlideStep) * sampleRate_));
            if (std::fabs(rampTime - lastGlideRampTime_) > 0.01 * std::max(rampTime, lastGlideRampTime_))
            {
                delaySm_.reset(sampleRate_, rampTime);
                lastGlideRampTime_ = rampTime;
            }
            delaySm_.setTargetValue(target);
        }

        void applyBlockRate_(const Params &p) noexcept
        {
            const double fc = std::clamp(static_cast<double>(p.dampHz), 20.0, 0.45 * sampleRate_);
            const double gw = std::tan(std::numbers::pi * fc / sampleRate_);
            dampGSm_.setTargetValue(static_cast<float>(gw / (1.0 + gw)));

            // DC blocker pole: 5 Hz. The blocker sits after the saturator
            // and compounds over more passes. The 5 Hz corner keeps the
            // loss at 40 Hz under 2 dB over 20 passes.
            dcR_ = static_cast<float>(std::exp(-2.0 * std::numbers::pi * 5.0 / sampleRate_));

            // Low cut: one-pole highpass, same topology as the damp filter.
            const double fcCut = std::clamp(static_cast<double>(p.loopCutHz), 20.0, 0.45 * sampleRate_);
            const double gwCut = std::tan(std::numbers::pi * fcCut / sampleRate_);
            cutGSm_.setTargetValue(static_cast<float>(gwCut / (1.0 + gwCut)));

            satOrder_ = std::clamp(p.satOrder, 0, 2);
            const float newSatLatency = (satOrder_ == 2)
                                            ? 1.0f
                                            : (satOrder_ == 1)
                                                  ? 0.5f
                                                  : 0.0f;
            satLatencySm_.setTargetValue(newSatLatency);

            const float clampedDrive = std::clamp(p.loopDrive, 0.1f, 16.0f);
            loopTrim_ = Math::loopTrim(clampedDrive);

            const float modRate = std::clamp(p.delayModRateHz, 0.01f, 10.0f);
            ouL_.setRate(sampleRate_, modRate);
            ouR_.setRate(sampleRate_, modRate);

            const float cents = std::clamp(p.delayModDepth, 0.0f, 50.0f);
            // Map the depth in cents to an RMS delay slope. A pitch reading
            // averages the slope over the tone period, so the scale uses the
            // windowed increment RMS of the OU process. The reference window
            // is 1 ms, the period of a 1 kHz tone.
            const double slopeTarget = static_cast<double>(cents) * (std::numbers::ln2 / 1200.0);
            const double tRef = std::max(1.0, std::round(sampleRate_ * 0.001));
            const double incRms = ouL_.windowedIncrementRms(tRef);
            modKSm_.setTargetValue(static_cast<float>(incRms > 0.0 ? slopeTarget / incRms : 0.0));
        }

        // Compute the equal-power rotation from the smoothed cross value.
        // theta = cross * pi/2. cos and sin are evaluated at block rate.
        void updateCrossRotation_() noexcept
        {
            const float cross = std::clamp(crossSm_.getCurrentValue(), 0.0f, 1.0f);
            const float theta = cross * (std::numbers::pi_v<float> * 0.5f);
            crossCos_ = mmCos(theta);
            crossSin_ = mmSin(theta);
        }

        float saturate_(Nonlinear::ADAA1<Nonlinear::TanhNL> &a1,
                        Nonlinear::ADAA2<Nonlinear::TanhNL> &a2,
                        float x) noexcept
        {
            switch (satOrder_)
            {
                case 2: return static_cast<float>(a2.process(x));
                case 1: return static_cast<float>(a1.process(x));
                default: return std::clamp(x, -1.0f, 1.0f);
            }
        }

        void processSampleScalar_(const float *in, const float *inR,
                                  float *wet, float *wetR,
                                  float d, float g, float drive,
                                  bool hasR, int mask,
                                  float fade, float baseT,
                                  float modL, float modR, float modMix) noexcept
        {
            const float makeup = 1.0f / drive;

            float tapL = 0.0f;
            float tapR = 0.0f;

            if (delayMode_ == 1)
            {
                const float gdBank = static_cast<float>(BBD::BrigadeLine::getBankGroupDelayAtDC(sampleRate_))
                                     + BBD::BrigadeLine::kSplitStepOffset;
                const float dBase = d - satLatency_ - fade * baseT - gdBank;
                if (hasR)
                {
                    const float modMean = 0.5f * (modL + modR);
                    const float dEffL = dBase + modMix * (modL - modMean) + modMean;
                    const float dEffR = dBase + modMix * (modR - modMean) + modMean;
                    bbdL_.setClockHz(BBD::ClockModel::clockFor(dEffL, sampleRate_));
                    bbdR_.setClockHz(BBD::ClockModel::clockFor(dEffR, sampleRate_));
                    tapL = expL_.processSample(bbdL_.readTap());
                    tapR = expR_.processSample(bbdR_.readTap());
                }
                else
                {
                    const float dEffL = d + modL - satLatency_ - fade * baseT - gdBank;
                    bbdL_.setClockHz(BBD::ClockModel::clockFor(dEffL, sampleRate_));
                    tapL = expL_.processSample(bbdL_.readTap());
                    tapR = tapL;
                }
            }
            else
            {
                const float readDelayL = std::max(kMinLoopDelay, d + modL - satLatency_ - fade * baseT);
                const float readDelayR = std::max(kMinLoopDelay, d + modR - satLatency_ - fade * baseT);

                tapL = FracDelayTap::read(ringL_, writeIdx_, readDelayL);
                tapR = hasR
                             ? FracDelayTap::read(ringR_, writeIdx_, readDelayR)
                             : tapL;
            }

            if (baseT > 0.0f) // diffuser running: diffuse, then fade-blend
            {
                float diffL = tapL;
                float diffR = tapR;
                diffuser_.processBlockRef(&diffL, hasR ? &diffR : nullptr, 1);
                tapL = tapL * (1.0f - fade) + diffL * fade;
                if (hasR)
                    tapR = tapR * (1.0f - fade) + diffR * fade;
                else
                    tapR = tapL; // mono: mirror the blended L
            }

            // Equal-power cross-feed rotation. Block-rate coefficients.
            const float mixL = hasR ? (crossCos_ * tapL + crossSin_ * tapR) : tapL;
            const float mixR = hasR ? (crossCos_ * tapR + crossSin_ * tapL) : tapL;
            const float vL = g * mixL;
            const float vR = g * mixR;

            const float sL = saturate_(adaa1L_, adaa2L_, drive * vL) * makeup;
            const float sR = hasR
                                 ? saturate_(adaa1R_, adaa2R_, drive * vR) * makeup
                                 : sL;

            dampL_ += dampG_ * (sL - dampL_);
            dampR_ += dampG_ * (sR - dampR_);

            cutLpL_ += cutG_ * (dampL_ - cutLpL_);
            cutLpR_ += cutG_ * (dampR_ - cutLpR_);
            const float cutL = dampL_ - cutLpL_;
            const float cutR = dampR_ - cutLpR_;

            const float hL = cutL - dcXL_ + dcR_ * dcYL_;
            dcXL_ = cutL;
            dcYL_ = hL;
            const float hR = cutR - dcXR_ + dcR_ * dcYR_;
            dcXR_ = cutR;
            dcYR_ = hR;

            float wL = *in + hL;
            if (!std::isfinite(wL)) wL = 0.0f;
            if (delayMode_ == 1)
            {
                const float wCompL = compL_.processSample(wL);
                bbdL_.writeSample(wCompL);
            }
            ringL_.writeBlock(&wL, writeIdx_, 1);
            ringL_.refreshMirror(writeIdx_, 1);

            if (hasR)
            {
                float wR = *inR + hR;
                if (!std::isfinite(wR)) wR = 0.0f;
                if (delayMode_ == 1)
                {
                    const float wCompR = compR_.processSample(wR);
                    bbdR_.writeSample(wCompR);
                }
                ringR_.writeBlock(&wR, writeIdx_, 1);
                ringR_.refreshMirror(writeIdx_, 1);
            }

            writeIdx_ = (writeIdx_ + 1) & mask;

            *wet = tapL * loopTrim_; // the blended (diffused when on) loop-tap stream
            if (hasR) *wetR = tapR * loopTrim_;
        }

        Pow2RingBuffer ringL_;
        Pow2RingBuffer ringR_;
        int writeIdx_ = 0;
        float maxDelay_ = 0.0f;
        double sampleRate_ = 48000.0;
        bool firstBlock_ = true;

        Smoothers::LinearSmoother<float> delaySm_;
        Smoothers::LinearSmoother<float> fbSm_;
        Smoothers::LinearSmoother<float> crossSm_;
        Smoothers::LinearSmoother<float> driveSm_;
        Smoothers::LinearSmoother<float> dampGSm_;
        Smoothers::LinearSmoother<float> cutGSm_;
        Smoothers::LinearSmoother<float> satLatencySm_;

        // Last glide ramp duration, in seconds.
        double lastGlideRampTime_ = 0.020;

        // block-rate coefficients
        float dampG_ = 0.0f;
        float cutG_ = 0.0f;
        float dcR_ = 0.999f;
        int satOrder_ = 2;
        float satLatency_ = 1.0f;
        float loopTrim_ = 1.0f;
        float crossCos_ = 1.0f; // block-rate rotation cos(theta)
        float crossSin_ = 0.0f; // block-rate rotation sin(theta)

        // per-channel loop state
        float dampL_ = 0.0f;
        float dampR_ = 0.0f;
        float cutLpL_ = 0.0f;
        float cutLpR_ = 0.0f;
        float dcXL_ = 0.0f;
        float dcYL_ = 0.0f;
        float dcXR_ = 0.0f;
        float dcYR_ = 0.0f;

        Nonlinear::ADAA1<Nonlinear::TanhNL> adaa1L_;
        Nonlinear::ADAA1<Nonlinear::TanhNL> adaa1R_;
        Nonlinear::ADAA2<Nonlinear::TanhNL> adaa2L_;
        Nonlinear::ADAA2<Nonlinear::TanhNL> adaa2R_;

        alignas(16) std::array<float, static_cast<std::size_t>(kMaxChunk) + Pow2RingBuffer::kTail> tapWinL_{};
        alignas(16) std::array<float, static_cast<std::size_t>(kMaxChunk) + Pow2RingBuffer::kTail> tapWinR_{};

        Diffusion::Diffuser diffuser_;
        BBD::BrigadeLine bbdL_;
        BBD::BrigadeLine bbdR_;
        BBD::CompressorCell compL_;
        BBD::CompressorCell compR_;
        BBD::ExpanderCell expL_;
        BBD::ExpanderCell expR_;
        std::vector<float> bbdHeapStorage_;
        int delayMode_ = 0;
        int lastDelayMode_ = 0;
        bool enableDiffuser_ = false;
        DiffuserState diffState_ = DiffuserState::Off;
        float diffFade_ = 0.0f; // 0 = raw tap, 1 = diffused tap

        // Per-channel modulation states. The generators share one seed
        // constant and differ by the stream index.
        Mod::OrnsteinUhlenbeck ouL_;
        Mod::OrnsteinUhlenbeck ouR_;
        Mod::Pcg32 rngL_;
        Mod::Pcg32 rngR_;
        Smoothers::LinearSmoother<float> modKSm_;
    };
}
#endif

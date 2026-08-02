#pragma once

#ifndef CHRONOS_FEEDBACK_DELAY_H
#define CHRONOS_FEEDBACK_DELAY_H

#include "BlockTapReader.h"
#include "Diffuser.h"
#include "FracDelayTap.h"
#include "LinearSmoother.h"
#include "Pow2RingBuffer.h"
#include "nonlinear/ADAA1.h"
#include "nonlinear/ADAA2.h"
#include "nonlinear/Nonlinearities.h"
#include "utils/memory/BumpArena.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numbers>

namespace MarsDSP::Delays {
    class FeedbackDelay {
    public:
        static constexpr float kMaxFeedback     = 1.2f;
        static constexpr float kMinLoopDelay    = 4.0f;   // > FracDelayTap's 3.0 contract

        static constexpr int   kMaxChunk    = 64;   // max sub-chunk length (ramp-array footprint)
        static constexpr int   kChunkGuard  = 6;    // interpolator window (base = wIdx - i - 3, len 6 ≤ kTail)

        struct Params
        {
            float delaySamples = 4800.0f;
            float feedback     = 0.0f;    // 0..kMaxFeedback; > 1 self-oscillates, bounded
            float dampHz       = 6000.0f; // one-pole lowpass in the loop
            float crossFeed    = 0.0f;    // 0 straight, 1 full ping-pong
            float loopDrive    = 1.0f;    // how hard repeats lean on the tanh ceiling
            int   satOrder     = 2;       // 0 hard, 1 ADAA1, 2 ADAA2
            float diffusion      = 0.7f;  // 0..1 -> allpass coeff 0..0.92
            float diffuserSize   = 0.5f;  // 0..1 (1 = full path length)
            float diffModDepth   = 16.0f; // samples, 0..62
            float diffModRateHz  = 0.5f;  // 0..8
            bool  enableDiffuser = false; // off by default
        };

        void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples) noexcept
        {
            prepareImpl_(sampleRate, maxBlockSize, maxDelaySamples, nullptr);
        }

        void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples,
                     Memory::BumpArena& arena) noexcept
        {
            prepareImpl_(sampleRate, maxBlockSize, maxDelaySamples, &arena);
        }

        static std::size_t ringStorageFloats(double sampleRate, int maxBlockSize,
                                             int maxDelaySamples) noexcept
        {
            const int minCap = maxDelaySamples + maxBlockSize
                             + Pow2RingBuffer::kTail + 8;
            return 2 * Pow2RingBuffer::arenaFloatsFor(minCap)
                 + Diffusion::Diffuser::ringStorageFloats(sampleRate);
        }

        void reset() noexcept
        {
            ringL_.clear();
            ringR_.clear();
            writeIdx_ = 0;
            adaa1L_.reset(); adaa1R_.reset();
            adaa2L_.reset(); adaa2R_.reset();
            dampL_ = dampR_ = 0.0f;
            dcXL_ = dcXR_ = dcYL_ = dcYR_ = 0.0f;
            diffuser_.reset();
            enableDiffuser_ = false;
            diffState_ = DiffuserState::Off;
            diffFade_ = 0.0f;
            firstBlock_ = true;
        }

        void resetParams(const Params& p) noexcept
        {
            applyBlockRate_(p);
            delaySm_.setCurrentAndTargetValue(clampDelay_(p.delaySamples));
            fbSm_.setCurrentAndTargetValue(std::clamp(p.feedback, 0.0f, kMaxFeedback));
            crossSm_.setCurrentAndTargetValue(std::clamp(p.crossFeed, 0.0f, 1.0f));
            driveSm_.setCurrentAndTargetValue(std::clamp(p.loopDrive, 0.1f, 16.0f));
            dampGSm_.setCurrentAndTargetValue(dampGSm_.getTargetValue());
            dampG_ = dampGSm_.getCurrentValue();
            satLatencySm_.setCurrentAndTargetValue(satLatencySm_.getTargetValue());
            satLatency_ = satLatencySm_.getCurrentValue();
            applyDiffuserParams_(p);
            diffuser_.prime();
            enableDiffuser_ = p.enableDiffuser;
            diffState_ = enableDiffuser_ ? DiffuserState::On : DiffuserState::Off;
            diffFade_ = enableDiffuser_ ? 1.0f : 0.0f;
            firstBlock_ = false;
        }

        void setParams(const Params& p) noexcept
        {
            if (firstBlock_) { resetParams(p); return; }
            applyBlockRate_(p);
            delaySm_.setTargetValue(clampDelay_(p.delaySamples));
            fbSm_.setTargetValue(std::clamp(p.feedback, 0.0f, kMaxFeedback));
            crossSm_.setTargetValue(std::clamp(p.crossFeed, 0.0f, 1.0f));
            driveSm_.setTargetValue(std::clamp(p.loopDrive, 0.1f, 16.0f));
            applyDiffuserParams_(p);
            enableDiffuser_ = p.enableDiffuser;
        }

        void process(const float* inL, const float* inR,
                     float* wetL, float* wetR, int n) noexcept
        {
            assert(inL != nullptr && wetL != nullptr);
            const bool hasR = (inR != nullptr && wetR != nullptr);
            const int  mask = ringL_.mask();

            diffuserTransition_();   // block-rate enable edge (primes on rising)

            int s = 0;
            while (s < n)
            {
                const int remaining = n - s;

                const float baseT = (diffState_ != DiffuserState::Off)
                    ? diffuser_.transportSamples() : 0.0f;

                const float dCur = delaySm_.getCurrentValue();
                const float dTgt = delaySm_.getTargetValue();
                const float satLatMax = std::max(satLatencySm_.getCurrentValue(),
                                                 satLatencySm_.getTargetValue());
                const float dMin = std::max(kMinLoopDelay,
                    std::min(dCur, dTgt) - satLatMax - baseT);

                int Lc = static_cast<int>(std::floor(dMin)) - kChunkGuard;
                Lc = std::clamp(Lc, 1, std::min(kMaxChunk, remaining));

                if (Lc < 4)
                {
                    // Per-sample scalar path (same code as processRef's body).
                    const bool runDiff = (diffState_ != DiffuserState::Off);
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float d     = delaySm_.getNextValue();
                        const float g     = fbSm_.getNextValue();
                        const float cross = crossSm_.getNextValue();
                        const float drive = driveSm_.getNextValue();
                        const float fade  = fadeStep_();
                        dampG_      = dampGSm_.getNextValue();
                        satLatency_ = satLatencySm_.getNextValue();
                        processSampleScalar_(inL + s + i, hasR ? inR + s + i : nullptr,
                                             wetL + s + i, hasR ? wetR + s + i : nullptr,
                                             d, g, cross, drive, hasR, mask,
                                             fade, runDiff ? baseT : 0.0f);
                    }
                    s += Lc;
                    continue;
                }

                alignas(16) float dR[kMaxChunk], gR[kMaxChunk],
                                 crossR[kMaxChunk], driveR[kMaxChunk], fadeR[kMaxChunk],
                                 dampGR[kMaxChunk], satLatR[kMaxChunk];
                const bool wasRunning = (diffState_ != DiffuserState::Off);
                for (int i = 0; i < Lc; ++i)
                {
                    dR[i]      = delaySm_.getNextValue();
                    gR[i]      = fbSm_.getNextValue();
                    crossR[i]  = crossSm_.getNextValue();
                    driveR[i]  = driveSm_.getNextValue();
                    fadeR[i]   = fadeStep_();
                    dampGR[i]  = dampGSm_.getNextValue();
                    satLatR[i] = satLatencySm_.getNextValue();
                }
                const bool runDiff = wasRunning || (diffState_ != DiffuserState::Off);

                alignas(16) float tapL[kMaxChunk], tapR[kMaxChunk];
                const bool settled = (dR[0] == dR[Lc - 1])
                                     && (fadeR[0] == fadeR[Lc - 1])
                                     && (satLatR[0] == satLatR[Lc - 1]);

                if (settled)
                {
                    const float readDelay = std::max(kMinLoopDelay,
                        dR[0] - satLatR[0] - fadeR[0] * baseT);
                    const auto  iInt = static_cast<int>(readDelay);
                    const float f = readDelay - static_cast<float>(iInt);
                    const FracDelayTap::Coeffs4 k = FracDelayTap::lagrange3(f);
                    const int base = (writeIdx_ - iInt - 3) & mask;
                    const int winLen = Lc + 6;
                    const M128 cf = MM(set_ps)(k.c4, k.c3, k.c2, k.c1);

                    const auto wL = BlockTapReader::acquireWindow(ringL_, base, winLen, tapWinL_.data());
                    const float* winL = wL.ptr;
                    for (int i = 0; i < Lc; ++i)
                    {
                        const M128 taps = MM(loadu_ps)(winL + i + 1);
                        const M128 prod = MM(mul_ps)(taps, cf);
                        const M128 sh1  = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
                        const M128 sh2  = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
                        tapL[i] = MM(cvtss_f32)(sh2);
                    }

                    if (hasR)
                    {
                        const auto wR = BlockTapReader::acquireWindow(ringR_, base, winLen, tapWinR_.data());
                        const float* winR = wR.ptr;
                        for (int i = 0; i < Lc; ++i)
                        {
                            const M128 taps = MM(loadu_ps)(winR + i + 1);
                            const M128 prod = MM(mul_ps)(taps, cf);
                            const M128 sh1  = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
                            const M128 sh2  = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
                            tapR[i] = MM(cvtss_f32)(sh2);
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float readDelay = std::max(kMinLoopDelay,
                            dR[i] - satLatR[i] - fadeR[i] * baseT);
                        tapL[i] = FracDelayTap::read(ringL_, writeIdx_ + i, readDelay);
                        if (hasR)
                            tapR[i] = FracDelayTap::read(ringR_, writeIdx_ + i, readDelay);
                    }
                }

                if (!hasR)
                    for (int i = 0; i < Lc; ++i) tapR[i] = tapL[i];

                if (runDiff)
                {
                    alignas(16) float rawL[kMaxChunk], rawR[kMaxChunk];
                    std::memcpy(rawL, tapL, static_cast<std::size_t>(Lc) * sizeof(float));
                    std::memcpy(rawR, tapR, static_cast<std::size_t>(Lc) * sizeof(float));
                    diffuser_.processBlock(tapL, hasR ? tapR : nullptr, Lc);
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float a = fadeR[i];
                        tapL[i] = rawL[i] * (1.0f - a) + tapL[i] * a;
                        if (hasR)
                            tapR[i] = rawR[i] * (1.0f - a) + tapR[i] * a;
                        else
                            tapR[i] = tapL[i];   // mono: mirror the blended L
                    }
                }

                alignas(16) float vL[kMaxChunk], vR[kMaxChunk];
                for (int i = 0; i < Lc; ++i)
                {
                    dampL_ += dampGR[i] * (tapL[i] - dampL_);
                    dampR_ += dampGR[i] * (tapR[i] - dampR_);

                    const float hL = dampL_ - dcXL_ + dcR_ * dcYL_;
                    dcXL_ = dampL_; dcYL_ = hL;
                    const float hR = dampR_ - dcXR_ + dcR_ * dcYR_;
                    dcXR_ = dampR_; dcYR_ = hR;

                    const float g = gR[i], cross = crossR[i];
                    vL[i] = g * ((1.0f - cross) * hL + cross * hR);
                    vR[i] = g * ((1.0f - cross) * hR + cross * hL);
                }

                alignas(16) float wL[kMaxChunk], wR[kMaxChunk];
                if (satOrder_ == 0)
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / driveR[i];
                        const float sL = std::clamp(driveR[i] * vL[i], -1.0f, 1.0f) * makeup;
                        const float sR = hasR ? std::clamp(driveR[i] * vR[i], -1.0f, 1.0f) * makeup : sL;
                        wL[i] = inL[s + i] + sL;
                        if (!std::isfinite(wL[i])) wL[i] = 0.0f;
                        if (hasR) { wR[i] = inR[s + i] + sR; if (!std::isfinite(wR[i])) wR[i] = 0.0f; }
                    }
                }
                else if (satOrder_ == 1)
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / driveR[i];
                        const float sL = static_cast<float>(adaa1L_.process(static_cast<double>(driveR[i] * vL[i]))) * makeup;
                        const float sR = hasR ? static_cast<float>(adaa1R_.process(static_cast<double>(driveR[i] * vR[i]))) * makeup : sL;
                        wL[i] = inL[s + i] + sL;
                        if (!std::isfinite(wL[i])) wL[i] = 0.0f;
                        if (hasR) { wR[i] = inR[s + i] + sR; if (!std::isfinite(wR[i])) wR[i] = 0.0f; }
                    }
                }
                else
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / driveR[i];
                        const float sL = static_cast<float>(adaa2L_.process(static_cast<double>(driveR[i] * vL[i]))) * makeup;
                        const float sR = hasR ? static_cast<float>(adaa2R_.process(static_cast<double>(driveR[i] * vR[i]))) * makeup : sL;
                        wL[i] = inL[s + i] + sL;
                        if (!std::isfinite(wL[i])) wL[i] = 0.0f;
                        if (hasR) { wR[i] = inR[s + i] + sR; if (!std::isfinite(wR[i])) wR[i] = 0.0f; }
                    }
                }

                ringL_.writeBlock(wL, writeIdx_, Lc);
                ringL_.refreshMirror(writeIdx_, Lc);
                if (hasR)
                {
                    ringR_.writeBlock(wR, writeIdx_, Lc);
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
        void processRef(const float* inL, const float* inR,
                        float* wetL, float* wetR, int n) noexcept
        {
            assert(inL != nullptr && wetL != nullptr);
            const bool hasR = (inR != nullptr && wetR != nullptr);
            const int  mask = ringL_.mask();

            diffuserTransition_();

            for (int s = 0; s < n; ++s)
            {
                const float baseT = (diffState_ != DiffuserState::Off)
                    ? diffuser_.transportSamples() : 0.0f;
                const float d     = delaySm_.getNextValue();
                const float g     = fbSm_.getNextValue();
                const float cross = crossSm_.getNextValue();
                const float drive = driveSm_.getNextValue();
                const float fade  = fadeStep_();
                dampG_      = dampGSm_.getNextValue();
                satLatency_ = satLatencySm_.getNextValue();
                processSampleScalar_(inL + s, hasR ? inR + s : nullptr,
                                     wetL + s, hasR ? wetR + s : nullptr,
                                     d, g, cross, drive, hasR, mask, fade, baseT);
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept { return 0; }
        [[nodiscard]] float getMaxDelay() const noexcept { return maxDelay_; }

        // Return the larger modulation oscillator magnitude.
        [[nodiscard]] double oscillatorMagnitude() const noexcept { return diffuser_.oscillatorMagnitude(); }

        // RMS ratio of tanh(k * x) to x for a 0.5-amplitude sine reference.
        // The loop output trim is pow(rmsRatio, -0.5). Computed by fixed
        // quadrature over one sine period.
        static float rmsRatioForDrive_(float k) noexcept
        {
            constexpr int N = 128;
            constexpr double kPi = 3.14159265358979323846;
            const double kd = static_cast<double>(k);
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
        static constexpr int   kDiffuserFadeSamples = 480;  // ~10 ms @48 kHz
        static constexpr float kDiffuserFadeInc =
            1.0f / static_cast<float>(kDiffuserFadeSamples);

        void diffuserTransition_() noexcept
        {
            const bool wantOn = enableDiffuser_;
            if (wantOn && diffState_ == DiffuserState::Off)
            {
                diffuser_.prime();   // no stale audio replays
                diffState_ = DiffuserState::FadingIn;
                diffFade_ = 0.0f;
            }
            else if (!wantOn && diffState_ == DiffuserState::On)
                diffState_ = DiffuserState::FadingOut;
            else if (wantOn && diffState_ == DiffuserState::FadingOut)
                diffState_ = DiffuserState::FadingIn;    // reverse: rings warm
            else if (!wantOn && diffState_ == DiffuserState::FadingIn)
                diffState_ = DiffuserState::FadingOut;   // reverse
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
            }
            else if (diffState_ == DiffuserState::FadingOut)
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

        void applyDiffuserParams_(const Params& p) noexcept
        {
            diffuser_.setDiffusion(p.diffusion);
            diffuser_.setSize(p.diffuserSize);
            diffuser_.setModDepthSamples(p.diffModDepth);
            diffuser_.setModRateHz(p.diffModRateHz);
        }

        void prepareImpl_(double sampleRate, int maxBlockSize, int maxDelaySamples,
                          Memory::BumpArena* arena) noexcept
        {
            assert(sampleRate > 0.0);
            assert(maxBlockSize > 0);
            assert(maxDelaySamples > static_cast<int>(kMinLoopDelay));

            sampleRate_ = sampleRate;
            const int minCap = maxDelaySamples + maxBlockSize
                             + Pow2RingBuffer::kTail + 8;
            if (arena != nullptr)
            {
                ringL_.prepare(minCap, *arena);
                ringR_.prepare(minCap, *arena);
                diffuser_.prepare(sampleRate, *arena);
            }
            else
            {
                ringL_.prepare(minCap);
                ringR_.prepare(minCap);
                diffuser_.prepare(sampleRate);
            }
            maxDelay_ = static_cast<float>(
                ringL_.getCapacity() - Pow2RingBuffer::kTail - 2);

            delaySm_.reset(sampleRate, 0.050);
            fbSm_.reset(sampleRate, 0.020);
            crossSm_.reset(sampleRate, 0.020);
            driveSm_.reset(sampleRate, 0.020);
            dampGSm_.reset(sampleRate, 0.020);
            satLatencySm_.reset(sampleRate, 0.010);
            reset();
        }

        float clampDelay_(float d) const noexcept
        {
            return std::clamp(d, kMinLoopDelay + 1.5f, maxDelay_);
        }

        void applyBlockRate_(const Params& p) noexcept
        {
            const double fc = std::clamp(static_cast<double>(p.dampHz),
                                         20.0, 0.45 * sampleRate_);
            const double gw = std::tan(std::numbers::pi * fc / sampleRate_);
            dampGSm_.setTargetValue(static_cast<float>(gw / (1.0 + gw)));

            // DC blocker pole: ~8 Hz, R = exp(-2*pi*fc/fs).
            dcR_ = static_cast<float>(
                std::exp(-2.0 * std::numbers::pi * 8.0 / sampleRate_));

            satOrder_ = std::clamp(p.satOrder, 0, 2);
            const float newSatLatency = (satOrder_ == 2) ? 1.0f
                                      : (satOrder_ == 1) ? 0.5f
                                                         : 0.0f;
            satLatencySm_.setTargetValue(newSatLatency);

            const float clampedDrive = std::clamp(p.loopDrive, 0.1f, 16.0f);
            loopTrim_ = std::pow(rmsRatioForDrive_(clampedDrive), -0.5f);
        }

        float saturate_(Nonlinear::ADAA1<Nonlinear::TanhNL>& a1,
                        Nonlinear::ADAA2<Nonlinear::TanhNL>& a2,
                        float x) noexcept
        {
            switch (satOrder_)
            {
                case 2:  return static_cast<float>(a2.process(static_cast<double>(x)));
                case 1:  return static_cast<float>(a1.process(static_cast<double>(x)));
                default: return std::clamp(x, -1.0f, 1.0f);
            }
        }

        void processSampleScalar_(const float* in, const float* inR,
                                   float* wet, float* wetR,
                                   float d, float g, float cross, float drive,
                                   bool hasR, int mask,
                                   float fade, float baseT) noexcept
        {
            const float makeup = 1.0f / drive;
            const float readDelay = std::max(kMinLoopDelay, d - satLatency_ - fade * baseT);

            float tapL = FracDelayTap::read(ringL_, writeIdx_, readDelay);
            float tapR = hasR
                ? FracDelayTap::read(ringR_, writeIdx_, readDelay)
                : tapL;

            if (baseT > 0.0f)   // diffuser running: diffuse, then fade-blend
            {
                float diffL = tapL;
                float diffR = tapR;
                diffuser_.processBlockRef(&diffL, hasR ? &diffR : nullptr, 1);
                tapL = tapL * (1.0f - fade) + diffL * fade;
                if (hasR)
                    tapR = tapR * (1.0f - fade) + diffR * fade;
                else
                    tapR = tapL;   // mono: mirror the blended L
            }

            dampL_ += dampG_ * (tapL - dampL_);
            dampR_ += dampG_ * (tapR - dampR_);

            const float hL = dampL_ - dcXL_ + dcR_ * dcYL_;
            dcXL_ = dampL_; dcYL_ = hL;
            const float hR = dampR_ - dcXR_ + dcR_ * dcYR_;
            dcXR_ = dampR_; dcYR_ = hR;

            const float vL = g * ((1.0f - cross) * hL + cross * hR);
            const float vR = g * ((1.0f - cross) * hR + cross * hL);

            const float sL = saturate_(adaa1L_, adaa2L_, drive * vL) * makeup;
            const float sR = hasR
                ? saturate_(adaa1R_, adaa2R_, drive * vR) * makeup
                : sL;

            float wL = *in + sL;
            if (!std::isfinite(wL)) wL = 0.0f;
            ringL_.writeBlock(&wL, writeIdx_, 1);
            ringL_.refreshMirror(writeIdx_, 1);

            if (hasR)
            {
                float wR = *inR + sR;
                if (!std::isfinite(wR)) wR = 0.0f;
                ringR_.writeBlock(&wR, writeIdx_, 1);
                ringR_.refreshMirror(writeIdx_, 1);
            }

            writeIdx_ = (writeIdx_ + 1) & mask;

            *wet = tapL * loopTrim_;   // the blended (diffused when on) loop-tap stream
            if (hasR) *wetR = tapR * loopTrim_;
        }

        Pow2RingBuffer ringL_;
        Pow2RingBuffer ringR_;
        int writeIdx_ = 0;
        float maxDelay_ = 0.0f;
        double sampleRate_ = 48000.0;
        bool  firstBlock_ = true;

        Smoothers::LinearSmoother<float> delaySm_;
        Smoothers::LinearSmoother<float> fbSm_;
        Smoothers::LinearSmoother<float> crossSm_;
        Smoothers::LinearSmoother<float> driveSm_;
        Smoothers::LinearSmoother<float> dampGSm_;
        Smoothers::LinearSmoother<float> satLatencySm_;

        // block-rate coefficients
        float dampG_ = 0.0f;
        float dcR_   = 0.999f;
        int   satOrder_ = 2;
        float satLatency_ = 1.0f;
        float loopTrim_ = 1.0f;

        // per-channel loop state
        float dampL_ = 0.0f;
        float dampR_ = 0.0f;
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
        bool enableDiffuser_ = false;
        DiffuserState diffState_ = DiffuserState::Off;
        float diffFade_ = 0.0f;   // 0 = raw tap, 1 = diffused tap
    };
}
#endif

#pragma once

#ifndef CHRONOS_FEEDBACK_DELAY_H
#define CHRONOS_FEEDBACK_DELAY_H

#include "FracDelayTap.h"
#include "LinearSmoother.h"
#include "Pow2RingBuffer.h"
#include "nonlinear/ADAA1.h"
#include "nonlinear/ADAA2.h"
#include "nonlinear/Nonlinearities.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>

namespace MarsDSP::Delays {

    class FeedbackDelay {
    public:
        static constexpr float kMaxFeedback     = 1.2f;
        static constexpr float kMinLoopDelay    = 4.0f;   // > FracDelayTap's 3.0 contract
        static constexpr float kMinDriveMakeup  = 1.0f;

        struct Params
        {
            float delaySamples = 4800.0f;
            float feedback     = 0.0f;    // 0..kMaxFeedback; > 1 self-oscillates, bounded
            float dampHz       = 6000.0f; // one-pole lowpass in the loop
            float crossFeed    = 0.0f;    // 0 straight, 1 full ping-pong
            float loopDrive    = 1.0f;    // how hard repeats lean on the tanh ceiling
            int   satOrder     = 2;       // 0 = hard bypass sat, 1 = ADAA1, 2 = ADAA2
        };

        void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples) noexcept
        {
            assert(sampleRate > 0.0);
            assert(maxBlockSize > 0);
            assert(maxDelaySamples > static_cast<int>(kMinLoopDelay));

            sampleRate_ = sampleRate;
            const int minCap = maxDelaySamples + maxBlockSize
                             + Pow2RingBuffer::kTail + 8;
            ringL_.prepare(minCap);
            ringR_.prepare(minCap);
            maxDelay_ = static_cast<float>(
                ringL_.getCapacity() - Pow2RingBuffer::kTail - 2);

            delaySm_.reset(sampleRate, 0.050);
            fbSm_.reset(sampleRate, 0.020);
            crossSm_.reset(sampleRate, 0.020);
            driveSm_.reset(sampleRate, 0.020);
            reset();
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
            firstBlock_ = true;
        }

        void resetParams(const Params& p) noexcept
        {
            applyBlockRate_(p);
            delaySm_.setCurrentAndTargetValue(clampDelay_(p.delaySamples));
            fbSm_.setCurrentAndTargetValue(std::clamp(p.feedback, 0.0f, kMaxFeedback));
            crossSm_.setCurrentAndTargetValue(std::clamp(p.crossFeed, 0.0f, 1.0f));
            driveSm_.setCurrentAndTargetValue(std::clamp(p.loopDrive, 0.1f, 16.0f));
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
        }

        void process(const float* inL, const float* inR,
                     float* wetL, float* wetR, int n) noexcept
        {
            assert(inL != nullptr && wetL != nullptr);
            const bool hasR = (inR != nullptr && wetR != nullptr);
            const int  mask = ringL_.mask();

            for (int s = 0; s < n; ++s)
            {
                const float d     = delaySm_.getNextValue();
                const float g     = fbSm_.getNextValue();
                const float cross = crossSm_.getNextValue();
                const float drive = driveSm_.getNextValue();
                const float makeup = 1.0f / std::max(drive, kMinDriveMakeup);

                const float readDelay =
                    std::max(kMinLoopDelay, d - satLatency_);

                const float tapL = FracDelayTap::read(ringL_, writeIdx_, readDelay);
                const float tapR = hasR
                    ? FracDelayTap::read(ringR_, writeIdx_, readDelay)
                    : tapL;

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

                float wL = inL[s] + sL;
                if (!std::isfinite(wL)) wL = 0.0f;
                ringL_.writeBlock(&wL, writeIdx_, 1);
                ringL_.refreshMirror(writeIdx_, 1);

                if (hasR)
                {
                    float wR = inR[s] + sR;
                    if (!std::isfinite(wR)) wR = 0.0f;
                    ringR_.writeBlock(&wR, writeIdx_, 1);
                    ringR_.refreshMirror(writeIdx_, 1);
                }

                writeIdx_ = (writeIdx_ + 1) & mask;

                wetL[s] = tapL;
                if (hasR) wetR[s] = tapR;
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept { return 0; }

    private:
        float clampDelay_(float d) const noexcept
        {
            return std::clamp(d, kMinLoopDelay + 1.5f, maxDelay_);
        }

        void applyBlockRate_(const Params& p) noexcept
        {
            const double fc = std::clamp(static_cast<double>(p.dampHz),
                                         20.0, 0.45 * sampleRate_);
            const double gw = std::tan(std::numbers::pi * fc / sampleRate_);
            dampG_ = static_cast<float>(gw / (1.0 + gw));

            // DC blocker pole: ~8 Hz, R = exp(-2*pi*fc/fs).
            dcR_ = static_cast<float>(
                std::exp(-2.0 * std::numbers::pi * 8.0 / sampleRate_));

            satOrder_ = std::clamp(p.satOrder, 0, 2);
            satLatency_ = (satOrder_ == 2) ? 1.0f
                        : (satOrder_ == 1) ? 0.5f
                                           : 0.0f;
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

        Pow2RingBuffer ringL_, ringR_;
        int   writeIdx_ = 0;
        float maxDelay_ = 0.0f;
        double sampleRate_ = 48000.0;
        bool  firstBlock_ = true;

        Smoothers::LinearSmoother<float> delaySm_, fbSm_, crossSm_, driveSm_;

        // block-rate coefficients
        float dampG_ = 0.0f;
        float dcR_   = 0.999f;
        int   satOrder_ = 2;
        float satLatency_ = 1.0f;

        // per-channel loop state
        float dampL_ = 0.0f, dampR_ = 0.0f;
        float dcXL_ = 0.0f, dcYL_ = 0.0f, dcXR_ = 0.0f, dcYR_ = 0.0f;

        Nonlinear::ADAA1<Nonlinear::TanhNL> adaa1L_, adaa1R_;
        Nonlinear::ADAA2<Nonlinear::TanhNL> adaa2L_, adaa2R_;
    };
}
#endif

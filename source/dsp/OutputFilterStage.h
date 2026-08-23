#pragma once

#ifndef CHRONOS_OUTPUT_FILTER_STAGE_H
#define CHRONOS_OUTPUT_FILTER_STAGE_H

#include "SallenKeyLPF.h"
#include "SallenKeyHPF.h"
#include "StateVariable.h"
#include "OnePoleSmoother.h"
#include "nonlinear/ADAA1.h"
#include "nonlinear/Nonlinearities.h"
#include "simd/Config.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace MarsDSP::Filters
{
    /** Output filter stage with two topologies.
     *  The Digital topology runs a stereo SIMD state-variable HPF
     *  and LPF. The Analog topology runs scalar Sallen-Key sections
     *  with an ADAA1 tanh saturator on each filter output. The mode
     *  switch is a 20 ms linear crossfade. Both modes are zero latency.
     */
    class OutputFilterStage
    {
    public:
        enum class Mode
        {
            Digital = 0,
            Analog = 1
        };

        void prepare (double sampleRate, int numChannels)
        {
            sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
            numChannels_ = numChannels > 1 ? 2 : 1;

            svfHpf_.reset();
            svfLpf_.reset();

            skHpfL_.prepare (sampleRate_);
            skHpfR_.prepare (sampleRate_);
            skLpfL_.prepare (sampleRate_);
            skLpfR_.prepare (sampleRate_);

            adaaHpfL_.reset();
            adaaHpfR_.reset();
            adaaLpfL_.reset();
            adaaLpfR_.reset();

            fadeLengthSamples_ = std::max (1, static_cast<int> (std::round (0.02 * sampleRate_)));
            fadeStep_ = 1.0f / static_cast<float> (fadeLengthSamples_ > 1 ? fadeLengthSamples_ - 1 : 1);
            fadePos_ = 0.0f;
            targetMode_ = Mode::Digital;

            hpfSm_.reset (sampleRate_, 0.01, 32);
            lpfSm_.reset (sampleRate_, 0.01, 32);
            hpfSm_.setCurrentAndTargetValue (20.0f);
            lpfSm_.setCurrentAndTargetValue (20000.0f);

            lastHpfHz_ = -1.0f;
            lastLpfHz_ = -1.0f;
            hpfHz_ = 20.0f;
            lpfHz_ = 20000.0f;

            reset();
        }

        void reset()
        {
            svfHpf_.reset();
            svfLpf_.reset();

            skHpfL_.reset();
            skHpfR_.reset();
            skLpfL_.reset();
            skLpfR_.reset();

            adaaHpfL_.reset();
            adaaHpfR_.reset();
            adaaLpfL_.reset();
            adaaLpfR_.reset();

            fadePos_ = (targetMode_ == Mode::Analog) ? 1.0f : 0.0f;
            hpfSm_.setCurrentAndTargetValue (hpfHz_);
            lpfSm_.setCurrentAndTargetValue (lpfHz_);

            lastHpfHz_ = -1.0f;
            lastLpfHz_ = -1.0f;
        }

        void setMode (Mode m)
        {
            const bool atDigital = (fadePos_ <= 0.0f);
            const bool atAnalog = (fadePos_ >= 1.0f);
            if (atDigital && m == Mode::Analog)
                resetAnalogPath_();
            else if (atAnalog && m == Mode::Digital)
                resetDigitalPath_();
            targetMode_ = m;
        }

        // Snap to a mode with no crossfade. The engine calls this on reset.
        void setModeImmediate (Mode m)
        {
            fadePos_ = (m == Mode::Analog) ? 1.0f : 0.0f;
            targetMode_ = m;
            if (m == Mode::Digital)
                resetAnalogPath_();
            else
                resetDigitalPath_();
        }

        void setCutoffs (float hpfHz, float lpfHz)
        {
            hpfHz_ = hpfHz;
            lpfHz_ = lpfHz;
            hpfSm_.setTargetValue (hpfHz);
            lpfSm_.setTargetValue (lpfHz);
        }

        void process (const float* inL, const float* inR, float* outL, float* outR, int n) noexcept
        {
            if (n <= 0)
                return;

            const bool hasR = (numChannels_ > 1 && inR != nullptr && outR != nullptr);
            const float targetPos = (targetMode_ == Mode::Analog) ? 1.0f : 0.0f;

            for (int offset = 0; offset < n;)
            {
                const int subBlock = std::min (32, n - offset);

                const bool digEngaged = (fadePos_ < 1.0f) || (targetMode_ == Mode::Digital);
                const bool anaEngaged = (fadePos_ > 0.0f) || (targetMode_ == Mode::Analog);

                // The Digital SVF ramps its coefficients across the sub-block.
                if (digEngaged)
                {
                    svfHpf_.setCoeffForBlock (SimdSVF::SVFType::HighPass, sampleRate_, hpfHz_, svfQ_, 0.0, subBlock);
                    svfLpf_.setCoeffForBlock (SimdSVF::SVFType::LowPass, sampleRate_, lpfHz_, svfQ_, 0.0, subBlock);
                }

                // The Analog coefficients follow a 10 ms one-pole. The
                // 1e-4 relative guard skips the setParams call at rest.
                if (anaEngaged)
                {
                    hpfSm_.processN (subBlock);
                    lpfSm_.processN (subBlock);

                    const float hpfSolved = hpfSm_.getCurrentValue();
                    const float lpfSolved = lpfSm_.getCurrentValue();
                    const bool hpfMoved = (lastHpfHz_ <= 0.0f)
                                          || (std::fabs (hpfSolved - lastHpfHz_) / lastHpfHz_ > 1e-4f);
                    const bool lpfMoved = (lastLpfHz_ <= 0.0f)
                                          || (std::fabs (lpfSolved - lastLpfHz_) / lastLpfHz_ > 1e-4f);

                    if (hpfMoved)
                    {
                        skHpfL_.setParams (hpfSolved, static_cast<float> (svfQ_));
                        if (hasR)
                            skHpfR_.setParams (hpfSolved, static_cast<float> (svfQ_));
                        lastHpfHz_ = hpfSolved;
                    }

                    if (lpfMoved)
                    {
                        skLpfL_.setParams (lpfSolved, static_cast<float> (svfQ_));
                        if (hasR)
                            skLpfR_.setParams (lpfSolved, static_cast<float> (svfQ_));
                        lastLpfHz_ = lpfSolved;
                    }
                }

                for (int s = 0; s < subBlock; ++s)
                {
                    const int idx = offset + s;
                    float xL = inL[idx];
                    float xR = hasR ? inR[idx] : 0.0f;

                    // Non-finite hygiene on inputs
                    if (!std::isfinite (xL))
                    {
                        xL = 0.0f;
                        skHpfL_.reset();
                        skLpfL_.reset();
                        adaaHpfL_.reset();
                        adaaLpfL_.reset();
                    }
                    if (hasR && !std::isfinite (xR))
                    {
                        xR = 0.0f;
                        skHpfR_.reset();
                        skLpfR_.reset();
                        adaaHpfR_.reset();
                        adaaLpfR_.reset();
                    }

                    const bool digRun = (fadePos_ < 1.0f);
                    const bool anaRun = (fadePos_ > 0.0f);

                    float digL = 0.0f;
                    float digR = 0.0f;
                    float anaL = 0.0f;
                    float anaR = 0.0f;

                    if (digRun)
                    {
                        const M128 wetV = MM(set_ps) (0.0f, 0.0f, hasR ? xR : 0.0f, xL);
                        const M128 hpV = svfHpf_.processBlockStep (wetV);
                        const M128 lpV = svfLpf_.processBlockStep (hpV);
                        alignas(16) std::array<float, 4> outV {};
                        MM(store_ps) (outV.data(), lpV);
                        digL = outV[0];
                        if (hasR) digR = outV[1];
                    }

                    if (anaRun)
                    {
                        // High-pass filter
                        float hpOutL = skHpfL_.processSample (xL);
                        if (!std::isfinite (hpOutL)) { hpOutL = 0.0f; skHpfL_.reset(); }
                        hpOutL = static_cast<float> (adaaHpfL_.process (static_cast<double> (hpOutL * invRail_))) * kRail_;

                        // Low-pass filter
                        float lpOutL = skLpfL_.processSample (hpOutL);
                        if (!std::isfinite (lpOutL)) { lpOutL = 0.0f; skLpfL_.reset(); }
                        lpOutL = static_cast<float> (adaaLpfL_.process (static_cast<double> (lpOutL * invRail_))) * kRail_;

                        anaL = lpOutL;

                        if (hasR)
                        {
                            float hpOutR = skHpfR_.processSample (xR);
                            if (!std::isfinite (hpOutR)) { hpOutR = 0.0f; skHpfR_.reset(); }
                            hpOutR = static_cast<float> (adaaHpfR_.process (static_cast<double> (hpOutR * invRail_))) * kRail_;

                            float lpOutR = skLpfR_.processSample (hpOutR);
                            if (!std::isfinite (lpOutR)) { lpOutR = 0.0f; skLpfR_.reset(); }
                            lpOutR = static_cast<float> (adaaLpfR_.process (static_cast<double> (lpOutR * invRail_))) * kRail_;

                            anaR = lpOutR;
                        }
                    }

                    if (digRun && anaRun)
                    {
                        const float inv = 1.0f - fadePos_;
                        outL[idx] = inv * digL + fadePos_ * anaL;
                        if (hasR) outR[idx] = inv * digR + fadePos_ * anaR;
                    }
                    else if (digRun)
                    {
                        outL[idx] = digL;
                        if (hasR) outR[idx] = digR;
                    }
                    else
                    {
                        outL[idx] = anaL;
                        if (hasR) outR[idx] = anaR;
                    }

                    if (fadePos_ < targetPos)
                        fadePos_ = std::min (fadePos_ + fadeStep_, targetPos);
                    else if (fadePos_ > targetPos)
                        fadePos_ = std::max (fadePos_ - fadeStep_, targetPos);
                }

                offset += subBlock;
            }
        }

    private:
        void resetAnalogPath_() noexcept
        {
            skHpfL_.reset();
            skHpfR_.reset();
            skLpfL_.reset();
            skLpfR_.reset();
            adaaHpfL_.reset();
            adaaHpfR_.reset();
            adaaLpfL_.reset();
            adaaLpfR_.reset();
            lastHpfHz_ = -1.0f;
            lastLpfHz_ = -1.0f;
        }

        void resetDigitalPath_() noexcept
        {
            svfHpf_.reset();
            svfLpf_.reset();
        }

        double sampleRate_ { 48000.0 };
        int numChannels_ { 2 };
        static constexpr double svfQ_ { 0.7071 };

        static constexpr float kRail_ { 4.0f };
        static constexpr float invRail_ { 1.0f / kRail_ };

        SimdSVF svfHpf_ {};
        SimdSVF svfLpf_ {};

        SallenKeyHPF skHpfL_ {};
        SallenKeyHPF skHpfR_ {};
        SallenKeyLPF skLpfL_ {};
        SallenKeyLPF skLpfR_ {};

        Nonlinear::ADAA1<Nonlinear::TanhNL> adaaHpfL_ {};
        Nonlinear::ADAA1<Nonlinear::TanhNL> adaaHpfR_ {};
        Nonlinear::ADAA1<Nonlinear::TanhNL> adaaLpfL_ {};
        Nonlinear::ADAA1<Nonlinear::TanhNL> adaaLpfR_ {};

        Mode targetMode_ { Mode::Digital };
        float fadePos_ { 0.0f };
        float fadeStep_ { 1.0f / 959.0f };
        int fadeLengthSamples_ { 960 };

        Smoothers::OnePoleSmoother<float> hpfSm_ {};
        Smoothers::OnePoleSmoother<float> lpfSm_ {};

        float hpfHz_ { 20.0f };
        float lpfHz_ { 20000.0f };
        float lastHpfHz_ { -1.0f };
        float lastLpfHz_ { -1.0f };
    };
}
#endif

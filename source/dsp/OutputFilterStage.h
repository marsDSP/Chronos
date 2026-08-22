#pragma once

#ifndef CHRONOS_OUTPUT_FILTER_STAGE_H
#define CHRONOS_OUTPUT_FILTER_STAGE_H

#include "SallenKeyLPF.h"
#include "SallenKeyHPF.h"
#include "StateVariable.h"
#include "nonlinear/ADAA1.h"
#include "nonlinear/Nonlinearities.h"
#include "simd/Config.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace MarsDSP::Filters
{
    /** Output filter stage with two topologies.
     *  Digital runs a stereo-pair SIMD state-variable HPF/LPF cascade.
     *  Analog runs scalar wave-digital Sallen-Key sections with an
     *  out-of-loop ADAA1 tanh saturator per filter output. The mode switch
     *  is a 20 ms linear crossfade; both modes are zero latency.
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
            fadeStep_ = 0;
            isFading_ = false;
            currentMode_ = Mode::Digital;
            targetMode_ = Mode::Digital;

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

            fadeStep_ = 0;
            isFading_ = false;
            currentMode_ = targetMode_;

            lastHpfHz_ = -1.0f;
            lastLpfHz_ = -1.0f;
        }

        void setMode (Mode m)
        {
            if (m == targetMode_ && !isFading_)
                return;

            targetMode_ = m;
            if (currentMode_ != targetMode_)
            {
                isFading_ = true;
                fadeStep_ = 0;

                // Reset incoming path at start of crossfade
                if (targetMode_ == Mode::Analog)
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
                else
                {
                    svfHpf_.reset();
                    svfLpf_.reset();
                }
            }
        }

        void setCutoffs (float hpfHz, float lpfHz)
        {
            hpfHz_ = hpfHz;
            lpfHz_ = lpfHz;
        }

        void process (const float* inL, const float* inR, float* outL, float* outR, int n) noexcept
        {
            if (n <= 0)
                return;

            const bool hasR = (numChannels_ > 1 && inR != nullptr && outR != nullptr);

            for (int offset = 0; offset < n;)
            {
                const int subBlock = std::min (32, n - offset);

                // Update Digital mode SVF coefficients once per sub-block
                if (currentMode_ == Mode::Digital || isFading_)
                {
                    svfHpf_.setCoeffForBlock (SimdSVF::SVFType::HighPass, sampleRate_, hpfHz_, svfQ_, 0.0, subBlock);
                    svfLpf_.setCoeffForBlock (SimdSVF::SVFType::LowPass, sampleRate_, lpfHz_, svfQ_, 0.0, subBlock);
                }

                // Update Analog mode Sallen-Key coefficients if moved > 0.05%
                if (currentMode_ == Mode::Analog || isFading_)
                {
                    const bool hpfMoved = (lastHpfHz_ <= 0.0f) || (std::fabs (hpfHz_ - lastHpfHz_) / lastHpfHz_ > 0.0005f);
                    const bool lpfMoved = (lastLpfHz_ <= 0.0f) || (std::fabs (lpfHz_ - lastLpfHz_) / lastLpfHz_ > 0.0005f);

                    if (hpfMoved)
                    {
                        skHpfL_.setParams (hpfHz_, static_cast<float> (svfQ_));
                        if (hasR)
                            skHpfR_.setParams (hpfHz_, static_cast<float> (svfQ_));
                        lastHpfHz_ = hpfHz_;
                    }

                    if (lpfMoved)
                    {
                        skLpfL_.setParams (lpfHz_, static_cast<float> (svfQ_));
                        if (hasR)
                            skLpfR_.setParams (lpfHz_, static_cast<float> (svfQ_));
                        lastLpfHz_ = lpfHz_;
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

                    float digL = 0.0f;
                    float digR = 0.0f;
                    float anaL = 0.0f;
                    float anaR = 0.0f;

                    if (currentMode_ == Mode::Digital || isFading_)
                    {
                        const M128 wetV = MM(set_ps) (0.0f, 0.0f, hasR ? xR : 0.0f, xL);
                        const M128 hpV = svfHpf_.processBlockStep (wetV);
                        const M128 lpV = svfLpf_.processBlockStep (hpV);
                        alignas(16) std::array<float, 4> outV {};
                        MM(store_ps) (outV.data(), lpV);
                        digL = outV[0];
                        if (hasR) digR = outV[1];
                    }

                    if (currentMode_ == Mode::Analog || isFading_)
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

                    if (isFading_)
                    {
                        const float alpha = static_cast<float> (fadeStep_) / static_cast<float> (fadeLengthSamples_);
                        const float fromL = (currentMode_ == Mode::Digital) ? digL : anaL;
                        const float fromR = (currentMode_ == Mode::Digital) ? digR : anaR;
                        const float toL   = (targetMode_ == Mode::Digital) ? digL : anaL;
                        const float toR   = (targetMode_ == Mode::Digital) ? digR : anaR;

                        outL[idx] = (1.0f - alpha) * fromL + alpha * toL;
                        if (hasR) outR[idx] = (1.0f - alpha) * fromR + alpha * toR;

                        ++fadeStep_;
                        if (fadeStep_ >= fadeLengthSamples_)
                        {
                            isFading_ = false;
                            currentMode_ = targetMode_;
                            fadeStep_ = 0;
                        }
                    }
                    else
                    {
                        if (currentMode_ == Mode::Digital)
                        {
                            outL[idx] = digL;
                            if (hasR) outR[idx] = digR;
                        }
                        else
                        {
                            outL[idx] = anaL;
                            if (hasR) outR[idx] = anaR;
                        }
                    }
                }

                offset += subBlock;
            }
        }

    private:
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

        Mode currentMode_ { Mode::Digital };
        Mode targetMode_ { Mode::Digital };
        bool isFading_ { false };
        int fadeStep_ { 0 };
        int fadeLengthSamples_ { 960 };

        float hpfHz_ { 20.0f };
        float lpfHz_ { 20000.0f };
        float lastHpfHz_ { -1.0f };
        float lastLpfHz_ { -1.0f };
    };
}
#endif

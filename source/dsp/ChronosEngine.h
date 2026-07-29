#pragma once

#ifndef CHRONOS_CHRONOS_ENGINE_H
#define CHRONOS_CHRONOS_ENGINE_H

#include "SimdDelayLine.h"
#include "StateVariable.h"
#include "LinearSmoother.h"
#include "nonlinear/ADAA1.h"
#include "nonlinear/ADAA2.h"
#include "nonlinear/Nonlinearities.h"
#include "align/SaturatorAlign.h"
#include "DelayInterpolator.h"
#include "math/Trigonometry.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace MarsDSP {
    class ChronosEngine {
    public:
        struct Params
        {
            float delaySamples;
            float driveLin;
            float mix;
            float gainLin;
            float hpfHz;
            float lpfHz;
            int bits;
            int adaaOrder;
            Delays::Interpolation interp;
        };

        void prepare(double sampleRate, int maxBlockSize, int numChannels) noexcept
        {
            assert(sampleRate > 0.0);
            assert(maxBlockSize > 0);
            assert(numChannels == 1 || numChannels == 2);

            sampleRate_ = sampleRate;
            numChannels_ = numChannels;
            wetBufCapacity_ = std::max(1, 2 * maxBlockSize);

            delayLine_.prepare(sampleRate, wetBufCapacity_, 5000.0f);
            wetBufL_.resize(static_cast<std::size_t>(wetBufCapacity_));
            wetBufR_.resize(static_cast<std::size_t>(wetBufCapacity_));

            constexpr double kRampSeconds = 0.02;
            gainSmoother_.reset(sampleRate, kRampSeconds);
            bitsSmoother_.reset(sampleRate, kRampSeconds);
            hpfSmoother_.reset(sampleRate, kRampSeconds);
            lpfSmoother_.reset(sampleRate, kRampSeconds);
            mixSmoother_.reset(sampleRate, kRampSeconds);
            driveSmoother_.reset(sampleRate, kRampSeconds);

            reset();
        }

        void reset() noexcept
        {
            delayLine_.reset();
            hpf_.reset();
            lpf_.reset();
            adaa1L_.reset();
            adaa1R_.reset();
            adaa2L_.reset();
            adaa2R_.reset();
            alignL_.reset();
            alignR_.reset();


            smoothedGain_ = 0.0f;
            smoothedBits_ = 0;
            smoothedHpf_ = 0.0f;
            smoothedLpf_ = 0.0f;
            smoothedMix_ = 0.0f;
            smoothedDrive_ = 0.0f;
        }

        void resetParams(const Params &p) noexcept
        {
            delaySamples_ = p.delaySamples;
            adaaOrder_ = p.adaaOrder;
            interp_ = p.interp;
            delayLine_.setInterpolation(p.interp);

            gainSmoother_.setCurrentAndTargetValue(p.gainLin);
            bitsSmoother_.setCurrentAndTargetValue(static_cast<float>(p.bits));
            hpfSmoother_.setCurrentAndTargetValue(p.hpfHz);
            lpfSmoother_.setCurrentAndTargetValue(p.lpfHz);
            mixSmoother_.setCurrentAndTargetValue(p.mix);
            driveSmoother_.setCurrentAndTargetValue(p.driveLin);
        }

        void setParams(const Params &p) noexcept
        {
            delaySamples_ = p.delaySamples;
            adaaOrder_ = p.adaaOrder;
            interp_ = p.interp;
            delayLine_.setInterpolation(p.interp);

            gainSmoother_.setTargetValue(p.gainLin);
            bitsSmoother_.setTargetValue(static_cast<float>(p.bits));
            hpfSmoother_.setTargetValue(p.hpfHz);
            lpfSmoother_.setTargetValue(p.lpfHz);
            mixSmoother_.setTargetValue(p.mix);
            driveSmoother_.setTargetValue(p.driveLin);
        }

        void process(float *const*io, int numChannels, int numSamples) noexcept
        {
            assert(io != nullptr);
            assert(io[0] != nullptr);
            assert(numChannels == 1 || numChannels == 2);
            if (numSamples <= 0) return;

            const double fsSafe = sampleRate_ > 0.0 ? sampleRate_ : 48000.0;

            float *data0 = io[0];
            float *data1 = numChannels > 1 ? io[1] : nullptr;

            for (int offset = 0; offset < numSamples;)
            {
                const int chunk = std::min(wetBufCapacity_, numSamples - offset);

                delayLine_.process(data0 + offset,
                                   data1 != nullptr ? data1 + offset : nullptr,
                                   wetBufL_.data(),
                                   data1 != nullptr ? wetBufR_.data() : nullptr,
                                   chunk, delaySamples_, delaySamples_);

                hpf_.setCoeffForBlock(SVF::SVFType::HighPass, fsSafe, smoothedHpf_, svfQ_, 0.0, chunk);
                lpf_.setCoeffForBlock(SVF::SVFType::LowPass, fsSafe, smoothedLpf_, svfQ_, 0.0, chunk);

                alignL_.setMode(adaaOrder_);
                alignR_.setMode(adaaOrder_);

                for (int s = 0; s < chunk; ++s)
                {
                    smoothen_();

                    const float driveLin = smoothedDrive_;
                    const float mixNorm = smoothedMix_ * 0.01f;
                    const float theta = mixNorm * (std::numbers::pi_v<float> * 0.5f);

                    const float dryGain = mmCos(theta);
                    const float wetGain = mmSin(theta);

                    const float dry0 = data0[offset + s];
                    const float dry0a = alignL_.processDry(dry0);

                    float wet0 = wetBufL_[static_cast<std::size_t>(s)];

                    float dry1 = 0.0f;
                    float dry1a = 0.0f;
                    float wet1 = 0.0f;
                    if (data1 != nullptr)
                    {
                        dry1 = data1[offset + s];
                        dry1a = alignR_.processDry(dry1);
                        wet1 = wetBufR_[static_cast<std::size_t>(s)];
                    }

                    float sat0;
                    float sat1 = 0.0f;
                    switch (adaaOrder_)
                    {
                        case 0:
                            sat0 = wet0;
                            if (data1 != nullptr) sat1 = wet1;
                            break;
                        case 1:
                            sat0 = static_cast<float>(adaa1L_.process(driveLin * wet0));
                            if (data1 != nullptr) sat1 = static_cast<float>(adaa1R_.process(driveLin * wet1));
                            break;
                        default:
                            sat0 = static_cast<float>(adaa2L_.process(driveLin * wet0));
                            if (data1 != nullptr) sat1 = static_cast<float>(adaa2R_.process(driveLin * wet1));
                            break;
                    }

                    sat0 = alignL_.processWet(sat0);
                    if (data1 != nullptr) sat1 = alignR_.processWet(sat1);

                    const M128 wetV = MM(set_ps)(0.0f, 0.0f, sat1, sat0);
                    const M128 hpV = hpf_.processBlockStep(wetV);
                    const M128 lpV = lpf_.processBlockStep(hpV);
                    alignas(16) std::array<float, 4> out;
                    MM(store_ps)(out.data(), lpV);

                    data0[offset + s] = dry0a * dryGain + out[0] * wetGain;
                    if (data1 != nullptr) data1[offset + s] = dry1a * dryGain + out[1] * wetGain;

                    const float gainLin = smoothedGain_;
                    const float lsb = std::ldexp(1.0f, 1 - smoothedBits_);

                    for (int ch = 0; ch < numChannels; ++ch)
                    {
                        auto *data = io[ch];
                        auto &state = ch == 0 ? xorshiftL_ : xorshiftR_;
                        const float scaled = data[offset + s] * gainLin;
                        const float dither = (nextUniform(state) - nextUniform(state)) * lsb;
                        data[offset + s] = std::round((scaled + dither) / lsb) * lsb;
                    }
                }
                offset += chunk;
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept
        {
            return Align::SaturatorAlign::kBudget;
        }

        [[nodiscard]] int getWetBufCapacity() const noexcept { return wetBufCapacity_; }

        void setDitherSeeds(std::uint32_t l, std::uint32_t r) noexcept
        {
            xorshiftL_ = l;
            xorshiftR_ = r;
        }

    private:
        void smoothen_() noexcept
        {
            smoothedGain_ = gainSmoother_.getNextValue();
            smoothedBits_ = static_cast<int>(bitsSmoother_.getNextValue());
            smoothedHpf_ = hpfSmoother_.getNextValue();
            smoothedLpf_ = lpfSmoother_.getNextValue();
            smoothedMix_ = mixSmoother_.getNextValue();
            smoothedDrive_ = driveSmoother_.getNextValue();
        }

        Delays::SimdDelayLine delayLine_;
        std::vector<float> wetBufL_;
        std::vector<float> wetBufR_;
        int wetBufCapacity_{0};

        using SVF = Filters::SimdSVF;
        SVF hpf_;
        SVF lpf_;
        static constexpr double svfQ_{0.7071};

        Nonlinear::ADAA1<Nonlinear::TanhNL> adaa1L_;
        Nonlinear::ADAA1<Nonlinear::TanhNL> adaa1R_;
        Nonlinear::ADAA2<Nonlinear::TanhNL> adaa2L_;
        Nonlinear::ADAA2<Nonlinear::TanhNL> adaa2R_;

        Align::SaturatorAlign alignL_;
        Align::SaturatorAlign alignR_;

        std::uint32_t xorshiftL_{0x12345678u};
        std::uint32_t xorshiftR_{0x9abcdef0u};

        static float nextUniform(std::uint32_t &state) noexcept
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
        }

        Smoothers::LinearSmoother<float> gainSmoother_;
        Smoothers::LinearSmoother<float> bitsSmoother_;
        Smoothers::LinearSmoother<float> hpfSmoother_;
        Smoothers::LinearSmoother<float> lpfSmoother_;
        Smoothers::LinearSmoother<float> mixSmoother_;
        Smoothers::LinearSmoother<float> driveSmoother_;

        float smoothedGain_{};
        int smoothedBits_{};
        float smoothedHpf_{};
        float smoothedLpf_{};
        float smoothedMix_{};
        float smoothedDrive_{};

        double sampleRate_{0.0};
        int numChannels_{0};
        float delaySamples_{0.0f};
        int adaaOrder_{2};

        Delays::Interpolation interp_{Delays::Interpolation::Lagrange5th};
    };
}
#endif

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

            // preallocate all scratch, 16-byte aligned, sized to wetBufCapacity_.
            driveRamp_     .resize(static_cast<std::size_t>(wetBufCapacity_));
            hpfRamp_       .resize(static_cast<std::size_t>(wetBufCapacity_));
            lpfRamp_       .resize(static_cast<std::size_t>(wetBufCapacity_));
            thetaRamp_     .resize(static_cast<std::size_t>(wetBufCapacity_));
            gainRamp_      .resize(static_cast<std::size_t>(wetBufCapacity_));
            lsbRamp_       .resize(static_cast<std::size_t>(wetBufCapacity_));
            satL_          .resize(static_cast<std::size_t>(wetBufCapacity_));
            satR_          .resize(static_cast<std::size_t>(wetBufCapacity_));
            alignedDryL_   .resize(static_cast<std::size_t>(wetBufCapacity_));
            alignedDryR_   .resize(static_cast<std::size_t>(wetBufCapacity_));
            wetPostSvfL_   .resize(static_cast<std::size_t>(wetBufCapacity_));
            wetPostSvfR_   .resize(static_cast<std::size_t>(wetBufCapacity_));
            bypassDryInL_  .resize(static_cast<std::size_t>(wetBufCapacity_));
            bypassDryInR_  .resize(static_cast<std::size_t>(wetBufCapacity_));
            blendL_        .resize(static_cast<std::size_t>(wetBufCapacity_));
            blendR_        .resize(static_cast<std::size_t>(wetBufCapacity_));

            bypassSmoother_.reset(sampleRate, 0.01);
            bypassDryL_.reset();
            bypassDryR_.reset();
            bypassDryL_.setDelay(latencySamples());
            bypassDryR_.setDelay(latencySamples());

            constexpr double kRampSeconds = 0.02;
            gainSmoother_.reset(sampleRate, kRampSeconds);
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
            adaa1L_.reset(); adaa1R_.reset();
            adaa2L_.reset(); adaa2R_.reset();
            alignL_.reset(); alignR_.reset();
            bypassDryL_.reset(); bypassDryR_.reset();
            bypassDryL_.setDelay(latencySamples());
            bypassDryR_.setDelay(latencySamples());
            bypassSmoother_.setCurrentAndTargetValue(0.0f);
            bypassTarget_ = 0.0f;

            smoothedGain_ = 0.0f;
            smoothedBits_ = 0;
            smoothedHpf_ = 0.0f;
            smoothedLpf_ = 0.0f;
            smoothedMix_ = 0.0f;
            smoothedDrive_ = 0.0f;
            // hpf/lpf smoothed values start at 0.0f here, but resetParams
            // sets them to the raw parameter values so the SVF is configured
            // correctly from the first block (not from 0.0f).
        }

        void resetParams(const Params &p) noexcept
        {
            delaySamples_ = p.delaySamples;
            adaaOrder_ = p.adaaOrder;
            interp_ = p.interp;
            delayLine_.setInterpolation(p.interp);

            // init the smoothed values AND snap the smoothers to
            // the raw parameters, so the SVF is configured from the correct
            // cutoff on the first block (previously the SVF saw 0.0f on the
            // first block after prepare).
            smoothedHpf_ = p.hpfHz;
            smoothedLpf_ = p.lpfHz;

            gainSmoother_.setCurrentAndTargetValue(p.gainLin);
            smoothedBits_ = p.bits;
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
            smoothedBits_ = p.bits;
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
            const bool hasR = data1 != nullptr;

            for (int offset = 0; offset < numSamples;)
            {
                const int chunk = std::min(wetBufCapacity_, numSamples - offset);

                // ── 1. Delay line (block-rate) ────────────────────────────
                delayLine_.process(data0 + offset,
                                   hasR ? data1 + offset : nullptr,
                                   wetBufL_.data(),
                                   hasR ? wetBufR_.data() : nullptr,
                                   chunk, delaySamples_, delaySamples_);

                // bits is block-rate int, no smoother. lsb is computed
                // once per block
                const float blockLsb = std::ldexp(1.0f, 1 - smoothedBits_);
                for (int s = 0; s < chunk; ++s)
                {
                    smoothen_();
                    driveRamp_[static_cast<std::size_t>(s)] = smoothedDrive_;
                    hpfRamp_[static_cast<std::size_t>(s)] = smoothedHpf_;
                    lpfRamp_[static_cast<std::size_t>(s)] = smoothedLpf_;
                    thetaRamp_[static_cast<std::size_t>(s)] =
                        (smoothedMix_ * 0.01f) * (std::numbers::pi_v<float> * 0.5f);
                    gainRamp_[static_cast<std::size_t>(s)] = smoothedGain_;
                    lsbRamp_[static_cast<std::size_t>(s)] = blockLsb;
                }

                // ── 3. SVF coefficients (this block's start cutoff) ───────
                hpf_.setCoeffForBlock(SVF::SVFType::HighPass, fsSafe, hpfRamp_[0], svfQ_, 0.0, chunk);
                lpf_.setCoeffForBlock(SVF::SVFType::LowPass, fsSafe, lpfRamp_[0], svfQ_, 0.0, chunk);

                // ── 4. Align mode (once per chunk) ────────────────────────
                alignL_.setMode(adaaOrder_);
                alignR_.setMode(adaaOrder_);

                // ── 5. Align dry stage (stateful: ShortDelay ring) ────────
                for (int s = 0; s < chunk; ++s)
                {
                    bypassDryInL_[static_cast<std::size_t>(s)] = data0[offset + s];
                    if (hasR)
                        bypassDryInR_[static_cast<std::size_t>(s)] = data1[offset + s];
                    alignedDryL_[static_cast<std::size_t>(s)] =
                        alignL_.processDry(data0[offset + s]);
                    if (hasR)
                        alignedDryR_[static_cast<std::size_t>(s)] =
                            alignR_.processDry(data1[offset + s]);
                }

                // ── 6. ADAA + align wet stage (stateful) ──────────────────
                for (int s = 0; s < chunk; ++s)
                {
                    const auto u = static_cast<std::size_t>(s);
                    const float drv = driveRamp_[u];
                    const float wet0 = wetBufL_[u];
                    const float wet1 = hasR ? wetBufR_[u] : 0.0f;

                    float sat0;
                    float sat1 = 0.0f;
                    switch (adaaOrder_)
                    {
                        case 0:
                            sat0 = wet0;
                            if (hasR) sat1 = wet1;
                            break;
                        case 1:
                            sat0 = static_cast<float>(adaa1L_.process(driveRamp_[u] * wet0));
                            if (hasR) sat1 = static_cast<float>(adaa1R_.process(driveRamp_[u] * wet1));
                            break;
                        default:
                            sat0 = static_cast<float>(adaa2L_.process(driveRamp_[u] * wet0));
                            if (hasR) sat1 = static_cast<float>(adaa2R_.process(driveRamp_[u] * wet1));
                            break;
                    }
                    (void)drv;

                    satL_[u] = alignL_.processWet(sat0);
                    if (hasR) satR_[u] = alignR_.processWet(sat1);
                }

                // ── 7. SVF stage (stateful: IIR + coefficient ramp) ───────
                for (int s = 0; s < chunk; ++s)
                {
                    const auto u = static_cast<std::size_t>(s);
                    const M128 wetV = MM(set_ps)(0.0f, 0.0f,
                                                  hasR ? satR_[u] : 0.0f, satL_[u]);
                    const M128 hpV = hpf_.processBlockStep(wetV);
                    const M128 lpV = lpf_.processBlockStep(hpV);
                    alignas(16) std::array<float, 4> out;
                    MM(store_ps)(out.data(), lpV);
                    wetPostSvfL_[u] = out[0];
                    if (hasR) wetPostSvfR_[u] = out[1];
                }

                // ── 8. Crossfade stage (stateless) ────────────────────────
                // 4-wide SIMD. Endpoint clamp from D1 applies to both
                // paths. If mix is at an endpoint, skip trig entirely.
                const float mixVal = mixSmoother_.getCurrentValue();
                const bool fullDry = (mixVal <= 0.0f);
                const bool fullWet = (mixVal >= 100.0f);

                if (fullDry)
                {
                    for (int s = 0; s < chunk; ++s)
                    {
                        const auto u = static_cast<std::size_t>(s);
                        data0[offset + s] = alignedDryL_[u];
                        if (hasR) data1[offset + s] = alignedDryR_[u];
                    }
                }
                else if (fullWet)
                {
                    for (int s = 0; s < chunk; ++s)
                    {
                        const auto u = static_cast<std::size_t>(s);
                        data0[offset + s] = wetPostSvfL_[u];
                        if (hasR) data1[offset + s] = wetPostSvfR_[u];
                    }
                }
                else
                {
                    // SIMD path
                    const int jFull = chunk & ~3;
                    for (int s = 0; s + 4 <= chunk; s += 4)
                    {
                        const auto u = static_cast<std::size_t>(s);
                        const M128 vTheta = MM(loadu_ps)(thetaRamp_.data() + s);
                        const M128 vCos = mmCos(vTheta);
                        const M128 vSin = mmSin(vTheta);
                        // L: dry*cos + wet*sin
                        const M128 vDryL = MM(loadu_ps)(alignedDryL_.data() + s);
                        const M128 vWetL = MM(loadu_ps)(wetPostSvfL_.data() + s);
                        const M128 vOutL = FMADD(vDryL, vCos, MM(mul_ps)(vWetL, vSin));
                        MM(storeu_ps)(data0 + offset + s, vOutL);
                        if (hasR)
                        {
                            const M128 vDryR = MM(loadu_ps)(alignedDryR_.data() + s);
                            const M128 vWetR = MM(loadu_ps)(wetPostSvfR_.data() + s);
                            const M128 vOutR = FMADD(vDryR, vCos, MM(mul_ps)(vWetR, vSin));
                            MM(storeu_ps)(data1 + offset + s, vOutR);
                        }
                    }
                    // Scalar tail (same scalar overload, not a second approx)
                    for (int s = jFull; s < chunk; ++s)
                    {
                        const auto u = static_cast<std::size_t>(s);
                        const float dryGain = mmCos(thetaRamp_[u]);
                        const float wetGain = mmSin(thetaRamp_[u]);
                        data0[offset + s] = alignedDryL_[u] * dryGain + wetPostSvfL_[u] * wetGain;
                        if (hasR) data1[offset + s] = alignedDryR_[u] * dryGain + wetPostSvfR_[u] * wetGain;
                    }
                }

                // ── 9a. Bypass blend (scalar, stateful: dry delay + fade) ──
                for (int s = 0; s < chunk; ++s)
                {
                    const auto u = static_cast<std::size_t>(s);
                    const float bypassAmt = bypassSmoother_.getNextValue();
                    {
                        const float rawIn = bypassDryInL_[u];
                        const float dryDelayed = bypassDryL_.process(rawIn);
                        blendL_[u] = data0[offset + s] * (1.0f - bypassAmt)
                                   + dryDelayed * bypassAmt;
                    }
                    if (hasR)
                    {
                        const float rawIn = bypassDryInR_[u];
                        const float dryDelayed = bypassDryR_.process(rawIn);
                        blendR_[u] = data1[offset + s] * (1.0f - bypassAmt)
                                   + dryDelayed * bypassAmt;
                    }
                }

                // ── 9b. Gain + TPDF dither + quant (V2 SIMD) ──────────────
                // lsb is block-constant. Rounding: emulate
                // round-half-away-from-zero via trunc(x + copysign(0.5, x)).
                const M128 vLsb = MM(set1_ps)(blockLsb);
                const M128 vInvLsb = MM(set1_ps)(1.0f / blockLsb);
                const M128 vHalf = MM(set1_ps)(0.5f);
                const M128 vSignMask = MM(set1_ps)(-0.0f);
                const int jFull = chunk & ~3;
                for (int s = 0; s + 4 <= chunk; s += 4)
                {
                    const auto u = static_cast<std::size_t>(s);
                    // L
                    {
                        const M128 vBlend = MM(loadu_ps)(blendL_.data() + s);
                        const M128 vGain = MM(loadu_ps)(gainRamp_.data() + s);
                        const M128 vScaled = MM(mul_ps)(vBlend, vGain);
                        const M128 vD1 = nextUniformSimd_(xorshiftSimdL_);
                        const M128 vD2 = nextUniformSimd_(xorshiftSimdL_);
                        const M128 vDither = MM(mul_ps)(MM(sub_ps)(vD1, vD2), vLsb);
                        const M128 vQ = MM(mul_ps)(MM(add_ps)(vScaled, vDither), vInvLsb);
                        // round-half-away-from-zero: trunc(q + copysign(0.5, q))
                        const M128 vSign = MM(and_ps)(vQ, vSignMask);
                        const M128 vShifted = MM(add_ps)(vQ, MM(or_ps)(vHalf, vSign));
                        // trunc via cvtt+cvte (SSE2, no SSE4.1 needed)
                        const M128I vInt = MM(cvttps_epi32)(vShifted);
                        const M128 vRounded = MM(cvtepi32_ps)(vInt);
                        MM(storeu_ps)(data0 + offset + s, MM(mul_ps)(vRounded, vLsb));
                    }
                    if (hasR)
                    {
                        const M128 vBlend = MM(loadu_ps)(blendR_.data() + s);
                        const M128 vGain = MM(loadu_ps)(gainRamp_.data() + s);
                        const M128 vScaled = MM(mul_ps)(vBlend, vGain);
                        const M128 vD1 = nextUniformSimd_(xorshiftSimdR_);
                        const M128 vD2 = nextUniformSimd_(xorshiftSimdR_);
                        const M128 vDither = MM(mul_ps)(MM(sub_ps)(vD1, vD2), vLsb);
                        const M128 vQ = MM(mul_ps)(MM(add_ps)(vScaled, vDither), vInvLsb);
                        const M128 vSign = MM(and_ps)(vQ, vSignMask);
                        const M128 vShifted = MM(add_ps)(vQ, MM(or_ps)(vHalf, vSign));
                        const M128I vInt = MM(cvttps_epi32)(vShifted);
                        const M128 vRounded = MM(cvtepi32_ps)(vInt);
                        MM(storeu_ps)(data1 + offset + s, MM(mul_ps)(vRounded, vLsb));
                    }
                }
                // Scalar tail
                for (int s = jFull; s < chunk; ++s)
                {
                    const auto u = static_cast<std::size_t>(s);
                    const float gainLin = gainRamp_[u];
                    {
                        const float scaled = blendL_[u] * gainLin;
                        const float dither = (nextUniform(xorshiftL_) - nextUniform(xorshiftL_)) * blockLsb;
                        data0[offset + s] = std::round((scaled + dither) / blockLsb) * blockLsb;
                    }
                    if (hasR)
                    {
                        const float scaled = blendR_[u] * gainLin;
                        const float dither = (nextUniform(xorshiftR_) - nextUniform(xorshiftR_)) * blockLsb;
                        data1[offset + s] = std::round((scaled + dither) / blockLsb) * blockLsb;
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
            // V2: seed 4-lane SIMD xorshift. Each lane gets a different
            // seed derived from the base seed. Zero-state guard: if a
            // lane is zero, set it to 1.
            std::uint32_t seedsL[4] = { l, l * 2654435761u + 1u, l * 40503u + 2u, l * 2246822519u + 3u };
            std::uint32_t seedsR[4] = { r, r * 2654435761u + 1u, r * 40503u + 2u, r * 2246822519u + 3u };
            for (int i = 0; i < 4; ++i)
            {
                if (seedsL[i] == 0u) seedsL[i] = 1u;
                if (seedsR[i] == 0u) seedsR[i] = 1u;
            }
            alignas(16) std::uint32_t vL[4] = { seedsL[0], seedsL[1], seedsL[2], seedsL[3] };
            alignas(16) std::uint32_t vR[4] = { seedsR[0], seedsR[1], seedsR[2], seedsR[3] };
            xorshiftSimdL_ = MM(load_si128)(reinterpret_cast<const M128I*>(vL));
            xorshiftSimdR_ = MM(load_si128)(reinterpret_cast<const M128I*>(vR));
        }

        void setBypass(bool bypassed) noexcept
        {
            const float newTarget = bypassed ? 1.0f : 0.0f;
            if (newTarget != bypassTarget_)
            {
                bypassTarget_ = newTarget;
                bypassSmoother_.setTargetValue(newTarget);
            }
        }

    private:
        void smoothen_() noexcept
        {
            smoothedGain_ = gainSmoother_.getNextValue();
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

        // V2: 4-lane SIMD xorshift32. Each lane is independent.
        M128I xorshiftSimdL_{};
        M128I xorshiftSimdR_{};

        static M128 nextUniformSimd_(M128I& state) noexcept
        {
            // xorshift32 on all 4 lanes: x ^= x<<13; x ^= x>>17; x ^= x<<5
            state = MM(xor_si128)(state, MM(slli_epi32)(state, 13));
            state = MM(xor_si128)(state, MM(srli_epi32)(state, 17));
            state = MM(xor_si128)(state, MM(slli_epi32)(state, 5));
            // top 24 bits -> float in [0, 1)
            const M128I shifted = MM(srli_epi32)(state, 8);
            const M128 asFloat = MM(cvtepi32_ps)(shifted);
            return MM(mul_ps)(asFloat, MM(set1_ps)(1.0f / 16777216.0f));
        }

        Smoothers::LinearSmoother<float> gainSmoother_;
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

        Smoothers::LinearSmoother<float> bypassSmoother_;
        float bypassTarget_{0.0f};
        Align::ShortDelay<Align::SaturatorAlign::kBudget> bypassDryL_;
        Align::ShortDelay<Align::SaturatorAlign::kBudget> bypassDryR_;

        // preallocated scratch (sized to wetBufCapacity_ in prepare)
        std::vector<float> driveRamp_;
        std::vector<float> hpfRamp_;
        std::vector<float> lpfRamp_;
        std::vector<float> thetaRamp_;
        std::vector<float> gainRamp_;
        std::vector<float> lsbRamp_;
        std::vector<float> satL_;
        std::vector<float> satR_;
        std::vector<float> alignedDryL_;
        std::vector<float> alignedDryR_;
        std::vector<float> wetPostSvfL_;
        std::vector<float> wetPostSvfR_;
        std::vector<float> bypassDryInL_;
        std::vector<float> bypassDryInR_;
        std::vector<float> blendL_;
        std::vector<float> blendR_;

        double sampleRate_{0.0};
        int numChannels_{0};
        float delaySamples_{0.0f};
        int adaaOrder_{2};
        Delays::Interpolation interp_{Delays::Interpolation::Lagrange5th};
    };
}
#endif

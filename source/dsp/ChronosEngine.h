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
#include "../math/Trigonometry.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace MarsDSP {
    class ChronosEngine {
    public:
        struct Params {
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

        // ── Lifecycle ────────────────────────────────────────────────────────
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

            gainSmoother_.setCurrentAndTargetValue(0.0f);
            bitsSmoother_.setCurrentAndTargetValue(0.0f);
            mixSmoother_.setCurrentAndTargetValue(0.0f);
            driveSmoother_.setCurrentAndTargetValue(0.0f);
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

        // ── Process (S2 stub — not implemented yet) ──────────────────────────

        void process(float *const* /*io*/, int /*numChannels*/, int /*numSamples*/) noexcept
        {
            assert(!"ChronosEngine::process not implemented (skeleton)");
        }

        // ── Queries ──────────────────────────────────────────────────────────
        [[nodiscard]] static constexpr int latencySamples() noexcept
        {
            return Align::SaturatorAlign::kBudget;
        }

        [[nodiscard]] int getWetBufCapacity() const noexcept { return wetBufCapacity_; }

        // Dither seed control (for testability. chain_parity needs
        // deterministic dither to compare engine vs reference bit-exactly)
        void setDitherSeeds(std::uint32_t l, std::uint32_t r) noexcept
        {
            xorshiftL_ = l;
            xorshiftR_ = r;
        }

    private:
        // ── DSP members ────────────────────────
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

        // ── Dither RNG ─────────────────────────
        std::uint32_t xorshiftL_{0x12345678u};
        std::uint32_t xorshiftR_{0x9abcdef0u};

        static float nextUniform(std::uint32_t &state) noexcept
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
        }

        // ── Per-sample smoothers ───────────────────
        Smoothers::LinearSmoother<float> gainSmoother_;
        Smoothers::LinearSmoother<float> bitsSmoother_;
        Smoothers::LinearSmoother<float> hpfSmoother_;
        Smoothers::LinearSmoother<float> lpfSmoother_;
        Smoothers::LinearSmoother<float> mixSmoother_;
        Smoothers::LinearSmoother<float> driveSmoother_;

        // ── Prepared state ───────────────────────────────────────────────────
        double sampleRate_{0.0};
        int numChannels_{0};
        float delaySamples_{0.0f}; // block-rate, unsmoothed
        int adaaOrder_{2}; // default ADAA2
        Delays::Interpolation interp_{Delays::Interpolation::Lagrange5th};
    };
}
#endif

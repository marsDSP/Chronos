#pragma once

#ifndef CHRONOS_MODULATION_H
#define CHRONOS_MODULATION_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>

// Random modulation sources.
// Pcg32 is a PCG-XSH-RR generator with per-instance state.
// OrnsteinUhlenbeck is an OU process with the exact discrete step.
// The Gaussian step uses a sum of four uniforms. It needs no branch.

namespace MarsDSP::Mod {

    class Pcg32 {
    public:
        // Seed from one constant and a stream index. The stream index
        // decorrelates two generators that share the constant.
        void seed(std::uint64_t seedValue, std::uint64_t stream) noexcept
        {
            state_ = 0;
            inc_ = (stream << 1u) | 1u;
            next();
            state_ += seedValue;
            next();
        }

        std::uint32_t next() noexcept
        {
            const std::uint64_t old = state_;
            state_ = old * 6364136223846793005uLL + inc_;
            const auto xorshifted = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
            const auto rot = static_cast<std::uint32_t>(old >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
        }

        // Return a uniform float in [0, 1).
        float nextUniform() noexcept
        {
            return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f);
        }

    private:
        std::uint64_t state_ = 0;
        std::uint64_t inc_ = 1;
    };

    class OrnsteinUhlenbeck {
    public:
        static constexpr double kClamp = 4.0; // state bound in sigmas

        void setRate(double sampleRate, double rateHz) noexcept
        {
            a_ = std::exp(-2.0 * std::numbers::pi * rateHz / sampleRate);
            s_ = std::sqrt(1.0 - a_ * a_);
        }

        void reset() noexcept { x_ = 0.0; }

        [[nodiscard]] double state() const noexcept { return x_; }

        // Advance the process once per sample. The state stays inside
        // kClamp sigmas so the delay read guard has a finite bound.
        float next(Pcg32& rng) noexcept
        {
            const float g = (rng.nextUniform() + rng.nextUniform()
                             + rng.nextUniform() + rng.nextUniform() - 2.0f)
                            * kGaussNorm;
            x_ = std::clamp(a_ * x_ + s_ * static_cast<double>(g), -kClamp, kClamp);
            return static_cast<float>(x_);
        }

        // Return the RMS of the increment that a windowed average shows,
        // per sample. The window length is in samples. The OU increment
        // depends on the window length because the process is not smooth.
        [[nodiscard]] double windowedIncrementRms(double windowSamples) const noexcept
        {
            return std::sqrt(2.0 * (1.0 - std::pow(a_, windowSamples))) / windowSamples;
        }

    private:
        static constexpr float kGaussNorm = 1.7320508075688772f; // sqrt(3)

        double a_ = 1.0;
        double s_ = 0.0;
        double x_ = 0.0;
    };
}
#endif

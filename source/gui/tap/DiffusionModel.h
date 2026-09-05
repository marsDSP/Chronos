#pragma once

#ifndef CHRONOS_DIFFUSION_MODEL_H
#define CHRONOS_DIFFUSION_MODEL_H

#include "../../dsp/Diffuser.h"

#include <array>
#include <cmath>

namespace MarsDSP::GUI {

// A model of the shipped Diffuser impulse response, for the halo
// and the pad cloud. Header-only, JUCE-free. It calls the public
// statics of Diffusion::Diffuser and caches the 18 section lengths.
class DiffusionModel {
public:
    explicit DiffusionModel(double sampleRate)
    {
        Diffusion::Diffuser::computeSectionLens(sampleRate, lenL_.data(), lenR_.data());
        sampleRate_ = sampleRate;
        // Cache the reference spread at full diffusion and full size.
        sigmaRef_ = sigma1Unchecked(1.0f, 1.0f);
    }

    // One-pass energy spread in seconds. Zero at diffusion 0.
    [[nodiscard]] float sigma1Seconds(float diffusion01, float size01) const noexcept
    {
        if (diffusion01 <= 0.0f)
            return 0.0f;
        return sigma1Unchecked(diffusion01, size01);
    }

    // One-pass energy centroid in seconds. Computed from the cached
    // section lengths, the same path as Diffuser::baseTransportSamples.
    [[nodiscard]] float transportSeconds(float size01) const noexcept
    {
        const float s = std::clamp(size01, 0.0f, 1.0f);
        auto sumBank = [&](const std::array<int, Diffusion::Diffuser::kNumDelaysPerBank>& len) noexcept
        {
            float sum = 0.0f;
            for (int i = 0; i < Diffusion::Diffuser::kNumPlainSections; ++i)
            {
                const auto lenF = static_cast<float>(len[static_cast<std::size_t>(i)]);
                float eff = Diffusion::Diffuser::effLen(lenF, s);
                eff = std::nearbyintf(eff);
                eff = std::clamp(eff, Diffusion::Diffuser::kMinDelay, lenF);
                sum += eff;
            }
            for (int i = 0; i < Diffusion::Diffuser::kNumNestedSections; ++i)
            {
                const auto lenOutF = static_cast<float>(len[static_cast<std::size_t>(3 + 2 * i)]);
                const auto lenInF  = static_cast<float>(len[static_cast<std::size_t>(4 + 2 * i)]);
                float effOut = Diffusion::Diffuser::effLen(lenOutF, s);
                effOut = std::nearbyintf(effOut);
                effOut = std::clamp(effOut, Diffusion::Diffuser::kMinDelay, lenOutF);
                float effIn = Diffusion::Diffuser::effLen(lenInF, s);
                effIn = std::nearbyintf(effIn);
                effIn = std::clamp(effIn, Diffusion::Diffuser::kMinDelay, lenInF);
                sum += (effOut + effIn);
            }
            return sum;
        };
        const float meanSamples = 0.5f * (sumBank(lenL_) + sumBank(lenR_));
        return meanSamples / static_cast<float>(sampleRate_);
    }

    // The reference spread at full diffusion and full size.
    [[nodiscard]] float sigmaRef() const noexcept { return sigmaRef_; }

private:
    // Sum the section variances. A first-order allpass with delay D
    // and coefficient g has the energy-time variance 2 g^2 D^2 / (1 - g^2)
    // about its centroid D. The cascade spread is the root of the
    // summed section variances. Each nested pair is one section of
    // delay effLen(outer) + effLen(inner) and coefficient
    // kSectionGain[3+i] * kMaxCoefficient * diffusion.
    [[nodiscard]] float sigma1Unchecked(float diffusion01, float size01) const noexcept
    {
        const float g = Diffusion::Diffuser::kMaxCoefficient * std::clamp(diffusion01, 0.0f, 1.0f);
        const float g2 = g * g;
        const float denom = std::max(1.0f - g2, 1e-9f);

        auto sectionVariance = [&](float lenF) -> float
        {
            const float eff = Diffusion::Diffuser::effLen(lenF, size01);
            const float d = std::clamp(std::nearbyintf(eff),
                                       Diffusion::Diffuser::kMinDelay, lenF);
            return 2.0f * g2 * d * d / denom;
        };

        float sum = 0.0f;
        // Plain sections 0..2.
        for (int i = 0; i < Diffusion::Diffuser::kNumPlainSections; ++i)
        {
            const auto lenL = static_cast<float>(lenL_[static_cast<std::size_t>(i)]);
            const auto lenR = static_cast<float>(lenR_[static_cast<std::size_t>(i)]);
            sum += sectionVariance(lenL) + sectionVariance(lenR);
        }
        // Nested pairs 3..5.
        for (int i = 0; i < Diffusion::Diffuser::kNumNestedSections; ++i)
        {
            const auto lenOutL = static_cast<float>(lenL_[static_cast<std::size_t>(3 + 2 * i)]);
            const auto lenInL  = static_cast<float>(lenL_[static_cast<std::size_t>(4 + 2 * i)]);
            const auto lenOutR = static_cast<float>(lenR_[static_cast<std::size_t>(3 + 2 * i)]);
            const auto lenInR  = static_cast<float>(lenR_[static_cast<std::size_t>(4 + 2 * i)]);
            const float gPair = Diffusion::Diffuser::kSectionGain[static_cast<std::size_t>(3 + i)] * g;
            const float gPair2 = gPair * gPair;
            const float denomPair = std::max(1.0f - gPair2, 1e-9f);
            auto pairVar = [&](float outF, float inF) -> float
            {
                const float dOut = std::clamp(std::nearbyintf(Diffusion::Diffuser::effLen(outF, size01)),
                                             Diffusion::Diffuser::kMinDelay, outF);
                const float dIn  = std::clamp(std::nearbyintf(Diffusion::Diffuser::effLen(inF, size01)),
                                             Diffusion::Diffuser::kMinDelay, inF);
                const float d = dOut + dIn;
                return 2.0f * gPair2 * d * d / denomPair;
            };
            sum += pairVar(lenOutL, lenInL) + pairVar(lenOutR, lenInR);
        }

        // The spread is the root of the summed variances, divided by the
        // sample rate, then the mean of the two banks.
        const float sigmaL = std::sqrt(sum) / static_cast<float>(sampleRate_);
        return sigmaL;
    }

    double sampleRate_ = 48000.0;
    std::array<int, Diffusion::Diffuser::kNumDelaysPerBank> lenL_ {};
    std::array<int, Diffusion::Diffuser::kNumDelaysPerBank> lenR_ {};
    float sigmaRef_ = 1.0f;
};

} // namespace MarsDSP::GUI

#endif

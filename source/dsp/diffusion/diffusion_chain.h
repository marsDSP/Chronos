#pragma once

#ifndef CHRONOS_DIFFUSION_CHAIN_H
#define CHRONOS_DIFFUSION_CHAIN_H

// ============================================================================
//  diffusion_chain.h
// ----------------------------------------------------------------------------
//  Cascade of K DiffusionStage instances. Each stage gets a different random
//  seed (derived from a base seed + stage index) and a per-stage diffusion
//  window multiplier so the cascade produces exponentially denser reflections
//  without wrap-around artefacts.
//
//  The default window-multiplier schedule doubles the window on each stage
//  (0.5, 1.0, 2.0, 4.0, ...), which is the classic Dattorro / Signalsmith
//  setup: successive stages do coarser-and-coarser mixing.
// ============================================================================

#include <array>

#include "diffusion_stage.h"

namespace MarsDSP::DSP::Diffusion
{
    // ----------------------------------------------------------------------------
    //  DiffusionChain<SampleType, ChannelCount, StageCount, MaxDelaySamplesPerChannel>
    // ----------------------------------------------------------------------------
    template <typename SampleType, std::size_t ChannelCount, std::size_t StageCount, std::size_t MaxDelaySamplesPerChannel>
    class DiffusionChain
    {
    public:
        using StageType = DiffusionStage<SampleType, ChannelCount, MaxDelaySamplesPerChannel>;

        DiffusionChain() = default;

        // ------------------------------------------------------------------
        //  Seed every stage's coefficients. Pass a single base seed; each
        //  stage gets its own distinct seed derived from the base + its
        //  index so the stages decorrelate from each other.
        // ------------------------------------------------------------------
        void prepare(double        hostSampleRateInHz,
                     std::uint32_t baseRandomSeed = 0xA5B7C9D2u) noexcept
        {
            for (std::size_t stageIndex = 0;
                 stageIndex < StageCount;
                 ++stageIndex)
            {
                // Mix the base seed with the stage index so each stage
                // samples a different coefficient pattern without needing
                // a true RNG hierarchy.
                const std::uint32_t perStageSeed = baseRandomSeed ^ (static_cast<std::uint32_t>(stageIndex + 1) * 0x9E3779B9u);

                stages[stageIndex].prepare(hostSampleRateInHz, perStageSeed);

                // Default window-multiplier schedule: geometric doubling
                // around a normalised centre.
                perStageWindowMultipliers[stageIndex] = static_cast<SampleType>(std::pow(2.0, static_cast<double>(stageIndex)
                                                            - static_cast<double>(StageCount) * 0.5
                                                            - 0.5));
            }
        }

        // Flush every stage's delay state; doesn't touch seeded coefficients.
        void reset() noexcept
        {
            for (auto& stage : stages)
                stage.reset();
        }

        // ------------------------------------------------------------------
        //  Set the overall diffusion window. Each stage's per-stage
        //  multiplier is applied to this value.
        // ------------------------------------------------------------------
        void setDiffusionWindowInMilliseconds(SampleType diffusionWindowMilliseconds) noexcept
        {
            for (std::size_t stageIndex = 0;
                 stageIndex < StageCount;
                 ++stageIndex)
            {
                stages[stageIndex].setDiffusionWindowInMilliseconds(diffusionWindowMilliseconds * perStageWindowMultipliers[stageIndex]);
            }
        }

        // Process a single N-channel vector through every stage in order.
        void processSingleVectorInPlace(SampleType* channelVector) noexcept
        {
            for (auto& stage : stages)
                stage.processSingleVectorInPlace(channelVector);
        }

        [[nodiscard]] constexpr std::size_t getStageCount() const noexcept
        {
            return StageCount;
        }

    private:
        std::array<StageType, StageCount>   stages{};
        std::array<SampleType, StageCount>  perStageWindowMultipliers{};
    };
}
#endif

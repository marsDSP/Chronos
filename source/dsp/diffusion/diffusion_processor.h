#pragma once

#ifndef CHRONOS_DIFFUSION_PROCESSOR_H
#define CHRONOS_DIFFUSION_PROCESSOR_H

// ============================================================================
//  diffusion_processor.h
// ----------------------------------------------------------------------------
//  Stereo-in, stereo-out wrapper around DiffusionChain. Responsible for:
//    - splitting each stereo input pair into an N-channel diffusion vector
//      through StereoMultiChannelBridge,
//    - driving the DiffusionChain with that vector,
//    - collapsing the chain's output back to stereo,
//    - dry/wet mixing the result against the raw input according to the
//      user-facing "diffusion amount" parameter.
//
//  Designed to live inside the delay engine's feedback path so the amount
//  parameter can be smoothed per-block by the engine's existing lipol.
// ============================================================================

#include <algorithm>
#include <type_traits>

#include "diffusion_chain.h"
#include "stereo_multichannel_bridge.h"

namespace MarsDSP::DSP::Diffusion
{
    // ----------------------------------------------------------------------------
    //  Configuration defaults. Kept as free constants so tests can
    //  instantiate the processor with the same sizes Chronos uses.
    // ----------------------------------------------------------------------------
    inline constexpr std::size_t kDefaultDiffusionChannelCount             = 8;
    inline constexpr std::size_t kDefaultDiffusionStageCount               = 4;

    // 8192 samples at 96 kHz = ~85 ms of slack per channel per stage; plenty
    // for the sub-200 ms diffusion windows we allow through the parameter.
    inline constexpr std::size_t kDefaultMaxDiffusionDelaySamplesPerChannel = 1u << 13;

    // ----------------------------------------------------------------------------
    //  DiffusionProcessor
    //
    //  Default instantiation matches Chronos's expected topology: float
    //  samples, 8 internal channels, 4 diffusion stages, 8192 samples per
    //  channel per stage. Any of those can be changed by supplying explicit
    //  template args if the delay engine grows.
    // ----------------------------------------------------------------------------
    template <typename SampleType = float,
              std::size_t ChannelCount         = kDefaultDiffusionChannelCount,
              std::size_t StageCount           = kDefaultDiffusionStageCount,
              std::size_t MaxDelaySamplesPerCh = kDefaultMaxDiffusionDelaySamplesPerChannel>
    class DiffusionProcessor
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "DiffusionProcessor requires a floating-point sample type.");

        using DiffusionChainType = DiffusionChain<SampleType, ChannelCount, StageCount, MaxDelaySamplesPerCh>;
        using StereoBridgeType = StereoMultiChannelBridge<SampleType, ChannelCount>;

        DiffusionProcessor() = default;

        void prepare(double        hostSampleRateInHz,
                     std::uint32_t baseRandomSeed = 0xA5B7C9D2u) noexcept
        {
            diffusionChain.prepare(hostSampleRateInHz, baseRandomSeed);
            applyCurrentDiffusionWindowToChain();
        }

        void reset() noexcept
        {
            diffusionChain.reset();
        }

        // User-facing "amount" control: 0 = fully bypassed (returns dry
        // input unchanged), 1 = fully wet (returns diffusion output only).
        // Intermediate values linearly cross-fade between dry and wet.
        void setDiffusionAmount(SampleType normalisedAmount) noexcept
        {
            currentDiffusionAmount = std::clamp(normalisedAmount,
                                                static_cast<SampleType>(0),
                                                static_cast<SampleType>(1));
        }

        // User-facing "size" control: the overall diffusion window in
        // milliseconds. Each stage's window multiplier is applied to this
        // value by DiffusionChain internally.
        void setDiffusionSizeInMilliseconds(SampleType diffusionSizeMilliseconds) noexcept
        {
            currentDiffusionSizeMilliseconds = std::max(diffusionSizeMilliseconds, static_cast<SampleType>(0.1));
            applyCurrentDiffusionWindowToChain();
        }

        [[nodiscard]] SampleType getDiffusionAmount() const noexcept
        {
            return currentDiffusionAmount;
        }

        [[nodiscard]] SampleType getDiffusionSizeInMilliseconds() const noexcept
        {
            return currentDiffusionSizeMilliseconds;
        }

        // Process one stereo sample pair in place. When diffusionAmount
        // is 0, this is branch-free pass-through. When it's above 0, the
        // input is split into N channels, diffused, collapsed back, and
        // mixed against the dry input per the amount.
        void processSingleStereoPairInPlace(SampleType& leftSampleInOut, SampleType& rightSampleInOut) noexcept
        {
            if (currentDiffusionAmount <= static_cast<SampleType>(0))
                return;

            const SampleType dryLeftInputSample  = leftSampleInOut;
            const SampleType dryRightInputSample = rightSampleInOut;

            SampleType multiChannelVector[ChannelCount];
            stereoBridge.splitStereoPairIntoMultiChannelVector(dryLeftInputSample,
                                                               dryRightInputSample,
                                                               multiChannelVector);

            diffusionChain.processSingleVectorInPlace(multiChannelVector);

            SampleType wetLeftOutputSample  = SampleType{};
            SampleType wetRightOutputSample = SampleType{};

            stereoBridge.collapseMultiChannelVectorToStereoPair(multiChannelVector,
                                                                wetLeftOutputSample,
                                                                wetRightOutputSample);

            const SampleType drySideAmount = static_cast<SampleType>(1) - currentDiffusionAmount;

            leftSampleInOut  = dryLeftInputSample  * drySideAmount + wetLeftOutputSample  * currentDiffusionAmount;
            rightSampleInOut = dryRightInputSample * drySideAmount + wetRightOutputSample * currentDiffusionAmount;
        }

    private:
        void applyCurrentDiffusionWindowToChain() noexcept
        {
            diffusionChain.setDiffusionWindowInMilliseconds(currentDiffusionSizeMilliseconds);
        }

        DiffusionChainType diffusionChain{};
        StereoBridgeType   stereoBridge{};

        SampleType currentDiffusionAmount           { static_cast<SampleType>(0)  };
        SampleType currentDiffusionSizeMilliseconds { static_cast<SampleType>(50) };
    };
}
#endif

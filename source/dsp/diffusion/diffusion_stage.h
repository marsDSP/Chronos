#pragma once

#ifndef CHRONOS_DIFFUSION_STAGE_H
#define CHRONOS_DIFFUSION_STAGE_H

// ============================================================================
//  diffusion_stage.h
// ----------------------------------------------------------------------------
//  One stage of a Dattorro / Signalsmith-style diffusion chain. Per step:
//
//     1) Write the incoming N-channel vector into the stage's delay bank.
//     2) Read each output channel from a different delay line, at a
//        channel-specific random delay length (evenly spread across the
//        available diffusion window).
//     3) Apply a Hadamard mixing matrix to the N samples (all-to-all mix).
//     4) Multiply each channel by a fixed +/-1 polarity flip.
//
//  The random delays + channel shuffle + polarity pattern are seeded once
//  at prepare() time so the stage is stable and reproducible across runs.
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <random>
#include <type_traits>

#include "multichannel_static_delay_bank.h"
#include "orthogonal_mixing_matrices.h"

namespace MarsDSP::DSP::Diffusion
{
    // ----------------------------------------------------------------------------
    //  DiffusionStage<SampleType, ChannelCount, MaxDelaySamplesPerChannel>
    //
    //  ChannelCount must be a power of 2 (Hadamard requirement).
    //  MaxDelaySamplesPerChannel must also be a power of 2 (ring bank
    //  requirement) and should be >= the maximum diffusion time (in samples)
    //  the caller ever wants to set.
    // ----------------------------------------------------------------------------
    template <typename SampleType, std::size_t ChannelCount, std::size_t MaxDelaySamplesPerChannel>
    class DiffusionStage
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "DiffusionStage requires a floating-point sample type.");
        static_assert(kIsPowerOfTwo<ChannelCount>,
                      "DiffusionStage requires ChannelCount to be a power of 2.");

        using DelayBankType = MultiChannelStaticDelayBank<SampleType, ChannelCount, MaxDelaySamplesPerChannel>;
        using HadamardMixerType = OrthogonalHadamardMixer<SampleType, ChannelCount>;

        DiffusionStage() = default;

        // ------------------------------------------------------------------
        //  Seed the stage's random delay multipliers, polarity flips and
        //  channel-swap permutation, all reproducibly from
        //  'stageRandomSeed'. Different stages in a chain should pass
        //  different seeds (or have their seed derived from the stage index
        //  by the parent chain) so the diffusion pattern decorrelates.
        // ------------------------------------------------------------------
        void prepare(double       hostSampleRateInHz,
                     std::uint32_t stageRandomSeed) noexcept
        {
            sampleRateOverOneThousand = static_cast<SampleType>(hostSampleRateInHz / 1000.0);

            std::mt19937 deterministicRandomEngine(stageRandomSeed);

            // Channel swap permutation: a random perm of [0..N-1], so each
            // output channel reads from a different delay line than it
            // wrote to.
            std::iota(perOutputChannelReadSource.begin(),
                      perOutputChannelReadSource.end(),
                      std::size_t{0});

            std::shuffle(perOutputChannelReadSource.begin(),
                         perOutputChannelReadSource.end(),
                         deterministicRandomEngine);

            // Per-channel delay multipliers: each in an equally-spaced
            // sub-range [k/(N+1), (k+1)/(N+1)] so delays are spread out
            // rather than clustering together.
            std::uniform_real_distribution uniformZeroOne(0.0, 1.0);
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                const double subRangeLow = static_cast<double>(channelIndex + 1) / static_cast<double>(ChannelCount + 1);
                const double subRangeHigh = static_cast<double>(channelIndex + 2) / static_cast<double>(ChannelCount + 1);

                const double drawnMultiplier = subRangeLow + (subRangeHigh - subRangeLow) * uniformZeroOne(deterministicRandomEngine);

                perChannelDelayFractionOfWindow[channelIndex] = static_cast<SampleType>(drawnMultiplier);
            }

            // Polarity flips: independent fair coin per channel.
            std::uniform_int_distribution coinFlip(0, 1);
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                perChannelPolarityFlip[channelIndex] = (coinFlip(deterministicRandomEngine) == 0)
                                                            ? static_cast<SampleType>(1)
                                                            : static_cast<SampleType>(-1);
            }

            delayBank.reset();
            setDiffusionWindowInMilliseconds(static_cast<SampleType>(50.0));
        }

        // Flush stored samples; doesn't touch the seeded coefficients.
        void reset() noexcept
        {
            delayBank.reset();
        }

        // ------------------------------------------------------------------
        //  Set the diffusion window (the longest delay any channel can
        //  take). Each channel ends up with its own delay that is
        //  perChannelDelayFractionOfWindow[i] * diffusionWindow; the
        //  multipliers are in [1/(N+1), 1] so this is an approximate
        //  upper bound, not exact.
        // ------------------------------------------------------------------
        void setDiffusionWindowInMilliseconds(SampleType diffusionWindowMilliseconds) noexcept
        {
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                const SampleType delayInSamplesFloat = perChannelDelayFractionOfWindow[channelIndex]
                                                        * diffusionWindowMilliseconds
                                                        * sampleRateOverOneThousand;

                auto delayInSamples = static_cast<std::size_t>(delayInSamplesFloat);

                // Clamp into the legal [1, MaxDelaySamplesPerChannel-1]
                // range. The >=1 bound is required by the delay bank; the
                // upper bound prevents wrap-around aliasing.
                if (delayInSamples < 1)
                    delayInSamples = 1;

                if (delayInSamples >= DelayBankType::kMaxSamplesPerLine)
                    delayInSamples = DelayBankType::kMaxSamplesPerLine - 1;

                perChannelReadDelayInSamples[channelIndex] = delayInSamples;
            }
        }

        // ------------------------------------------------------------------
        //  Process one N-channel vector in place. 'channelVector' must
        //  point to exactly ChannelCount floats; they are both input and
        //  output for this sample step.
        // ------------------------------------------------------------------
        void processSingleVectorInPlace(SampleType* channelVector) noexcept
        {
            // Step 1: write input into each channel's ring buffer.
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                delayBank.writeSampleToChannel(channelIndex, channelVector[channelIndex]);
            }
            delayBank.advanceSharedWriteIndex();

            // Step 2: read output from the shuffled source channels at
            // per-channel delays. Write back into channelVector so downstream
            // steps see the delayed + shuffled vector.
            std::array<SampleType, ChannelCount> shuffledReadBuffer{};
            for (std::size_t outputChannelIndex = 0;
                 outputChannelIndex < ChannelCount;
                 ++outputChannelIndex)
            {
                const std::size_t readSourceChannel    = perOutputChannelReadSource[outputChannelIndex];
                const std::size_t readDelayInSamples   = perChannelReadDelayInSamples[outputChannelIndex];
                shuffledReadBuffer[outputChannelIndex] = delayBank.readDelayedSampleFromChannel(readSourceChannel,
                                                                                                readDelayInSamples);
            }

            // Step 3: Hadamard mix (all channels to all channels).
            HadamardMixerType::applyInPlace(shuffledReadBuffer.data());

            // Step 4: polarity flip.
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                channelVector[channelIndex] = shuffledReadBuffer[channelIndex] * perChannelPolarityFlip[channelIndex];
            }
        }

    private:
        DelayBankType delayBank{};

        // Random / deterministic diffusion tables. All filled during prepare().
        std::array<std::size_t, ChannelCount> perOutputChannelReadSource{};
        std::array<SampleType,  ChannelCount> perChannelDelayFractionOfWindow{};
        std::array<std::size_t, ChannelCount> perChannelReadDelayInSamples{};
        std::array<SampleType,  ChannelCount> perChannelPolarityFlip{};

        SampleType sampleRateOverOneThousand { static_cast<SampleType>(48.0) };
    };
}
#endif

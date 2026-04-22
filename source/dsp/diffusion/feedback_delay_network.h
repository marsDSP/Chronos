#pragma once

#ifndef CHRONOS_FEEDBACK_DELAY_NETWORK_H
#define CHRONOS_FEEDBACK_DELAY_NETWORK_H

// ============================================================================
//  feedback_delay_network.h
// ----------------------------------------------------------------------------
//  Lossless-Householder-mixed Feedback Delay Network. Each channel has:
//    - its own independent delay line (shared bank),
//    - a first-order shelf filter in its feedback path for low/high-
//      frequency T60 control,
//    - a per-channel random delay multiplier (seeded at prepare() time)
//      so delays are spread across an equally-spaced range.
//
//  Per-sample step:
//      1) Read a delayed sample from each channel's ring.
//      2) Apply the Householder mixing matrix to those samples.
//      3) Run each mixed channel through its shelf filter.
//      4) Write (input[i] + shelfOut[i]) back into delay[i].
// ============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>

#include "feedback_shelf_filter.h"
#include "multichannel_static_delay_bank.h"
#include "orthogonal_mixing_matrices.h"

namespace MarsDSP::DSP::Diffusion
{
    // ----------------------------------------------------------------------------
    //  FeedbackDelayNetwork<SampleType, ChannelCount, MaxDelaySamplesPerCh>
    // ----------------------------------------------------------------------------
    template <typename SampleType, std::size_t ChannelCount, std::size_t MaxDelaySamplesPerChannel>
    class FeedbackDelayNetwork
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "FeedbackDelayNetwork requires a floating-point sample type.");
        static_assert(ChannelCount >= 2,
                      "FeedbackDelayNetwork requires ChannelCount >= 2.");

        using DelayBankType      = MultiChannelStaticDelayBank<SampleType, ChannelCount, MaxDelaySamplesPerChannel>;
        using HouseholderMixer   = OrthogonalHouseholderMixer<SampleType, ChannelCount>;
        using ChannelShelfFilter = FeedbackShelfFilter<SampleType>;

        FeedbackDelayNetwork() = default;

        // ------------------------------------------------------------------
        //  Seed per-channel delay multipliers and reset state.
        // ------------------------------------------------------------------
        void prepare(double        hostSampleRateInHz,
                     std::uint32_t fdnRandomSeed = 0xF1CEE3B7u) noexcept
        {
            hostSampleRate = hostSampleRateInHz;
            sampleRateOverOneThousand = static_cast<SampleType>(hostSampleRateInHz / 1000.0);

            std::mt19937 deterministicRandomEngine(fdnRandomSeed);

            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                const double subRangeLow  = static_cast<double>(channelIndex + 1) / static_cast<double>(ChannelCount + 1);
                const double subRangeHigh = static_cast<double>(channelIndex + 2) / static_cast<double>(ChannelCount + 1);

                std::uniform_real_distribution dist(subRangeLow, subRangeHigh);
                perChannelDelayFractionOfWindow[channelIndex] = static_cast<SampleType>(dist(deterministicRandomEngine));
            }

            delayBank.reset();
            for (auto& shelf : perChannelShelfFilters)
                shelf.reset();

            setDelayTimeInMilliseconds(static_cast<SampleType>(80));
            setDecayTimeInMilliseconds(static_cast<SampleType>(1500),
                                       static_cast<SampleType>(800),
                                       static_cast<SampleType>(2500));
        }

        void reset() noexcept
        {
            delayBank.reset();
            for (auto& shelf : perChannelShelfFilters)
                shelf.reset();
        }

        // Set the longest delay time in milliseconds. Each channel ends up
        // with its own shorter delay derived from the random multiplier.
        void setDelayTimeInMilliseconds(
            SampleType longestDelayInMilliseconds) noexcept
        {
            currentLongestDelayMilliseconds = std::max(longestDelayInMilliseconds, static_cast<SampleType>(1));

            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                const SampleType delayInSamplesFloat = perChannelDelayFractionOfWindow[channelIndex]
                                                        * currentLongestDelayMilliseconds
                                                        * sampleRateOverOneThousand;

                auto delayInSamples = static_cast<std::size_t>(delayInSamplesFloat);

                if (delayInSamples < 1)
                    delayInSamples = 1;

                if (delayInSamples >= DelayBankType::kMaxSamplesPerLine)
                    delayInSamples = DelayBankType::kMaxSamplesPerLine - 1;

                perChannelReadDelayInSamples[channelIndex] = delayInSamples;
                perChannelDelayInMilliseconds[channelIndex] = static_cast<SampleType>(delayInSamples) / sampleRateOverOneThousand;
            }
        }

        // Compute and apply shelf coefficients so the low- and high-
        // frequency RT60 match the requested values.
        void setDecayTimeInMilliseconds(
            SampleType lowFrequencyT60InMilliseconds,
            SampleType highFrequencyT60InMilliseconds,
            SampleType crossoverFrequencyHz) noexcept
        {
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                const SampleType channelDelayMs = perChannelDelayInMilliseconds[channelIndex];

                const SampleType lowGain = calculateGainForT60(lowFrequencyT60InMilliseconds, channelDelayMs);
                const SampleType highGain = calculateGainForT60(highFrequencyT60InMilliseconds, channelDelayMs);

                perChannelShelfFilters[channelIndex].computeCoefficientsForTargetGains(lowGain,
                                                                                       highGain,
                                                                                       crossoverFrequencyHz,
                                                                                       hostSampleRate);
            }
        }

        // Per-sample process. 'channelInputVector' and 'channelOutputVector'
        // must point to ChannelCount floats. Output is the pre-mix delayed
        // sample (the signal the caller wants to send out of the reverb);
        // feedback (including input injection) is written back to the delay
        // bank internally.
        void processSingleVector(const SampleType* channelInputVector,
                                 SampleType*       channelOutputVector) noexcept
        {
            // 1) Read delayed samples from each channel.
            std::array<SampleType, ChannelCount> delayedChannelReads{};
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                delayedChannelReads[channelIndex] = delayBank.readDelayedSampleFromChannel(channelIndex,
                                                            perChannelReadDelayInSamples[channelIndex]);
            }

            // 2) Apply Householder mix (lossless, all-to-all reflection).
            HouseholderMixer::applyInPlace(delayedChannelReads.data());

            // 3) Per-channel shelf in the feedback path, scaled so T60 hits.
            std::array<SampleType, ChannelCount> shelfFilteredFeedback{};
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                shelfFilteredFeedback[channelIndex] = perChannelShelfFilters[channelIndex]
                                                        .processSingleSample(delayedChannelReads[channelIndex]);
            }

            // 4) Write (input + feedback) back to the delay bank.
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                const SampleType writeBackSample = channelInputVector[channelIndex] + shelfFilteredFeedback[channelIndex];
                delayBank.writeSampleToChannel(channelIndex, writeBackSample);
            }
            delayBank.advanceSharedWriteIndex();

            // Output the mixed delayed reads (pre-shelf) so the caller gets
            // the full reverberant field without the damping already folded
            // in a second time at the output. This matches LEX480's
            // "return outData.data()" after mixing matrix.
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                channelOutputVector[channelIndex] = delayedChannelReads[channelIndex];
            }
        }

        [[nodiscard]] SampleType getChannelDelayInMilliseconds(
            std::size_t channelIndex) const noexcept
        {
            return perChannelDelayInMilliseconds[channelIndex];
        }

    private:
        // Analytic gain-per-bounce needed for an N-bounce T60.
        //   |g|^n = 10^(-60/20) = 0.001 when n = T60 / delayTime
        static SampleType calculateGainForT60(SampleType t60Milliseconds,
                                              SampleType delayMilliseconds) noexcept
        {
            if (delayMilliseconds <= static_cast<SampleType>(0))
                return static_cast<SampleType>(0);

            const SampleType numberOfBounces = t60Milliseconds / delayMilliseconds;

            if (numberOfBounces <= static_cast<SampleType>(0))
                return static_cast<SampleType>(0);

            return static_cast<SampleType>(std::pow(0.001, 1.0 / static_cast<double>(numberOfBounces)));
        }

        DelayBankType delayBank{};
        std::array<ChannelShelfFilter, ChannelCount> perChannelShelfFilters{};

        std::array<SampleType,  ChannelCount> perChannelDelayFractionOfWindow{};
        std::array<std::size_t, ChannelCount> perChannelReadDelayInSamples{};
        std::array<SampleType,  ChannelCount> perChannelDelayInMilliseconds{};

        double     hostSampleRate                  {48000.0};
        SampleType sampleRateOverOneThousand       {static_cast<SampleType>(48)};
        SampleType currentLongestDelayMilliseconds {static_cast<SampleType>(80)};
    };
}
#endif

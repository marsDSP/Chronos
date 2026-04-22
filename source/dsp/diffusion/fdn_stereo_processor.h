#pragma once

#ifndef CHRONOS_FDN_STEREO_PROCESSOR_H
#define CHRONOS_FDN_STEREO_PROCESSOR_H

// ============================================================================
//  fdn_stereo_processor.h
// ----------------------------------------------------------------------------
//  Stereo-in, stereo-out wrapper around FeedbackDelayNetwork. Responsible for:
//    - spreading each stereo input pair to ChannelCount FDN channels,
//    - driving the FDN,
//    - collapsing the N-channel output back to stereo,
//    - dry/wet mixing via a fdnAmount parameter.
//
//  Drops in downstream of DiffusionProcessor inside the delay engine's
//  feedback path to give Chronos a Lexicon-style late-reverb tail.
// ============================================================================

#include <algorithm>
#include <type_traits>

#include "feedback_delay_network.h"
#include "stereo_multichannel_bridge.h"

namespace MarsDSP::DSP::Diffusion
{
    // ----------------------------------------------------------------------------
    //  Default sizes. 8 channels matches the diffusion processor so the two
    //  can share the StereoMultiChannelBridge pattern. 16384 samples per
    //  channel at 96 kHz is ~170 ms, enough for any realistic maximum FDN
    //  delay below the clamp applied in setFdnSizeInMilliseconds().
    // ----------------------------------------------------------------------------
    inline constexpr std::size_t kDefaultFdnChannelCount              = 8;
    inline constexpr std::size_t kDefaultMaxFdnDelaySamplesPerChannel = 1u << 14;

    template <typename SampleType = float,
              std::size_t ChannelCount         = kDefaultFdnChannelCount,
              std::size_t MaxDelaySamplesPerCh = kDefaultMaxFdnDelaySamplesPerChannel>
    class FdnStereoProcessor
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "FdnStereoProcessor requires a floating-point sample type.");

        using FdnType          = FeedbackDelayNetwork<SampleType, ChannelCount,
                                                      MaxDelaySamplesPerCh>;
        using StereoBridgeType = StereoMultiChannelBridge<SampleType, ChannelCount>;

        FdnStereoProcessor() = default;

        void prepare(double        hostSampleRateInHz,
                     std::uint32_t fdnRandomSeed = 0xF1CEE3B7u) noexcept
        {
            feedbackDelayNetwork.prepare(hostSampleRateInHz, fdnRandomSeed);
            feedbackDelayNetwork.setDelayTimeInMilliseconds(currentFdnSizeMilliseconds);
            feedbackDelayNetwork.setDecayTimeInMilliseconds(currentFdnDecayMilliseconds,
                                                            currentFdnDecayMilliseconds * static_cast<SampleType>(0.6),
                                                            currentFdnDampingCrossoverHz);
        }

        void reset() noexcept
        {
            feedbackDelayNetwork.reset();
        }

        void setFdnAmount(SampleType normalisedAmount) noexcept
        {
            currentFdnAmount = std::clamp(normalisedAmount,
                                          static_cast<SampleType>(0),
                                          static_cast<SampleType>(1));
        }

        void setFdnSizeInMilliseconds(SampleType longestDelayMs) noexcept
        {
            currentFdnSizeMilliseconds = std::clamp(longestDelayMs,
                                                    static_cast<SampleType>(10),
                                                    static_cast<SampleType>(200));

            feedbackDelayNetwork.setDelayTimeInMilliseconds(currentFdnSizeMilliseconds);

            // Rescaling delays changes the per-channel bounce count so
            // shelf gains need to be recomputed from the cached T60.
            applyCurrentDecayToFdn();
        }

        void setFdnDecayInMilliseconds(SampleType decayMs) noexcept
        {
            currentFdnDecayMilliseconds = std::clamp(decayMs,
                                                     static_cast<SampleType>(100),
                                                     static_cast<SampleType>(15000));
            applyCurrentDecayToFdn();
        }

        void setFdnDampingCrossoverInHz(SampleType crossoverHz) noexcept
        {
            currentFdnDampingCrossoverHz = std::clamp(crossoverHz,
                                                      static_cast<SampleType>(200),
                                                      static_cast<SampleType>(18000));
            applyCurrentDecayToFdn();
        }

        // Process one stereo pair in place. Bypass when amount == 0.
        void processSingleStereoPairInPlace(SampleType& leftSampleInOut,
                                            SampleType& rightSampleInOut) noexcept
        {
            if (currentFdnAmount <= static_cast<SampleType>(0))
                return;

            const SampleType dryLeft  = leftSampleInOut;
            const SampleType dryRight = rightSampleInOut;

            SampleType fdnInputVector[ChannelCount]{};
            SampleType fdnOutputVector[ChannelCount]{};

            stereoBridge.splitStereoPairIntoMultiChannelVector(dryLeft, dryRight, fdnInputVector);

            feedbackDelayNetwork.processSingleVector(fdnInputVector, fdnOutputVector);

            SampleType wetLeft  = SampleType{};
            SampleType wetRight = SampleType{};

            stereoBridge.collapseMultiChannelVectorToStereoPair(fdnOutputVector, wetLeft, wetRight);

            const SampleType drySide = static_cast<SampleType>(1) - currentFdnAmount;

            leftSampleInOut  = dryLeft  * drySide + wetLeft  * currentFdnAmount;
            rightSampleInOut = dryRight * drySide + wetRight * currentFdnAmount;
        }

        [[nodiscard]] SampleType getFdnAmount() const noexcept
        {
            return currentFdnAmount;
        }

    private:
        void applyCurrentDecayToFdn() noexcept
        {
            // Low freq T60 = decay; high freq T60 = 60% of decay (adds
            // natural HF damping without needing a separate parameter).
            feedbackDelayNetwork.setDecayTimeInMilliseconds(currentFdnDecayMilliseconds,
                                                            currentFdnDecayMilliseconds * static_cast<SampleType>(0.6),
                                                            currentFdnDampingCrossoverHz);
        }

        FdnType          feedbackDelayNetwork{};
        StereoBridgeType stereoBridge{};

        SampleType currentFdnAmount             {static_cast<SampleType>(0)};
        SampleType currentFdnSizeMilliseconds   {static_cast<SampleType>(80)};
        SampleType currentFdnDecayMilliseconds  {static_cast<SampleType>(1500)};
        SampleType currentFdnDampingCrossoverHz {static_cast<SampleType>(2500)};
    };
}
#endif

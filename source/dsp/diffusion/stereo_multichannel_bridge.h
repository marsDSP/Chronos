#pragma once

#ifndef CHRONOS_STEREO_MULTICHANNEL_BRIDGE_H
#define CHRONOS_STEREO_MULTICHANNEL_BRIDGE_H

// ============================================================================
//  stereo_multichannel_bridge.h
// ----------------------------------------------------------------------------
//  Convert a stereo sample pair (L, R) into an N-channel diffusion vector
//  and back, without destroying stereo information. The "split" direction
//  gives each of the N internal channels a unique mix of L and R with a
//  fixed angular spread across the circle; the "collapse" direction
//  averages over the same pattern so that collapse(split(L, R)) = (L, R)
//  exactly for any N.
// ============================================================================

#include <array>
#include <cmath>
#include <type_traits>

namespace MarsDSP::DSP::Diffusion
{
    // ----------------------------------------------------------------------------
    //  StereoMultiChannelBridge<SampleType, ChannelCount>
    //
    //  Splits (L, R) into a vector of ChannelCount channels using a cosine
    //  / sine spread around the unit circle:
    //     channel_k = cos(theta_k) * L + sin(theta_k) * R
    //  with theta_k spaced so channel 0 is purely L and channel N/2 is
    //  purely R.
    //
    //  Collapse is the transpose (times 2/N) so that collapse ° split is
    //  the identity:
    //     L = (2/N) * sum_k cos(theta_k) * channel_k
    //     R = (2/N) * sum_k sin(theta_k) * channel_k
    //
    //  Proof: sum_k cos(theta_k)^2 = N/2 and sum_k cos(theta_k)sin(theta_k) = 0
    //  for theta_k = k * pi / N, so the transpose * split on (L, 0) gives
    //  (L * N/2, 0), and scaling by 2/N recovers L.
    // ----------------------------------------------------------------------------
    template <typename SampleType, std::size_t ChannelCount>
    class StereoMultiChannelBridge
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "StereoMultiChannelBridge requires a floating-point sample type.");
        static_assert(ChannelCount >= 2,
                      "StereoMultiChannelBridge requires at least 2 channels.");

        StereoMultiChannelBridge() noexcept
        {
            fillSpreadCoefficientTables();
        }

        // Split (L, R) into 'channelVector'. channelVector must point to
        // exactly ChannelCount floats.
        void splitStereoPairIntoMultiChannelVector(
            SampleType   leftInputSample,
            SampleType   rightInputSample,
            SampleType*  channelVector) const noexcept
        {
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                channelVector[channelIndex] = perChannelLeftSpreadCoefficient[channelIndex]  * leftInputSample
                                                + perChannelRightSpreadCoefficient[channelIndex] * rightInputSample;
            }
        }

        // Collapse 'channelVector' (exactly ChannelCount floats) into a
        // stereo pair. The collapse is scaled so that
        //    collapse(split(L, R)) == (L, R)
        // exactly, for any ChannelCount >= 2.
        void collapseMultiChannelVectorToStereoPair(
            const SampleType* channelVector,
            SampleType&       leftOutputSample,
            SampleType&       rightOutputSample) const noexcept
        {
            SampleType accumulatedLeft  = SampleType{};
            SampleType accumulatedRight = SampleType{};

            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                accumulatedLeft  += perChannelLeftSpreadCoefficient[channelIndex]  * channelVector[channelIndex];
                accumulatedRight += perChannelRightSpreadCoefficient[channelIndex] * channelVector[channelIndex];
            }

            const SampleType collapseNormalisation = static_cast<SampleType>(2) / static_cast<SampleType>(ChannelCount);

            leftOutputSample  = accumulatedLeft  * collapseNormalisation;
            rightOutputSample = accumulatedRight * collapseNormalisation;
        }

    private:
        void fillSpreadCoefficientTables() noexcept
        {
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                const double angleRadians = static_cast<double>(channelIndex) * M_PI / static_cast<double>(ChannelCount);

                perChannelLeftSpreadCoefficient[channelIndex]  = static_cast<SampleType>(std::cos(angleRadians));
                perChannelRightSpreadCoefficient[channelIndex] = static_cast<SampleType>(std::sin(angleRadians));
            }
        }

        std::array<SampleType, ChannelCount> perChannelLeftSpreadCoefficient{};
        std::array<SampleType, ChannelCount> perChannelRightSpreadCoefficient{};
    };
}
#endif

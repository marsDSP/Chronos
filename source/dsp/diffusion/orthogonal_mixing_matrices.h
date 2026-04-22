#pragma once

#ifndef CHRONOS_ORTHOGONAL_MIXING_MATRICES_H
#define CHRONOS_ORTHOGONAL_MIXING_MATRICES_H

// ============================================================================
//  orthogonal_mixing_matrices.h
// ----------------------------------------------------------------------------
//  Two unit-gain-preserving mixing matrices used by the diffusion chain and
//  the feedback delay network:
//
//    * OrthogonalHadamardMixer<N> - In-place +/-1 sign-pattern mixer built
//      by the classic fast recursive Hadamard construction. Scales the
//      output by 1/sqrt(N) so the transform is unitary (||Hx|| = ||x||).
//      Requires N to be a power of 2.
//
//    * OrthogonalHouseholderMixer<N> - Dense reflection mixer of the form
//      y = x - (2/N) * sum(x) * 1 .  Well-known all-to-all mixer used by
//      Signalsmith / Dattorro topologies; scalar sum + bias subtract so
//      extremely cheap. Works for any N >= 2.
// ============================================================================

#include <limits>
#include <type_traits>

namespace MarsDSP::DSP::Diffusion
{
    // ----------------------------------------------------------------------------
    //  Compile-time helper: is N a power of 2?
    //  Hadamard requires N = 2^k.
    // ----------------------------------------------------------------------------
    template<std::size_t N>
    inline constexpr bool kIsPowerOfTwo = (N > 0) && ((N & (N - 1)) == 0);

    // Newton-Raphson constexpr sqrt, used only at compile time for mixer
    // scaling factors (std::sqrt is not constexpr before C++26).
    namespace detail
    {
        constexpr double constexprSqrtNewtonRaphson(double x, double previousGuess, double currentGuess) noexcept
        {
            return currentGuess == previousGuess
                       ? currentGuess
                       : constexprSqrtNewtonRaphson(
                           x, currentGuess,
                           0.5 * (currentGuess + x / currentGuess));
        }

        constexpr double constexprSqrt(double x) noexcept
        {
            return (x >= 0.0 && x < std::numeric_limits<double>::infinity())
                       ? constexprSqrtNewtonRaphson(x, 0.0, x)
                       : std::numeric_limits<double>::quiet_NaN();
        }
    }

    // ----------------------------------------------------------------------------
    //  OrthogonalHadamardMixer<ChannelCount>
    //
    //  y = (1/sqrt(N)) * H_N * x, where H_N is the N-th order +/-1 Hadamard
    //  matrix constructed recursively.
    //
    //  The recursive core computes the unscaled transform in O(N log N)
    //  sum/diff operations; the scaling pass at the end applies the unitary
    //  1/sqrt(N) factor to make the output norm match the input norm.
    // ----------------------------------------------------------------------------
    template<typename SampleType, std::size_t ChannelCount>
    struct OrthogonalHadamardMixer
    {
        static_assert(std::is_floating_point_v<SampleType>,
                      "OrthogonalHadamardMixer requires a floating-point sample type.");
        static_assert(kIsPowerOfTwo<ChannelCount>,
                      "OrthogonalHadamardMixer requires ChannelCount to be a power of 2.");

    private:
        static constexpr SampleType kUnitaryScalingFactor = static_cast<SampleType>(1) / static_cast<SampleType>(
                                                            detail::constexprSqrt(static_cast<double>(ChannelCount)));

    public:
        // Recursive unscaled butterfly (internal, but kept public so the
        // half-size template instantiations can invoke each other through
        // the recursion chain). Writes into 'destination'; may alias
        // 'source' since the algorithm only reads each slot once per level.
        static void applyUnscaledRecursiveButterfly(SampleType *destination, const SampleType *source) noexcept
        {
            if constexpr (ChannelCount == 1)
            {
                destination[0] = source[0];
            }
            else if constexpr (ChannelCount == 2)
            {
                const SampleType sum = source[0] + source[1];
                const SampleType difference = source[0] - source[1];
                destination[0] = sum;
                destination[1] = difference;
            }
            else
            {
                static constexpr std::size_t halfChannelCount = ChannelCount / 2;

                // Two unscaled half-size Hadamards, then combine halves by
                // sum / difference.
                OrthogonalHadamardMixer<SampleType, halfChannelCount>
                        ::applyUnscaledRecursiveButterfly(destination, source);
                OrthogonalHadamardMixer<SampleType, halfChannelCount>
                        ::applyUnscaledRecursiveButterfly(destination + halfChannelCount, source + halfChannelCount);

                for (std::size_t channelIndex = 0;
                     channelIndex < halfChannelCount;
                     ++channelIndex)
                {
                    const SampleType firstHalfValue = destination[channelIndex];
                    const SampleType secondHalfValue = destination[channelIndex + halfChannelCount];
                    destination[channelIndex] = firstHalfValue + secondHalfValue;
                    destination[channelIndex + halfChannelCount] = firstHalfValue - secondHalfValue;
                }
            }
        }

        // Apply the unitary Hadamard transform in place.
        static void applyInPlace(SampleType *channelValues) noexcept
        {
            applyUnscaledRecursiveButterfly(channelValues, channelValues);

            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                channelValues[channelIndex] *= kUnitaryScalingFactor;
            }
        }

        // Apply the unitary Hadamard transform out of place; 'destination'
        // and 'source' must not overlap.
        static void applyOutOfPlace(SampleType *destination, const SampleType *source) noexcept
        {
            applyUnscaledRecursiveButterfly(destination, source);

            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                destination[channelIndex] *= kUnitaryScalingFactor;
            }
        }

        [[nodiscard]] static constexpr SampleType unitaryScalingFactor() noexcept
        {
            return kUnitaryScalingFactor;
        }
    };

    // ----------------------------------------------------------------------------
    //  OrthogonalHouseholderMixer<ChannelCount>
    //
    //  y = x - (2/N) * (sum_k x_k) * 1   where 1 is the all-ones vector.
    //
    //  This is a norm-preserving reflection: the reflector vector is
    //  u = 1/sqrt(N) * [1, 1, ..., 1], and the transform is
    //    H = I - 2 u u^T.
    //  Because ||u|| = 1, H is orthogonal and ||Hx|| = ||x|| exactly.
    // ----------------------------------------------------------------------------
    template<typename SampleType, std::size_t ChannelCount>
    struct OrthogonalHouseholderMixer
    {
        static_assert(std::is_floating_point_v<SampleType>,
                      "OrthogonalHouseholderMixer requires a floating-point sample type.");
        static_assert(ChannelCount >= 2,
                      "OrthogonalHouseholderMixer requires ChannelCount >= 2.");

    private:
        static constexpr SampleType kHouseholderBiasFactor = static_cast<SampleType>(-2) / static_cast<SampleType>(ChannelCount);

    public:
        // Apply the Householder reflection in place.
        static void applyInPlace(SampleType *channelValues) noexcept
        {
            SampleType accumulatedSumAcrossChannels = SampleType{};
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                accumulatedSumAcrossChannels += channelValues[channelIndex];
            }

            const SampleType broadcastBias =
                    accumulatedSumAcrossChannels * kHouseholderBiasFactor;

            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                channelValues[channelIndex] += broadcastBias;
            }
        }

        // Apply the Householder reflection out of place; 'destination' and
        // 'source' MAY overlap / alias, but the result is well defined as
        // if applyInPlace had been called on a copy.
        static void applyOutOfPlace(SampleType *destination, const SampleType *source) noexcept
        {
            SampleType accumulatedSumAcrossChannels = SampleType{};
            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                accumulatedSumAcrossChannels += source[channelIndex];
            }

            const SampleType broadcastBias = accumulatedSumAcrossChannels * kHouseholderBiasFactor;

            for (std::size_t channelIndex = 0;
                 channelIndex < ChannelCount;
                 ++channelIndex)
            {
                destination[channelIndex] = source[channelIndex] + broadcastBias;
            }
        }

        [[nodiscard]] static constexpr SampleType householderBiasFactor() noexcept
        {
            return kHouseholderBiasFactor;
        }
    };
}
#endif

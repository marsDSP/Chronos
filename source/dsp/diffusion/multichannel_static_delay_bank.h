#pragma once

#ifndef CHRONOS_MULTICHANNEL_STATIC_DELAY_BANK_H
#define CHRONOS_MULTICHANNEL_STATIC_DELAY_BANK_H

// ============================================================================
//  multichannel_static_delay_bank.h
// ----------------------------------------------------------------------------
//  Bank of N independent short delay lines, each sized up to a power-of-2
//  compile-time maximum so the modulo wrap degenerates to a cheap mask. Used
//  as storage by the diffusion stages (one line per parallel diffusion
//  channel) and by the feedback delay network.
// ============================================================================

#include <array>
#include <cassert>
#include <type_traits>

namespace MarsDSP::DSP::Diffusion
{
    // ----------------------------------------------------------------------------
    //  MultiChannelStaticDelayBank<SampleType, ChannelCount, MaxSamplesPerChannel>
    //
    //  Requires MaxSamplesPerChannel to be a power of 2 so wrap-around can
    //  use bitmask arithmetic. Each channel shares a single write pointer
    //  (callers advance it together) but stores its own independent ring
    //  buffer, so read pointers can be arbitrary per channel.
    // ----------------------------------------------------------------------------
    template <typename SampleType, std::size_t ChannelCount, std::size_t MaxSamplesPerChannel>
    class MultiChannelStaticDelayBank
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "MultiChannelStaticDelayBank requires a floating-point sample type.");
        static_assert((MaxSamplesPerChannel > 0) && ((MaxSamplesPerChannel & (MaxSamplesPerChannel - 1)) == 0),
                      "MultiChannelStaticDelayBank requires "
                      "MaxSamplesPerChannel to be a power of 2.");

        static constexpr std::size_t kChannelCount        = ChannelCount;
        static constexpr std::size_t kMaxSamplesPerLine   = MaxSamplesPerChannel;
        static constexpr std::size_t kWrapAroundBitMask   = MaxSamplesPerChannel - 1;

        MultiChannelStaticDelayBank() = default;

        // Zero every sample in every channel's ring buffer, and reset the
        // shared write index to zero. Safe to call whenever stale state
        // might remain (transport restart, plugin bypass, etc.).
        void reset() noexcept
        {
            for (auto& perChannelRing : perChannelRingBuffers)
                perChannelRing.fill(SampleType{});

            sharedWriteIndex = 0;
        }

        // Push a single sample into a single channel's ring. The write
        // index is shared across channels so callers drive it via
        // advanceSharedWriteIndex().
        void writeSampleToChannel(std::size_t channelIndex, SampleType  inputSample) noexcept
        {
            assert(channelIndex < ChannelCount);
            perChannelRingBuffers[channelIndex][sharedWriteIndex] = inputSample;
        }

        // Read a single sample from a single channel at the given delay
        // (in whole samples). readDelayInSamples MUST be in [1, MaxSamplesPerChannel-1].
        [[nodiscard]] SampleType readDelayedSampleFromChannel(std::size_t channelIndex, std::size_t readDelayInSamples) const noexcept
        {
            assert(channelIndex < ChannelCount);
            assert(readDelayInSamples >= 1);
            assert(readDelayInSamples <  MaxSamplesPerChannel);

            const std::size_t readIndex = (sharedWriteIndex + MaxSamplesPerChannel - readDelayInSamples) & kWrapAroundBitMask;
            return perChannelRingBuffers[channelIndex][readIndex];
        }

        // Advance the shared write index by one sample, wrapping.
        // Callers typically call this once after writing all channels for
        // the current sample step.
        void advanceSharedWriteIndex() noexcept
        {
            sharedWriteIndex = (sharedWriteIndex + 1) & kWrapAroundBitMask;
        }

        // Return the shared write pointer (exposed mostly so tests can
        // inspect internal state; not normally needed in process loops).
        [[nodiscard]] std::size_t getSharedWriteIndex() const noexcept
        {
            return sharedWriteIndex;
        }

    private:
        // Per-channel ring buffers, each exactly MaxSamplesPerChannel long
        // (no doubling / mirror trick because we're integer-delay only).
        std::array<std::array<SampleType, MaxSamplesPerChannel>, ChannelCount>
            perChannelRingBuffers{};

        std::size_t sharedWriteIndex{0};
    };
}
#endif

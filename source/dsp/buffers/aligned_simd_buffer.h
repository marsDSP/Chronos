#pragma once

#ifndef CHRONOS_ALIGNED_SIMD_BUFFER_H
#define CHRONOS_ALIGNED_SIMD_BUFFER_H

// ============================================================================
//  aligned_simd_buffer.h
// ----------------------------------------------------------------------------
//  Heap-allocated multichannel sample buffer whose backing storage is aligned
//  to the SIMD batch alignment of xsimd's default architecture. Per-channel
//  sample counts are rounded up to a whole batch so every channel pointer is
//  xsimd-load/store-safe even when the caller asks for a sample count that
//  isn't a multiple of the vector width.
// ============================================================================

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

#include <xsimd/xsimd.hpp>

#include "aligned_simd_buffer_detail.h"

namespace MarsDSP::DSP::AlignedBuffers
{
    // ----------------------------------------------------------------------------
    //  AlignedSIMDBuffer
    //
    //  Owns its backing std::vector (allocated through xsimd's aligned
    //  allocator). Channels share the same contiguous block of memory; each
    //  channel's base pointer is the block start plus channelIndex * padded
    //  sample count. This layout lets us hand out a plain SampleType** to
    //  code that expects the JUCE-style array-of-channel-pointers layout.
    // ----------------------------------------------------------------------------
    template<typename SampleType, std::size_t alignmentInBytes = xsimd::default_arch::alignment()>
    class AlignedSIMDBuffer
    {
    public:
        using ElementType = SampleType;

        static constexpr int kMaximumNumberOfChannels = kMaxChannelsInAlignedBuffer;

        AlignedSIMDBuffer() = default;

        AlignedSIMDBuffer(int desiredNumberOfChannels, int desiredNumberOfSamples)
        {
            setMaxSize(desiredNumberOfChannels, desiredNumberOfSamples);
        }

        AlignedSIMDBuffer(AlignedSIMDBuffer &&) noexcept = default;
        AlignedSIMDBuffer &operator=(AlignedSIMDBuffer &&) noexcept = default;
        AlignedSIMDBuffer(const AlignedSIMDBuffer &) = delete;
        AlignedSIMDBuffer &operator=(const AlignedSIMDBuffer &) = delete;

        // ------------------------------------------------------------------
        //  Resize the underlying storage. Allocates max(numChannels, 1) *
        //  paddedSampleCount elements in one contiguous aligned vector and
        //  rewrites the per-channel pointer table. Also resets the active
        //  (current) size to the requested dimensions.
        // ------------------------------------------------------------------
        void setMaxSize(int desiredNumberOfChannels, int desiredNumberOfSamples)
        {
            assert(desiredNumberOfChannels >= 0 && desiredNumberOfChannels <= kMaximumNumberOfChannels);

            const int effectiveNumberOfChannels = std::max(desiredNumberOfChannels, 1);
            const int effectiveNumberOfSamples  = std::max(desiredNumberOfSamples,  0);

            // For float / double buffers, pad the per-channel sample count
            // up to the next SIMD batch width so channel base pointers stay
            // aligned and so unaligned end-of-channel tail reads are safe.
            int paddedNumberOfSamples = effectiveNumberOfSamples;

            if constexpr (Detail::isFloatOrDouble<SampleType>)
            {
                static constexpr int kSIMDBatchWidth = static_cast<int>(xsimd::batch<SampleType>::size);
                paddedNumberOfSamples = ceilingDivide(effectiveNumberOfSamples, kSIMDBatchWidth) * kSIMDBatchWidth;
            }

            rawContiguousStorage.clear();
            hasBeenFullyCleared = true;
            currentNumberOfChannels = 0;
            currentNumberOfSamples = 0;

            rawContiguousStorage.resize(static_cast<std::size_t>(effectiveNumberOfChannels)
                                        * static_cast<std::size_t>(paddedNumberOfSamples), SampleType{});

            std::fill(perChannelBasePointers.begin(), perChannelBasePointers.end(), nullptr);

            for (int channelIndex = 0;
                 channelIndex < effectiveNumberOfChannels;
                 ++channelIndex)
            {
                perChannelBasePointers[static_cast<std::size_t>(channelIndex)] = rawContiguousStorage.data()
                                                                                    + static_cast<std::size_t>(channelIndex)
                                                                                    * static_cast<std::size_t>(paddedNumberOfSamples);
            }

            paddedNumberOfSamplesPerChannel = paddedNumberOfSamples;
            setCurrentSize(effectiveNumberOfChannels, effectiveNumberOfSamples);
        }

        // ------------------------------------------------------------------
        //  Change the "in-use" region of a buffer whose max size is already
        //  enough. Zero-fills any newly-exposed channels / samples so
        //  growing a region never leaves stale data in the tail.
        // ------------------------------------------------------------------
        void setCurrentSize(int desiredNumberOfChannels, int desiredNumberOfSamples) noexcept
        {
            assert(static_cast<std::size_t>(desiredNumberOfChannels)
                    * static_cast<std::size_t>(desiredNumberOfSamples)
                    <= rawContiguousStorage.size());

            const bool numberOfSamplesIsIncreasing = desiredNumberOfSamples > currentNumberOfSamples;
            const bool numberOfChannelsIsIncreasing = desiredNumberOfChannels > currentNumberOfChannels;

            if (numberOfSamplesIsIncreasing)
                Detail::clearChannelRange(perChannelBasePointers.data(),
                                          0,
                                          currentNumberOfChannels,
                                          currentNumberOfSamples,
                                          desiredNumberOfSamples);

            if (numberOfChannelsIsIncreasing)
                Detail::clearChannelRange(perChannelBasePointers.data(),
                                          currentNumberOfChannels,
                                          desiredNumberOfChannels,
                                          0,
                                          desiredNumberOfSamples);

            currentNumberOfChannels = desiredNumberOfChannels;
            currentNumberOfSamples  = desiredNumberOfSamples;
        }

        // Zero every active sample in every active channel. No-op if
        // already cleared since last write.
        void clearAllSamples() noexcept
        {
            if (hasBeenFullyCleared)
                return;

            Detail::clearChannelRange(perChannelBasePointers.data(), 0, currentNumberOfChannels, 0, currentNumberOfSamples);
            hasBeenFullyCleared = true;
        }

        [[nodiscard]] int getNumChannels() const noexcept
        {
            return currentNumberOfChannels;
        }

        [[nodiscard]] int getNumSamples() const noexcept
        {
            return currentNumberOfSamples;
        }

        // Per-channel samples actually allocated in memory (>= getNumSamples()
        // because of SIMD padding). Exposed so code that wants to write
        // past the logical end (e.g., Lagrange tail mirrors in the delay
        // engine) can know how much slack it has.
        [[nodiscard]] int getPaddedNumSamplesPerChannel() const noexcept
        {
            return paddedNumberOfSamplesPerChannel;
        }

        // ------------------------------------------------------------------
        //  Mutable / const raw pointer access. hasBeenFullyCleared is
        //  opportunistically cleared on any mutable request so subsequent
        //  clearAllSamples() calls do useful work.
        // ------------------------------------------------------------------
        [[nodiscard]] SampleType *getWritePointer(int channelIndex) noexcept
        {
            hasBeenFullyCleared = false;
            return perChannelBasePointers[static_cast<std::size_t>(channelIndex)];
        }

        [[nodiscard]] const SampleType *getReadPointer(int channelIndex) const noexcept
        {
            return perChannelBasePointers[static_cast<std::size_t>(channelIndex)];
        }

        [[nodiscard]] std::span<SampleType> getWriteSpan(int channelIndex) noexcept
        {
            hasBeenFullyCleared = false;

            return
            {
                perChannelBasePointers[static_cast<std::size_t>(channelIndex)],
                static_cast<std::size_t>(currentNumberOfSamples)
            };
        }

        [[nodiscard]] std::span<const SampleType> getReadSpan(int channelIndex) const noexcept
        {
            return
            {
                perChannelBasePointers[static_cast<std::size_t>(channelIndex)],
                static_cast<std::size_t>(currentNumberOfSamples)
            };
        }

        [[nodiscard]] SampleType **getArrayOfWritePointers() noexcept
        {
            hasBeenFullyCleared = false;
            return perChannelBasePointers.data();
        }

        [[nodiscard]] const SampleType *const*getArrayOfReadPointers() const noexcept
        {
            return perChannelBasePointers.data();
        }

    private:
        int currentNumberOfChannels{0};
        int currentNumberOfSamples{0};
        int paddedNumberOfSamplesPerChannel{0};
        bool hasBeenFullyCleared{true};

        // xsimd::aligned_allocator honours alignmentInBytes on every heap
        // allocation; combined with the per-channel padding above, this
        // guarantees every channel base pointer is aligned.
        std::vector<SampleType, xsimd::aligned_allocator<SampleType, alignmentInBytes> > rawContiguousStorage{};

        // Fixed-width table of channel base pointers; matches the
        // AlignedSIMDBufferView layout so views can be constructed in O(1)
        // by copying the pointer table.
        std::array<SampleType *, kMaximumNumberOfChannels> perChannelBasePointers{};
    };
}
#endif

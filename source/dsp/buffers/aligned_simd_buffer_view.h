#pragma once

#ifndef CHRONOS_ALIGNED_SIMD_BUFFER_VIEW_H
#define CHRONOS_ALIGNED_SIMD_BUFFER_VIEW_H

// ============================================================================
//  aligned_simd_buffer_view.h
// ----------------------------------------------------------------------------
//  Lightweight, non-owning view over an existing multichannel sample layout.
//  Mirrors the shape of juce::dsp::AudioBlock so callers that used to pass a
//  JUCE block can construct an aligned_simd_buffer_view from the same
//  channel-pointer array and pass that into the delay engine instead.
//
//  Does NOT depend on JUCE; the raw-pointer-array constructor is the
//  universal adapter. ChronosProcessor builds one from
//  buffer.getArrayOfWritePointers().
// ============================================================================

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

#include "aligned_simd_buffer.h"
#include "aligned_simd_buffer_detail.h"

namespace MarsDSP::DSP::AlignedBuffers
{
    // ----------------------------------------------------------------------------
    //  AlignedSIMDBufferView
    //
    //  Template on SampleType (supports const / non-const). Keeps only a
    //  fixed-size table of channel pointers + a sample offset + a sample
    //  count. Copying / moving the view does not touch sample data.
    // ----------------------------------------------------------------------------
    template<typename SampleType>
    class AlignedSIMDBufferView
    {
    public:
        using ElementType = SampleType;

        static constexpr int kMaximumNumberOfChannels = kMaxChannelsInAlignedBuffer;

        AlignedSIMDBufferView() = default;

        // ------------------------------------------------------------------
        //  Build a view directly over a raw array of channel base pointers
        //  (JUCE-style SampleType**). The caller-owned pointers must stay
        //  alive as long as the view is in use.
        // ------------------------------------------------------------------
        AlignedSIMDBufferView(SampleType *const*externalChannelBasePointers,
                              int numberOfChannelsInView,
                              int numberOfSamplesPerChannelInView,
                              int sampleOffsetIntoEachChannel = 0) noexcept
                              : numberOfChannels(numberOfChannelsInView),
                                numberOfSamples(numberOfSamplesPerChannelInView)
        {
            assert(numberOfChannels >= 0 && numberOfChannels <= kMaximumNumberOfChannels);
            assert(numberOfSamples >= 0);

            for (int channelIndex = 0;
                 channelIndex < numberOfChannels;
                 ++channelIndex)
            {
                perChannelPointers[static_cast<std::size_t>(channelIndex)] =
                        externalChannelBasePointers[channelIndex]
                        + sampleOffsetIntoEachChannel;
            }
        }

        // ------------------------------------------------------------------
        //  Construct a view over every channel / sample of an
        //  AlignedSIMDBuffer that owns the underlying storage. Convenient
        //  for code paths that already hold an AlignedSIMDBuffer by
        //  reference (e.g., the delay engine's circular buffer).
        // ------------------------------------------------------------------
        template<typename BufferSampleType,
                 typename = std::enable_if_t<std::is_same_v<std::remove_const_t<SampleType>,
                                                            std::remove_const_t<BufferSampleType>>>>

        explicit AlignedSIMDBufferView(AlignedSIMDBuffer<std::remove_const_t<BufferSampleType>> &backingBuffer,
            int sampleOffsetIntoEachChannel = 0,
            int desiredNumberOfSamplesInView = -1,
            int desiredStartChannelIndex = 0,
            int desiredNumberOfChannelsInView = -1) noexcept
            : numberOfChannels(desiredNumberOfChannelsInView < 0 ? (backingBuffer.getNumChannels() - desiredStartChannelIndex)
                                                                 : desiredNumberOfChannelsInView),
              numberOfSamples(desiredNumberOfSamplesInView < 0   ? (backingBuffer.getNumSamples() - sampleOffsetIntoEachChannel)
                                                                 : desiredNumberOfSamplesInView)
        {
            assert(backingBuffer.getNumChannels() >= desiredStartChannelIndex + numberOfChannels);
            assert(backingBuffer.getNumSamples() >= sampleOffsetIntoEachChannel + numberOfSamples);

            auto *const*sourcePointerArray = backingBuffer.getArrayOfWritePointers();
        }

        AlignedSIMDBufferView(const AlignedSIMDBufferView &) = default;

        AlignedSIMDBufferView &operator=(const AlignedSIMDBufferView &) = default;

        AlignedSIMDBufferView(AlignedSIMDBufferView &&) = default;

        AlignedSIMDBufferView &operator=(AlignedSIMDBufferView &&) = default;

        [[nodiscard]] int getNumChannels() const noexcept
        {
            return numberOfChannels;
        }

        [[nodiscard]] int getNumSamples() const noexcept
        {
            return numberOfSamples;
        }

        [[nodiscard]] SampleType *getChannelPointer(int channelIndex) const noexcept
        {
            return perChannelPointers[static_cast<std::size_t>(channelIndex)];
        }

        [[nodiscard]] std::span<SampleType> getChannelSpan(int channelIndex) const noexcept
        {
            return {
                perChannelPointers[static_cast<std::size_t>(channelIndex)],
                static_cast<std::size_t>(numberOfSamples)
            };
        }

        // Sub-view: take a sample-range slice off the front of every
        // channel. Useful for block-within-block work (partial overlap,
        // tail processing, etc.); mirrors juce::dsp::AudioBlock::getSubBlock.
        [[nodiscard]] AlignedSIMDBufferView getSubRange(int sampleOffsetIntoView, int numberOfSamplesInSubRange) const noexcept
        {
            assert(sampleOffsetIntoView >= 0 && sampleOffsetIntoView <= numberOfSamples);
            assert(sampleOffsetIntoView + numberOfSamplesInSubRange <= numberOfSamples);

            AlignedSIMDBufferView subView;
            subView.numberOfChannels = numberOfChannels;
            subView.numberOfSamples = numberOfSamplesInSubRange;
            for (int channelIndex = 0;
                 channelIndex < numberOfChannels;
                 ++channelIndex)
            {
                subView.perChannelPointers[static_cast<std::size_t>(channelIndex)] =
                        perChannelPointers[static_cast<std::size_t>(channelIndex)]
                        + sampleOffsetIntoView;
            }
            return subView;
        }

    private:
        int numberOfChannels{0};
        int numberOfSamples{0};

        std::array<SampleType *, kMaximumNumberOfChannels>
        perChannelPointers{};
    };
}
#endif

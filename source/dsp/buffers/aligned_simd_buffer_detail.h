#pragma once

#ifndef CHRONOS_ALIGNED_SIMD_BUFFER_DETAIL_H
#define CHRONOS_ALIGNED_SIMD_BUFFER_DETAIL_H

// ============================================================================
//  aligned_simd_buffer_detail.h
// ----------------------------------------------------------------------------
//  Internal helpers for aligned_simd_buffer / aligned_simd_buffer_view.
// ============================================================================
#include <algorithm>
#include <type_traits>

namespace MarsDSP::DSP::AlignedBuffers::inline Detail
{
    // ------------------------------------------------------------------
    //  Compile-time constant: maximum number of channels any
    //  AlignedSIMDBuffer / AlignedSIMDBufferView can ever address. Kept
    //  here so both classes agree on the array size of their channel
    //  pointer tables.
    // ------------------------------------------------------------------
    inline constexpr int kMaxChannelsInAlignedBuffer = 32;

    // Ceiling division for integer types; used for padding a requested
    // number of samples up to the next whole SIMD batch.
    template <typename IntegerType>
    constexpr IntegerType ceilingDivide(IntegerType numerator, IntegerType denominator) noexcept
    {
        return (numerator + denominator - 1) / denominator;
    }

    // Trait: the buffer's SIMD padding logic only kicks in for the two
    // floating-point sample types we actually use in the plugin. Integer
    // buffers are left unpadded.
    template <typename T>
    inline constexpr bool isFloatOrDouble = std::is_same_v<T, float> || std::is_same_v<T, double>;

    // Zero a [startSampleIndex, endSampleIndex) window in every channel
    // between [startChannelIndex, endChannelIndex). Null pointers are
    // silently skipped so partially-initialised channel arrays are safe.
    template <typename SampleType>
    void clearChannelRange(SampleType* const* channelPointerArray,
                           int startChannelIndex,
                           int endChannelIndex,
                           int startSampleIndex,
                           int endSampleIndex) noexcept
    {
        for (int channelIndex = startChannelIndex;
             channelIndex < endChannelIndex;
             ++channelIndex)
        {
            SampleType* const channelBasePointer = channelPointerArray[channelIndex];
            if (channelBasePointer == nullptr)
                continue;

            std::fill(channelBasePointer + startSampleIndex,
                      channelBasePointer + endSampleIndex,
                      SampleType{});
        }
    }
}
#endif

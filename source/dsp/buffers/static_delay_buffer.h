#pragma once

#ifndef CHRONOS_STATIC_DELAY_BUFFER_H
#define CHRONOS_STATIC_DELAY_BUFFER_H

// ============================================================================
//  static_delay_buffer.h
// ----------------------------------------------------------------------------
//  Single-channel power-of-two ring buffer with an optional mirror half, so
//  an interpolator's read window never straddles the wrap boundary.
//
//  The interpolation type is a template parameter taken from
//  source/dsp/engine/delay/delay_interpolator.h, so callers pick quality
//  (None / Linear / Lagrange3rd / Lagrange5th) without reimplementing the
//  math at each call-site.
//
//  Usage:
//
//      using Line = Buffers::StaticDelayBuffer<float,
//                                              InterpolationTypes::Lagrange5th,
//                                              1 << 14>;
//      Line line;
//      line.reset();
//
//      // Write sample at current writeIndex, then decrement writeIndex.
//      line.pushSample(inputSample, writeIndex);
//      Line::decrementPointer(writeIndex);
//
//      // Read at a fractional delay in samples.
//      const float readIndex = Line::getReadPointer(writeIndex, delayInSamples);
//      const float y         = line.popSample(readIndex);
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "dsp/engine/delay/delay_interpolator.h"

namespace MarsDSP::DSP::Buffers
{
    // ----------------------------------------------------------------------------
    //  StaticDelayBuffer<SampleType, InterpolationType, MaxDelaySamples, StorageType>
    //
    //  MaxDelaySamples MUST be a power of two so the decrementPointer helper can
    //  stay branch-free and callers can mask instead of modulo.
    // ----------------------------------------------------------------------------
    template<typename SampleType,
        typename InterpolationType = None,
        std::size_t MaxDelaySamples = (1u << 14),
        typename StorageType = SampleType>
    class StaticDelayBuffer
    {
    public:
        using NumericType = SampleType;

        static_assert(std::is_floating_point_v<SampleType>,
                      "StaticDelayBuffer requires a floating-point sample type.");
        static_assert(MaxDelaySamples > 0
                      && (MaxDelaySamples & (MaxDelaySamples - 1)) == 0,
                      "StaticDelayBuffer requires MaxDelaySamples to be a power of two.");

        // Does the interpolator need a multi-sample read window? None = no.
        static constexpr bool kInterpolatorNeedsMirror =
                !std::is_same_v<InterpolationType, None>;

        // For interpolated types we keep a mirror copy of the buffer right
        // after the primary one so a read of N consecutive samples starting
        // anywhere in [0, MaxDelaySamples) never has to branch on wrap.
        static constexpr std::size_t kStorageLength =
                kInterpolatorNeedsMirror ? (2 * MaxDelaySamples) : MaxDelaySamples;

        static constexpr std::size_t kMaxDelaySamples = MaxDelaySamples;

        StaticDelayBuffer() = default;

        // ------------------------------------------------------------------
        //  Zero every stored sample.
        // ------------------------------------------------------------------
        void reset() noexcept
        {
            std::fill(buffer.begin(), buffer.end(), StorageType{});
        }

        // ------------------------------------------------------------------
        //  Write one sample at writeIndex. For mirrored storage, also write
        //  to writeIndex + MaxDelaySamples so the interpolator window is
        //  contiguous across the wrap.
        // ------------------------------------------------------------------
        void pushSample(SampleType inputSample, int writeIndex) noexcept
        {
            assert(writeIndex >= 0 && writeIndex < static_cast<int>(MaxDelaySamples));

            buffer[static_cast<std::size_t>(writeIndex)] =
                    static_cast<StorageType>(inputSample);

            if constexpr (kInterpolatorNeedsMirror)
            {
                buffer[static_cast<std::size_t>(writeIndex)
                       + MaxDelaySamples] = static_cast<StorageType>(inputSample);
            }
        }

        // ------------------------------------------------------------------
        //  popSample overloads (one per interpolator class of behavior).
        //  The SFINAE-selected overload is the one whose signature matches
        //  the interpolator's call() arity, so the delay type can be
        //  templated on any InterpolationType from delay_interpolator.h
        //  without a wrapper switch.
        // ------------------------------------------------------------------

        // Fractional-delay popSample for all non-None interpolators (Linear,
        // Lagrange3rd, Lagrange5th, ...). The interpolator class does all
        // the read-window math; the buffer just hands it a pointer + the
        // integer / fractional delay.
        template<typename IT = InterpolationType,
            std::enable_if_t<!std::is_same_v<IT, None>, int> = 0>
        SampleType popSample(NumericType readIndex) noexcept
        {
            assert(readIndex >= NumericType{0}
                && readIndex < static_cast<NumericType>(MaxDelaySamples));

            const int integerReadIndex = static_cast<int>(readIndex);
            const NumericType fractionalPart = readIndex
                                               - static_cast<NumericType>(integerReadIndex);

            return interpolator.template call<SampleType, NumericType, StorageType>(
                buffer.data(), integerReadIndex, fractionalPart);
        }

        // Integer popSample for the None specialization. Takes a float-typed
        // readIndex for API symmetry but just casts it.
        template<typename IT = InterpolationType,
            std::enable_if_t<std::is_same_v<IT, None>, int> = 0>
        SampleType popSample(NumericType readIndex) noexcept
        {
            assert(readIndex >= NumericType{0}
                && readIndex < static_cast<NumericType>(MaxDelaySamples));
            return interpolator.template call<SampleType, NumericType, StorageType>(
                buffer.data(), static_cast<int>(readIndex));
        }

        // Convenience overload when the delay is already an int.
        SampleType popSample(int readIndex) noexcept
        {
            assert(readIndex >= 0 && readIndex < static_cast<int>(MaxDelaySamples));
            return static_cast<SampleType>(buffer[static_cast<std::size_t>(readIndex)]);
        }

        // ------------------------------------------------------------------
        //  Pointer helpers. decrementPointer walks any index (int or float)
        //  down by one sample, wrapping at MaxDelaySamples. getReadPointer
        //  converts a write index + positive delay into a read index.
        // ------------------------------------------------------------------
        template<typename T>
        static void decrementPointer(T &index) noexcept
        {
            index += static_cast<T>(MaxDelaySamples - 1);
            if (index >= static_cast<T>(MaxDelaySamples))
                index -= static_cast<T>(MaxDelaySamples);
        }

        static NumericType getReadPointer(int writeIndex,
                                          NumericType delayInSamples) noexcept
        {
            delayInSamples = std::max(static_cast<NumericType>(1), delayInSamples);
            const NumericType readIndex = std::fmod(
                static_cast<NumericType>(writeIndex) + delayInSamples,
                static_cast<NumericType>(MaxDelaySamples));
            assert(readIndex >= NumericType{0}
                && readIndex < static_cast<NumericType>(MaxDelaySamples));
            return readIndex;
        }

        static int getReadPointer(int writeIndex, int delayInSamples) noexcept
        {
            delayInSamples = std::max(1, delayInSamples);
            const int readIndex =
                    (writeIndex + delayInSamples) % static_cast<int>(MaxDelaySamples);
            assert(readIndex >= 0 && readIndex < static_cast<int>(MaxDelaySamples));
            return readIndex;
        }

    private:
        InterpolationType interpolator{};
        std::array<StorageType, kStorageLength> buffer{};
    };
}
#endif

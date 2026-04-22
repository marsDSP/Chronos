#pragma once

#ifndef CHRONOS_DELAY_INTERPOLATOR_H
#define CHRONOS_DELAY_INTERPOLATOR_H

// ============================================================================
//  delay_interpolator.h
// ----------------------------------------------------------------------------
//  Family of delay-line interpolator tag types. Every tag exposes the
//  same two-function API so a StaticDelayBuffer (or any other delay-line
//  abstraction) can template against an interpolation type and call it
//  uniformly:
//
//      void  updateInternalVariables (int& delayIntOffset, T& delayFrac);
//      SampleType call (const StorageType* buffer,
//                       int                delayInt,
//                       NumericType        delayFrac,
//                       const SampleType&  filterState = {});
// ============================================================================

namespace MarsDSP::DSP::inline InterpolationTypes
{
    // ------------------------------------------------------------------------
    //  None - zero-order (integer-index) "interpolation". Just reads a sample.
    // ------------------------------------------------------------------------
    struct None
    {
        template <typename T>
        void updateInternalVariables (int& /*delayIntOffset*/, T& /*delayFrac*/) noexcept {}

        template <typename SampleType,
                  typename NumericType  = SampleType,
                  typename StorageType  = SampleType>
        SampleType call (const StorageType* buffer,
                         int                delayInt,
                         NumericType        /*delayFrac*/ = {},
                         const SampleType&  /*state*/    = {}) const noexcept
        {
            return static_cast<SampleType> (buffer[delayInt]);
        }
    };

    // ------------------------------------------------------------------------
    //  Linear - 2-tap linear interpolation between buffer[idx] and buffer[idx+1].
    //  |H(w)| = 1 at DC, 0 at Nyquist. Cheap and guaranteed non-expanding.
    // ------------------------------------------------------------------------
    struct Linear
    {
        template <typename T>
        void updateInternalVariables (int& /*delayIntOffset*/, T& /*delayFrac*/) noexcept {}

        template <typename SampleType,
                  typename NumericType,
                  typename StorageType = SampleType>
        SampleType call (const StorageType* buffer,
                         int                delayInt,
                         NumericType        delayFrac,
                         const SampleType&  /*state*/ = {}) const noexcept
        {
            const auto value1 = static_cast<SampleType> (buffer[delayInt]);
            const auto value2 = static_cast<SampleType> (buffer[delayInt + 1]);
            return value1 + static_cast<SampleType> (delayFrac) * (value2 - value1);
        }
    };

    // ------------------------------------------------------------------------
    //  Lagrange3rd - 4-tap 3rd-order Lagrange interpolation.
    // ------------------------------------------------------------------------
    struct Lagrange3rd
    {
        template <typename T>
        void updateInternalVariables (int& delayIntOffset, T& delayFrac) noexcept
        {
            if (delayIntOffset >= 1)
            {
                ++delayFrac;
                --delayIntOffset;
            }
        }

        template <typename SampleType,
                  typename NumericType,
                  typename StorageType = SampleType>
        SampleType call (const StorageType* buffer,
                         int                delayInt,
                         NumericType        delayFrac,
                         const SampleType&  /*state*/ = {}) const noexcept
        {
            const auto v1 = static_cast<SampleType> (buffer[delayInt]);
            const auto v2 = static_cast<SampleType> (buffer[delayInt + 1]);
            const auto v3 = static_cast<SampleType> (buffer[delayInt + 2]);
            const auto v4 = static_cast<SampleType> (buffer[delayInt + 3]);

            const auto d1 = delayFrac - static_cast<NumericType> (1.0);
            const auto d2 = delayFrac - static_cast<NumericType> (2.0);
            const auto d3 = delayFrac - static_cast<NumericType> (3.0);

            const auto c1 = -d1 * d2 * d3 / static_cast<NumericType> (6.0);
            const auto c2 =  d2 * d3 * static_cast<NumericType> (0.5);
            const auto c3 = -d1 * d3 * static_cast<NumericType> (0.5);
            const auto c4 =  d1 * d2 / static_cast<NumericType> (6.0);

            return v1 * c1
                 + static_cast<SampleType> (delayFrac) * (v2 * c2 + v3 * c3 + v4 * c4);
        }
    };

    // ------------------------------------------------------------------------
    //  Lagrange5th - 6-tap 5th-order Lagrange interpolation.
    // ------------------------------------------------------------------------
    struct Lagrange5th
    {
        template <typename T>
        void updateInternalVariables (int& delayIntOffset, T& delayFrac) noexcept
        {
            if (delayIntOffset >= 2)
            {
                delayFrac      += static_cast<T> (2);
                delayIntOffset -= 2;
            }
        }

        template <typename SampleType,
                  typename NumericType,
                  typename StorageType = SampleType>
        SampleType call (const StorageType* buffer,
                         int                delayInt,
                         NumericType        delayFrac,
                         const SampleType&  /*state*/ = {}) const noexcept
        {
            const auto v1 = static_cast<SampleType> (buffer[delayInt]);
            const auto v2 = static_cast<SampleType> (buffer[delayInt + 1]);
            const auto v3 = static_cast<SampleType> (buffer[delayInt + 2]);
            const auto v4 = static_cast<SampleType> (buffer[delayInt + 3]);
            const auto v5 = static_cast<SampleType> (buffer[delayInt + 4]);
            const auto v6 = static_cast<SampleType> (buffer[delayInt + 5]);

            const auto d1 = delayFrac - static_cast<NumericType> (1.0);
            const auto d2 = delayFrac - static_cast<NumericType> (2.0);
            const auto d3 = delayFrac - static_cast<NumericType> (3.0);
            const auto d4 = delayFrac - static_cast<NumericType> (4.0);
            const auto d5 = delayFrac - static_cast<NumericType> (5.0);

            const auto c1 = -d1 * d2 * d3 * d4 * d5 / static_cast<NumericType> (120.0);
            const auto c2 =  d2 * d3 * d4 * d5      / static_cast<NumericType> ( 24.0);
            const auto c3 = -d1 * d3 * d4 * d5      / static_cast<NumericType> ( 12.0);
            const auto c4 =  d1 * d2 * d4 * d5      / static_cast<NumericType> ( 12.0);
            const auto c5 = -d1 * d2 * d3 * d5      / static_cast<NumericType> ( 24.0);
            const auto c6 =  d1 * d2 * d3 * d4      / static_cast<NumericType> (120.0);

            return v1 * c1
                 + static_cast<SampleType> (delayFrac)
                   * (v2 * c2 + v3 * c3 + v4 * c4 + v5 * c5 + v6 * c6);
        }
    };
}

#endif

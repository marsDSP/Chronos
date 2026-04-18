#pragma once

#ifndef CHRONOS_DELAY_INTERPOLATOR_H
#define CHRONOS_DELAY_INTERPOLATOR_H

namespace MarsDSP::DSP::inline InterpolationTypes
{
    struct None
    {
        template<typename T>
        void write(int&, T&)
        {
        }

        template<typename SampleType, typename NumericType, typename StorageType = SampleType>
        SampleType read(const SampleType *buffer, int delayInt, NumericType = {}, const SampleType& = {})
        {
            return static_cast<SampleType>(buffer[delayInt]);
        }
    };

    struct Linear
    {
        template<typename T>
        void write(int&, T&)
        {
        }

        template<typename SampleType, typename NumericType, typename StorageType = SampleType>
        SampleType read(const StorageType *buffer, int delayInt, NumericType delayFrac, const SampleType& = {})
        {
            auto idx1 = delayInt;
            auto idx2 = idx1 + 1;

            auto val1 = static_cast<SampleType>(buffer[idx1]);
            auto val2 = static_cast<SampleType>(buffer[idx2]);

            return val1 + static_cast<SampleType>(delayFrac) * (val2 - val1);
        }
    };

    struct Lagrange3rd
    {
        template<typename T>
        void write(int& delayIntOffset, T& delayFrac)
        {
            if (delayIntOffset >= 1)
            {
                ++delayFrac;
                delayIntOffset--;
            }
        }

        template<typename SampleType, typename NumericType, typename StorageType = SampleType>
        SampleType read(const StorageType *buffer, int delayInt, NumericType delayFrac, const SampleType& = {})
        {
            auto idx1 = delayInt;
            auto idx2 = idx1 + 1;
            auto idx3 = idx2 + 1;
            auto idx4 = idx3 + 1;

            auto val1 = static_cast<SampleType>(buffer[idx1]);
            auto val2 = static_cast<SampleType>(buffer[idx2]);
            auto val3 = static_cast<SampleType>(buffer[idx3]);
            auto val4 = static_cast<SampleType>(buffer[idx4]);

            auto d1 = delayFrac - static_cast<NumericType>(1.0);
            auto d2 = delayFrac - static_cast<NumericType>(2.0);
            auto d3 = delayFrac - static_cast<NumericType>(3.0);

            auto c1 = -d1 * d2 * d3 / static_cast<NumericType>(6.0);
            auto c2 = d2 * d3 * static_cast<NumericType>(0.5);
            auto c3 = -d1 * d3 * static_cast<NumericType>(0.5);
            auto c4 = d1 * d2 / static_cast<NumericType>(6.0);

            return val1 * c1 + static_cast<SampleType>(delayFrac) * (val2 * c2 + val3 * c3 + val4 * c4);
        }
    };

    struct Lagrange5th
    {
        template<typename T>
        void write(int& delayIntOffset, T& delayFrac)
        {
            if (delayIntOffset >= 2)
            {
                delayFrac += static_cast<T>(2);
                delayIntOffset -= 2;
            }
        }

        template<typename SampleType, typename NumericType, typename StorageType = SampleType>
        SampleType read(const StorageType *buffer, int delayInt, NumericType delayFrac, const SampleType& = {})
        {
            auto idx1 = delayInt;
            auto idx2 = idx1 + 1;
            auto idx3 = idx2 + 1;
            auto idx4 = idx3 + 1;
            auto idx5 = idx4 + 1;
            auto idx6 = idx5 + 1;

            auto val1 = static_cast<SampleType>(buffer[idx1]);
            auto val2 = static_cast<SampleType>(buffer[idx2]);
            auto val3 = static_cast<SampleType>(buffer[idx3]);
            auto val4 = static_cast<SampleType>(buffer[idx4]);
            auto val5 = static_cast<SampleType>(buffer[idx5]);
            auto val6 = static_cast<SampleType>(buffer[idx6]);

            auto d1 = delayFrac - static_cast<NumericType>(1.0);
            auto d2 = delayFrac - static_cast<NumericType>(2.0);
            auto d3 = delayFrac - static_cast<NumericType>(3.0);
            auto d4 = delayFrac - static_cast<NumericType>(4.0);
            auto d5 = delayFrac - static_cast<NumericType>(5.0);

            auto c1 = -d1 * d2 * d3 * d4 * d5 / static_cast<NumericType>(120.0);
            auto c2 = d2 * d3 * d4 * d5 / static_cast<NumericType>(24.0);
            auto c3 = -d1 * d3 * d4 * d5 / static_cast<NumericType>(12.0);
            auto c4 = d1 * d2 * d4 * d5 / static_cast<NumericType>(12.0);
            auto c5 = -d1 * d2 * d3 * d5 / static_cast<NumericType>(24.0);
            auto c6 = d1 * d2 * d3 * d4 / static_cast<NumericType>(120.0);

            return val1 * c1 + static_cast<SampleType>(delayFrac) * (val2 * c2 + val3 * c3 + val4 * c4 + val5 * c5 + val6 * c6);
        }
    };
}
#endif
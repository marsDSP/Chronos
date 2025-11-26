#pragma once

#ifndef CONVERTERS_H
#define CONVERTERS_H

#include <Includes.h>

namespace MarsDSP
{
    // Methods to convert various unit values from one to the other based on user input.
    class Converter
    {
    public:

        Converter() = default;
        ~Converter() = default;

        static juce::String stringFromMilliseconds(float value, int)
        {
            if (value < 10.0f)
                return juce::String(value, 2) + "ms";
            else if (value < 100.0f)
                return juce::String(value, 1) + "ms";
            else if (value < 1000.0f)
                return juce::String(static_cast<int>(value)) + "ms";
            else
                return juce::String(value * 0.001f, 2) + "s";
        }

        static juce::String stringFromHz(float value, int)
        {
            if (value < 1000.0f)
                return juce::String (static_cast<int>(value)) + "Hz";
            else if (value < 10000.0f)
                return juce::String (value / 1000.0f, 2) + "kHz";
            else
                return juce::String (value / 1000.0f, 1) + "kHz";
        }

        // Converts user input from string (keyboard input) to ms/s
        static float millisecondsFromString( const juce::String& text)
        {
            float value = text.getFloatValue();
            if (!text.endsWithIgnoreCase("ms"))
            {
                if (text.endsWithIgnoreCase("s") || value < minDelayTime)
                {
                    return value * 1000.0f;
                }
            }
            return value;
        }

        static float hzFromString(const juce::String& str)
        {
            float value = str.getFloatValue();
            if (value < 20.0f)
            {
                return value * 1000.0f;
            }
            return value;
        }

        static juce::String stringFromDecibels(float value, int)
        {
            return juce::String(value, 1) + "dB";
        }

        static juce::String stringFromPercent(float value, int)
        {
            return juce::String(static_cast<int>(value)) + "%";
        }

    private:

        static constexpr float minDelayTime = 5.0f;
        static constexpr float maxDelayTime = 5000.0f;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Converter)
    };

    class TempoSync
    {
    public:

        std::array<double, 16> noteLengthMultipliers =
        {
            0.125,                  //  0 = 1/32
            0.5 / 3.0,              //  1 = 1/16T
            0.1875,                 //  2 = 1/32.
            0.25,                   //  3 = 1/16
            1.0 / 3.0,              //  4 = 1/8T
            0.375,                  //  5 = 1/16.
            0.5,                    //  6 = 1/8
            2.0 / 3.0,              //  7 = 1/4T
            0.75,                   //  8 = 1/8.
            1.0,                    //  9 = 1/4
            4.0 / 3.0,              //  10 = 1/2T
            1.5,                    //  11 = 1/4.
            2.0,                    //  12 = 1/2
            8.0 / 3.0,              //  13 = 1/1T
            3.0,                    //  14 = 1/2.
            4.0,                    //  15 = 1/1
        };

        void reset() noexcept
        {
            bpm = 120.00;
        }

        void update(const juce::AudioPlayHead* playHead) noexcept
        {
            reset();

            if (playHead == nullptr) { return; }

            const auto opt = playHead->getPosition();

            if (!opt.hasValue()) { return; }

            const auto& pos = *opt;

            if (pos.getBpm().hasValue())
            {
                bpm = *pos.getBpm();
            }
        }

        double getTempo() const noexcept
        {
            return bpm;
        }

        double getMillisecondsFromNoteLength(const int index) const noexcept
        {
            return 60000.0 * noteLengthMultipliers[static_cast<size_t>(index)] / bpm;
        }

    private:

        double bpm { 120.0 };

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TempoSync)
    };
}

#endif
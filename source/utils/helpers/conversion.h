#pragma once

#ifndef CHRONOS_CONVERSION_H
#define CHRONOS_CONVERSION_H

#include <JuceHeader.h>

namespace MarsDSP::inline Utils {

    inline String freqHzToString (float freqHz)
    {
        if (freqHz <= 1000.0f)
            return String (freqHz, 2, false) + "Hz";

        return String (freqHz / 1000.0f, 2, false) + "kHz";
    }

    inline float stringToFreqHz (const String &string)
    {
        auto freqHz = string.getFloatValue();

        if (string.getLastCharacter() == 'k' || string.endsWith("kHz")
                                             || string.endsWith("khz"))
            freqHz *= 1000.0f;

        return freqHz;
    }

    inline String percentToString (float percentage)
    {
        auto percentString = String(static_cast<int>(percentage * 100.0f));

        return percentString + "%";
    }

    inline float stringToPercent (const String &string)
    {
        return string.getFloatValue() / 100.0f;
    }

    inline String gainInDecibelToString (float gainDecibel)
    {
        auto gainString = String(gainDecibel, 2, false);

        return gainString + "dB";
    }

    inline float stringToGainInDecibel(const String &string)
    {
        return string.getFloatValue();
    }

    inline String ratioToString (float ratio)
    {
        auto ratioString = String(ratio, 2, false);

        return ratioString + " : 1";
    }

    inline String timeMsToString (float timeMs)
    {
        if (timeMs < 1000.0f)
            return String(timeMs, 2, false) + " ms";

        auto timeSString = String(timeMs / 1000.0f, 2, false);

        return timeSString + " s";
    }

    inline float stringToTimeMs (const String &string)
    {

        auto timeMs = string.getFloatValue();
        if (string.endsWith(" s") || string.endsWith(" S")
                                  || string.endsWith(" seconds")
                                  || string.endsWith(" Seconds"))
            timeMs *= 1000.0f;

        return timeMs;
    }

    // add one for +/- ct here if necessary

    template <int NumDecimalPlaces>
    String floatToStringDecimal (float floatUnit)
    {
        return { floatUnit, NumDecimalPlaces, false};
    }

    inline String floatToString (float floatUnit)
    {
        return floatToStringDecimal<2>(floatUnit);
    }

    inline float stringToFloat (const String &string)
    {
        return string.getFloatValue();
    }

    inline Colour hexToRGB (const String &hex)
    {
        auto cleanHex = hex.trim();

        if (cleanHex.startsWith ("#"))
            cleanHex = cleanHex.substring (1);

        else if (cleanHex.startsWith ("0x"))
            cleanHex = cleanHex.substring (2);

        if (cleanHex.length() == 6)
            return Colour::fromString ("FF" + cleanHex);

        return Colour::fromString (hex);
    }
}
#endif
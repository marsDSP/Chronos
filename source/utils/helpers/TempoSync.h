#pragma once

#ifndef CHRONOS_TEMPOSYNC_H
#define CHRONOS_TEMPOSYNC_H

#include <array>
#include <string_view>
#include <utility>

namespace MarsDSP::Utils::Helpers::inline TempoSync
{
    enum class TempoSyncInterval : int
    {
        WholeNoteStraight = 0,
        WholeNoteDotted,
        WholeNoteTriplet,

        HalfNoteStraight,
        HalfNoteDotted,
        HalfNoteTriplet,

        QuarterNoteStraight,
        QuarterNoteDotted,
        QuarterNoteTriplet,

        EighthNoteStraight,
        EighthNoteDotted,
        EighthNoteTriplet,

        SixteenthNoteStraight,
        SixteenthNoteDotted,
        SixteenthNoteTriplet,

        ThirtySecondNoteStraight,
        ThirtySecondNoteDotted,
        ThirtySecondNoteTriplet,

        NumberOfIntervals
    };

    inline constexpr std::array
        kNominalNoteValues = {
            0.5, 0.5, 0.5,    // 1/1
            1.0, 1.0, 1.0,    // 1/2
            2.0, 2.0, 2.0,    // 1/4
            4.0, 4.0, 4.0,    // 8th
            8.0, 8.0, 8.0,    // 16th
            16.0, 16.0, 16.0, // 32nd
        };

    inline constexpr std::array
        kStyleMultipliers = {
            1.0, 1.5, 2.0 / 3.0,  // whole
            1.0, 1.5, 2.0 / 3.0,  // half
            1.0, 1.5, 2.0 / 3.0,  // quarter
            1.0, 1.5, 2.0 / 3.0,  // eighth
            1.0, 1.5, 2.0 / 3.0,  // sixteenth
            1.0, 1.5, 2.0 / 3.0,  // thirty-second
        };

    inline constexpr std::array<std::string_view, static_cast<std::size_t>(std::to_underlying(TempoSyncInterval::NumberOfIntervals))>
    kSyncIntervalDisplayLabels = {
            "1/1",   "1/1 .",   "1/1 T",
            "1/2",   "1/2 .",   "1/2 T",
            "1/4",   "1/4 .",   "1/4 T",
            "1/8",   "1/8 .",   "1/8 T",
            "1/16",  "1/16 .",  "1/16 T",
            "1/32",  "1/32 .",  "1/32 T",
        };

    inline double convertTempoSyncIntervalToMilliseconds(TempoSyncInterval tempoSyncInterval, double beatsPerMinute) noexcept
    {
        if (beatsPerMinute <= 0.0)
            return 0.0;

        const auto index = static_cast<std::size_t>(std::to_underlying(tempoSyncInterval));

        if (index >= kNominalNoteValues.size())
            return 0.0;

        // ms per 1/4 at given BPM
        const double millisecondsPerQuarterNote = 60000.0 / beatsPerMinute;

        const double nominalNoteValueHere = kNominalNoteValues[index];
        const double quarterToSelectedRatio = 2.0 / nominalNoteValueHere;

        const double styleFactor = kStyleMultipliers[index];

        return millisecondsPerQuarterNote * quarterToSelectedRatio * styleFactor;
    }

    [[nodiscard]] constexpr int getNumberOfTempoSyncIntervals() noexcept
    {
        return std::to_underlying(TempoSyncInterval::NumberOfIntervals);
    }

    [[nodiscard]] inline std::string_view getTempoSyncIntervalDisplayLabel(TempoSyncInterval tempoSyncInterval) noexcept
    {
        const auto index = static_cast<std::size_t>(std::to_underlying(tempoSyncInterval));
        return index < kSyncIntervalDisplayLabels.size()
                   ? kSyncIntervalDisplayLabels[index]
                   : std::string_view{};
    }
}
#endif
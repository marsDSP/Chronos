#pragma once

#ifndef CHRONOS_TEMPO_SYNC_H
#define CHRONOS_TEMPO_SYNC_H

// ============================================================================
//  tempo_sync.h
// ----------------------------------------------------------------------------
//  Musical-interval -> milliseconds helper for tempo-synced delay. Keeps the
//  enumeration of user-selectable intervals together with the conversion
//  function and a UI label table so the plugin layer can just iterate over
//  kSyncIntervalDisplayLabels to build a ComboBox.
//
//  Chosen scheme: straight / dotted / triplet variants of each nominal
//  note value from whole to 1/32. A "dotted" interval is 1.5x its straight
//  counterpart; a "triplet" is 2/3x. Delay time is measured as the
//  interval between successive beats of the selected note value.
//
//  For a straight quarter note at 120 BPM:
//    period = 60000 ms/min / 120 beats/min = 500 ms.
//  General formula for any nominal value n (where "quarter" corresponds
//  to n = 1):
//    periodMs = (60000 / bpm) * (4 / n) * style
//  with  n     = 0.5 whole, 1 quarter, 2 eighth, 4 sixteenth, 8 thirty2nd
//  and   style = 1.0 straight, 1.5 dotted, 2.0/3.0 triplet.
// ============================================================================

#include <array>
#include <string_view>

namespace MarsDSP::Utils::inline TempoSync
{
    // ----------------------------------------------------------------------------
    //  TempoSyncInterval
    //
    //  Enumerates every musical-division option the UI combobox exposes.
    //  The underlying integer value doubles as an array index into every
    //  parallel table below, so additions must keep the order / entries in
    //  sync.
    // ----------------------------------------------------------------------------
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

    // Nominal note value for each enum entry. Denominator style: 0.5 =
    // whole, 1 = quarter, 2 = eighth, etc.
    inline constexpr std::array
        kNominalNoteValues = {
            // whole group
            0.5, 0.5, 0.5,
            // half group
            1.0, 1.0, 1.0,
            // quarter group
            2.0, 2.0, 2.0,
            // eighth group
            4.0, 4.0, 4.0,
            // sixteenth group
            8.0, 8.0, 8.0,
            // thirty-second group
            16.0, 16.0, 16.0,
        };

    // Style multiplier: 1.0 for straight, 1.5 for dotted, 2/3 for triplet.
    inline constexpr std::array
        kStyleMultipliers = {
            1.0, 1.5, 2.0 / 3.0,  // whole
            1.0, 1.5, 2.0 / 3.0,  // half
            1.0, 1.5, 2.0 / 3.0,  // quarter
            1.0, 1.5, 2.0 / 3.0,  // eighth
            1.0, 1.5, 2.0 / 3.0,  // sixteenth
            1.0, 1.5, 2.0 / 3.0,  // thirty-second
        };

    // UI label table. Index by static_cast<int>(TempoSyncInterval::X).
    // Kept as string_view so there's no runtime construction cost.
    inline constexpr std::array<std::string_view, static_cast<std::size_t>(TempoSyncInterval::NumberOfIntervals)>
        kSyncIntervalDisplayLabels = {
            "1/1",   "1/1 .",   "1/1 T",
            "1/2",   "1/2 .",   "1/2 T",
            "1/4",   "1/4 .",   "1/4 T",
            "1/8",   "1/8 .",   "1/8 T",
            "1/16",  "1/16 .",  "1/16 T",
            "1/32",  "1/32 .",  "1/32 T",
        };

    // ----------------------------------------------------------------------------
    //  convertTempoSyncIntervalToMilliseconds
    //
    //  Given a musical-division selection and a tempo in BPM, return the
    //  corresponding delay period in milliseconds.
    // ----------------------------------------------------------------------------
    inline double convertTempoSyncIntervalToMilliseconds(TempoSyncInterval tempoSyncInterval, double beatsPerMinute) noexcept
    {
        if (beatsPerMinute <= 0.0)
            return 0.0;

        const auto index = static_cast<std::size_t>(tempoSyncInterval);

        if (index >= kNominalNoteValues.size())
            return 0.0;

        // milliseconds per quarter note at the given BPM.
        const double millisecondsPerQuarterNote = 60000.0 / beatsPerMinute;

        // A "quarter note" sits at nominalNoteValue == 2 in our table (the
        // "2.0" entry); a whole note has nominalNoteValue 0.5. The
        // conversion factor is therefore 2 / nominalNoteValue.
        const double nominalNoteValueHere = kNominalNoteValues[index];
        const double quarterToSelectedRatio = 2.0 / nominalNoteValueHere;

        const double styleFactor = kStyleMultipliers[index];

        return millisecondsPerQuarterNote * quarterToSelectedRatio * styleFactor;
    }

    [[nodiscard]] constexpr int getNumberOfTempoSyncIntervals() noexcept
    {
        return static_cast<int>(TempoSyncInterval::NumberOfIntervals);
    }

    [[nodiscard]] inline std::string_view getTempoSyncIntervalDisplayLabel(TempoSyncInterval tempoSyncInterval) noexcept
    {
        const auto index = static_cast<std::size_t>(tempoSyncInterval);
        return index < kSyncIntervalDisplayLabels.size()
                   ? kSyncIntervalDisplayLabels[index]
                   : std::string_view{};
    }
}
#endif

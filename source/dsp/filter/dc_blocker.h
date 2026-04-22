#pragma once

#ifndef CHRONOS_DC_BLOCKER_H
#define CHRONOS_DC_BLOCKER_H

// ============================================================================
//  dc_blocker.h
// ----------------------------------------------------------------------------
//  First-order DC-blocking highpass used as a safety filter in the feedback
//  write-back path. Transfer function:
//
//      H(z) = (1 - z^-1) / (1 - R * z^-1)       (here R = 0.995)
//
//  which is just:
//
//      y[n] = (x[n] - x[n-1]) + R * y[n-1]
//
//  The corner is at ~(1 - R) * fs / (2 pi); with R = 0.995 and fs = 48 kHz
//  that's roughly 38 Hz - still below the fundamentals of normal musical
//  content but fast enough that the filter's own ring-down (~1200 samples
//  to -54 dB) fits inside the delay engine's predicted tail budget so
//  ringoutSamples() stays an accurate silence-prediction.
// ============================================================================

#include <type_traits>

namespace MarsDSP::DSP::Filters
{
    template <typename SampleType = float>
    class DcBlocker
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "DcBlocker requires a floating-point sample type.");

        DcBlocker() = default;

        // ------------------------------------------------------------------
        //  Flush all state. Call from prepareToPlay / host reset so the
        //  first sample after a transport restart doesn't see a stale
        //  y[-1] value.
        // ------------------------------------------------------------------
        void reset() noexcept
        {
            previousInputSample         = SampleType{0};
            previousOutputSample        = SampleType{0};
            firstSampleAfterResetFlag   = true;
        }

        // ------------------------------------------------------------------
        //  Process a single sample. The first sample after a reset gates
        //  the (x - x_{-1}) differencing so we don't produce a spurious
        //  kick from the initial-zero state.
        // ------------------------------------------------------------------
        SampleType processSingleSample(SampleType inputSample) noexcept
        {
            const SampleType inputDifference = firstSampleAfterResetFlag
                                                ? SampleType{0}
                                                : inputSample - previousInputSample;

            // Pole location R. Chosen to place the corner well below any
            // musical content (~38 Hz at 48 kHz) while keeping the filter's
            // impulse-response decay short enough to fit inside the delay
            // engine's ringoutSamples() margin - otherwise a transport stop
            // with feedback = 0 would still show an audible tail from the
            // blocker's own y[n-1] decay.
            constexpr SampleType dcBlockerPoleRadius = static_cast<SampleType>(0.995);

            const SampleType outputSample = inputDifference
                                                + dcBlockerPoleRadius * previousOutputSample;

            previousInputSample       = inputSample;
            previousOutputSample      = outputSample;
            firstSampleAfterResetFlag = false;

            return outputSample;
        }

    private:
        SampleType previousInputSample        {static_cast<SampleType>(0)};
        SampleType previousOutputSample       {static_cast<SampleType>(0)};
        bool       firstSampleAfterResetFlag  {true};
    };
}
#endif

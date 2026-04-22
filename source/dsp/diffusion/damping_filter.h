#pragma once

#ifndef CHRONOS_DAMPING_FILTER_H
#define CHRONOS_DAMPING_FILTER_H

// ============================================================================
//  damping_filter.h
// ----------------------------------------------------------------------------
//  One-pole damping filter used inside each ChronosReverb block. A single
//  filter instance can be run as either a lowpass (for HF damping) or a
//  highpass (for LF damping) depending on which process method the caller
//  invokes.
//
//  The underlying recurrence is the classic single-pole smoother
//
//      a0[n] = a0[n-1] * c + x[n] * (1 - c)
//
//  which, interpreted as a lowpass, has cutoff controlled by 'c' in
//  [0, 1]. In ChronosReverb 'c' is derived from the user's damping percent knob
//  and clamped into [0.01, 0.99] inside the caller's inner loop so the
//  filter stays safely away from the unit circle without degenerating to a
//  pass-through / full-hold extreme.
//
//  The lowpass output is the state variable directly, while the highpass
//  output is recovered as (input - lowpass_output) via the allpass
//  complementary-pair identity. Because the two variants share storage,
//  the same instance cannot be simultaneously used in both modes - callers
//  that need both modes on the same signal instantiate two filters (as
//  ChronosReverb does for HF damping and LF damping in each block).
// ============================================================================

#include <type_traits>

namespace MarsDSP::DSP::ChronosReverb
{
    // ----------------------------------------------------------------------------
    //  ChronosReverbOnePoleDampingFilter<SampleType>
    //
    //  SampleType - floating-point sample type.
    // ----------------------------------------------------------------------------
    template <typename SampleType>
    class ChronosReverbOnePoleDampingFilter
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "ChronosReverbOnePoleDampingFilter requires a floating-point sample type.");

        ChronosReverbOnePoleDampingFilter() noexcept
            : previousLowpassStateSample(static_cast<SampleType>(0)) {}

        // ------------------------------------------------------------------
        //  reset: zero the running state so the next sample emerges purely
        //  from the input rather than from any stale tail.
        // ------------------------------------------------------------------
        void reset() noexcept
        {
            previousLowpassStateSample = static_cast<SampleType>(0);
        }

        // ------------------------------------------------------------------
        //  processSingleSampleAsLowpass: run one sample as a lowpass.
        //
        //  'lowpassMemoryCoefficient' in [0, 1) sets how much of the
        //  previous output state is blended with the new input sample.
        //  0 → pass-through, 1 → infinite hold. The caller clamps this
        //  into [0.01, 0.99] before passing it in.
        // ------------------------------------------------------------------
        SampleType processSingleSampleAsLowpass(SampleType inputSample,
                                                SampleType lowpassMemoryCoefficient) noexcept
        {
            previousLowpassStateSample =
                previousLowpassStateSample * lowpassMemoryCoefficient
                + inputSample * (static_cast<SampleType>(1) - lowpassMemoryCoefficient);
            return previousLowpassStateSample;
        }

        // ------------------------------------------------------------------
        //  processSingleSampleAsHighpass: run one sample as a highpass.
        //
        //  Derivation: the complementary-pair identity for a first-order
        //  filter says lowpass(x) + highpass(x) = x, so we can compute the
        //  HP output as x - lowpass(x). Here we run the smoother with
        //  (1 - c) swapped with c compared to the lowpass call so a
        //  single "0%..100%" user knob range sweeps the HP cutoff
        //  monotonically in the same direction the LP variant does with
        //  its own knob.
        // ------------------------------------------------------------------
        SampleType processSingleSampleAsHighpass(SampleType inputSample,
                                                 SampleType highpassMemoryCoefficient) noexcept
        {
            previousLowpassStateSample =
                previousLowpassStateSample
                    * (static_cast<SampleType>(1) - highpassMemoryCoefficient)
                + inputSample * highpassMemoryCoefficient;
            return inputSample - previousLowpassStateSample;
        }

    private:
        // Running state. Used as the lowpass output directly, or as the
        // lowpass-complement when run in highpass mode.
        SampleType previousLowpassStateSample;
    };
}
#endif

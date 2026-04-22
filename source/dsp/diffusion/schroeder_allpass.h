#pragma once

#ifndef CHRONOS_SCHROEDER_ALLPASS_H
#define CHRONOS_SCHROEDER_ALLPASS_H

// ============================================================================
//  schroeder_allpass.h
// ----------------------------------------------------------------------------
//  Single-input / single-output Schroeder-style allpass section built
//  around an integer-length ring buffer. It is used twice in the
//  ChronosReverb topology:
//
//    * four of them are chained in series at the network's input to
//      spread transients ("input diffusers")
//    * two of them live inside each of the four parallel reverb blocks to
//      thicken the tail ("loop allpasses" driven by the "buildup" knob).
//
//  The transfer function is
//
//      H(z) = ( -coefficient + z^-N ) / ( 1 - coefficient * z^-N )
//
//  which has unit magnitude at every frequency: the section only rotates
//  phase without changing spectrum. In practice the feedback multiplier
//  'coefficient' must be held strictly below 1 so the single-pole
//  resonance at z^-N doesn't become a sustained ringing. The caller
//  clamps the effective value via a 0.7 scaling of the user's
//  "diffusion" / "buildup" knob.
//
//  The underlying storage is one plain float array whose length is set at
//  prepare() time, plus an integer write index. We keep the full
//  MAX_RING_BUFFER_CAPACITY of floats live per instance - trading memory
//  for a branch-free process() that never touches the heap.
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace MarsDSP::DSP::ChronosReverb
{
    // ----------------------------------------------------------------------------
    //  ChronosReverbSchroederAllpassDelayLine<SampleType, MaxRingBufferCapacity>
    //
    //  SampleType                - floating-point sample type (float / double)
    //  MaxRingBufferCapacity     - upper bound on the allpass delay length
    //                              in samples. Must accommodate the
    //                              longest per-channel length the
    //                              ChronosReverbStereoProcessor ever
    //                              sets with its size-scale ceiling on
    //                              a 48 kHz host (≈ 170 ms worth of
    //                              samples at the default ceiling).
    // ----------------------------------------------------------------------------
    template <typename SampleType,
              std::size_t MaxRingBufferCapacity = (1u << 17)>  // 131072
    class ChronosReverbSchroederAllpassDelayLine
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "ChronosReverbSchroederAllpassDelayLine requires a floating-point sample type.");
        static_assert(MaxRingBufferCapacity > 0,
                      "ChronosReverbSchroederAllpassDelayLine requires a non-zero capacity.");

        static constexpr std::size_t kMaxRingBufferCapacity = MaxRingBufferCapacity;

        ChronosReverbSchroederAllpassDelayLine() noexcept
        {
            reset();
            currentLengthInSamples   = 1;
            currentRingBufferCursor  = 0;
        }

        // ------------------------------------------------------------------
        //  reset: zero every stored sample and return the cursor to the
        //  start of the ring buffer. Used on transport restart, plugin
        //  bypass and whenever stale state should be flushed.
        // ------------------------------------------------------------------
        void reset() noexcept
        {
            std::memset(ringBuffer.data(),
                        0,
                        MaxRingBufferCapacity * sizeof(SampleType));
            currentRingBufferCursor = 0;
        }

        // ------------------------------------------------------------------
        //  setLengthInSamples: change the effective tap position of the
        //  allpass. Must be strictly less than MaxRingBufferCapacity so
        //  the mask-free compare-and-wrap logic stays in bounds; we
        //  clamp defensively in case a caller derives a length from a
        //  parameter that hasn't been range-checked yet.
        //
        //  Changing the length does NOT clear the ring. A gently changing
        //  length produces a mild pitch/phase artifact as old samples
        //  continue to recirculate under a new tap time; holding the
        //  coefficient reasonably small keeps the artifact inaudible.
        // ------------------------------------------------------------------
        void setLengthInSamples(int newLengthInSamples) noexcept
        {
            currentLengthInSamples = std::clamp(newLengthInSamples,
                                                1,
                                                static_cast<int>(MaxRingBufferCapacity) - 1);
        }

        [[nodiscard]] int getLengthInSamples() const noexcept
        {
            return currentLengthInSamples;
        }

        // ------------------------------------------------------------------
        //  processSingleSample: run one sample through the section.
        //
        //  The canonical Schroeder allpass recurrence is
        //
        //      v[n]   = x[n] - coefficient * v[n-N]
        //      y[n]   = v[n-N] + coefficient * v[n]
        //
        //  where v is the internal state that is stored in the delay line.
        //  We read the oldest stored value, compute the new state
        //  `delayInput = x - coeff * oldest`, then emit
        //  `oldest + coeff * delayInput` and write `delayInput` into the
        //  slot we just read. The section advances its cursor by one
        //  sample before every read so the "slot we just read" is the
        //  cell we're about to overwrite, which lets the whole operation
        //  share a single indexed access per sample.
        //
        //  'feedbackCoefficient' is the section's internal gain. Stable
        //  operation requires |coefficient| < 1. The caller (the
        //  ChronosReverbStereoProcessor) multiplies the user's
        //  "diffusion" / "buildup" percent knobs by 0.7 to keep
        //  coefficients well inside the unit circle.
        // ------------------------------------------------------------------
        SampleType processSingleSample(SampleType inputSample,
                                       SampleType feedbackCoefficient) noexcept
        {
            // Advance the cursor first. If we're about to step past the
            // section's active length, wrap to zero.
            ++currentRingBufferCursor;
            if (currentRingBufferCursor >= currentLengthInSamples)
                currentRingBufferCursor = 0;

            // Load the oldest sample currently stored at the cursor. This
            // represents v[n-N] in the canonical recurrence.
            const SampleType oldestStoredSample =
                ringBuffer[static_cast<std::size_t>(currentRingBufferCursor)];

            // Compute the new state value that will enter the delay line.
            // This is v[n] in the canonical form.
            const SampleType newStateSampleEnteringDelay =
                inputSample - feedbackCoefficient * oldestStoredSample;

            // Compute the emitted output sample. This is y[n] in the
            // canonical form.
            const SampleType outputSample =
                oldestStoredSample + feedbackCoefficient * newStateSampleEnteringDelay;

            // Write the new state back into the slot we just read, so
            // N samples from now it will be produced again as oldest.
            ringBuffer[static_cast<std::size_t>(currentRingBufferCursor)] =
                newStateSampleEnteringDelay;

            return outputSample;
        }

    private:
        // The float array is exactly MaxRingBufferCapacity long and is
        // allocated as a class member rather than a heap buffer, which
        // trades memory for a branch-free process() that never touches
        // the heap in the audio thread.
        std::array<SampleType, MaxRingBufferCapacity> ringBuffer{};

        // The section's currently-active delay length, in samples. Must be
        // kept strictly less than MaxRingBufferCapacity.
        int currentLengthInSamples{1};

        // Running cursor into 'ringBuffer'. Always advanced one step before
        // every read/write so the oldest stored sample is always the cell
        // at 'currentRingBufferCursor' after the pre-increment.
        int currentRingBufferCursor{0};
    };
}
#endif

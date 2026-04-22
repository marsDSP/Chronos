#pragma once

#ifndef CHRONOS_PREDELAY_BUFFER_H
#define CHRONOS_PREDELAY_BUFFER_H

// ============================================================================
//  predelay_buffer.h
// ----------------------------------------------------------------------------
//  Predelay ring used at the ChronosReverb network's input. Provides a
//  long, single-channel integer-delay line that lets the user add up to a
//  few hundred milliseconds of dry -> wet offset between the source and
//  the start of the reverb build-up.
//
//  The buffer is a plain wrap-around linear array with a running write
//  cursor. Unlike the Schroeder allpasses inside the loop, the predelay
//  never has to read at multiple tap offsets simultaneously, so the class
//  exposes a single process(inputSample, tapInSamples) method that writes
//  the new sample and returns the sample that is currently 'tapInSamples'
//  old.
//
//  The capacity is template-configurable so a host running at 384 kHz with
//  a user-set max of 1.0 s predelay can ask for a 384000-sample buffer.
//  In practice ChronosReverb clamps the user's "predelay" seconds
//  parameter so the tap never exceeds 3/4 of the allocation (see
//  kMaxLegalTapInSamples below).
// ============================================================================

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace MarsDSP::DSP::ChronosReverb
{
    // ----------------------------------------------------------------------------
    //  ChronosReverbPredelayCircularBuffer<SampleType, MaxBufferCapacity>
    //
    //  SampleType          - floating-point sample type.
    //  MaxBufferCapacity   - upper bound on the number of samples of
    //                        predelay this instance can store. The
    //                        default 48000 * 8 * 4 = 1,536,000 covers
    //                        384 kHz hosts with headroom.
    // ----------------------------------------------------------------------------
    template <typename SampleType,
              std::size_t MaxBufferCapacity = (48000u * 8u * 4u)>
    class ChronosReverbPredelayCircularBuffer
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "ChronosReverbPredelayCircularBuffer requires a floating-point sample type.");
        static_assert(MaxBufferCapacity > 0,
                      "ChronosReverbPredelayCircularBuffer requires a non-zero capacity.");

        // Leave a 25% safety margin between the maximum valid predelay
        // tap and the total buffer length. Exposing this here means
        // callers can query the effective "maximum tap in samples"
        // rather than re-deriving it from the template parameter.
        static constexpr std::size_t kMaxBufferCapacity     = MaxBufferCapacity;
        static constexpr std::size_t kMaxLegalTapInSamples  = MaxBufferCapacity * 3u / 4u;

        ChronosReverbPredelayCircularBuffer() noexcept
        {
            reset();
        }

        // ------------------------------------------------------------------
        //  reset: zero every stored sample and return the cursor to the
        //  start of the buffer.
        // ------------------------------------------------------------------
        void reset() noexcept
        {
            std::memset(circularBuffer.data(),
                        0,
                        MaxBufferCapacity * sizeof(SampleType));
            writeCursor = 0;
        }

        // ------------------------------------------------------------------
        //  processSingleSample: advance the write cursor, read the sample
        //  that is 'tapInSamples' old, and write the new input sample into
        //  the just-advanced slot.
        //
        //  'tapInSamples' must be in [1, kMaxLegalTapInSamples]. The caller
        //  should already have clamped it; we clamp defensively anyway so
        //  a bad input can never index outside the buffer.
        // ------------------------------------------------------------------
        SampleType processSingleSample(SampleType inputSample,
                                       int        tapInSamples) noexcept
        {
            // Advance the running cursor by one sample. We wrap via a
            // compare-and-zero rather than a mask because MaxBufferCapacity
            // is not constrained to be a power of two.
            ++writeCursor;
            if (writeCursor == static_cast<int>(MaxBufferCapacity))
                writeCursor = 0;

            // Resolve the tap: we want the sample 'tapInSamples' ago, i.e.
            // at absolute slot (writeCursor - tapInSamples). Clamp the tap
            // into a safe range in case the caller passes 0 / negative /
            // over-capacity values, and add MaxBufferCapacity before the
            // wrap to keep the subtraction non-negative.
            const int clampedTap = std::clamp(tapInSamples,
                                              1,
                                              static_cast<int>(kMaxLegalTapInSamples));
            int tapSlot = writeCursor - clampedTap;
            while (tapSlot < 0)
                tapSlot += static_cast<int>(MaxBufferCapacity);

            const SampleType delayedSample =
                circularBuffer[static_cast<std::size_t>(tapSlot)];

            // Overwrite the just-advanced slot with the incoming sample
            // so it becomes available to future reads.
            circularBuffer[static_cast<std::size_t>(writeCursor)] = inputSample;

            return delayedSample;
        }

    private:
        // Ring-buffer storage. Allocated on-object so the audio thread
        // never has to touch the heap.
        std::array<SampleType, MaxBufferCapacity> circularBuffer{};

        // Running write cursor in [0, MaxBufferCapacity).
        int writeCursor{0};
    };
}
#endif

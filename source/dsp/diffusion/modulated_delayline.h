#pragma once

#ifndef CHRONOS_MODULATED_DELAY_LINE_H
#define CHRONOS_MODULATED_DELAY_LINE_H

// ============================================================================
//  modulated_delayline.h
// ----------------------------------------------------------------------------
//  LFO-modulated dual-tap delay line. Each of the four reverb blocks in
//  the ChronosReverb topology owns one of these.
//
//  Per per-sample tick this class:
//    1) writes the input sample into a length-power-of-two ring buffer,
//    2) reads two fixed-tap offsets (one steers the left stereo output,
//       the other steers the right), returned via reference parameters,
//    3) reads a third, LFO-modulated tap at the main loop delay length
//       (plus a small LFO offset) and returns it as the recirculating
//       feedback sample that continues around the loop.
//
//  The LFO offset is expressed in fixed-point Q8.8 samples - a 24-bit
//  signed integer whose low 8 bits encode the fractional sample. Keeping
//  the offset in fixed point lets the hot inner loop stay in integer math
//  except for the final linear interpolation.
//
//  The ring buffer is always the same power-of-two length given by the
//  MaxDelayCapacityInSamples template parameter, so wrap-around can use a
//  single bitmask. The actual active delay length is set dynamically by
//  setLoopDelayLengthInSamples() and must be strictly less than the
//  template capacity.
// ============================================================================

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace MarsDSP::DSP::ChronosReverb
{
    // ----------------------------------------------------------------------------
    //  ChronosReverbStereoTappedModulatedDelayLine<SampleType, MaxDelayCapacityInSamples>
    //
    //  SampleType                 - floating-point sample type.
    //  MaxDelayCapacityInSamples  - power-of-two length of the ring buffer,
    //                               in samples. Must be >= the longest
    //                               delay time the
    //                               ChronosReverbStereoProcessor ever
    //                               configures.
    // ----------------------------------------------------------------------------
    template <typename SampleType,
              std::size_t MaxDelayCapacityInSamples = (1u << 17)>
    class ChronosReverbStereoTappedModulatedDelayLine
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "ChronosReverbStereoTappedModulatedDelayLine requires a floating-point sample type.");
        static_assert(MaxDelayCapacityInSamples > 0
                      && (MaxDelayCapacityInSamples & (MaxDelayCapacityInSamples - 1u)) == 0u,
                      "ChronosReverbStereoTappedModulatedDelayLine requires MaxDelayCapacityInSamples "
                      "to be a power of two so the ring buffer can wrap with a bitmask.");

        // ------------------------------------------------------------------
        //  Fixed-point resolution of the modulation tap offset:
        //  8 fractional bits per whole sample, so modulation_q8.8 = 256
        //  means "one sample forward".
        // ------------------------------------------------------------------
        static constexpr int kModulationSubSampleBits     = 8;
        static constexpr int kModulationSubSampleRange    = 1 << kModulationSubSampleBits;
        static constexpr int kModulationSubSampleBitMask  = kModulationSubSampleRange - 1;

        static constexpr std::size_t kMaxDelayCapacityInSamples = MaxDelayCapacityInSamples;
        static constexpr std::size_t kRingBufferBitMask         = MaxDelayCapacityInSamples - 1u;

        ChronosReverbStereoTappedModulatedDelayLine() noexcept
        {
            reset();
            activeLoopDelayLengthInSamples = 1;
        }

        // ------------------------------------------------------------------
        //  reset: zero every stored sample and return the write cursor to
        //  the start of the ring buffer.
        // ------------------------------------------------------------------
        void reset() noexcept
        {
            std::memset(ringBuffer.data(),
                        0,
                        MaxDelayCapacityInSamples * sizeof(SampleType));
            writeCursor = 0;
        }

        // ------------------------------------------------------------------
        //  setLoopDelayLengthInSamples: update the length of the main
        //  recirculating tap. This does not affect the left / right tap
        //  positions, which are addressed absolutely per-call.
        // ------------------------------------------------------------------
        void setLoopDelayLengthInSamples(int newLoopDelayInSamples) noexcept
        {
            activeLoopDelayLengthInSamples =
                std::clamp(newLoopDelayInSamples,
                           1,
                           static_cast<int>(MaxDelayCapacityInSamples) - 1);
        }

        [[nodiscard]] int getLoopDelayLengthInSamples() const noexcept
        {
            return activeLoopDelayLengthInSamples;
        }

        // ------------------------------------------------------------------
        //  processSingleSample: run one sample through the delay line.
        //
        //  Parameters:
        //    inputSample                - the sample to be written into
        //                                 the line this step
        //    leftStereoTapInSamples     - integer offset where the left
        //                                 output tap should read
        //    leftStereoTapOutputSample  - reference, receives the value
        //                                 read at leftStereoTap
        //    rightStereoTapInSamples    - integer offset where the right
        //                                 output tap should read
        //    rightStereoTapOutputSample - reference, receives the value
        //                                 read at rightStereoTap
        //    modulationOffsetInQ8Samples - Q8.8 signed offset applied on
        //                                  top of the main loop delay
        //                                  length so the recirculating
        //                                  feedback tap is LFO-modulated.
        //                                  The sign determines whether
        //                                  the tap moves deeper or
        //                                  shallower in time.
        //
        //  Returns the sample value at the LFO-modulated loop tap; this is
        //  the signal that the caller feeds back into the next reverb
        //  block in the chain.
        //
        //  Implementation uses linear interpolation between the two
        //  integer slots bracketing the modulated tap, weighted by the
        //  fractional part of the Q8 offset.
        // ------------------------------------------------------------------
        SampleType processSingleSample(SampleType  inputSample,
                                       int         leftStereoTapInSamples,
                                       SampleType& leftStereoTapOutputSample,
                                       int         rightStereoTapInSamples,
                                       SampleType& rightStereoTapOutputSample,
                                       int         modulationOffsetInQ8Samples) noexcept
        {
            // Advance write cursor (and mask for wrap) before any read,
            // so the slot we're about to overwrite is always the oldest.
            writeCursor = (writeCursor + 1) & static_cast<int>(kRingBufferBitMask);

            // Stereo taps: pure integer reads.
            const int leftTapSlot =
                (writeCursor - leftStereoTapInSamples) & static_cast<int>(kRingBufferBitMask);
            const int rightTapSlot =
                (writeCursor - rightStereoTapInSamples) & static_cast<int>(kRingBufferBitMask);

            leftStereoTapOutputSample  = ringBuffer[static_cast<std::size_t>(leftTapSlot)];
            rightStereoTapOutputSample = ringBuffer[static_cast<std::size_t>(rightTapSlot)];

            // LFO-modulated loop tap, fractional. Split the Q8 offset
            // into integer + fractional parts:
            //   integerOffset = modulation >> 8   (signed arithmetic shift)
            //   fractionalNumerator = modulation & 0xff
            //   interpWeight1 = fractionalNumerator / 256
            //   interpWeight2 = (256 - fractionalNumerator) / 256
            //
            // Slot for sample "at loop length + integerOffset + 1":
            //   primary = (writeCursor - activeLoopDelayLength
            //              + integerOffset + 1) & mask
            // Slot for sample "at loop length + integerOffset":
            //   secondary = (writeCursor - activeLoopDelayLength
            //                + integerOffset)     & mask
            //
            // Linearly interpolate between the two using
            // fractionalNumerator's weight on primary and its complement
            // on secondary.
            const int modulationIntegerOffsetInSamples =
                modulationOffsetInQ8Samples >> kModulationSubSampleBits;
            const int modulationFractionalNumerator =
                modulationOffsetInQ8Samples & kModulationSubSampleBitMask;
            const int modulationFractionalComplement =
                kModulationSubSampleRange - modulationFractionalNumerator;

            const int primaryModulatedTapSlot =
                (writeCursor - activeLoopDelayLengthInSamples
                 + modulationIntegerOffsetInSamples + 1)
                & static_cast<int>(kRingBufferBitMask);
            const int secondaryModulatedTapSlot =
                (writeCursor - activeLoopDelayLengthInSamples
                 + modulationIntegerOffsetInSamples)
                & static_cast<int>(kRingBufferBitMask);

            const SampleType primarySample =
                ringBuffer[static_cast<std::size_t>(primaryModulatedTapSlot)];
            const SampleType secondarySample =
                ringBuffer[static_cast<std::size_t>(secondaryModulatedTapSlot)];

            const SampleType fractionalReconstructionMultiplier =
                static_cast<SampleType>(1)
                / static_cast<SampleType>(kModulationSubSampleRange);

            const SampleType interpolatedLoopTapSample =
                (primarySample   * static_cast<SampleType>(modulationFractionalNumerator)
                 + secondarySample * static_cast<SampleType>(modulationFractionalComplement))
                * fractionalReconstructionMultiplier;

            // Commit the incoming sample into the just-advanced slot so
            // future reads can see it.
            ringBuffer[static_cast<std::size_t>(writeCursor)] = inputSample;

            return interpolatedLoopTapSample;
        }

    private:
        // Storage. Allocated on-object so the audio thread never has to
        // touch the heap.
        std::array<SampleType, MaxDelayCapacityInSamples> ringBuffer{};

        // Running write cursor in [0, MaxDelayCapacityInSamples).
        int writeCursor{0};

        // Active length of the main recirculating loop tap, in whole
        // samples. The modulation offset is added on top of this.
        int activeLoopDelayLengthInSamples{1};
    };
}
#endif

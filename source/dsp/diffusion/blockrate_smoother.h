#pragma once

#ifndef CHRONOS_BLOCK_RATE_LINEAR_SMOOTHER_H
#define CHRONOS_BLOCK_RATE_LINEAR_SMOOTHER_H

// ============================================================================
//  blockrate_smoother.h
// ----------------------------------------------------------------------------
//  Used to ramp the decay multiplier, diffusion, buildup, damping,
//  and modulation parameters from the previous block's value to the
//  current block's value over the duration of one processing block.
//
//      ChronosReverbBlockRateLinearSmoother<float> diffusionSmoother;
//      diffusionSmoother.prepareForBlockOfSamples(blockSize);     // once
//
//      // every audio block:
//      diffusionSmoother.setTargetValue(diffusionKnobValue);
//      for (int n = 0; n < blockSize; ++n)
//      {
//          useValue(diffusionSmoother.currentValue);
//          diffusionSmoother.advanceByOneSample();
//      }
//
//  The smoother exposes 'currentValue' as a plain public member because
//  the inner ChronosReverb loop reads it on every sample - inlining this
//  read matters for the tight per-sample hot loop so we don't wrap it in
//  a getter.
//
//  First-run behaviour: until setTargetValue() is called for the first
//  time, currentValue is 0 and the smoother stays at 0. The default-
//  constructed state initialises v to 0 and only snaps on snapToValue().
// ============================================================================

#include <algorithm>
#include <type_traits>

namespace MarsDSP::DSP::ChronosReverb
{
    // ----------------------------------------------------------------------------
    //  ChronosReverbBlockRateLinearSmoother<SampleType>
    // ----------------------------------------------------------------------------
    template <typename SampleType>
    class ChronosReverbBlockRateLinearSmoother
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "ChronosReverbBlockRateLinearSmoother requires a floating-point sample type.");

        // ------------------------------------------------------------------
        //  The instantaneous smoother output, updated by
        //  advanceByOneSample() once per audio sample. Kept public for
        //  hot-loop readability so the inner ChronosReverb loop can read
        //  it without going through a getter.
        // ------------------------------------------------------------------
        SampleType currentValue { static_cast<SampleType>(0) };

        // ------------------------------------------------------------------
        //  prepareForBlockOfSamples: tell the smoother how many samples of
        //  ramp to produce between successive setTargetValue() calls. Must
        //  be called at least once before setTargetValue(), and re-called
        //  any time the host's block size changes.
        // ------------------------------------------------------------------
        void prepareForBlockOfSamples(int blockSizeInSamples) noexcept
        {
            cachedBlockSizeInSamples         = std::max(1, blockSizeInSamples);
            cachedInverseBlockSizeInSamples  =
                static_cast<SampleType>(1) / static_cast<SampleType>(cachedBlockSizeInSamples);
            perSampleIncrement               = static_cast<SampleType>(0);
        }

        // ------------------------------------------------------------------
        //  snapToValue: jump currentValue to 'newValue' immediately and
        //  cancel any ramp that might have been in flight. Used at
        //  prepare() / reset() so the first block doesn't fade from 0.
        // ------------------------------------------------------------------
        void snapToValue(SampleType newValue) noexcept
        {
            currentValue       = newValue;
            cachedTargetValue  = newValue;
            perSampleIncrement = static_cast<SampleType>(0);
        }

        // ------------------------------------------------------------------
        //  setTargetValue: announce the value the smoother should have
        //  reached by the end of the block. Produces a straight linear
        //  ramp from the current value to the target, distributed evenly
        //  over cachedBlockSizeInSamples per-sample increments.
        // ------------------------------------------------------------------
        void setTargetValue(SampleType newTargetValue) noexcept
        {
            cachedTargetValue  = newTargetValue;
            perSampleIncrement = (newTargetValue - currentValue) * cachedInverseBlockSizeInSamples;
        }

        // ------------------------------------------------------------------
        //  advanceByOneSample: called once per audio sample in the inner
        //  loop. Adds the precomputed per-sample increment to currentValue.
        //  At the end of the block, currentValue == cachedTargetValue.
        // ------------------------------------------------------------------
        void advanceByOneSample() noexcept
        {
            currentValue += perSampleIncrement;
        }

        [[nodiscard]] SampleType getTargetValue() const noexcept { return cachedTargetValue; }

    private:
        SampleType cachedTargetValue             { static_cast<SampleType>(0) };
        SampleType perSampleIncrement            { static_cast<SampleType>(0) };
        SampleType cachedInverseBlockSizeInSamples { static_cast<SampleType>(1) };
        int        cachedBlockSizeInSamples      { 1 };
    };
}
#endif

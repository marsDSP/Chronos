#pragma once

#ifndef CHRONOS_CROSSFADE_BYPASS_ENGINE_H
#define CHRONOS_CROSSFADE_BYPASS_ENGINE_H

// ============================================================================
//  crossfade_bypass_engine.h
// ----------------------------------------------------------------------------
//  Click-free bypass helper for any block-based processor in Chronos. Sits
//  at the top and bottom of a processor's block-process routine and, when
//  its "is active" flag transitions (active -> bypassed or bypassed ->
//  active), performs a linear crossfade between the unprocessed input and
//  the processed output over a single block. Outside transitions it is a
//  branch-free pass-through (the caller either processes normally or
//  skips processing entirely).
//
//  Usage pattern:
//
//      const bool processorShouldRunThisBlock =
//          bypassEngine.captureDryInputAndDecideWhetherToProcess(
//              ioView, processorIsActive);
//
//      if (processorShouldRunThisBlock)
//          // ... run the wrapped processor over ioView here ...
//
//      bypassEngine.crossfadeWithCapturedDryInputIfTransitioning(
//          ioView, processorIsActive);
// ============================================================================

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "dsp/buffers/aligned_simd_buffer.h"
#include "dsp/buffers/aligned_simd_buffer_view.h"

namespace MarsDSP::DSP::Bypass
{
    // ----------------------------------------------------------------------------
    //  CrossfadeBypassEngine<SampleType>
    //
    //  One instance per wrapped processor. The caller drives it with a
    //  boolean "active" flag each block; transitions between true and false
    //  cause a one-block linear crossfade between captured dry input and
    //  processed output. Stable state (no transition) is a zero-cost
    //  pass-through.
    // ----------------------------------------------------------------------------
    template <typename SampleType = float>
    class CrossfadeBypassEngine
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "CrossfadeBypassEngine requires a floating-point sample type.");

        using AlignedBufferType = AlignedBuffers::AlignedSIMDBuffer<SampleType>;
        using AlignedBufferViewType = AlignedBuffers::AlignedSIMDBufferView<SampleType>;

        CrossfadeBypassEngine() = default;

        // ------------------------------------------------------------------
        //  Allocate the fade buffer and latch the initial "active" state.
        //  Must be called on prepareToPlay() before any audio flows.
        // ------------------------------------------------------------------
        void prepare(int         maximumBlockSizeInSamples,
                     int         numberOfChannels,
                     bool        initialProcessorIsActiveState) noexcept
        {
            previousProcessorIsActiveState = initialProcessorIsActiveState;
            fadeBuffer.setMaxSize(numberOfChannels, maximumBlockSizeInSamples);
            fadeBuffer.setCurrentSize(numberOfChannels, maximumBlockSizeInSamples);
            fadeBuffer.clearAllSamples();
        }

        // Reset latched state without touching the allocation. Typical call
        // site is the host's own reset() / transport stop path.
        void reset(bool currentProcessorIsActiveState) noexcept
        {
            previousProcessorIsActiveState = currentProcessorIsActiveState;
            fadeBuffer.clearAllSamples();
        }

        // ------------------------------------------------------------------
        //  Capture the dry input to the fade buffer if the active-state flag
        //  just changed, and tell the caller whether the wrapped processor
        //  should actually run this block. Returns:
        //    * false - no need to run the processor (we've been bypassed for
        //      at least two consecutive blocks with no state change).
        //    * true  - either the processor is currently active, or we're
        //      transitioning and the caller needs to produce a processed
        //      output so the crossfade has something to fade against.
        // ------------------------------------------------------------------
        [[nodiscard]] bool captureDryInputAndDecideWhetherToProcess(
            const AlignedBufferViewType& ioBlockView,
            bool                         currentProcessorIsActiveState) noexcept
        {
            // Steady-state bypass: both previous and current blocks are
            // inactive -> skip the processor entirely.
            if (!currentProcessorIsActiveState && !previousProcessorIsActiveState)
            {
                return false;
            }

            // Transition block: stash a snapshot of the dry input so the
            // output crossfade has a reference.
            if (currentProcessorIsActiveState != previousProcessorIsActiveState)
            {
                copyInputBlockIntoFadeBuffer(ioBlockView);
            }

            return true;
        }

        // ------------------------------------------------------------------
        //  If a transition happened this block, crossfade the processed
        //  output with the dry input snapshot captured in
        //  captureDryInputAndDecideWhetherToProcess(). Otherwise this call
        //  is a zero-cost no-op.
        // ------------------------------------------------------------------
        void crossfadeWithCapturedDryInputIfTransitioning(
            const AlignedBufferViewType& ioBlockView,
            bool                         currentProcessorIsActiveState) noexcept
        {
            if (currentProcessorIsActiveState == previousProcessorIsActiveState)
                return;

            const int numberOfChannels = ioBlockView.getNumChannels();
            const int numberOfSamples  = ioBlockView.getNumSamples();

            if (numberOfSamples <= 0)
                return;

            // Gain schedule: when turning ON (active now, bypassed before)
            // we fade FROM dry INTO processed, so startGain = 0, endGain = 1.
            // When turning OFF (bypassed now, active before) we fade FROM
            // processed INTO dry, so startGain = 1, endGain = 0.
            const SampleType gainAppliedToProcessedAtBlockStart = currentProcessorIsActiveState
                                                                    ? static_cast<SampleType>(0)
                                                                    : static_cast<SampleType>(1);
            const SampleType gainAppliedToProcessedAtBlockEnd = static_cast<SampleType>(1) - gainAppliedToProcessedAtBlockStart;

            const SampleType perSampleGainIncrement = (gainAppliedToProcessedAtBlockEnd - gainAppliedToProcessedAtBlockStart)
                                                        / static_cast<SampleType>(numberOfSamples);

            for (int channelIndex = 0;
                 channelIndex < numberOfChannels;
                 ++channelIndex)
            {
                SampleType* const processedChannelPointer = ioBlockView.getChannelPointer(channelIndex);
                const SampleType* const dryChannelPointer = fadeBuffer.getReadPointer(channelIndex);

                if (processedChannelPointer == nullptr || dryChannelPointer == nullptr)
                    continue;

                SampleType processedGainRamp = gainAppliedToProcessedAtBlockStart;
                for (int sampleIndex = 0;
                     sampleIndex < numberOfSamples;
                     ++sampleIndex)
                {
                    const SampleType dryGainRamp = static_cast<SampleType>(1) - processedGainRamp;

                    processedChannelPointer[sampleIndex] =
                        processedChannelPointer[sampleIndex] * processedGainRamp
                      + dryChannelPointer[sampleIndex]       * dryGainRamp;

                    processedGainRamp += perSampleGainIncrement;
                }
            }

            previousProcessorIsActiveState = currentProcessorIsActiveState;
        }

        [[nodiscard]] bool getPreviousProcessorIsActiveState() const noexcept
        {
            return previousProcessorIsActiveState;
        }

    private:
        void copyInputBlockIntoFadeBuffer(const AlignedBufferViewType& ioBlockView) noexcept
        {
            const int numberOfChannels = ioBlockView.getNumChannels();
            const int numberOfSamples  = ioBlockView.getNumSamples();

            if (numberOfSamples <= 0)
                return;

            for (int channelIndex = 0;
                 channelIndex < numberOfChannels;
                 ++channelIndex)
            {
                const SampleType* const sourceChannelPointer = ioBlockView.getChannelPointer(channelIndex);
                SampleType* const destinationChannelPointer = fadeBuffer.getWritePointer(channelIndex);

                if (sourceChannelPointer == nullptr || destinationChannelPointer == nullptr)
                    continue;

                std::memcpy(destinationChannelPointer, sourceChannelPointer, static_cast<std::size_t>(numberOfSamples) * sizeof(SampleType));
            }
        }

        // Heap-allocated, xsimd-aligned scratch for the pre-transition
        // snapshot of the dry input. Reused across transitions.
        AlignedBufferType fadeBuffer{};

        // State latched on each call: tracks whether the processor was
        // active during the previous block so transitions can be detected.
        bool previousProcessorIsActiveState { false };
    };
}
#endif

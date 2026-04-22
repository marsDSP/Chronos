#pragma once

#ifndef CHRONOS_STEREO_PROCESSOR_H
#define CHRONOS_STEREO_PROCESSOR_H

// ============================================================================
//  stereo_processor.h
// ----------------------------------------------------------------------------
//  Port of sst-effects' ChronosReverb (sst-effects/include/sst/effects/ChronosReverb.h)
//  template class, adapted to the Chronos DSP namespace and rewritten with
//  explicit, verbose names for every member and every intermediate
//  variable in the inner loop. All underlying building blocks
//  (Schroeder allpass, predelay, one-pole damping, dual-tap delay,
//  quadrature LFO, block-rate smoother) live in their own headers with
//  similarly verbose names; see this file's #includes.
//
//  Topology (per sample tick):
//
//    // --- feed for the reverb loop (mono, matches the SST reference) ---
//    monoInput          = 0.5 * (inputL + inputR)
//    predelayed         = predelay.process(monoInput, userPredelayInSamples)
//    loopFeed           = predelayed
//    for i in 0..3:                     // 4 input diffusers in series
//        loopFeed       = inputAllpass[i].process(loopFeed, diffusion07)
//
//    // --- stereo diffusor on the direct delay taps -----------------------
//    // This is the path that makes the reverb mix knob behave like a
//    // diffusor on the delay taps themselves. Each stereo channel runs
//    // its own chain of 4 Schroeder allpasses (same Schroeder lengths as
//    // the loop feed, but independent state) so the direct output is a
//    // smeared-but-stereo version of the input. At mix = 0 the direct
//    // taps pass through cleanly; at higher mix values the diffused
//    // stereo signal (plus the reverb loop's tail) is blended in.
//
//    diffusedDirectLeft  = inputL
//    diffusedDirectRight = inputR
//    for i in 0..3:
//        diffusedDirectLeft  = directAllpassL[i].process(diffusedDirectLeft,  diffusion07)
//        diffusedDirectRight = directAllpassR[i].process(diffusedDirectRight, diffusion07)
//
//    // --- reverb loop (4 blocks in series, recirculating through state) --
//    recirculating      = networkState      // from last sample's tail
//    loopTailLeft       = 0
//    loopTailRight      = 0
//
//    lfoQuartet[0..3]   = (cos, sin, -cos, -sin)  // from the LFO this step
//
//    for b in 0..3:                     // 4 reverb blocks in series
//        recirculating += loopFeed
//        for c in 0..1:                 // 2 loop allpasses per block
//            recirculating = blockAllpass[b][c].process(recirculating, buildup07)
//        recirculating  = hfDamper[b].lowpass (recirculating, hfDampCoef)
//        recirculating  = lfDamper[b].highpass(recirculating, lfDampCoef)
//
//        mod_q8 = int(modulationDepthInSamples * lfoQuartet[b] * 256)
//        recirculating = delay[b].process(recirculating,
//                                         leftTap[b],  leftTapOut,
//                                         rightTap[b], rightTapOut,
//                                         mod_q8)
//
//        loopTailLeft   += leftTapOut  * leftTapGain[b]
//        loopTailRight  += rightTapOut * rightTapGain[b]
//
//        recirculating *= decayPerBlockMultiplier
//
//    // --- wet signal = stereo-diffused delay tap + reverb loop tail -----
//    wetL = diffusedDirectLeft  + loopTailLeft
//    wetR = diffusedDirectRight + loopTailRight
//    networkState = recirculating
//
//    // --- additive send: mix is the master reverb-intensity control ---
//    outL = dryInputLeft  + mix * wetL
//    outR = dryInputRight + mix * wetR
//
//  Because the final stage is a PURE SEND (dry is passed through
//  unattenuated, and the wet contribution is scaled by mix), the mix
//  knob is the only knob that can make the reverb audible: at mix = 0
//  the output is bit-identical to the input regardless of every other
//  reverb parameter (roomSize, decayTime, diffusion, buildup,
//  modulation, damping, predelay). Combined with the engine-level
//  reverbSendBypassEngine - which short-circuits the whole
//  processBlockInPlace call when the Reverb Bypass flag is on OR
//  mix == 0 - this means the reverb is fully removable with either
//  the mix knob at zero or the Reverb Bypass toggle.
//
//  Parameter meanings (verbose names used in this class's public setters;
//  see setter comments for the exact legal ranges):
//
//    roomSize       - bipolar percent. The actual size scale is
//                     2^(roomSize) so 0 = default size, +1 = 2x, -1 = 0.5x.
//    decayTime      - log2-seconds. Actual decay time in seconds is
//                     2^(decayTime). SST uses the range [-4, 6] → ~0.0625 s
//                     up to 64 s.
//    predelay       - log2-seconds. Same convention as decayTime.
//    diffusion      - 0..1 knob. Input-diffuser allpass coefficient is
//                     0.7 * this value.
//    buildup        - 0..1 knob. Loop-allpass coefficient is 0.7 * this.
//    modulation     - 0..1 knob. Tap-modulation depth in samples is
//                     modulation * sr * 0.001 * 5 (i.e. up to 5 ms).
//    hfDamping      - 0..1 knob. Lowpass memory coefficient in the loop
//                     is 0.8 * this.
//    lfDamping      - 0..1 knob. Highpass memory coefficient is 0.2 * this.
//    mix            - 0..1 knob. Master reverb-send level. 0 = no
//                     reverb contribution (output == input exactly),
//                     1 = full reverb contribution added on top of
//                     dry. This is the only knob (besides the Reverb
//                     Bypass toggle) that can introduce reverb into
//                     the output.
//    width          - -1..1 (signed percent). Mid/side rotation applied
//                     to the wet signal before the mix.
//
//  Size invariants: the internal building blocks are templated on a
//  MaxDelayCapacity large enough to hold SST's longest tap
//  (≈ 0.5 s × 8x size scale at 384 kHz). We expose that capacity via the
//  template parameter kMaxInternalCapacityInSamples below.
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "blockrate_smoother.h"
#include "damping_filter.h"
#include "predelay_buffer.h"
#include "quadrature_oscillator.h"
#include "schroeder_allpass.h"
#include "modulated_delayline.h"

namespace MarsDSP::DSP::ChronosReverb
{
    // ----------------------------------------------------------------------------
    //  ChronosReverbStereoProcessor<SampleType>
    //
    //  Stereo in / stereo out. Maintains its own allpass / delay state
    //  across processBlockInPlace calls. Configure via the public setters
    //  (setRoomSize, setDecayTime, ...) before each block.
    // ----------------------------------------------------------------------------
    template <typename SampleType>
    class ChronosReverbStereoProcessor
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "ChronosReverbStereoProcessor requires a floating-point sample type.");

        // Internal constants, matching SST's defines. Renamed verbosely.
        static constexpr int         kNumberOfReverbBlocksInLoop       = 4;
        static constexpr int         kNumberOfInputDiffuserAllpasses   = 4;
        static constexpr int         kNumberOfAllpassesPerBlock        = 2;

        // Internal delay-line capacity (per ring, power of two so the
        // tapped delay line's wrap is a bitmask). 32768 samples is ~680 ms
        // at 48 kHz / ~170 ms at 192 kHz - enough for SST's longest tap
        // (122.6 ms * sizeScale) at any practical size scale. Intentionally
        // smaller than the SST reference value (131072) so the full
        // processor fits comfortably inside DelayEngine by value instead
        // of forcing a heap allocation.
        static constexpr std::size_t kMaxInternalCapacityInSamples     = (1u << 15);

        // Predelay ring capacity. 192000 samples = 4 s @ 48 kHz / 2 s @
        // 96 kHz / 1 s @ 192 kHz. Covers the SST predelay parameter
        // range (-8..1 log2-seconds => 4 ms..2 s) at every supported
        // host sample rate without blowing the DelayEngine up memory-wise.
        static constexpr std::size_t kMaxPredelayCapacityInSamples     = (48000u * 4u);

        // -60 dB linear gain target used in the per-block decay formula.
        // SST: `db60 = powf(10, 0.05 * -60) = 1e-3`.
        static constexpr double      kMinusSixtyDecibelsAsLinearGain   = 0.001;

        // Tap gain schedule, copied directly from SST's setvars().
        static constexpr SampleType  kFixedLeftTapGains[kNumberOfReverbBlocksInLoop] = {
            static_cast<SampleType>(1.5 / 4.0),
            static_cast<SampleType>(1.2 / 4.0),
            static_cast<SampleType>(1.0 / 4.0),
            static_cast<SampleType>(0.8 / 4.0),
        };
        static constexpr SampleType  kFixedRightTapGains[kNumberOfReverbBlocksInLoop] = {
            static_cast<SampleType>(1.5 / 4.0),
            static_cast<SampleType>(1.2 / 4.0),
            static_cast<SampleType>(1.0 / 4.0),
            static_cast<SampleType>(0.8 / 4.0),
        };

        // Per-block delay lengths (in ms at size scale = 1.0), directly
        // from SST's calc_size() hard-coded table.
        static constexpr double kBaseLeftTapTimesInMilliseconds [kNumberOfReverbBlocksInLoop] =
            {  80.3,  59.3,  97.7, 122.6 };
        static constexpr double kBaseRightTapTimesInMilliseconds[kNumberOfReverbBlocksInLoop] =
            {  35.5, 101.6,  73.9,  80.3 };

        static constexpr double kBaseInputAllpassLengthsInMilliseconds[kNumberOfInputDiffuserAllpasses] =
            { 4.76, 6.81, 10.13, 16.72 };

        // Per-block allpass + loop-delay table (ms at size = 1.0). Rows
        // are blocks, columns are the two loop allpasses per block plus a
        // final loop-delay column. This is a verbatim copy of the numbers
        // in SST's calc_size().
        static constexpr double kBaseLoopAllpassLengthsInMilliseconds
            [kNumberOfReverbBlocksInLoop][kNumberOfAllpassesPerBlock] =
        {
            { 38.2, 53.4 },
            { 44.0, 41.0 },
            { 48.3, 60.5 },
            { 38.9, 42.2 },
        };
        static constexpr double kBaseLoopDelayLengthsInMilliseconds
            [kNumberOfReverbBlocksInLoop] =
            { 178.8, 126.5, 106.1, 139.4 };

        // SST's hard-coded "nominal loop time at scale=1" used in the
        // per-block decay-multiplier formula. 0.5508 s.
        static constexpr double kNominalLoopTimeAtSizeScaleOneInSeconds = 0.5508;

        // The LFO defaults to 0.25 Hz (SST: `2^-2` Hz).
        static constexpr double kDefaultLfoFrequencyInHertz = 0.25;

        ChronosReverbStereoProcessor() noexcept = default;

        // ------------------------------------------------------------------
        //  prepare: configure sample-rate-dependent state. Must be called
        //  before the first processBlockInPlace() and any time the host
        //  sample rate or maximum block size changes.
        // ------------------------------------------------------------------
        void prepare(double hostSampleRateInHz,
                     int    maximumBlockSizeInSamples) noexcept
        {
            cachedHostSampleRateInHz       = hostSampleRateInHz;
            cachedMaximumBlockSizeInSamples = std::max(1, maximumBlockSizeInSamples);

            decayPerBlockMultiplierSmoother.prepareForBlockOfSamples(cachedMaximumBlockSizeInSamples);
            diffusionCoefficientSmoother   .prepareForBlockOfSamples(cachedMaximumBlockSizeInSamples);
            buildupCoefficientSmoother     .prepareForBlockOfSamples(cachedMaximumBlockSizeInSamples);
            hfDampingCoefficientSmoother   .prepareForBlockOfSamples(cachedMaximumBlockSizeInSamples);
            lfDampingCoefficientSmoother   .prepareForBlockOfSamples(cachedMaximumBlockSizeInSamples);
            modulationDepthSmoother        .prepareForBlockOfSamples(cachedMaximumBlockSizeInSamples);
            wetDryMixSmoother              .prepareForBlockOfSamples(cachedMaximumBlockSizeInSamples);

            // Default LFO rate (SST default = 0.25 Hz quadrature).
            quadratureModulationOscillator.setRateInRadiansPerSample(
                static_cast<SampleType>(2.0 * M_PI
                                        * kDefaultLfoFrequencyInHertz
                                        / hostSampleRateInHz));
            quadratureModulationOscillator.snapToCosineSineUnitPhasor();

            recalculateAllDelayLengthsFromSizeScale(static_cast<SampleType>(1));
            recalculateDecayPerBlockMultiplierFromCurrentParameters();

            reset();
        }

        // ------------------------------------------------------------------
        //  reset: flush every stored sample and running filter state.
        // ------------------------------------------------------------------
        void reset() noexcept
        {
            runningRecirculatingNetworkStateSample = static_cast<SampleType>(0);
            predelayCircularBuffer.reset();
            for (auto& ap : inputDiffuserAllpasses)             ap.reset();
            for (auto& ap : stereoDirectDiffuserAllpassesLeft)  ap.reset();
            for (auto& ap : stereoDirectDiffuserAllpassesRight) ap.reset();
            for (auto& row : blockLoopAllpasses)
                for (auto& ap : row)                            ap.reset();
            for (auto& dmp : blockHfDampingFilters)             dmp.reset();
            for (auto& dmp : blockLfDampingFilters)             dmp.reset();
            for (auto& dln : blockLoopDelayLines)               dln.reset();

            decayPerBlockMultiplierSmoother.snapToValue(cachedDecayPerBlockMultiplier);
            diffusionCoefficientSmoother   .snapToValue(
                static_cast<SampleType>(0.7) * userDiffusionNormalised);
            buildupCoefficientSmoother     .snapToValue(
                static_cast<SampleType>(0.7) * userBuildupNormalised);
            hfDampingCoefficientSmoother   .snapToValue(
                static_cast<SampleType>(0.8) * userHighFrequencyDampingNormalised);
            lfDampingCoefficientSmoother   .snapToValue(
                static_cast<SampleType>(0.2) * userLowFrequencyDampingNormalised);
            modulationDepthSmoother        .snapToValue(
                computeModulationDepthInSamples(userModulationNormalised));
            wetDryMixSmoother              .snapToValue(userMixNormalised);

            quadratureModulationOscillator.snapToCosineSineUnitPhasor();
        }

        // ------------------------------------------------------------------
        //  Parameter setters. All values get cached here and only read by
        //  the smoothers at the top of processBlockInPlace(), so callers
        //  can push new targets as often as they like without affecting
        //  the audio thread mid-block.
        // ------------------------------------------------------------------
        void setRoomSize           (SampleType roomSizeBipolarExponent) noexcept
        {
            userRoomSizeExponent = roomSizeBipolarExponent;
        }
        void setDecayTime          (SampleType decayTimeLog2Seconds) noexcept
        {
            userDecayTimeLog2Seconds = decayTimeLog2Seconds;
        }
        void setPredelayTime       (SampleType predelayLog2Seconds) noexcept
        {
            userPredelayLog2Seconds = predelayLog2Seconds;
        }
        void setDiffusion          (SampleType diffusionNormalised) noexcept
        {
            userDiffusionNormalised = std::clamp(diffusionNormalised,
                                                 static_cast<SampleType>(0),
                                                 static_cast<SampleType>(1));
        }
        void setBuildup            (SampleType buildupNormalised) noexcept
        {
            userBuildupNormalised = std::clamp(buildupNormalised,
                                               static_cast<SampleType>(0),
                                               static_cast<SampleType>(1));
        }
        void setModulation         (SampleType modulationNormalised) noexcept
        {
            userModulationNormalised = std::clamp(modulationNormalised,
                                                  static_cast<SampleType>(0),
                                                  static_cast<SampleType>(1));
        }
        void setHighFrequencyDamping(SampleType highFrequencyDampingNormalised) noexcept
        {
            userHighFrequencyDampingNormalised = std::clamp(highFrequencyDampingNormalised,
                                                            static_cast<SampleType>(0),
                                                            static_cast<SampleType>(1));
        }
        void setLowFrequencyDamping(SampleType lowFrequencyDampingNormalised) noexcept
        {
            userLowFrequencyDampingNormalised = std::clamp(lowFrequencyDampingNormalised,
                                                           static_cast<SampleType>(0),
                                                           static_cast<SampleType>(1));
        }
        void setMix                (SampleType mixNormalised) noexcept
        {
            userMixNormalised = std::clamp(mixNormalised,
                                           static_cast<SampleType>(0),
                                           static_cast<SampleType>(1));
        }

        // ------------------------------------------------------------------
        //  processBlockInPlace: run 'numSamplesInBlock' stereo samples
        //  through the reverb in place. The wet signal (stereo-diffused
        //  delay tap + loop tail) is ADDED on top of dry with a mix-
        //  scaled gain, so the mix knob is the sole master intensity
        //  control. mix = 0 returns the input unchanged regardless of
        //  every other reverb parameter.
        // ------------------------------------------------------------------
        void processBlockInPlace(SampleType* __restrict leftChannelInOut,
                                 SampleType* __restrict rightChannelInOut,
                                 int                    numSamplesInBlock) noexcept
        {
            assert(numSamplesInBlock <= cachedMaximumBlockSizeInSamples);

            // Hard short-circuit: mix = 0 means "no reverb contribution
            // whatsoever". Skip every internal update so none of the
            // other reverb parameters (roomSize, decayTime, diffusion,
            // buildup, modulation, damping, predelay) can modify the
            // internal state OR the output buffer. The buffer passes
            // through bit-exact, and when the user ramps mix back up
            // the wet-dry smoother starts from zero rather than an
            // orphan positive value.
            if (userMixNormalised <= static_cast<SampleType>(0))
            {
                wetDryMixSmoother.snapToValue(static_cast<SampleType>(0));
                return;
            }

            // 1) Refresh per-block state from the cached user parameters.
            const SampleType sizeScale =
                static_cast<SampleType>(std::pow(2.0,
                                                 static_cast<double>(userRoomSizeExponent)));
            recalculateAllDelayLengthsFromSizeScale(sizeScale);
            recalculateDecayPerBlockMultiplierFromCurrentParameters();

            decayPerBlockMultiplierSmoother.setTargetValue(cachedDecayPerBlockMultiplier);
            diffusionCoefficientSmoother   .setTargetValue(
                static_cast<SampleType>(0.7) * userDiffusionNormalised);
            buildupCoefficientSmoother     .setTargetValue(
                static_cast<SampleType>(0.7) * userBuildupNormalised);
            hfDampingCoefficientSmoother   .setTargetValue(
                static_cast<SampleType>(0.8) * userHighFrequencyDampingNormalised);
            lfDampingCoefficientSmoother   .setTargetValue(
                static_cast<SampleType>(0.2) * userLowFrequencyDampingNormalised);
            modulationDepthSmoother        .setTargetValue(
                computeModulationDepthInSamples(userModulationNormalised));
            wetDryMixSmoother              .setTargetValue(userMixNormalised);

            // Predelay tap in whole samples.
            const int predelayTapInSamples = std::clamp(
                static_cast<int>(cachedHostSampleRateInHz
                                 * std::pow(2.0, static_cast<double>(userPredelayLog2Seconds))),
                1,
                static_cast<int>(predelayCircularBuffer.kMaxLegalTapInSamples));

            // 2) Per-sample loop.
            for (int sampleIndexInBlock = 0;
                 sampleIndexInBlock < numSamplesInBlock;
                 ++sampleIndexInBlock)
            {
                const SampleType dryLeftInputSample  = leftChannelInOut [sampleIndexInBlock];
                const SampleType dryRightInputSample = rightChannelInOut[sampleIndexInBlock];

                const SampleType monoizedInputSample =
                    static_cast<SampleType>(0.5) * (dryLeftInputSample + dryRightInputSample);

                // 2a) Predelay (mono, feeds the reverb loop only - the
                //     direct stereo-diffusor path below operates on the
                //     current-sample stereo input so it acts as an
                //     insert effect on the delay taps).
                SampleType inputBusSample =
                    predelayCircularBuffer.processSingleSample(monoizedInputSample,
                                                               predelayTapInSamples);

                const SampleType currentDiffusionCoefficient =
                    diffusionCoefficientSmoother.currentValue;

                // 2b) Four input diffusers in series. This feeds the
                //     reverb loop below with a mono, decorrelated
                //     version of the predelayed input (matches the SST
                //     Reverb2 reference topology).
                for (auto& inputAllpass : inputDiffuserAllpasses)
                    inputBusSample = inputAllpass.processSingleSample(inputBusSample,
                                                                      currentDiffusionCoefficient);

                // 2b.stereo) Parallel stereo-diffusor path. Each channel
                //     runs its own chain of 4 Schroeder allpasses with
                //     the same lengths as the mono chain but independent
                //     state, so the result is a smeared-but-stereo copy
                //     of the live delay taps. When the reverb mix is 0
                //     these samples contribute nothing; as mix rises
                //     they smoothly diffuse the direct signal before the
                //     loop tail gets blended on top.
                SampleType stereoDiffusedLeftSample  = dryLeftInputSample;
                SampleType stereoDiffusedRightSample = dryRightInputSample;
                for (int stereoDiffuserIndex = 0;
                     stereoDiffuserIndex < kNumberOfInputDiffuserAllpasses;
                     ++stereoDiffuserIndex)
                {
                    stereoDiffusedLeftSample =
                        stereoDirectDiffuserAllpassesLeft[stereoDiffuserIndex]
                            .processSingleSample(stereoDiffusedLeftSample,
                                                 currentDiffusionCoefficient);
                    stereoDiffusedRightSample =
                        stereoDirectDiffuserAllpassesRight[stereoDiffuserIndex]
                            .processSingleSample(stereoDiffusedRightSample,
                                                 currentDiffusionCoefficient);
                }

                // 2c) Four parallel-in-series reverb blocks. Each block
                //     adds the diffused input to the recirculating state,
                //     runs 2 allpasses, damps low + high, then grabs stereo
                //     taps from its LFO-modulated delay line and adds
                //     their contribution (with fixed gains) into the
                //     stereo accumulators.
                SampleType recirculatingSample = runningRecirculatingNetworkStateSample;
                SampleType leftAccumulatedWet  = static_cast<SampleType>(0);
                SampleType rightAccumulatedWet = static_cast<SampleType>(0);

                // Quartet of 90-degree-offset LFO waveforms, one per block.
                const SampleType lfoCosineSample = quadratureModulationOscillator.getCosineSample();
                const SampleType lfoSineSample   = quadratureModulationOscillator.getSineSample();
                const SampleType perBlockLfoQuartet[kNumberOfReverbBlocksInLoop] = {
                     lfoCosineSample,
                     lfoSineSample,
                    -lfoCosineSample,
                    -lfoSineSample,
                };

                // Clamp damping coefficients to [0.01, 0.99] on every
                // sample: both one-pole filters degenerate to invalid
                // behaviour at the endpoints.
                const SampleType highFrequencyDampingMemoryCoefficient =
                    std::clamp(hfDampingCoefficientSmoother.currentValue,
                               static_cast<SampleType>(0.01),
                               static_cast<SampleType>(0.99));
                const SampleType lowFrequencyDampingMemoryCoefficient =
                    std::clamp(lfDampingCoefficientSmoother.currentValue,
                               static_cast<SampleType>(0.01),
                               static_cast<SampleType>(0.99));
                const SampleType currentBuildupCoefficient =
                    buildupCoefficientSmoother.currentValue;
                const SampleType currentModulationDepthInSamples =
                    modulationDepthSmoother.currentValue;
                const SampleType currentDecayPerBlockMultiplier =
                    decayPerBlockMultiplierSmoother.currentValue;

                for (int blockIndex = 0;
                     blockIndex < kNumberOfReverbBlocksInLoop;
                     ++blockIndex)
                {
                    // Inject the diffused input sample into this block.
                    recirculatingSample += inputBusSample;

                    // Two loop allpasses per block.
                    for (int loopAllpassIndex = 0;
                         loopAllpassIndex < kNumberOfAllpassesPerBlock;
                         ++loopAllpassIndex)
                    {
                        recirculatingSample =
                            blockLoopAllpasses[blockIndex][loopAllpassIndex]
                                .processSingleSample(recirculatingSample,
                                                     currentBuildupCoefficient);
                    }

                    // High-frequency damping (lowpass) and low-frequency
                    // damping (highpass) in the recirculating path.
                    recirculatingSample =
                        blockHfDampingFilters[blockIndex]
                            .processSingleSampleAsLowpass(recirculatingSample,
                                                          highFrequencyDampingMemoryCoefficient);
                    recirculatingSample =
                        blockLfDampingFilters[blockIndex]
                            .processSingleSampleAsHighpass(recirculatingSample,
                                                           lowFrequencyDampingMemoryCoefficient);

                    // LFO-modulated loop delay with stereo output taps.
                    // The modulation offset is in Q8.8 samples; convert
                    // from the smoother's sample-domain depth and the
                    // block's LFO sample by scaling by 256.
                    const int modulationOffsetInQ8Samples = static_cast<int>(
                        currentModulationDepthInSamples
                        * perBlockLfoQuartet[blockIndex]
                        * static_cast<SampleType>(
                              ChronosReverbStereoTappedModulatedDelayLine<SampleType,
                                  kMaxInternalCapacityInSamples>
                                      ::kModulationSubSampleRange));

                    SampleType leftStereoTapOutputSample  = static_cast<SampleType>(0);
                    SampleType rightStereoTapOutputSample = static_cast<SampleType>(0);
                    recirculatingSample =
                        blockLoopDelayLines[blockIndex].processSingleSample(
                            recirculatingSample,
                            cachedLeftTapDelayLengthsInSamples [blockIndex],
                            leftStereoTapOutputSample,
                            cachedRightTapDelayLengthsInSamples[blockIndex],
                            rightStereoTapOutputSample,
                            modulationOffsetInQ8Samples);

                    leftAccumulatedWet  +=
                        leftStereoTapOutputSample  * kFixedLeftTapGains [blockIndex];
                    rightAccumulatedWet +=
                        rightStereoTapOutputSample * kFixedRightTapGains[blockIndex];

                    // Decay-per-block multiplier: applied after every
                    // block so the effective per-loop gain is
                    // multiplier^4. See the SST reverb loop.
                    recirculatingSample *= currentDecayPerBlockMultiplier;
                }

                // Commit new network state for next sample.
                runningRecirculatingNetworkStateSample = recirculatingSample;

                // 2d) Additive send. The wet signal (stereo-diffused
                //     delay tap + loop-reverb tail) is ADDED on top of
                //     the untouched dry input, scaled by the mix
                //     smoother's ramped output. Consequences:
                //       * mix = 0 -> output == input exactly, so no
                //         other reverb parameter can affect the delay
                //         taps (diffusion, buildup, roomSize,
                //         decayTime, damping, modulation, predelay
                //         are all bypassable by the mix knob alone).
                //       * DelayEngine wraps this call in
                //         reverbSendBypassEngine with the gate
                //         (!reverbBypassed && mix > 0), so the
                //         Reverb Bypass flag removes the reverb
                //         entirely.
                const SampleType wetWeight = wetDryMixSmoother.currentValue;

                const SampleType wetLeftSample =
                    stereoDiffusedLeftSample  + leftAccumulatedWet;
                const SampleType wetRightSample =
                    stereoDiffusedRightSample + rightAccumulatedWet;

                leftChannelInOut [sampleIndexInBlock] =
                    dryLeftInputSample  + wetLeftSample  * wetWeight;
                rightChannelInOut[sampleIndexInBlock] =
                    dryRightInputSample + wetRightSample * wetWeight;

                // 2e) Per-sample advance for all the smoothers + the LFO.
                decayPerBlockMultiplierSmoother.advanceByOneSample();
                diffusionCoefficientSmoother   .advanceByOneSample();
                buildupCoefficientSmoother     .advanceByOneSample();
                hfDampingCoefficientSmoother   .advanceByOneSample();
                lfDampingCoefficientSmoother   .advanceByOneSample();
                modulationDepthSmoother        .advanceByOneSample();
                wetDryMixSmoother              .advanceByOneSample();
                quadratureModulationOscillator .advancePhaseByOneSample();
            }
        }

    private:
        // ------------------------------------------------------------------
        //  Helper: convert the user modulation knob into a depth in
        //  samples. Matches SST's
        //      modulation * sampleRate * 0.001 * 5
        //  i.e. up to 5 ms of delay deviation at full depth.
        // ------------------------------------------------------------------
        [[nodiscard]] SampleType computeModulationDepthInSamples(
            SampleType modulationNormalised) const noexcept
        {
            return modulationNormalised
                 * static_cast<SampleType>(cachedHostSampleRateInHz)
                 * static_cast<SampleType>(0.001)
                 * static_cast<SampleType>(5);
        }

        // ------------------------------------------------------------------
        //  Helper: recompute all per-block delay lengths at the given size
        //  scale (= 2^roomSize). Called from processBlockInPlace() every
        //  block so OU / knob changes reach the topology.
        // ------------------------------------------------------------------
        void recalculateAllDelayLengthsFromSizeScale(SampleType sizeScale) noexcept
        {
            const double sampleRateTimesSizeScale =
                cachedHostSampleRateInHz * static_cast<double>(sizeScale);

            // Input diffusers. The mono chain feeds the reverb loop,
            // while the two stereo chains sit on the direct delay-tap
            // path and give the mix knob its diffusor character. All
            // three chains share the same Schroeder lengths so their
            // smearing follows the same time / pitch profile.
            for (int i = 0; i < kNumberOfInputDiffuserAllpasses; ++i)
            {
                const int allpassLengthInSamples =
                    millisecondsToSamples(kBaseInputAllpassLengthsInMilliseconds[i],
                                          sampleRateTimesSizeScale);
                inputDiffuserAllpasses            [i].setLengthInSamples(allpassLengthInSamples);
                stereoDirectDiffuserAllpassesLeft [i].setLengthInSamples(allpassLengthInSamples);
                stereoDirectDiffuserAllpassesRight[i].setLengthInSamples(allpassLengthInSamples);
            }

            // Per-block allpasses + loop delays + stereo tap offsets.
            for (int b = 0; b < kNumberOfReverbBlocksInLoop; ++b)
            {
                for (int c = 0; c < kNumberOfAllpassesPerBlock; ++c)
                {
                    blockLoopAllpasses[b][c].setLengthInSamples(
                        millisecondsToSamples(kBaseLoopAllpassLengthsInMilliseconds[b][c],
                                              sampleRateTimesSizeScale));
                }

                const int loopDelaySamples =
                    millisecondsToSamples(kBaseLoopDelayLengthsInMilliseconds[b],
                                          sampleRateTimesSizeScale);
                blockLoopDelayLines[b].setLoopDelayLengthInSamples(loopDelaySamples);

                cachedLeftTapDelayLengthsInSamples[b] =
                    millisecondsToSamples(kBaseLeftTapTimesInMilliseconds[b],
                                          sampleRateTimesSizeScale);
                cachedRightTapDelayLengthsInSamples[b] =
                    millisecondsToSamples(kBaseRightTapTimesInMilliseconds[b],
                                          sampleRateTimesSizeScale);
            }
        }

        // ------------------------------------------------------------------
        //  Helper: the per-block decay multiplier is
        //     db60 ^ (loopTime / (4 * 2^decayTime))
        //  where loopTime is the nominal 0.5508 s (size = 1) times the
        //  current size scale. This matches the SST closed-form solution
        //  for a 4-block chain to hit -60 dB at the target decay time.
        // ------------------------------------------------------------------
        void recalculateDecayPerBlockMultiplierFromCurrentParameters() noexcept
        {
            const double sizeScale = std::pow(2.0,
                                              static_cast<double>(userRoomSizeExponent));
            const double loopTimeInSeconds =
                kNominalLoopTimeAtSizeScaleOneInSeconds * sizeScale;

            const double decayTimeInSeconds =
                std::pow(2.0, static_cast<double>(userDecayTimeLog2Seconds));

            const double safeDenominator =
                std::max(4.0 * decayTimeInSeconds, 1e-9);

            cachedDecayPerBlockMultiplier = static_cast<SampleType>(
                std::pow(kMinusSixtyDecibelsAsLinearGain,
                         loopTimeInSeconds / safeDenominator));
        }

        // ------------------------------------------------------------------
        //  Helper: convert an SST-style millisecond length (at size = 1)
        //  into samples at the current host sample rate, clamped so the
        //  result never exceeds the internal delay capacity.
        // ------------------------------------------------------------------
        [[nodiscard]] static int millisecondsToSamples(double lengthInMilliseconds,
                                                       double sampleRateTimesSizeScale) noexcept
        {
            const double lengthInSamples =
                sampleRateTimesSizeScale * lengthInMilliseconds * 0.001;
            const int clamped = static_cast<int>(std::clamp(
                lengthInSamples,
                1.0,
                static_cast<double>(kMaxInternalCapacityInSamples - 2)));
            return clamped;
        }

        // ------------------------------------------------------------------
        //  Cached host state.
        // ------------------------------------------------------------------
        double cachedHostSampleRateInHz        { 48000.0 };
        int    cachedMaximumBlockSizeInSamples { 1 };
        SampleType cachedDecayPerBlockMultiplier { static_cast<SampleType>(0) };
        std::array<int, kNumberOfReverbBlocksInLoop> cachedLeftTapDelayLengthsInSamples{};
        std::array<int, kNumberOfReverbBlocksInLoop> cachedRightTapDelayLengthsInSamples{};

        // ------------------------------------------------------------------
        //  User parameter cache (updated by setters, read at the top of
        //  processBlockInPlace).
        // ------------------------------------------------------------------
        SampleType userRoomSizeExponent               { static_cast<SampleType>(0)    };
        SampleType userDecayTimeLog2Seconds           { static_cast<SampleType>(0.75) };
        SampleType userPredelayLog2Seconds            { static_cast<SampleType>(-4)   };
        SampleType userDiffusionNormalised            { static_cast<SampleType>(1)    };
        SampleType userBuildupNormalised              { static_cast<SampleType>(1)    };
        SampleType userModulationNormalised           { static_cast<SampleType>(0.5)  };
        SampleType userHighFrequencyDampingNormalised { static_cast<SampleType>(0.2)  };
        SampleType userLowFrequencyDampingNormalised  { static_cast<SampleType>(0.2)  };
        SampleType userMixNormalised                  { static_cast<SampleType>(0.33) };

        // ------------------------------------------------------------------
        //  Block-rate smoothers (one per audio-rate parameter).
        // ------------------------------------------------------------------
        ChronosReverbBlockRateLinearSmoother<SampleType> decayPerBlockMultiplierSmoother;
        ChronosReverbBlockRateLinearSmoother<SampleType> diffusionCoefficientSmoother;
        ChronosReverbBlockRateLinearSmoother<SampleType> buildupCoefficientSmoother;
        ChronosReverbBlockRateLinearSmoother<SampleType> hfDampingCoefficientSmoother;
        ChronosReverbBlockRateLinearSmoother<SampleType> lfDampingCoefficientSmoother;
        ChronosReverbBlockRateLinearSmoother<SampleType> modulationDepthSmoother;
        ChronosReverbBlockRateLinearSmoother<SampleType> wetDryMixSmoother;

        // ------------------------------------------------------------------
        //  Network state.
        // ------------------------------------------------------------------
        SampleType runningRecirculatingNetworkStateSample { static_cast<SampleType>(0) };

        ChronosReverbPredelayCircularBuffer<SampleType, kMaxPredelayCapacityInSamples>
            predelayCircularBuffer;

        // Mono input diffusers that feed the reverb loop. Match the
        // SST Reverb2 reference topology exactly.
        std::array<ChronosReverbSchroederAllpassDelayLine<SampleType, kMaxInternalCapacityInSamples>,
                   kNumberOfInputDiffuserAllpasses>
            inputDiffuserAllpasses{};

        // Parallel stereo input-diffuser chains applied directly to the
        // stereo input (= the delay taps when ChronosReverb is wired as
        // a post-delay send inside DelayEngine). These make the reverb
        // mix knob behave as a stereo diffusor on the delay taps rather
        // than a pure parallel reverb send; see the topology comment
        // at the top of this file.
        std::array<ChronosReverbSchroederAllpassDelayLine<SampleType, kMaxInternalCapacityInSamples>,
                   kNumberOfInputDiffuserAllpasses>
            stereoDirectDiffuserAllpassesLeft{};
        std::array<ChronosReverbSchroederAllpassDelayLine<SampleType, kMaxInternalCapacityInSamples>,
                   kNumberOfInputDiffuserAllpasses>
            stereoDirectDiffuserAllpassesRight{};

        std::array<std::array<ChronosReverbSchroederAllpassDelayLine<SampleType, kMaxInternalCapacityInSamples>,
                              kNumberOfAllpassesPerBlock>,
                   kNumberOfReverbBlocksInLoop>
            blockLoopAllpasses{};

        std::array<ChronosReverbOnePoleDampingFilter<SampleType>, kNumberOfReverbBlocksInLoop>
            blockHfDampingFilters{};
        std::array<ChronosReverbOnePoleDampingFilter<SampleType>, kNumberOfReverbBlocksInLoop>
            blockLfDampingFilters{};

        std::array<ChronosReverbStereoTappedModulatedDelayLine<SampleType, kMaxInternalCapacityInSamples>,
                   kNumberOfReverbBlocksInLoop>
            blockLoopDelayLines{};

        ChronosReverbQuadratureModulationOscillator<SampleType> quadratureModulationOscillator;
    };

    // --------------------------------------------------------------------------
    //  Out-of-class definitions of constexpr static arrays (required by the
    //  ODR until C++17 inline-in-class convention is available to all
    //  compilation units referencing them).
    // --------------------------------------------------------------------------
    template <typename SampleType>
    constexpr SampleType ChronosReverbStereoProcessor<SampleType>
        ::kFixedLeftTapGains[kNumberOfReverbBlocksInLoop];
    template <typename SampleType>
    constexpr SampleType ChronosReverbStereoProcessor<SampleType>
        ::kFixedRightTapGains[kNumberOfReverbBlocksInLoop];
    template <typename SampleType>
    constexpr double ChronosReverbStereoProcessor<SampleType>
        ::kBaseLeftTapTimesInMilliseconds[kNumberOfReverbBlocksInLoop];
    template <typename SampleType>
    constexpr double ChronosReverbStereoProcessor<SampleType>
        ::kBaseRightTapTimesInMilliseconds[kNumberOfReverbBlocksInLoop];
    template <typename SampleType>
    constexpr double ChronosReverbStereoProcessor<SampleType>
        ::kBaseInputAllpassLengthsInMilliseconds[kNumberOfInputDiffuserAllpasses];
    template <typename SampleType>
    constexpr double ChronosReverbStereoProcessor<SampleType>
        ::kBaseLoopAllpassLengthsInMilliseconds
            [kNumberOfReverbBlocksInLoop]
            [kNumberOfAllpassesPerBlock];
    template <typename SampleType>
    constexpr double ChronosReverbStereoProcessor<SampleType>
        ::kBaseLoopDelayLengthsInMilliseconds[kNumberOfReverbBlocksInLoop];
}
#endif

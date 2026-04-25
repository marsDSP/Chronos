 #pragma once

#ifndef CHRONOS_FLUTTER_ENGINE_H
#define CHRONOS_FLUTTER_ENGINE_H

// ============================================================================
//  flutter_engine.h
// ----------------------------------------------------------------------------
//  Sum of three cosines at f, 2f, 3f with fixed phase offsets approximates
//
//  Block interpolation
//  -------------------
//  The audio path now drives the engine once per host block via
//  resolveBlock(numSamples). Internally the engine evaluates the LFO at
//  kSubBlockSize-sample boundaries (16 samples by default — the same rate
//  the delay engine's tap modulator was already using) and linearly
//  interpolates within each sub-block, filling a per-channel SIMD_M128
//  array `acLines[ch]`. The line is the equivalent of a multi-segment
//  BlockLerpSIMD: one linear segment per sub-block, end-of-segment
//  values stitched into the start of the next so block boundaries are
//  C0-continuous.
//
//  Audio-rate consumers can read the precomputed line per quad
//  (acQuad(q, ch)) or per sample (acAt(n, ch)). The delay engine's
//  tap-modulation crossfade — which already operates at sub-block
//  granularity — reads the exact end-of-sub-block AC value via
//  subBlockEndAc(idx, ch) so behaviour is bit-identical to the
//  per-sub-block processBlock(subN, ch) it replaces.
//
//  Legacy nextSample(channel) and processBlock(int, size_t) APIs are
//  preserved for the wow/flutter characterisation harness, which
//  exercises the LFO independently of the audio path.
// ============================================================================

#include <JuceHeader.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "dsp/math/simd/simd_config.h"

namespace MarsDSP::DSP::Modulation
{
    class FlutterEngine
    {
    public:
        // Largest pow-2 host block we support internally. The acLines store
        // one SIMD_M128 per quad of the host block, padded up to this
        // capacity so resolveBlock has nothing to allocate.
        static constexpr int kMaxBlockSize = 4096;

        // Sub-block resolution at which the LFO is re-evaluated. Matches
        // delay_engine.h::kTapModulationSubBlockSize so the sub-block-end
        // AC samples line up with the Lagrange tap reader.
        static constexpr int kSubBlockSize = 16;

        static constexpr int kMaxQuads     = kMaxBlockSize / 4;
        static constexpr int kMaxSubBlocks = (kMaxBlockSize + kSubBlockSize - 1) / kSubBlockSize;

        FlutterEngine() = default;

        void prepare (double sampleRateHz, int maxBlockSize, int numChannels)
        {
            ignoreUnused (maxBlockSize);

            sampleRate = static_cast<float>(sampleRateHz);

            const auto numCh = static_cast<size_t>(std::max (numChannels, 1));

            depthSlew.resize (numCh);
            for (auto& slew : depthSlew)
            {
                slew.reset (sampleRateHz, 0.05);
                slew.setCurrentAndTargetValue (kDepthFloor);
            }

            phase1.assign (numCh, 0.0f);
            phase2.assign (numCh, 0.0f);
            phase3.assign (numCh, 0.0f);
            blockEndAc.assign (numCh, 0.0f);

            acLines    .assign (numCh, AcLine{});
            subEndLines.assign (numCh, SubEndLine{});

            // Canonical tape-capstan f / 2f / 3f triplet plus a DC bias,
            // expressed directly in samples-of-delay. Chronos's audio
            // path consumes the engine's output as samples, so no
            // sample-rate scaling belongs here — the constants are the
            // peak (per-component) delay swing the tap reader sees.
            //
            // Earlier revisions multiplied by `1000.0f / sampleRate`,
            // which collapsed the Σ peak to ~6 samples at 48 kHz — well
            // below the perception threshold for delay modulation — and
            // shrank further at higher sample rates. With the raw
            // constants the Σ peak is ~289 samples (~6 ms at 48 kHz)
            // before the DC offset, which lands flutter cleanly in the
            // audible range.
            amplitude1 = -230.0f;
            amplitude2 =  -80.0f;
            amplitude3 =  -99.0f;
            dcOffsetInSamples = 350.0f;
        }

        // rateNormalised in [0, 1] maps to 0.1..100 Hz.
        // depthNormalised in [0, 1] maps to the aggregate flutter amplitude.
        void prepareBlock (float rateNormalised, float depthNormalised) noexcept
        {
            // depth^2 * 0.5 → depthTarget = depth * sqrt(0.5) ≈ depth * 0.707.
            // The previous formula (depth^3 * 81/625) capped depthTarget at
            // 0.36 at full knob (depth^1.5 * 0.36), leaving the effect at
            // <13% of amplitude at 50% knob. The new curve is linear in depth
            // with a 0.707 ceiling, doubling audibility across the whole range.
            const float skewedDepth = std::pow (depthNormalised, 2.0f) * 0.5f;
            const float depthTarget = std::sqrt (juce::jmax (0.0f, skewedDepth));
            for (auto& slew : depthSlew)
                slew.setTargetValue (juce::jmax (kDepthFloor, depthTarget));

            const float rateHz = 0.1f * std::pow (1000.0f, rateNormalised);
            angleDelta1 = juce::MathConstants<float>::twoPi * rateHz / sampleRate;
            angleDelta2 = 2.0f * angleDelta1;
            angleDelta3 = 3.0f * angleDelta1;
        }

        // Single-shot block resolver. Advances every channel's phase by the
        // full host block, evaluates the LFO at kSubBlockSize boundaries, and
        // fills acLines[ch] with a piecewise-linear SIMD ramp plus
        // subEndLines[ch] with the exact end-of-sub-block AC values. Audio
        // path consumers read via acQuad/acAt or subBlockEndAc.
        void resolveBlock (int numSamples) noexcept
        {
            if (numSamples <= 0) return;

            const int safeBlock = std::min (numSamples, kMaxBlockSize);
            numSubBlocksThisCall = 0;

            for (size_t ch = 0; ch < blockEndAc.size(); ++ch)
                resolveChannel (ch, safeBlock);
        }

        // Per-quad SIMD read of the precomputed AC line for channel `ch`.
        // q must be in [0, numSamples/4) of the most recent resolveBlock
        // call. Behaviour for out-of-range q is undefined.
        [[nodiscard]] SIMD_M128 acQuad (int q, size_t ch) const noexcept
        {
            assert (q >= 0 && q < kMaxQuads);
            return acLines[ch].quads[q];
        }

        // Per-sample scalar read for the SIMD-tail of the audio loop.
        [[nodiscard]] float acAt (int n, size_t ch) const noexcept
        {
            assert (n >= 0 && n < kMaxBlockSize);
            const int q    = n >> 2;
            const int lane = n & 3;
            alignas (16) float lanes[4];
            SIMD_MM (store_ps) (lanes, acLines[ch].quads[q]);
            return lanes[lane];
        }

        // Exact end-of-sub-block AC value for sub-block index `idx`. Mirrors
        // the value the legacy processBlock(subN, ch) returned at the end of
        // its sub-block, so existing per-sub-block crossfade consumers stay
        // bit-identical.
        [[nodiscard]] float subBlockEndAc (int idx, size_t ch) const noexcept
        {
            assert (idx >= 0 && idx < kMaxSubBlocks);
            return subEndLines[ch].endAc[idx];
        }

        [[nodiscard]] int getNumSubBlocks() const noexcept { return numSubBlocksThisCall; }

        // Per-sample pair: the AC wobble and the constant DC offset.
        // Both are in samples-of-delay; callers typically add them and
        // apply to a tap-position reader. Used by characterisation
        // harnesses; the audio hot path prefers resolveBlock + acQuad/acAt.
        inline std::pair<float, float> nextSample (size_t channelIndex) noexcept
        {
            phase1[channelIndex] += angleDelta1;
            phase2[channelIndex] += angleDelta2;
            phase3[channelIndex] += angleDelta3;

            const float depth = depthSlew[channelIndex].getNextValue();
            const float ac = depth * (amplitude1 * std::cos (phase1[channelIndex] + kPhaseOffset1)
                                    + amplitude2 * std::cos (phase2[channelIndex] + kPhaseOffset2)
                                    + amplitude3 * std::cos (phase3[channelIndex] + kPhaseOffset3));
            blockEndAc[channelIndex] = ac;
            return { ac, dcOffsetInSamples };
        }

        // Block-rate fast path. Advances every phase by the full block's
        // worth in one step and returns the (ac, dc) pair sampled at the
        // end of the block. Kept for callers that want a single end-of-block
        // value without resolving the per-quad line.
        inline std::pair<float, float> processBlock (int numSamples, size_t channelIndex) noexcept
        {
            const float phaseAdvance1 = angleDelta1 * static_cast<float>(numSamples);
            phase1[channelIndex] += phaseAdvance1;
            phase2[channelIndex] += phaseAdvance1 * 2.0f;
            phase3[channelIndex] += phaseAdvance1 * 3.0f;

            const float depth = depthSlew[channelIndex].skip (numSamples);
            const float ac = depth * (amplitude1 * std::cos (phase1[channelIndex] + kPhaseOffset1)
                                    + amplitude2 * std::cos (phase2[channelIndex] + kPhaseOffset2)
                                    + amplitude3 * std::cos (phase3[channelIndex] + kPhaseOffset3));
            blockEndAc[channelIndex] = ac;
            return { ac, dcOffsetInSamples };
        }

        [[nodiscard]] float getBlockEndAc (size_t channelIndex) const noexcept
        {
            return blockEndAc[channelIndex];
        }

        [[nodiscard]] float getDcOffsetInSamples() const noexcept
        {
            return dcOffsetInSamples;
        }

        // Advance phase without emitting an output sample.
        inline void skipSample (size_t channelIndex) noexcept
        {
            phase1[channelIndex] += angleDelta1;
            phase2[channelIndex] += angleDelta2;
            phase3[channelIndex] += angleDelta3;
            depthSlew[channelIndex].getNextValue();
        }

        inline void wrapPhase (size_t channelIndex) noexcept
        {
            constexpr float twoPi = juce::MathConstants<float>::twoPi;
            for (float* p : { &phase1[channelIndex], &phase2[channelIndex], &phase3[channelIndex] })
            {
                while (*p >= twoPi) *p -= twoPi;
                while (*p <  0.0f)  *p += twoPi;
            }
        }

        [[nodiscard]] bool isSilent() const noexcept
        {
            return depthSlew.empty()
                || depthSlew.front().getTargetValue() == kDepthFloor;
        }

        void reset() noexcept
        {
            for (auto& slew : depthSlew)
                slew.setCurrentAndTargetValue (kDepthFloor);
            std::fill (phase1.begin(),     phase1.end(),     0.0f);
            std::fill (phase2.begin(),     phase2.end(),     0.0f);
            std::fill (phase3.begin(),     phase3.end(),     0.0f);
            std::fill (blockEndAc.begin(), blockEndAc.end(), 0.0f);

            const auto zero = SIMD_MM (setzero_ps) ();
            for (auto& line : acLines)
                std::fill (line.quads, line.quads + kMaxQuads, zero);
            for (auto& subLine : subEndLines)
                std::fill (subLine.endAc, subLine.endAc + kMaxSubBlocks, 0.0f);
            numSubBlocksThisCall = 0;
        }

    private:
        void resolveChannel (size_t ch, int safeBlock) noexcept
        {
            float startAc = blockEndAc[ch];

            int sampleIdx = 0;
            int subIdx    = 0;
            while (sampleIdx < safeBlock)
            {
                const int subN = std::min (kSubBlockSize, safeBlock - sampleIdx);

                // Advance every phase track by the full sub-block in one step.
                phase1[ch] += angleDelta1 * static_cast<float>(subN);
                phase2[ch] += angleDelta2 * static_cast<float>(subN);
                phase3[ch] += angleDelta3 * static_cast<float>(subN);

                const float depthAtSubEnd = depthSlew[ch].skip (subN);
                const float endAc = depthAtSubEnd
                                  * (amplitude1 * std::cos (phase1[ch] + kPhaseOffset1)
                                   + amplitude2 * std::cos (phase2[ch] + kPhaseOffset2)
                                   + amplitude3 * std::cos (phase3[ch] + kPhaseOffset3));

                fillSubBlockLine (ch, sampleIdx, subN, startAc, endAc);
                subEndLines[ch].endAc[subIdx] = endAc;

                startAc = endAc;
                sampleIdx += subN;
                ++subIdx;
            }

            blockEndAc[ch] = startAc;

            // resolveBlock loops over all channels; only the last channel's
            // count survives, but every channel walks the same sub-block
            // partition so this is consistent.
            numSubBlocksThisCall = subIdx;
        }

        // Fill the per-quad acLine[ch] for a single sub-block of length subN
        // with a linear ramp from startAc (sample sampleIdx) to endAc
        // (sample sampleIdx + subN). Convention matches the existing tap-
        // modulation sub-block crossfade in delay_engine.h: lane k of quad q
        // gets
        //     ac = startAc + ((q << 2) + k) / subN · (endAc - startAc)
        // so lane 0 of the first quad equals startAc and lane 3 of the last
        // quad equals startAc + (subN - 1) / subN · (endAc - startAc), i.e.
        // ALMOST endAc; the next sub-block's startAc closes the small jump.
        void fillSubBlockLine (size_t ch, int sampleIdx, int subN,
                               float startAc, float endAc) noexcept
        {
            const float invSub  = 1.0f / static_cast<float>(subN);
            const auto  vBase   = SIMD_MM (set1_ps) (startAc);
            const auto  vStep   = SIMD_MM (set1_ps) ((endAc - startAc) * invSub);
            const auto  vLane   = SIMD_MM (setr_ps) (0.0f, 1.0f, 2.0f, 3.0f);

            const int subQuads = subN >> 2;
            const int subTail  = subN - (subQuads << 2);
            const int lineQ0   = sampleIdx >> 2;

            for (int sq = 0; sq < subQuads; ++sq)
            {
                const auto vK   = SIMD_MM (set1_ps) (static_cast<float>(sq << 2));
                const auto vIdx = SIMD_MM (add_ps) (vK, vLane);
                const auto vAc  = SIMD_MM (add_ps) (vBase, SIMD_MM (mul_ps) (vStep, vIdx));
                acLines[ch].quads[lineQ0 + sq] = vAc;
            }

            // Sub-block lengths are always multiples of 4 except possibly the
            // very last sub-block of an oddly-sized host block. Build a
            // partial quad scalar-by-scalar in that case.
            if (subTail != 0)
            {
                alignas (16) float tail[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                const float stepScalar  = (endAc - startAc) * invSub;
                const int   tailStartK  = subQuads << 2;
                for (int k = 0; k < subTail; ++k)
                    tail[k] = startAc + stepScalar * static_cast<float>(tailStartK + k);
                acLines[ch].quads[lineQ0 + subQuads] = SIMD_MM (load_ps) (tail);
            }
        }

        static constexpr float kDepthFloor   = 0.001f;
        static constexpr float kPhaseOffset1 = 0.0f;
        static constexpr float kPhaseOffset2 = 13.0f * juce::MathConstants<float>::pi / 4.0f;
        static constexpr float kPhaseOffset3 = -juce::MathConstants<float>::pi / 10.0f;

        float sampleRate        { 48000.0f };
        float amplitude1        { 0.0f };
        float amplitude2        { 0.0f };
        float amplitude3        { 0.0f };
        float dcOffsetInSamples { 0.0f };

        float angleDelta1 { 0.0f };
        float angleDelta2 { 0.0f };
        float angleDelta3 { 0.0f };

        std::vector<float> phase1;
        std::vector<float> phase2;
        std::vector<float> phase3;
        std::vector<float> blockEndAc;
        std::vector<juce::SmoothedValue<float, juce::ValueSmoothingTypes::Multiplicative>> depthSlew;

        // Per-channel piecewise-linear AC line for the most recent host block.
        // Aligned so SIMD load/store of the underlying SIMD_M128 quads is
        // safe regardless of std::vector storage policy.
        struct alignas (16) AcLine
        {
            SIMD_M128 quads[kMaxQuads]{};
        };

        // Per-channel exact end-of-sub-block AC values for the most recent
        // resolveBlock(numSamples). subEndLines[ch].endAc[idx] is the LFO's
        // exact value at sample (idx + 1) * kSubBlockSize - 1 (or the end
        // of the partial last sub-block if the block isn't a multiple of
        // kSubBlockSize).
        struct alignas (16) SubEndLine
        {
            float endAc[kMaxSubBlocks]{};
        };

        std::vector<AcLine>     acLines;
        std::vector<SubEndLine> subEndLines;
        int                     numSubBlocksThisCall { 0 };

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FlutterEngine)
    };
}
#endif

#pragma once

#ifndef CHRONOS_FEEDBACK_DELAY_H
#define CHRONOS_FEEDBACK_DELAY_H

#include "BlockTapReader.h"
#include "FracDelayTap.h"
#include "LinearSmoother.h"
#include "Pow2RingBuffer.h"
#include "nonlinear/ADAA1.h"
#include "nonlinear/ADAA2.h"
#include "nonlinear/Nonlinearities.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numbers>

namespace MarsDSP::Delays {
    class FeedbackDelay {
    public:
        static constexpr float kMaxFeedback     = 1.2f;
        static constexpr float kMinLoopDelay    = 4.0f;   // > FracDelayTap's 3.0 contract
        static constexpr float kMinDriveMakeup  = 1.0f;

        static constexpr int   kMaxChunk    = 64;   // max sub-chunk length (ramp-array footprint)
        static constexpr int   kChunkGuard  = 6;    // interpolator window (base = wIdx - i - 3, len 6 ≤ kTail)

        struct Params
        {
            float delaySamples = 4800.0f;
            float feedback     = 0.0f;    // 0..kMaxFeedback; > 1 self-oscillates, bounded
            float dampHz       = 6000.0f; // one-pole lowpass in the loop
            float crossFeed    = 0.0f;    // 0 straight, 1 full ping-pong
            float loopDrive    = 1.0f;    // how hard repeats lean on the tanh ceiling
            int   satOrder     = 2;       // 0 = hard bypass sat, 1 = ADAA1, 2 = ADAA2
        };

        // Callers must pass the contractual max delay, not a pow2 capacity.
        // prepare() adds (maxBlockSize + kTail + 8) and bit-ceils the sum into
        // the ring capacity; passing an already-pow2 capacity double-rounds it
        // (e.g. 262144 -> 524288, a 2x waste). ChronosEngine passes
        // SimdDelayLine::getMaxDelaySamples() (the pre-rounding max, 240000 @
        // 48 kHz/5000 ms), which rounds once to 262144 = 1 MB/channel. The
        // invariant getMaxDelay() >= maxDelaySamples holds by construction:
        // capacity = bit_ceil(maxDelaySamples + maxBlockSize + 16) >= that sum,
        // so getMaxDelay() = capacity - 10 >= maxDelaySamples + maxBlockSize + 6.
        void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples) noexcept
        {
            assert(sampleRate > 0.0);
            assert(maxBlockSize > 0);
            assert(maxDelaySamples > static_cast<int>(kMinLoopDelay));

            sampleRate_ = sampleRate;
            const int minCap = maxDelaySamples + maxBlockSize
                             + Pow2RingBuffer::kTail + 8;
            ringL_.prepare(minCap);
            ringR_.prepare(minCap);
            maxDelay_ = static_cast<float>(
                ringL_.getCapacity() - Pow2RingBuffer::kTail - 2);

            delaySm_.reset(sampleRate, 0.050);
            fbSm_.reset(sampleRate, 0.020);
            crossSm_.reset(sampleRate, 0.020);
            driveSm_.reset(sampleRate, 0.020);
            reset();
        }

        void reset() noexcept
        {
            ringL_.clear();
            ringR_.clear();
            writeIdx_ = 0;
            adaa1L_.reset(); adaa1R_.reset();
            adaa2L_.reset(); adaa2R_.reset();
            dampL_ = dampR_ = 0.0f;
            dcXL_ = dcXR_ = dcYL_ = dcYR_ = 0.0f;
            firstBlock_ = true;
        }

        void resetParams(const Params& p) noexcept
        {
            applyBlockRate_(p);
            delaySm_.setCurrentAndTargetValue(clampDelay_(p.delaySamples));
            fbSm_.setCurrentAndTargetValue(std::clamp(p.feedback, 0.0f, kMaxFeedback));
            crossSm_.setCurrentAndTargetValue(std::clamp(p.crossFeed, 0.0f, 1.0f));
            driveSm_.setCurrentAndTargetValue(std::clamp(p.loopDrive, 0.1f, 16.0f));
            firstBlock_ = false;
        }

        void setParams(const Params& p) noexcept
        {
            if (firstBlock_) { resetParams(p); return; }
            applyBlockRate_(p);
            delaySm_.setTargetValue(clampDelay_(p.delaySamples));
            fbSm_.setTargetValue(std::clamp(p.feedback, 0.0f, kMaxFeedback));
            crossSm_.setTargetValue(std::clamp(p.crossFeed, 0.0f, 1.0f));
            driveSm_.setTargetValue(std::clamp(p.loopDrive, 0.1f, 16.0f));
        }

        // ──────────────────────────────────────────────────────────────────
        // chunked block processing over the loop-carried distance.
        //
        // The recursion is y[n] = x[n] + N(y[n − D]) — a loop-carried
        // dependency at distance D, where N is damp → DC → cross →
        // drive·tanh(ADAA) → makeup. SimdDelayLine's write-before-read contract
        // (commit the whole block, then read) is the exact negation of this
        // dataflow, which is why the feedback loop cannot be routed through it.
        // But a dependency at distance D licenses processing in chunks of
        // Lc ≤ D − guard samples: every read in the chunk lands ≥ D behind the
        // write head, so no read touches a write from the same chunk. This is
        // the same invariant Diffuser::chunk_ already relies on (kMinDelay 32 >
        // kChunk 16, proven by diffuser_parity); this commit applies it to the
        // feedback loop with a dynamic chunk length.
        //
        // Per sub-chunk:
        //  1. Advance the four smoothers into stack ramps dR[], gR[], crossR[],
        //     driveR[] (Lc values each, fixed-size alignas(16) arrays).
        //  2. Bulk tap read: Lc taps from both rings at the per-sample ramped
        //     delay. When the delay ramp is settled (dR[0] == dR[Lc−1]), hoist
        //     the window acquisition (one windowPtr / readWindow for the whole
        //     chunk via BlockTapReader::acquireWindow) and the Lagrange3
        //     coefficients, then read Lc taps in a tight per-sample dot loop
        //     from the hoisted window — same mul + horizontal-sum op order as
        //     FracDelayTap::read, so satOrder = 0 parity is bit-exact. When the
        //     ramp is moving, fall back to per-sample FracDelayTap::read (the
        //     base varies per sample).
        //  3. Scalar recursive chain over the chunk into w[]: damp one-pole,
        //     DC blocker, cross-mix, ADAA saturate, makeup, isfinite scrub.
        //     The satOrder_ switch is hoisted out of the sample loop (three
        //     specialized inner loops — it is block-rate state). These
        //     recursions are distance-1 in their own state (filter / ADAA
        //     history), not through the ring, so they are legal inside the
        //     chunk; they are irreducibly serial (ADAA is a nonlinear state
        //     recursion — no scan), so do not attempt to vectorize them. ADAA
        //     stays in double (conditioning analysis stands).
        //  4. Bulk write: one writeBlock(w, writeIdx_, Lc) + one
        //     refreshMirror(writeIdx_, Lc) per channel per chunk (replaces
        //     Lc single-float write+mirror pairs).
        //  5. Output: wet[s] = tap[s] from the bulk read.
        //
        // Lc = clamp(int(floor(dMin)) − kChunkGuard, 1, min(kMaxChunk, remaining))
        //   where dMin = max(kMinLoopDelay, min(dCur, dTgt) − satLatency_) — the
        //   minimum read delay over the next Lc smoother steps (the delay
        //   smoother ramps linearly, so min(dCur, dTgt) is the ramp minimum).
        //   kChunkGuard = 6 is the interpolator window (base = writeIdx − i − 3,
        //   window length 6 ≤ kTail). If Lc < 4, fall through to the per-sample
        //   scalar path for this sub-chunk (degenerate only when delay < ~10
        //   samples; kMinLoopDelay = 4 keeps it legal).
        // ──────────────────────────────────────────────────────────────────
        void process(const float* inL, const float* inR,
                     float* wetL, float* wetR, int n) noexcept
        {
            assert(inL != nullptr && wetL != nullptr);
            const bool hasR = (inR != nullptr && wetR != nullptr);
            const int  mask = ringL_.mask();

            int s = 0;
            while (s < n)
            {
                const int remaining = n - s;

                // dMin = minimum read delay over the next Lc smoother steps.
                // The delay smoother ramps linearly from dCur to dTgt, so
                // min(dCur, dTgt) is the ramp minimum.
                const float dCur = delaySm_.getCurrentValue();
                const float dTgt = delaySm_.getTargetValue();
                const float dMin = std::max(kMinLoopDelay,
                    std::min(dCur, dTgt) - satLatency_);

                int Lc = static_cast<int>(std::floor(dMin)) - kChunkGuard;
                Lc = std::clamp(Lc, 1, std::min(kMaxChunk, remaining));

                if (Lc < 4)
                {
                    // Per-sample scalar path (same code as processRef's body).
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float d     = delaySm_.getNextValue();
                        const float g     = fbSm_.getNextValue();
                        const float cross = crossSm_.getNextValue();
                        const float drive = driveSm_.getNextValue();
                        processSampleScalar_(inL + s + i, hasR ? inR + s + i : nullptr,
                                             wetL + s + i, hasR ? wetR + s + i : nullptr,
                                             d, g, cross, drive, hasR, mask);
                    }
                    s += Lc;
                    continue;
                }

                // 1. Advance smoothers into stack ramps.
                alignas(16) float dR[kMaxChunk], gR[kMaxChunk],
                                 crossR[kMaxChunk], driveR[kMaxChunk];
                for (int i = 0; i < Lc; ++i)
                {
                    dR[i]     = delaySm_.getNextValue();
                    gR[i]     = fbSm_.getNextValue();
                    crossR[i] = crossSm_.getNextValue();
                    driveR[i] = driveSm_.getNextValue();
                }

                // 2. Bulk tap read.
                alignas(16) float tapL[kMaxChunk], tapR[kMaxChunk];
                const bool settled = (dR[0] == dR[Lc - 1]);

                if (settled)
                {
                    // Hoist window + coefficients; per-sample dot from the
                    // hoisted window (same mul + horizontal-sum op order as
                    // FracDelayTap::read → bit-exact at satOrder = 0).
                    const float readDelay = std::max(kMinLoopDelay, dR[0] - satLatency_);
                    const auto  iInt = static_cast<int>(readDelay);
                    const float f = readDelay - static_cast<float>(iInt);
                    const FracDelayTap::Coeffs4 k = FracDelayTap::lagrange3(f);
                    const int base = (writeIdx_ - iInt - 3) & mask;
                    const int winLen = Lc + 6;
                    const M128 cf = MM(set_ps)(k.c4, k.c3, k.c2, k.c1);

                    const auto wL = BlockTapReader::acquireWindow(ringL_, base, winLen, tapWinL_.data());
                    const float* winL = wL.ptr;
                    for (int i = 0; i < Lc; ++i)
                    {
                        const M128 taps = MM(loadu_ps)(winL + i + 1);
                        const M128 prod = MM(mul_ps)(taps, cf);
                        const M128 sh1  = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
                        const M128 sh2  = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
                        tapL[i] = MM(cvtss_f32)(sh2);
                    }

                    if (hasR)
                    {
                        const auto wR = BlockTapReader::acquireWindow(ringR_, base, winLen, tapWinR_.data());
                        const float* winR = wR.ptr;
                        for (int i = 0; i < Lc; ++i)
                        {
                            const M128 taps = MM(loadu_ps)(winR + i + 1);
                            const M128 prod = MM(mul_ps)(taps, cf);
                            const M128 sh1  = MM(add_ps)(prod, MM(movehl_ps)(prod, prod));
                            const M128 sh2  = MM(add_ss)(sh1, MM(shuffle_ps)(sh1, sh1, MM_SHUFFLE(0, 0, 0, 1)));
                            tapR[i] = MM(cvtss_f32)(sh2);
                        }
                    }
                }
                else
                {
                    // Per-sample FracDelayTap::read (the base varies per sample).
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float readDelay = std::max(kMinLoopDelay, dR[i] - satLatency_);
                        tapL[i] = FracDelayTap::read(ringL_, writeIdx_ + i, readDelay);
                        if (hasR)
                            tapR[i] = FracDelayTap::read(ringR_, writeIdx_ + i, readDelay);
                    }
                }

                // When mono, tapR = tapL (same as processRef's
                // `tapR = hasR ? ... : tapL`). The damp/DC/cross chain reads
                // tapR[i] unconditionally to keep dampR_ tracking dampL_.
                if (!hasR)
                    for (int i = 0; i < Lc; ++i) tapR[i] = tapL[i];

                // 3. Scalar recursive chain: damp → DC → cross (common), then
                //    saturate with hoisted sat switch. vL[]/vR[] bridge the two
                //    loops (stack arrays — storing a float to memory and reading
                //    it back is bit-exact, so parity is preserved).
                alignas(16) float vL[kMaxChunk], vR[kMaxChunk];
                for (int i = 0; i < Lc; ++i)
                {
                    dampL_ += dampG_ * (tapL[i] - dampL_);
                    dampR_ += dampG_ * (tapR[i] - dampR_);

                    const float hL = dampL_ - dcXL_ + dcR_ * dcYL_;
                    dcXL_ = dampL_; dcYL_ = hL;
                    const float hR = dampR_ - dcXR_ + dcR_ * dcYR_;
                    dcXR_ = dampR_; dcYR_ = hR;

                    const float g = gR[i], cross = crossR[i];
                    vL[i] = g * ((1.0f - cross) * hL + cross * hR);
                    vR[i] = g * ((1.0f - cross) * hR + cross * hL);
                }

                alignas(16) float wL[kMaxChunk], wR[kMaxChunk];
                if (satOrder_ == 0)
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / std::max(driveR[i], kMinDriveMakeup);
                        const float sL = std::clamp(driveR[i] * vL[i], -1.0f, 1.0f) * makeup;
                        const float sR = hasR ? std::clamp(driveR[i] * vR[i], -1.0f, 1.0f) * makeup : sL;
                        wL[i] = inL[s + i] + sL;
                        if (!std::isfinite(wL[i])) wL[i] = 0.0f;
                        if (hasR) { wR[i] = inR[s + i] + sR; if (!std::isfinite(wR[i])) wR[i] = 0.0f; }
                    }
                }
                else if (satOrder_ == 1)
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / std::max(driveR[i], kMinDriveMakeup);
                        const float sL = static_cast<float>(adaa1L_.process(static_cast<double>(driveR[i] * vL[i]))) * makeup;
                        const float sR = hasR ? static_cast<float>(adaa1R_.process(static_cast<double>(driveR[i] * vR[i]))) * makeup : sL;
                        wL[i] = inL[s + i] + sL;
                        if (!std::isfinite(wL[i])) wL[i] = 0.0f;
                        if (hasR) { wR[i] = inR[s + i] + sR; if (!std::isfinite(wR[i])) wR[i] = 0.0f; }
                    }
                }
                else
                {
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float makeup = 1.0f / std::max(driveR[i], kMinDriveMakeup);
                        const float sL = static_cast<float>(adaa2L_.process(static_cast<double>(driveR[i] * vL[i]))) * makeup;
                        const float sR = hasR ? static_cast<float>(adaa2R_.process(static_cast<double>(driveR[i] * vR[i]))) * makeup : sL;
                        wL[i] = inL[s + i] + sL;
                        if (!std::isfinite(wL[i])) wL[i] = 0.0f;
                        if (hasR) { wR[i] = inR[s + i] + sR; if (!std::isfinite(wR[i])) wR[i] = 0.0f; }
                    }
                }

                // 4. Bulk write (one writeBlock + refreshMirror per channel).
                ringL_.writeBlock(wL, writeIdx_, Lc);
                ringL_.refreshMirror(writeIdx_, Lc);
                if (hasR)
                {
                    ringR_.writeBlock(wR, writeIdx_, Lc);
                    ringR_.refreshMirror(writeIdx_, Lc);
                }
                writeIdx_ = (writeIdx_ + Lc) & mask;

                // 5. Output: wet = taps.
                for (int i = 0; i < Lc; ++i)
                {
                    wetL[s + i] = tapL[i];
                    if (hasR) wetR[s + i] = tapR[i];
                }

                s += Lc;
            }
        }

        // reference only -- do not optimize, do not delete.
        void processRef(const float* inL, const float* inR,
                        float* wetL, float* wetR, int n) noexcept
        {
            assert(inL != nullptr && wetL != nullptr);
            const bool hasR = (inR != nullptr && wetR != nullptr);
            const int  mask = ringL_.mask();

            for (int s = 0; s < n; ++s)
            {
                const float d     = delaySm_.getNextValue();
                const float g     = fbSm_.getNextValue();
                const float cross = crossSm_.getNextValue();
                const float drive = driveSm_.getNextValue();
                processSampleScalar_(inL + s, hasR ? inR + s : nullptr,
                                     wetL + s, hasR ? wetR + s : nullptr,
                                     d, g, cross, drive, hasR, mask);
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept { return 0; }
        [[nodiscard]] float getMaxDelay() const noexcept { return maxDelay_; }

    private:
        float clampDelay_(float d) const noexcept
        {
            return std::clamp(d, kMinLoopDelay + 1.5f, maxDelay_);
        }

        void applyBlockRate_(const Params& p) noexcept
        {
            const double fc = std::clamp(static_cast<double>(p.dampHz),
                                         20.0, 0.45 * sampleRate_);
            const double gw = std::tan(std::numbers::pi * fc / sampleRate_);
            dampG_ = static_cast<float>(gw / (1.0 + gw));

            // DC blocker pole: ~8 Hz, R = exp(-2*pi*fc/fs).
            dcR_ = static_cast<float>(
                std::exp(-2.0 * std::numbers::pi * 8.0 / sampleRate_));

            satOrder_ = std::clamp(p.satOrder, 0, 2);
            satLatency_ = (satOrder_ == 2) ? 1.0f
                        : (satOrder_ == 1) ? 0.5f
                                           : 0.0f;
        }

        float saturate_(Nonlinear::ADAA1<Nonlinear::TanhNL>& a1,
                        Nonlinear::ADAA2<Nonlinear::TanhNL>& a2,
                        float x) noexcept
        {
            switch (satOrder_)
            {
                case 2:  return static_cast<float>(a2.process(static_cast<double>(x)));
                case 1:  return static_cast<float>(a1.process(static_cast<double>(x)));
                default: return std::clamp(x, -1.0f, 1.0f);
            }
        }

        // the one scalar per-sample implementation. Called by processRef
        // (the reference twin) and by the Lc < 4 fallback in process. Reads a
        // tap, runs the recursive chain (damp → DC → cross → saturate →
        // makeup), writes to the ring, and outputs the raw tap as wet.
        // Identical op order to the pre process() body.
        void processSampleScalar_(const float* in, const float* inR,
                                   float* wet, float* wetR,
                                   float d, float g, float cross, float drive,
                                   bool hasR, int mask) noexcept
        {
            const float makeup = 1.0f / std::max(drive, kMinDriveMakeup);
            const float readDelay =
                std::max(kMinLoopDelay, d - satLatency_);

            const float tapL = FracDelayTap::read(ringL_, writeIdx_, readDelay);
            const float tapR = hasR
                ? FracDelayTap::read(ringR_, writeIdx_, readDelay)
                : tapL;

            dampL_ += dampG_ * (tapL - dampL_);
            dampR_ += dampG_ * (tapR - dampR_);

            const float hL = dampL_ - dcXL_ + dcR_ * dcYL_;
            dcXL_ = dampL_; dcYL_ = hL;
            const float hR = dampR_ - dcXR_ + dcR_ * dcYR_;
            dcXR_ = dampR_; dcYR_ = hR;

            const float vL = g * ((1.0f - cross) * hL + cross * hR);
            const float vR = g * ((1.0f - cross) * hR + cross * hL);

            const float sL = saturate_(adaa1L_, adaa2L_, drive * vL) * makeup;
            const float sR = hasR
                ? saturate_(adaa1R_, adaa2R_, drive * vR) * makeup
                : sL;

            float wL = *in + sL;
            if (!std::isfinite(wL)) wL = 0.0f;
            ringL_.writeBlock(&wL, writeIdx_, 1);
            ringL_.refreshMirror(writeIdx_, 1);

            if (hasR)
            {
                float wR = *inR + sR;
                if (!std::isfinite(wR)) wR = 0.0f;
                ringR_.writeBlock(&wR, writeIdx_, 1);
                ringR_.refreshMirror(writeIdx_, 1);
            }

            writeIdx_ = (writeIdx_ + 1) & mask;

            *wet = tapL;
            if (hasR) *wetR = tapR;
        }

        Pow2RingBuffer ringL_, ringR_;
        int   writeIdx_ = 0;
        float maxDelay_ = 0.0f;
        double sampleRate_ = 48000.0;
        bool  firstBlock_ = true;

        Smoothers::LinearSmoother<float> delaySm_, fbSm_, crossSm_, driveSm_;

        // block-rate coefficients
        float dampG_ = 0.0f;
        float dcR_   = 0.999f;
        int   satOrder_ = 2;
        float satLatency_ = 1.0f;

        // per-channel loop state
        float dampL_ = 0.0f, dampR_ = 0.0f;
        float dcXL_ = 0.0f, dcYL_ = 0.0f, dcXR_ = 0.0f, dcYR_ = 0.0f;

        Nonlinear::ADAA1<Nonlinear::TanhNL> adaa1L_, adaa1R_;
        Nonlinear::ADAA2<Nonlinear::TanhNL> adaa2L_, adaa2R_;

        // scratch for the settled bulk-tap-read window fallback (when the
        // window wraps past capacity, readWindow copies into here). Sized for
        // the largest chunk: kMaxChunk + 6 taps ≤ kMaxChunk + kTail.
        alignas(16) std::array<float, static_cast<std::size_t>(kMaxChunk) + Pow2RingBuffer::kTail> tapWinL_{};
        alignas(16) std::array<float, static_cast<std::size_t>(kMaxChunk) + Pow2RingBuffer::kTail> tapWinR_{};
    };
}
#endif

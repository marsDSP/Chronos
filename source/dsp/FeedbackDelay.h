#pragma once

#ifndef CHRONOS_FEEDBACK_DELAY_H
#define CHRONOS_FEEDBACK_DELAY_H

#include "BlockTapReader.h"
#include "Diffuser.h"
#include "FracDelayTap.h"
#include "LinearSmoother.h"
#include "Pow2RingBuffer.h"
#include "nonlinear/ADAA1.h"
#include "nonlinear/ADAA2.h"
#include "nonlinear/Nonlinearities.h"
#include "utils/memory/BumpArena.h"

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
            int   satOrder     = 2;       // 0 hard, 1 ADAA1, 2 ADAA2
            // C7c: in-loop diffuser. The loop tap stream passes through the
            // diffuser before the recursion (and the output), so repeat n has
            // n diffusion passes (progressive wash). The loop tap is read at
            // d - satLatency_ - fade*baseTransport, where baseTransport is
            // the cascade's exact energy centroid at every g — so the loop
            // period stays exactly d per pass at all diffusion settings.
            bool  enableDiffuser = false;
            float diffusion      = 0.7f;  // 0..1 -> allpass coeff 0..0.92
            float diffuserSize   = 0.5f;  // 0..1 (0 = full path lengths)
            float diffModDepth   = 16.0f; // samples, 0..62
            float diffModRateHz  = 0.5f;  // 0..8
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
            prepareImpl_(sampleRate, maxBlockSize, maxDelaySamples, nullptr);
        }

        // C9b: carve both rings from the caller's arena (sized via
        // ringStorageFloats) instead of owning them.
        void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples,
                     Memory::BumpArena& arena) noexcept
        {
            prepareImpl_(sampleRate, maxBlockSize, maxDelaySamples, &arena);
        }

        // floats an arena must supply for both rings plus the in-loop
        // diffuser's 16 section rings: token-identical minCap arithmetic to
        // prepareImpl_, so the carve fits exactly.
        static std::size_t ringStorageFloats(double sampleRate, int maxBlockSize,
                                             int maxDelaySamples) noexcept
        {
            const int minCap = maxDelaySamples + maxBlockSize
                             + Pow2RingBuffer::kTail + 8;
            return 2 * Pow2RingBuffer::arenaFloatsFor(minCap)
                 + Diffusion::Diffuser::ringStorageFloats(sampleRate);
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
            diffuser_.reset();
            enableDiffuser_ = false;
            diffState_ = DiffuserState::Off;
            diffFade_ = 0.0f;
            firstBlock_ = true;
        }

        void resetParams(const Params& p) noexcept
        {
            applyBlockRate_(p);
            delaySm_.setCurrentAndTargetValue(clampDelay_(p.delaySamples));
            fbSm_.setCurrentAndTargetValue(std::clamp(p.feedback, 0.0f, kMaxFeedback));
            crossSm_.setCurrentAndTargetValue(std::clamp(p.crossFeed, 0.0f, 1.0f));
            driveSm_.setCurrentAndTargetValue(std::clamp(p.loopDrive, 0.1f, 16.0f));
            applyDiffuserParams_(p);
            diffuser_.prime();   // snap smoothers to the targets just set, clear rings
            enableDiffuser_ = p.enableDiffuser;
            diffState_ = enableDiffuser_ ? DiffuserState::On : DiffuserState::Off;
            diffFade_ = enableDiffuser_ ? 1.0f : 0.0f;
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
            applyDiffuserParams_(p);
            enableDiffuser_ = p.enableDiffuser;
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

            diffuserTransition_();   // block-rate enable edge (primes on rising)

            int s = 0;
            while (s < n)
            {
                const int remaining = n - s;

                // C7c: in-loop diffuser offset. The loop tap is read at
                // d - satLatency_ - fade*baseT, where baseT is the cascade's
                // exact energy centroid (g-invariant), so the loop period
                // stays exactly d per pass at all diffusion settings. The
                // guard uses the state at sub-chunk start (conservative
                // during fades: the full baseT is subtracted even though the
                // fade ramps up through the sub-chunk).
                const float baseT = (diffState_ != DiffuserState::Off)
                    ? diffuser_.transportSamples() : 0.0f;

                // dMin = minimum read delay over the next Lc smoother steps.
                // The loop tap itself is the most restrictive read (it carries
                // the diffuser offset).
                const float dCur = delaySm_.getCurrentValue();
                const float dTgt = delaySm_.getTargetValue();
                const float dMin = std::max(kMinLoopDelay,
                    std::min(dCur, dTgt) - satLatency_ - baseT);

                int Lc = static_cast<int>(std::floor(dMin)) - kChunkGuard;
                Lc = std::clamp(Lc, 1, std::min(kMaxChunk, remaining));

                if (Lc < 4)
                {
                    // Per-sample scalar path (same code as processRef's body).
                    const bool runDiff = (diffState_ != DiffuserState::Off);
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float d     = delaySm_.getNextValue();
                        const float g     = fbSm_.getNextValue();
                        const float cross = crossSm_.getNextValue();
                        const float drive = driveSm_.getNextValue();
                        const float fade  = fadeStep_();
                        processSampleScalar_(inL + s + i, hasR ? inR + s + i : nullptr,
                                             wetL + s + i, hasR ? wetR + s + i : nullptr,
                                             d, g, cross, drive, hasR, mask,
                                             fade, runDiff ? baseT : 0.0f);
                    }
                    s += Lc;
                    continue;
                }

                // 1. Advance smoothers into stack ramps. The diffuser fade is
                //    advanced here too (one step per sample, shared L/R).
                alignas(16) float dR[kMaxChunk], gR[kMaxChunk],
                                 crossR[kMaxChunk], driveR[kMaxChunk], fadeR[kMaxChunk];
                const bool wasRunning = (diffState_ != DiffuserState::Off);
                for (int i = 0; i < Lc; ++i)
                {
                    dR[i]     = delaySm_.getNextValue();
                    gR[i]     = fbSm_.getNextValue();
                    crossR[i] = crossSm_.getNextValue();
                    driveR[i] = driveSm_.getNextValue();
                    fadeR[i]  = fadeStep_();
                }
                // Run the diffuser for this sub-chunk if it was running at the
                // start OR is still running now (a fade may finish mid-chunk;
                // the leading samples still need diffused values to blend).
                const bool runDiff = wasRunning || (diffState_ != DiffuserState::Off);

                // 2. Bulk tap read (at d - satLatency_ - fade*baseT).
                alignas(16) float tapL[kMaxChunk], tapR[kMaxChunk];
                const bool settled = (dR[0] == dR[Lc - 1]) && (fadeR[0] == fadeR[Lc - 1]);

                if (settled)
                {
                    // Hoist window + coefficients; per-sample dot from the
                    // hoisted window (same mul + horizontal-sum op order as
                    // FracDelayTap::read → bit-exact at satOrder = 0).
                    const float readDelay = std::max(kMinLoopDelay,
                        dR[0] - satLatency_ - fadeR[0] * baseT);
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
                        const float readDelay = std::max(kMinLoopDelay,
                            dR[i] - satLatency_ - fadeR[i] * baseT);
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

                // 2b. C7c: in-loop diffuser pass + toggle blend. The tap stream
                //     passes through the diffuser BEFORE the recursion (and
                //     the output), so repeat n gets n diffusion passes — the
                //     progressive wash. The blend out = raw*(1-a) + diff*a
                //     shares the fade with the tap offset (fade*baseT above):
                //     the blend's energy centroid is fade*baseT exactly, so
                //     the comp stays exact throughout the fade. Scalar on
                //     purpose: identical op order to processSampleScalar_
                //     (bit-exact, no FMA-contraction ambiguity).
                if (runDiff)
                {
                    alignas(16) float rawL[kMaxChunk], rawR[kMaxChunk];
                    std::memcpy(rawL, tapL, static_cast<std::size_t>(Lc) * sizeof(float));
                    std::memcpy(rawR, tapR, static_cast<std::size_t>(Lc) * sizeof(float));
                    diffuser_.processBlock(tapL, hasR ? tapR : nullptr, Lc);
                    for (int i = 0; i < Lc; ++i)
                    {
                        const float a = fadeR[i];
                        tapL[i] = rawL[i] * (1.0f - a) + tapL[i] * a;
                        if (hasR)
                            tapR[i] = rawR[i] * (1.0f - a) + tapR[i] * a;
                        else
                            tapR[i] = tapL[i];   // mono: mirror the blended L
                    }
                }

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

                // 5. Output: wet = the (diffused, blended) loop-tap stream.
                //    When the diffuser is off this is the raw loop tap —
                //    pre-C7c behavior, bit-exact.
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

            diffuserTransition_();

            for (int s = 0; s < n; ++s)
            {
                const float baseT = (diffState_ != DiffuserState::Off)
                    ? diffuser_.transportSamples() : 0.0f;
                const float d     = delaySm_.getNextValue();
                const float g     = fbSm_.getNextValue();
                const float cross = crossSm_.getNextValue();
                const float drive = driveSm_.getNextValue();
                const float fade  = fadeStep_();
                processSampleScalar_(inL + s, hasR ? inR + s : nullptr,
                                     wetL + s, hasR ? wetR + s : nullptr,
                                     d, g, cross, drive, hasR, mask, fade, baseT);
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept { return 0; }
        [[nodiscard]] float getMaxDelay() const noexcept { return maxDelay_; }

    private:
        // C7c: diffuser enable-toggle state machine (same semantics the
        // engine-level machine had, but the fade now lives INSIDE the loop:
        // one variable drives both the tap-stream blend and the loop-tap
        // offset fade*baseT, which stays comp-exact throughout the fade).
        enum class DiffuserState { Off, FadingIn, On, FadingOut };
        static constexpr int   kDiffuserFadeSamples = 480;  // ~10 ms @48 kHz
        static constexpr float kDiffuserFadeInc =
            1.0f / static_cast<float>(kDiffuserFadeSamples);

        void diffuserTransition_() noexcept
        {
            const bool wantOn = enableDiffuser_;
            if (wantOn && diffState_ == DiffuserState::Off)
            {
                diffuser_.prime();   // no stale audio replays
                diffState_ = DiffuserState::FadingIn;
                diffFade_ = 0.0f;
            }
            else if (!wantOn && diffState_ == DiffuserState::On)
                diffState_ = DiffuserState::FadingOut;
            else if (wantOn && diffState_ == DiffuserState::FadingOut)
                diffState_ = DiffuserState::FadingIn;    // reverse: rings warm
            else if (!wantOn && diffState_ == DiffuserState::FadingIn)
                diffState_ = DiffuserState::FadingOut;   // reverse
        }

        // advance the fade one sample, returning the PRE-step weight for this
        // sample (0 = raw tap, 1 = diffused tap). Transitions to On/Off at
        // the rails.
        float fadeStep_() noexcept
        {
            const float a = diffFade_;
            if (diffState_ == DiffuserState::FadingIn)
            {
                diffFade_ += kDiffuserFadeInc;
                if (diffFade_ >= 1.0f)
                {
                    diffFade_ = 1.0f;
                    diffState_ = DiffuserState::On;
                }
            }
            else if (diffState_ == DiffuserState::FadingOut)
            {
                diffFade_ -= kDiffuserFadeInc;
                if (diffFade_ <= 0.0f)
                {
                    diffFade_ = 0.0f;
                    diffState_ = DiffuserState::Off;
                }
            }
            return a;
        }

        void applyDiffuserParams_(const Params& p) noexcept
        {
            diffuser_.setDiffusion(p.diffusion);
            diffuser_.setSize(p.diffuserSize);
            diffuser_.setModDepthSamples(p.diffModDepth);
            diffuser_.setModRateHz(p.diffModRateHz);
        }

        void prepareImpl_(double sampleRate, int maxBlockSize, int maxDelaySamples,
                          Memory::BumpArena* arena) noexcept
        {
            assert(sampleRate > 0.0);
            assert(maxBlockSize > 0);
            assert(maxDelaySamples > static_cast<int>(kMinLoopDelay));

            sampleRate_ = sampleRate;
            const int minCap = maxDelaySamples + maxBlockSize
                             + Pow2RingBuffer::kTail + 8;
            if (arena != nullptr)
            {
                ringL_.prepare(minCap, *arena);
                ringR_.prepare(minCap, *arena);
                diffuser_.prepare(sampleRate, *arena);
            }
            else
            {
                ringL_.prepare(minCap);
                ringR_.prepare(minCap);
                diffuser_.prepare(sampleRate);
            }
            maxDelay_ = static_cast<float>(
                ringL_.getCapacity() - Pow2RingBuffer::kTail - 2);

            delaySm_.reset(sampleRate, 0.050);
            fbSm_.reset(sampleRate, 0.020);
            crossSm_.reset(sampleRate, 0.020);
            driveSm_.reset(sampleRate, 0.020);
            reset();
        }

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
        // (the reference twin) and by the Lc < 4 fallback in process. Reads
        // the loop tap at d - satLatency_ - fade*baseT, passes it through the
        // in-loop diffuser (blended by the toggle fade), runs the recursive
        // chain (damp → DC → cross → saturate → makeup), writes to the ring,
        // and outputs the blended tap stream as wet. Identical op order to
        // the chunked process() body (the blend stays scalar so FMA
        // contraction cannot diverge between the two).
        void processSampleScalar_(const float* in, const float* inR,
                                   float* wet, float* wetR,
                                   float d, float g, float cross, float drive,
                                   bool hasR, int mask,
                                   float fade, float baseT) noexcept
        {
            const float makeup = 1.0f / std::max(drive, kMinDriveMakeup);
            const float readDelay =
                std::max(kMinLoopDelay, d - satLatency_ - fade * baseT);

            float tapL = FracDelayTap::read(ringL_, writeIdx_, readDelay);
            float tapR = hasR
                ? FracDelayTap::read(ringR_, writeIdx_, readDelay)
                : tapL;

            if (baseT > 0.0f)   // diffuser running: diffuse, then fade-blend
            {
                float diffL = tapL;
                float diffR = tapR;
                diffuser_.processBlockRef(&diffL, hasR ? &diffR : nullptr, 1);
                tapL = tapL * (1.0f - fade) + diffL * fade;
                if (hasR)
                    tapR = tapR * (1.0f - fade) + diffR * fade;
                else
                    tapR = tapL;   // mono: mirror the blended L
            }

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

            *wet = tapL;   // the blended (diffused when on) loop-tap stream
            if (hasR) *wetR = tapR;
        }

        Pow2RingBuffer ringL_;
        Pow2RingBuffer ringR_;
        int writeIdx_ = 0;
        float maxDelay_ = 0.0f;
        double sampleRate_ = 48000.0;
        bool  firstBlock_ = true;

        Smoothers::LinearSmoother<float> delaySm_;
        Smoothers::LinearSmoother<float> fbSm_;
        Smoothers::LinearSmoother<float> crossSm_;
        Smoothers::LinearSmoother<float> driveSm_;

        // block-rate coefficients
        float dampG_ = 0.0f;
        float dcR_   = 0.999f;
        int   satOrder_ = 2;
        float satLatency_ = 1.0f;

        // per-channel loop state
        float dampL_ = 0.0f;
        float dampR_ = 0.0f;
        float dcXL_ = 0.0f;
        float dcYL_ = 0.0f;
        float dcXR_ = 0.0f;
        float dcYR_ = 0.0f;

        Nonlinear::ADAA1<Nonlinear::TanhNL> adaa1L_;
        Nonlinear::ADAA1<Nonlinear::TanhNL> adaa1R_;
        Nonlinear::ADAA2<Nonlinear::TanhNL> adaa2L_;
        Nonlinear::ADAA2<Nonlinear::TanhNL> adaa2R_;

        // scratch for the settled bulk-tap-read window fallback (when the
        // window wraps past capacity, readWindow copies into here). Sized for
        // the largest chunk: kMaxChunk + 6 taps ≤ kMaxChunk + kTail.
        alignas(16) std::array<float, static_cast<std::size_t>(kMaxChunk) + Pow2RingBuffer::kTail> tapWinL_{};
        alignas(16) std::array<float, static_cast<std::size_t>(kMaxChunk) + Pow2RingBuffer::kTail> tapWinR_{};

        // C7c: the in-loop diffuser and its enable-toggle fade state. The
        // diffuser sits between the loop-tap read and the damp/DC/cross/sat
        // recursion; its output is both the wet stream and the recursion
        // input. Off (fade settled 0) = bypassed: raw tap, no diffuser work,
        // pre-C7c bit-exact behavior.
        Diffusion::Diffuser diffuser_;
        bool enableDiffuser_ = false;
        DiffuserState diffState_ = DiffuserState::Off;
        float diffFade_ = 0.0f;   // 0 = raw tap, 1 = diffused tap
    };
}
#endif

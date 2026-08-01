// tests/harnesses/dsp/fb_parity.cpp
// ──────────────────────────────────────────────────────────────────────────
// FeedbackDelay parity: process (chunked, C3) vs processRef (per-sample
// reference twin, // reference only -- do not optimize, do not delete).
//
// The chunked process() processes sub-chunks of Lc ≤ D − kChunkGuard samples
// (D = read delay), exploiting the loop-carried distance to read all taps
// before writing any. processRef() is the verbatim pre-C3 per-sample loop.
// Both share processSampleScalar_ (the one scalar implementation) for the
// Lc < 4 fallback, so the degenerate path is identical by construction.
//
// Parity subtlety (why the gate is split):
//  * satOrder = 0 (hard clamp, no FMA in the chain): the chunked bulk-tap-read
//    uses the identical FracDelayTap::read op order (mul + horizontal sum,
//    hoisted window when settled, per-sample FracDelayTap::read when moving).
//    The recursive chain (damp/DC/cross/saturate) is the same per-sample ops
//    in the same order. Storing vL[i] to a stack float and reading it back is
//    bit-exact. So satOrder = 0 requires BIT-EXACT parity — this validates the
//    chunk structure (indices, chunk boundaries, mirror refresh, write order)
//    exactly.
//  * satOrder ∈ {1,2}: ADAA is a nonlinear state recursion in double. Even
//    with identical per-sample ops, a recirculating system can compound
//    float32 rounding differences per loop pass if any op reorders. The
//    chunked path's per-sample ops are identical to processRef's, so we
//    EXPECT bit-exact, but we gate with a per-sample relative tolerance 1e-5
//    over the first 2·D samples (one loop period of recirculation) plus an
//    energy-envelope gate (per-1024-sample RMS ratio within ±0.1 dB) over
//    ≥ 10 loop passes at feedback 0.95 — divergence must stay bounded-noise,
//    not systematic. If the chunked path is truly bit-exact (same op order),
//    the tolerance is never reached; the envelope gate is a safety net.
//
// Matrix: delay ∈ {5, 7, 12, 48, 480, 4800} (first three force the degenerate
//   scalar path and the Lc boundary), feedback ∈ {0, 0.5, 0.95, 1.2}, cross ∈
//   {0, 0.37, 1}, block ∈ {1, 17, 64, 256, 512}, mono + stereo. Plus a
//   delay-automation ramp (sweep delay across blocks → mid-ramp smoother,
//   crossing chunk boundaries) and a dampHz/crossFeed automation case.
//
// C7 output-tap section: setOutputTapOffset(offset) on BOTH instances (the
// diffuser base-transport comp; the output tap reads the ring at
// d − satLatency − offset while the loop tap stays at d − satLatency). Cells
// are chosen to hit every output-tap path against processRef:
//   * offset 37.5 at delay 480/4800: settled bulk read AND the per-sample
//     walk (automation), Lc participates via dMinOut;
//   * offset 37.5 at delay 12: dMinOut clamps Lc < 4 → scalar fallback WITH
//     offset > 0 (per-sample FracDelayTap::read output tap);
//   * offset 100 at delay 48: output tap clamps at kMinLoopDelay (repeats
//     land late by the remainder — documented C7 clamp semantics);
//   * offset 2930 at delay 4800: realistic diffuser base transport (size 0).
// satOrder = 0 stays BIT-EXACT (the bulk read uses the identical mul +
// horizontal-sum op order as FracDelayTap::read, as in the loop-tap path).
//
// Conventions (matching latency_null_check / chain_parity): plain main(), exit
// code, printf, always-live CHECK/FAIL. Links SharedCode only; no JUCE.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/FeedbackDelay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {

constexpr double kFs        = 48000.0;
constexpr double kPi        = std::numbers::pi_v<double>;
constexpr int    kMaxDelay  = 262144;   // matches the engine's fb ring capacity
constexpr int    kTotal     = 1 << 16;  // 65536; > 10*4800 = 48000 for the energy envelope
constexpr float  kDampHz    = 6000.0f;
constexpr float  kLoopDrive = 3.981f;   // ~12 dB linear

using MarsDSP::Delays::FeedbackDelay;

const char* g_section = "(startup)";
int g_bitExactOk = 0;
int g_bitExactFail = 0;
int g_tolOk = 0;
int g_envOk = 0;
double g_worstRel = 0.0;
double g_worstEnvDb = 0.0;

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

struct Cfg
{
    int   delay;
    float feedback;
    float cross;
    int   satOrder;
    int   block;
    bool  stereo;
    float outOffset = 0.0f;   // C7: output-tap offset (0 = pre-C7 behavior)
};

// Per-1024-sample RMS of a buffer, in dB relative to a reference RMS.
double rmsDb(const std::vector<float>& x, int start, int len)
{
    double sumSq = 0.0;
    for (int i = 0; i < len; ++i)
        sumSq += static_cast<double>(x[static_cast<std::size_t>(start + i)]) *
                 static_cast<double>(x[static_cast<std::size_t>(start + i)]);
    return 20.0 * std::log10(std::sqrt(sumSq / static_cast<double>(len)) + 1e-30);
}

void runOne(const Cfg& c, bool automateDelay, bool automateDampCross)
{
    FeedbackDelay fast, ref;
    fast.prepare(kFs, c.block, kMaxDelay);
    ref.prepare(kFs, c.block, kMaxDelay);

    FeedbackDelay::Params p;
    p.delaySamples = static_cast<float>(c.delay);
    p.feedback     = c.feedback;
    p.crossFeed    = c.cross;
    p.dampHz       = kDampHz;
    p.loopDrive    = kLoopDrive;
    p.satOrder     = c.satOrder;
    fast.resetParams(p);
    ref.resetParams(p);
    fast.setOutputTapOffset(c.outOffset);   // C7: same offset on both, so the
    ref.setOutputTapOffset(c.outOffset);    // comparison isolates structure

    const bool hasR = c.stereo;
    std::vector<float> inL(static_cast<std::size_t>(kTotal));
    std::vector<float> inR(static_cast<std::size_t>(kTotal));
    for (int i = 0; i < kTotal; ++i)
    {
        const auto u = static_cast<std::size_t>(i);
        inL[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / kFs));
        inR[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 330.0 * static_cast<double>(i) / kFs));
    }

    std::vector<float> fL(static_cast<std::size_t>(kTotal));
    std::vector<float> fR(static_cast<std::size_t>(kTotal));
    std::vector<float> rL(static_cast<std::size_t>(kTotal));
    std::vector<float> rR(static_cast<std::size_t>(kTotal));

    for (int off = 0; off < kTotal; off += c.block)
    {
        const int n = std::min(c.block, kTotal - off);

        if (automateDelay)
        {
            // Sweep delay across blocks: oscillate ±40% around the base.
            const float frac = static_cast<float>(off) / static_cast<float>(kTotal);
            const float swing = 0.4f * std::sin(2.0f * std::numbers::pi_v<float> * frac);
            p.delaySamples = static_cast<float>(c.delay) * (1.0f + swing);
            p.delaySamples = std::max(p.delaySamples, FeedbackDelay::kMinLoopDelay + 2.0f);
            fast.setParams(p);
            ref.setParams(p);
        }
        if (automateDampCross)
        {
            // Oscillate dampHz and crossFeed across blocks.
            const float frac = static_cast<float>(off) / static_cast<float>(kTotal);
            p.dampHz    = 3000.0f + 6000.0f * (0.5f + 0.5f * std::sin(2.0f * std::numbers::pi_v<float> * frac));
            p.crossFeed = 0.5f + 0.5f * std::sin(2.0f * std::numbers::pi_v<float> * frac * 1.3f);
            p.crossFeed = std::clamp(p.crossFeed, 0.0f, 1.0f);
            fast.setParams(p);
            ref.setParams(p);
        }

        fast.process(inL.data() + off, hasR ? inR.data() + off : nullptr,
                     fL.data() + off, hasR ? fR.data() + off : nullptr, n);
        ref.processRef(inL.data() + off, hasR ? inR.data() + off : nullptr,
                       rL.data() + off, hasR ? rR.data() + off : nullptr, n);
    }

    // ── Compare ──
    const int D = c.delay;
    const int twoD = std::min(2 * D, kTotal);
    const bool bitExact = (c.satOrder == 0);

    if (bitExact)
    {
        bool ok = true;
        for (int i = 0; i < kTotal; ++i)
        {
            const auto u = static_cast<std::size_t>(i);
            if (fL[u] != rL[u]) { ok = false; FAIL("BIT-EXACT delay=%d fb=%.2f cross=%.2f sat=%d blk=%d ch=%d off=%.1f i=%d L: %g != %g",
                     c.delay, c.feedback, c.cross, c.satOrder, c.block, c.stereo?2:1, c.outOffset, i,
                     static_cast<double>(fL[u]), static_cast<double>(rL[u])); }
            if (hasR && fR[u] != rR[u]) { ok = false; FAIL("BIT-EXACT delay=%d fb=%.2f cross=%.2f sat=%d blk=%d ch=%d off=%.1f i=%d R: %g != %g",
                     c.delay, c.feedback, c.cross, c.satOrder, c.block, c.stereo?2:1, c.outOffset, i,
                     static_cast<double>(fR[u]), static_cast<double>(rR[u])); }
        }
        if (ok) ++g_bitExactOk;
    }
    else
    {
        // Per-sample relative tolerance over the first 2*D samples.
        bool tolOk = true;
        for (int i = 0; i < twoD; ++i)
        {
            const auto u = static_cast<std::size_t>(i);
            const float denom = std::max(std::fabs(rL[u]), 1e-6f);
            const float rel = std::fabs(fL[u] - rL[u]) / denom;
            g_worstRel = std::max(g_worstRel, static_cast<double>(rel));
            if (rel > 1e-5f) { tolOk = false; FAIL("TOL delay=%d fb=%.2f cross=%.2f sat=%d blk=%d i=%d L: rel=%.3e > 1e-5 (%g vs %g)",
                     c.delay, c.feedback, c.cross, c.satOrder, c.block, i,
                     static_cast<double>(rel), static_cast<double>(fL[u]), static_cast<double>(rL[u])); }
            if (hasR)
            {
                const float denomR = std::max(std::fabs(rR[u]), 1e-6f);
                const float relR = std::fabs(fR[u] - rR[u]) / denomR;
                g_worstRel = std::max(g_worstRel, static_cast<double>(relR));
                if (relR > 1e-5f) { tolOk = false; FAIL("TOL delay=%d fb=%.2f cross=%.2f sat=%d blk=%d i=%d R: rel=%.3e > 1e-5 (%g vs %g)",
                         c.delay, c.feedback, c.cross, c.satOrder, c.block, i,
                         static_cast<double>(relR), static_cast<double>(fR[u]), static_cast<double>(rR[u])); }
            }
        }
        if (tolOk) ++g_tolOk;

        // Energy envelope: per-1024 RMS ratio within ±0.1 dB over ≥ 10 loop
        // passes (only for feedback 0.95, where recirculation is strongest).
        if (std::fabs(c.feedback - 0.95f) < 0.01f)
        {
            bool envOk = true;
            const int winLen = 1024;
            const int minPasses = 10 * D;
            const int envEnd = std::min(minPasses, kTotal - winLen);
            for (int start = D; start + winLen <= envEnd; start += winLen)
            {
                const double dbF = rmsDb(fL, start, winLen);
                const double dbR = rmsDb(rL, start, winLen);
                const double devDb = dbF - dbR;
                g_worstEnvDb = std::max(g_worstEnvDb, std::fabs(devDb));
                if (std::fabs(devDb) > 0.1)
                {
                    envOk = false;
                    FAIL("ENV delay=%d fb=%.2f sat=%d blk=%d start=%d: %.3f dB > 0.1 dB",
                         c.delay, c.feedback, c.satOrder, c.block, start, devDb);
                }
            }
            if (envOk) ++g_envOk;
        }
    }
}

} // namespace

int main()
{
    std::printf("=== Chronos fb_parity (chunked process vs per-sample processRef) ===\n");
    std::printf("fs=%.0f  total=%d  maxDelay=%d\n\n", kFs, kTotal, kMaxDelay);

    const int   delays[6]  = { 5, 7, 12, 48, 480, 4800 };
    const float fbs[4]     = { 0.0f, 0.5f, 0.95f, 1.2f };
    const float cross[3]   = { 0.0f, 0.37f, 1.0f };
    const int   sats[3]    = { 0, 1, 2 };
    const int   blocks[5]  = { 1, 17, 64, 256, 512 };
    const bool  stereos[2] = { false, true };

    long configs = 0;
    for (int delay : delays)
    for (float fbk : fbs)
    for (float cr : cross)
    for (int sat : sats)
    for (int blk : blocks)
    for (bool stereo : stereos)
    {
        g_section = "matrix";
        runOne({ delay, fbk, cr, sat, blk, stereo }, false, false);
        ++configs;
    }

    // Delay-automation ramp (mid-ramp smoother, crossing chunk boundaries).
    g_section = "delay-automation";
    for (int sat : sats)
    for (int blk : { 17, 64, 256, 512 })
    for (bool stereo : stereos)
    {
        runOne({ 480, 0.95f, 0.37f, sat, blk, stereo }, true, false);
        ++configs;
    }

    // dampHz/crossFeed automation (block-rate coefficient changes).
    g_section = "damp/cross-automation";
    for (int sat : sats)
    for (int blk : { 64, 256, 512 })
    for (bool stereo : stereos)
    {
        runOne({ 480, 0.95f, 0.0f, sat, blk, stereo, 0.0f }, false, true);
        ++configs;
    }

    // C7 output-tap offset cells (see header). sat 0 bit-exact + sat 2 tol.
    g_section = "output-tap";
    for (int sat : { 0, 2 })
    {
        // settled bulk read + moving walk, Lc via dMinOut.
        for (float fbk : { 0.5f, 0.95f })
        for (int blk : { 17, 64, 256 })
        for (bool stereo : stereos)
        {
            runOne({ 480, fbk, 0.37f, sat, blk, stereo, 37.5f }, false, false);
            ++configs;
        }
        // scalar fallback WITH offset (dMinOut clamps Lc < 4).
        for (int blk : { 17, 64 })
        for (bool stereo : stereos)
        {
            runOne({ 12, 0.5f, 0.37f, sat, blk, stereo, 37.5f }, false, false);
            ++configs;
        }
        // output tap clamped at kMinLoopDelay (offset > delay − margin).
        for (int blk : { 17, 64 })
        for (bool stereo : stereos)
        {
            runOne({ 48, 0.5f, 0.37f, sat, blk, stereo, 100.0f }, false, false);
            ++configs;
        }
        // realistic diffuser base transport (size 0 → ~2930 samples).
        for (int blk : { 64, 256 })
        for (bool stereo : stereos)
        {
            runOne({ 4800, 0.95f, 0.37f, sat, blk, stereo, 2930.0f }, false, false);
            ++configs;
        }
        // delay automation with a live output-tap offset (moving walk).
        for (int blk : { 64, 256 })
        for (bool stereo : stereos)
        {
            runOne({ 480, 0.95f, 0.37f, sat, blk, stereo, 37.5f }, true, false);
            ++configs;
        }
    }

    std::printf("matrix (%ld configs):\n", configs);
    std::printf("  satOrder=0 bit-exact:    %d configs PASS\n", g_bitExactOk);
    std::printf("  satOrder 1/2 tolerance:  %d configs PASS (worst rel %.3e, gate 1e-5)\n", g_tolOk, g_worstRel);
    std::printf("  satOrder 1/2 energy env: %d configs PASS (worst %.4f dB, gate 0.1 dB)\n", g_envOk, g_worstEnvDb);
    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

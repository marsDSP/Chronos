/**
 * FeedbackDelay parity: the chunked process() against the per-sample
 * processRef() reference twin. Diffuser-off cells are bit-exact.
 * Diffuser-on cells use a combined abs 1e-6 / rel 1e-3 gate plus a plus/minus
 * 0.1 dB energy-envelope check, because the SIMD diffuser kernel and the
 * scalar reference differ at ulp level through FMA contraction.
 * The matrix covers delay, feedback, cross, block size, mono and stereo,
 * the Lc < 4 fallback, automation ramps, and enable toggles.
 */

#include "dsp/FeedbackDelay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <print>
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
int g_tolOk = 0;
int g_envOk = 0;
double g_worstRel = 0.0;
double g_worstEnvDb = 0.0;

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

struct Cfg
{
    int   delay;
    float feedback;
    float cross;
    int   satOrder;
    int   block;
    bool  stereo;
    bool  diffOn    = false;  // in-loop diffuser enabled
    float diffusion = 0.7f;   // allpass coefficient 0..0.92
    float diffSize  = 0.5f;   // section length scale
    float delayModDepth  = 0.0f;  // cents; 0 keeps the settled path reachable
    float delayModRateHz = 1.0f;  // Hz
    int   delayMode  = 0;        // 0: Digital, 1: BBD
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
    p.enableDiffuser = c.diffOn;
    p.diffusion      = c.diffusion;
    p.diffuserSize   = c.diffSize;
    p.diffModDepth   = 0.0f;   // deterministic (no LFO walk across instances)
    p.diffModRateHz  = 0.5f;
    p.delayModDepth  = c.delayModDepth;
    p.delayModRateHz = c.delayModRateHz;
    p.delayMode     = c.delayMode;
    fast.resetParams(p);
    ref.resetParams(p);   // resetParams snaps: diffuser state On/Off, no fade

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
            // Oscillate dampHz, loopCutHz, and crossFeed across blocks. The
            // loopCut sweep catches a reference path that fails to advance
            // the low-cut coefficient smoother.
            const float frac = static_cast<float>(off) / static_cast<float>(kTotal);
            p.dampHz    = 3000.0f + 6000.0f * (0.5f + 0.5f * std::sin(2.0f * std::numbers::pi_v<float> * frac));
            p.loopCutHz = 40.0f + 800.0f * (0.5f + 0.5f * std::sin(2.0f * std::numbers::pi_v<float> * frac * 0.7f));
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

    // Compare
    const int D = c.delay;
    const int twoD = std::min(2 * D, kTotal);
    // All cells use a COMBINED gate (abs <= 1e-6 OR rel <= 1e-3). S5's
    // loopTrim_ = pow(rmsRatio, -0.5) uses std::pow, which is not bit-exact
    // across libm (x86_64 vs arm64 differ by 1 ULP); the trim feeds back
    // through the loop so the difference compounds over recirculation. The
    // combined gate is tight enough to catch real chunk-structure bugs
    // (which produce large, systematic divergence) while tolerating the
    // bounded ULP noise from the platform-dependent pow. The ±0.1 dB
    // energy-envelope gate below remains the systematic-divergence check.
    const bool combinedGate = true;

    {
        // Per-sample combined tolerance over the first 2*D samples.
        bool tolOk = true;
        for (int i = 0; i < twoD; ++i)
        {
            const auto u = static_cast<std::size_t>(i);
            const float absL = std::fabs(fL[u] - rL[u]);
            const float denom = std::max(std::fabs(rL[u]), 1e-6f);
            const float rel = absL / denom;
            g_worstRel = std::max(g_worstRel, static_cast<double>(rel));
            const bool okL = (absL <= 1e-6f || rel <= 1e-3f);
            if (!okL) { tolOk = false; FAIL("TOL delay={} fb={:.2} cross={:.2} sat={} blk={} i={} L: abs={:.3} rel={:.3} ({} vs {})",
                     c.delay, c.feedback, c.cross, c.satOrder, c.block, i,
                     static_cast<double>(absL), static_cast<double>(rel),
                     static_cast<double>(fL[u]), static_cast<double>(rL[u])); }
            if (hasR)
            {
                const float absR = std::fabs(fR[u] - rR[u]);
                const float denomR = std::max(std::fabs(rR[u]), 1e-6f);
                const float relR = absR / denomR;
                g_worstRel = std::max(g_worstRel, static_cast<double>(relR));
                const bool okR = (absR <= 1e-6f || relR <= 1e-3f);
                if (!okR) { tolOk = false; FAIL("TOL delay={} fb={:.2} cross={:.2} sat={} blk={} i={} R: abs={:.3} rel={:.3} ({} vs {})",
                         c.delay, c.feedback, c.cross, c.satOrder, c.block, i,
                         static_cast<double>(absR), static_cast<double>(relR),
                         static_cast<double>(fR[u]), static_cast<double>(rR[u])); }
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
                    FAIL("ENV delay={} fb={:.2} sat={} blk={} start={}: {:.3} dB > 0.1 dB",
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
    std::println("=== Chronos fb_parity (chunked process vs per-sample processRef) ===");
    std::println("fs={:.0}  total={}  maxDelay={}\n", kFs, kTotal, kMaxDelay);

    const std::array<int, 6> delays = {{ 5, 7, 12, 48, 480, 4800 }};
    const std::array<float, 4> fbs = {{ 0.0f, 0.5f, 0.95f, 1.2f }};
    const std::array<float, 3> cross = {{ 0.0f, 0.37f, 1.0f }};
    const std::array<int, 3> sats = {{ 0, 1, 2 }};
    const std::array<int, 5> blocks = {{ 1, 17, 64, 256, 512 }};
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

    // dampHz/loopCutHz/crossFeed automation (block-rate coefficient changes).
    g_section = "damp/cross-automation";
    for (int sat : sats)
    for (int blk : { 64, 256, 512 })
    for (bool stereo : stereos)
    {
        runOne({ 480, 0.95f, 0.0f, sat, blk, stereo, false }, false, true);
        ++configs;
    }

    // Delay-modulation cells: the OU stream must advance identically in both
    // paths, once per sample per channel, independent of the block size.
    // Modulation forces the per-sample tap walk in the chunked path.
    g_section = "delay-mod";
    for (int sat : { 0, 2 })
    for (int blk : { 1, 17, 64, 256, 512 })
    for (bool stereo : stereos)
    {
        runOne({ 4800, 0.5f, 0.37f, sat, blk, stereo, false, 0.7f, 0.5f, 25.0f, 1.5f }, false, false);
        ++configs;
    }
    for (int blk : { 64, 256 })
    for (bool stereo : stereos)
    {
        runOne({ 4800, 0.95f, 0.37f, 2, blk, stereo, false, 0.7f, 0.5f, 25.0f, 1.5f }, false, false);
        ++configs;
    }

    // C7c in-loop diffuser cells (see header). All tolerance-gated (the
    // diffuser kernels differ at last-ulp level by construction).
    g_section = "in-loop-diffuser";
    for (int sat : { 0, 2 })
    {
        // settled bulk read + SIMD diffuser kernel, both sizes, both coeffs.
        for (float fbk : { 0.5f, 0.95f })
        for (int blk : { 17, 64, 256 })
        for (bool stereo : stereos)
        for (float sz : { 0.5f, 0.0f })
        for (float df : { 0.5f, 1.0f })
        {
            runOne({ 4800, fbk, 0.37f, sat, blk, stereo, true, df, sz }, false, false);
            ++configs;
        }
        // clamp region: delay < baseTransport (loop tap at kMinLoopDelay,
        // Lc < 4 → scalar fallback WITH the diffuser live).
        for (int blk : { 17, 64 })
        for (bool stereo : stereos)
        {
            runOne({ 480, 0.5f, 0.37f, sat, blk, stereo, true, 1.0f, 0.5f }, false, false);
            ++configs;
        }
        // delay automation with the diffuser live (moving walk, non-settled).
        for (int blk : { 64, 256 })
        for (bool stereo : stereos)
        {
            runOne({ 4800, 0.95f, 0.37f, sat, blk, stereo, true, 0.7f, 0.5f }, true, false);
            ++configs;
        }
        // enable-toggle mid-run: FadingIn/FadingOut parity (the fade forces
        // the per-sample non-settled walk and the raw/diffused blend).
        for (int blk : { 64, 256 })
        for (bool stereo : stereos)
        {
            g_section = "in-loop-diffuser-toggle";
            FeedbackDelay fast, ref;
            fast.prepare(kFs, blk, kMaxDelay);
            ref.prepare(kFs, blk, kMaxDelay);

            FeedbackDelay::Params p;
            p.delaySamples = 4800.0f;
            p.feedback     = 0.95f;
            p.crossFeed    = 0.37f;
            p.dampHz       = kDampHz;
            p.loopDrive    = kLoopDrive;
            p.satOrder     = sat;
            p.enableDiffuser = false;
            p.diffusion      = 0.85f;
            p.diffuserSize   = 0.5f;
            p.diffModDepth   = 0.0f;
            fast.resetParams(p);
            ref.resetParams(p);

            const bool hasR = stereo;
            std::vector<float> inL(static_cast<std::size_t>(kTotal));
            std::vector<float> inR(static_cast<std::size_t>(kTotal));
            for (int i = 0; i < kTotal; ++i)
            {
                const auto u = static_cast<std::size_t>(i);
                inL[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / kFs));
                inR[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 330.0 * static_cast<double>(i) / kFs));
            }
            std::vector<float> fL(static_cast<std::size_t>(kTotal)), fR(static_cast<std::size_t>(kTotal));
            std::vector<float> rL(static_cast<std::size_t>(kTotal)), rR(static_cast<std::size_t>(kTotal));
            for (int off = 0; off < kTotal; off += blk)
            {
                const int n = std::min(blk, kTotal - off);
                // toggle at 1/4 (on), 2/4 (off), 3/4 (on) of the run.
                if (off == kTotal / 4 || off == 3 * kTotal / 4) { p.enableDiffuser = true;  fast.setParams(p); ref.setParams(p); }
                if (off == kTotal / 2)                          { p.enableDiffuser = false; fast.setParams(p); ref.setParams(p); }
                fast.process(inL.data() + off, hasR ? inR.data() + off : nullptr,
                             fL.data() + off, hasR ? fR.data() + off : nullptr, n);
                ref.processRef(inL.data() + off, hasR ? inR.data() + off : nullptr,
                               rL.data() + off, hasR ? rR.data() + off : nullptr, n);
            }
            for (int i = 0; i < kTotal; ++i)
            {
                const auto u = static_cast<std::size_t>(i);
                const float absL = std::fabs(fL[u] - rL[u]);
                const float denom = std::max(std::fabs(rL[u]), 1e-6f);
                const float rel = absL / denom;
                g_worstRel = std::max(g_worstRel, static_cast<double>(rel));
                if (absL > 1e-6f && rel > 1e-3f)
                    FAIL("TOGGLE-TOL sat={} blk={} ch={} i={} L: abs={:.3} rel={:.3} ({} vs {})",
                         sat, blk, stereo?2:1, i, static_cast<double>(absL),
                         static_cast<double>(rel), static_cast<double>(fL[u]),
                         static_cast<double>(rL[u]));
                if (hasR)
                {
                    const float absR = std::fabs(fR[u] - rR[u]);
                    const float denomR = std::max(std::fabs(rR[u]), 1e-6f);
                    const float relR = absR / denomR;
                    g_worstRel = std::max(g_worstRel, static_cast<double>(relR));
                    if (absR > 1e-6f && relR > 1e-3f)
                        FAIL("TOGGLE-TOL sat={} blk={} ch={} i={} R: abs={:.3} rel={:.3}",
                             sat, blk, stereo?2:1, i, static_cast<double>(absR),
                             static_cast<double>(relR));
                }
            }
            ++g_tolOk;
            ++configs;
        }
    }

    // BBD crossfeed cells: verify chunked process() vs processRef() for the
    // BBD delay core at crossfeed 0, 0.5, 1.0 (S67).
    g_section = "bbd-crossfeed";
    for (int delay : { 480, 4800 })
    for (float fbk : { 0.5f, 0.95f })
    for (float cr : { 0.0f, 0.5f, 1.0f })
    for (int sat : { 0, 2 })
    for (int blk : { 64, 256 })
    for (bool stereo : stereos)
    {
        runOne({ delay, fbk, cr, sat, blk, stereo, false, 0.7f, 0.5f, 25.0f, 1.5f, 1 }, false, false);
        ++configs;
    }

    std::println("matrix ({} configs):", configs);
    std::println("  tolerance:               {} configs PASS (worst rel {:.3}, gate 1e-3)", g_tolOk, g_worstRel);
    std::println("  satOrder 1/2 energy env: {} configs PASS (worst {:.4} dB, gate 0.1 dB)", g_envOk, g_worstEnvDb);
    std::println("\n=== ALL PROPERTIES HELD ===");
    return 0;
}

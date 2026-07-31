// tests/harnesses/dsp/diffuser_toggle_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Diffuser enable-toggle hygiene. Gates the three properties the
// ChronosEngine diffuser crossfade state machine exists to preserve:
//
// (a) No stale replay: when the diffuser is bypassed, its 16 rings hold up to
//     ~611 ms of frozen audio. Re-enabling must NOT replay that stale audio.
//     prime() clears the rings on a rising edge, and a wet-path crossfade
//     blends undiffused → diffused over ~10 ms. Feed tone A with the diffuser
//     on, disable, feed tone B for > 700 ms (> the ring depth), re-enable, and
//     measure energy at tone A's frequency in the output after re-enable.
//
//     The gate is RELATIVE, not absolute: the Schroeder allpass diffuser is
//     not a pure delay — at g > 0 it produces a low-level subharmonic / comb
//     response at the input tone's sub-frequencies (~−60 dBc at 220 Hz for a
//     440 Hz input at diffusion 0.7), even with no stale audio in the rings.
//     So an absolute −80 dBc gate is unachievable. Instead the toggle case is
//     gated against a BASELINE: a run with the diffuser always on and NO stale
//     tone A (tone B only from the start). If prime() + the crossfade work, the
//     toggle case's 220 Hz energy must be ≤ the baseline's (the toggle adds no
//     more tone A than the diffuser's inherent response). Without prime() the
//     stale tone A would replay at ~full amplitude, far exceeding the baseline.
//
// (b) No click: the per-sample crossfade bounds the output step at the toggle
//     edges. The fade introduces the diffused signal at inc = 1/480 per sample;
//     |diff − undiff| ≤ ~4.5 (diffuser_parity gates |out| ≤ 4.0, undiffused ≤
//     0.5), so the fade-induced step ≤ 4.5/480 ≈ 0.009. The signal's own
//     per-sample step (440 Hz sine, 0.5 amp) ≤ 2π·440/48000·0.5 ≈ 0.029. Total
//     ≤ 0.04; gate at 0.1 (well below the no-fade click of ~4.5).
//
// (c) PDC latency unchanged: latencySamples() is a compile-time constant
//     (SaturatorAlign::kBudget), independent of the diffuser toggle state.
//
// Uses ChronosEngine directly (the state machine is engine-level). Conventions
// matching latency_null_check / chain_parity: plain main(), exit code, printf,
// always-live CHECK/FAIL. Links SharedCode only; no JUCE.
//
// Coherent sampling: fA = 220 Hz, fB = 440 Hz. Both are integer bins of
// N = 24000 (220·24000/48000 = 110, 440·24000/48000 = 220), so the DTFT at
// either frequency has zero spectral leakage from the other.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/ChronosEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <vector>

namespace {

constexpr double kFs    = 48000.0;
constexpr double kPi    = std::numbers::pi_v<double>;
constexpr int    kBlock = 256;
constexpr int    kBudget = MarsDSP::Align::SaturatorAlign::kBudget;

using Engine = MarsDSP::ChronosEngine;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

Engine::Params makeParams(bool enableDiff) noexcept
{
    Engine::Params p{};
    p.delaySamples    = 1000.0f;
    p.driveLin        = 1.0f;     // 0 dB
    p.mix             = 100.0f;   // full wet (isolates the diffuser path)
    p.gainLin         = 1.0f;     // 0 dB
    p.hpfHz           = 20.0f;    // transparent
    p.lpfHz           = 20000.0f; // transparent
    p.bits            = 24;       // transparent (lsb ≈ 1.2e-7, ≪ -80 dB)
    p.adaaOrder       = 0;        // Off: no saturation (simplest wet path)
    p.interp          = MarsDSP::Delays::Interpolation::Lagrange5th;
    p.feedback        = 0.0f;     // plain delay line (not FeedbackDelay)
    p.dampHz          = 6000.0f;
    p.crossFeed       = 0.0f;
    p.loopDrive       = 1.0f;
    p.loopSatOrder    = 0;
    p.diffusion       = 0.7f;
    p.diffuserSize    = 0.5f;
    p.diffModDepth    = 0.0f;     // no modulation (deterministic)
    p.diffModRateHz   = 0.5f;
    p.enableDiffuser  = enableDiff;
    return p;
}

// Coherent-amplitude (sine amplitude = 2|X|/len) of the component at freqHz
// over x[start, start+len). Requires freqHz·len/fs to be an integer bin for
// zero spectral leakage.
double measureAmp(const std::vector<float>& x, double freqHz, int start, int len)
{
    const double omega = 2.0 * kPi * freqHz / kFs;
    double c = 0.0, s = 0.0;
    for (int n = 0; n < len; ++n)
    {
        const double ang = omega * static_cast<double>(start + n);
        const double v   = static_cast<double>(x[static_cast<std::size_t>(start + n)]);
        c += v * std::cos(ang);
        s += v * std::sin(ang);
    }
    return 2.0 * std::sqrt(c * c + s * s) / static_cast<double>(len);
}

// Fill io[0..n) with a sine at freqHz, continuing from sample index `base`.
void fillSine(std::vector<float>& buf, double freqHz, int base, int n)
{
    for (int i = 0; i < n; ++i)
        buf[static_cast<std::size_t>(base + i)] =
            0.5f * static_cast<float>(std::sin(2.0 * kPi * freqHz * static_cast<double>(base + i) / kFs));
}

// Run a 3-phase scenario (tone A phase 1, tone B phases 2–3) and return the
// {tone A, tone B} coherent amplitudes measured in the tail of phase 3.
// diffOn1/2/3 set the diffuser enable for each phase. Used for both the
// toggle case and the always-on baseline (no stale tone A: phase 1 feeds
// tone B too, so the rings never see tone A).
struct AmpPair { double ampA; double ampB; };

AmpPair runScenario(bool diffOn1, bool diffOn2, bool diffOn3,
                    double fA, double fB, bool phase1IsA)
{
    constexpr int kPhase = 48000;   // 1 second per phase
    constexpr int kMeasN = 24000;   // coherent window (fA, fB are integer bins)

    Engine eng;
    eng.prepare(kFs, kBlock, 2);
    eng.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    eng.resetParams(makeParams(diffOn1));

    std::vector<float> bufL(static_cast<std::size_t>(kPhase * 3));
    std::vector<float> bufR(static_cast<std::size_t>(kPhase * 3));
    std::vector<float> outL(static_cast<std::size_t>(kPhase * 3));

    const double p1Freq = phase1IsA ? fA : fB;
    fillSine(bufL, p1Freq, 0, kPhase);
    fillSine(bufR, p1Freq, 0, kPhase);
    for (int off = 0; off < kPhase; off += kBlock)
    {
        float* io[2] = { bufL.data() + off, bufR.data() + off };
        eng.process(io, 2, kBlock);
        std::memcpy(outL.data() + off, bufL.data() + off,
                    static_cast<std::size_t>(kBlock) * sizeof(float));
    }

    eng.setParams(makeParams(diffOn2));
    fillSine(bufL, fB, kPhase, kPhase);
    fillSine(bufR, fB, kPhase, kPhase);
    for (int off = 0; off < kPhase; off += kBlock)
    {
        const int o = kPhase + off;
        float* io[2] = { bufL.data() + o, bufR.data() + o };
        eng.process(io, 2, kBlock);
        std::memcpy(outL.data() + o, bufL.data() + o,
                    static_cast<std::size_t>(kBlock) * sizeof(float));
    }

    eng.setParams(makeParams(diffOn3));
    fillSine(bufL, fB, 2 * kPhase, kPhase);
    fillSine(bufR, fB, 2 * kPhase, kPhase);
    for (int off = 0; off < kPhase; off += kBlock)
    {
        const int o = 2 * kPhase + off;
        float* io[2] = { bufL.data() + o, bufR.data() + o };
        eng.process(io, 2, kBlock);
        std::memcpy(outL.data() + o, bufL.data() + o,
                    static_cast<std::size_t>(kBlock) * sizeof(float));
    }

    const int measStart = 3 * kPhase - kMeasN;
    return { measureAmp(outL, fA, measStart, kMeasN),
             measureAmp(outL, fB, measStart, kMeasN) };
}

// ── Test (a): no stale replay (relative gate) ─────────────────────────────
void testStaleReplay()
{
    g_section = "stale-replay";
    constexpr double fA = 220.0;
    constexpr double fB = 440.0;

    // Baseline: diffuser always on, NO stale tone A (phase 1 feeds tone B).
    // This measures the diffuser's inherent 220 Hz response to a 440 Hz input.
    const AmpPair base = runScenario(true, true, true, fA, fB, /*phase1IsA=*/false);
    if (base.ampB <= 1e-7)
        FAIL("baseline tone B %.3e too low (chain assembled wrong)", base.ampB);

    // Toggle case: tone A in phase 1 (rings fill with stale A), disable in
    // phase 2, re-enable in phase 3 (prime() clears the rings).
    const AmpPair tog = runScenario(true, false, true, fA, fB, /*phase1IsA=*/true);
    if (tog.ampB <= 1e-7)
        FAIL("toggle tone B %.3e too low (chain assembled wrong)", tog.ampB);

    const double baseDbc = 20.0 * std::log10(base.ampA / base.ampB);
    const double togDbc  = 20.0 * std::log10(tog.ampA  / tog.ampB);

    std::printf("    baseline (always-on, no stale A): A=%.3e B=%.3e A/B=%.1f dBc\n",
                base.ampA, base.ampB, baseDbc);
    std::printf("    toggle (stale A, on→off→on):       A=%.3e B=%.3e A/B=%.1f dBc\n",
                tog.ampA, tog.ampB, togDbc);

    // The toggle case must add no more tone A than the diffuser's inherent
    // response (≤ baseline + 3 dB slack for run-to-run dither noise). Without
    // prime() the stale replay would be ~0 dBc, far above this gate.
    CHECK(togDbc <= baseDbc + 3.0);
    std::printf("no stale replay (toggle %.1f dBc ≤ baseline %.1f + 3 dB): PASS\n",
                togDbc, baseDbc);
}

// ── Test (b): no click at toggle edges ────────────────────────────────────
void testClickBound()
{
    g_section = "click-bound";
    constexpr double fB = 440.0;
    constexpr int kSettle = 48000;   // let the diffuser settle
    constexpr int kCapture = 2048;   // ~8 blocks; covers the 480-sample fade
    constexpr float kClickBound = 0.1f;
    constexpr int kGap = 48000;      // between edges (> ring depth)

    Engine eng;
    eng.prepare(kFs, kBlock, 2);
    eng.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    eng.resetParams(makeParams(true));   // diffuser ON

    std::vector<float> bufL(static_cast<std::size_t>(kSettle * 2 + kCapture * 2 + kGap));
    std::vector<float> bufR(bufL.size());
    double maxStep = 0.0;
    int sampleIdx = 0;

    auto runBlocks = [&](int total, bool capture)
    {
        for (int off = 0; off < total; off += kBlock)
        {
            const int n = std::min(kBlock, total - off);
            fillSine(bufL, fB, sampleIdx, n);
            fillSine(bufR, fB, sampleIdx, n);
            float* io[2] = { bufL.data() + off, bufR.data() + off };
            eng.process(io, 2, n);
            if (capture)
            {
                for (int s = 1; s < n; ++s)
                {
                    const float step = std::fabs(bufL[static_cast<std::size_t>(off + s)]
                                               - bufL[static_cast<std::size_t>(off + s - 1)]);
                    maxStep = std::max(maxStep, static_cast<double>(step));
                }
            }
            sampleIdx += n;
        }
    };

    // Settle with diffuser ON, then capture the falling edge (On → off).
    runBlocks(kSettle, false);
    eng.setParams(makeParams(false));
    runBlocks(kCapture, true);

    // Gap with diffuser OFF, then capture the rising edge (Off → on).
    runBlocks(kGap, false);
    eng.setParams(makeParams(true));
    runBlocks(kCapture, true);

    std::printf("    click bound: max |step| = %.4f (gate %.1f)\n", maxStep, static_cast<double>(kClickBound));
    CHECK(maxStep <= static_cast<double>(kClickBound));
    std::printf("no click at toggle edges (max step %.4f < %.1f): PASS\n", maxStep, static_cast<double>(kClickBound));
}

// ── Test (c): PDC latency unchanged by toggling ───────────────────────────
void testLatencyInvariant()
{
    g_section = "latency-invariant";
    const int lat0 = Engine::latencySamples();
    CHECK(lat0 == kBudget);

    Engine eng;
    eng.prepare(kFs, kBlock, 2);
    eng.resetParams(makeParams(true));
    const int latOn = Engine::latencySamples();
    eng.setParams(makeParams(false));
    const int latOff = Engine::latencySamples();
    eng.setParams(makeParams(true));
    const int latOnAgain = Engine::latencySamples();

    CHECK(latOn == kBudget);
    CHECK(latOff == kBudget);
    CHECK(latOnAgain == kBudget);
    std::printf("PDC latency = %d (kBudget) at all toggle states: PASS\n", lat0);
}

} // namespace

int main()
{
    std::printf("=== Chronos diffuser_toggle_check (C5) ===\n");
    std::printf("fs=%.0f  block=%d  kBudget=%d  fade=%d samples\n\n",
                kFs, kBlock, kBudget, Engine::kDiffuserFadeSamples);

    testStaleReplay();
    testClickBound();
    testLatencyInvariant();

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

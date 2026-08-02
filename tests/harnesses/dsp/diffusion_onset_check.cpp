
// mean-comp skew). Full wet (mix = 100), no saturation (adaaOrder = 0),
// transparent SVF, bits = 24. Each scenario runs in a fresh engine with
// ~200 ms of silence to let the size + delay smoothers settle before the
// impulse. Conventions matching latency_null_check / chain_parity.
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

constexpr double kFs     = 48000.0;
constexpr int    kBlock  = 256;
constexpr int    kBudget = MarsDSP::Align::SaturatorAlign::kBudget;
constexpr int    kDelay  = 40000;    // T: large enough for comp to fit at size ≥ 0.5
constexpr int    kSettle = 12000;    // 250 ms: > 50 ms size + 20 ms delay smoother
constexpr int    kCapture = 100000;  // ~2 s: delay (40k) + transport (≤3k) +
                                       // allpass tail at g=0.92 (energy e-fold
                                       // ~4k samples → converged) + margin
constexpr float  kOnsetFrac = 0.05f; // 5% of total energy
constexpr int    kOnsetGate = 144;   // 3 ms @48 kHz

using Engine = MarsDSP::ChronosEngine;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

Engine::Params makeParams(bool enableDiff, float diffusion, float size) noexcept
{
    Engine::Params p{};
    p.delaySamples    = static_cast<float>(kDelay);
    p.driveLin        = 1.0f;
    p.mix             = 100.0f;   // full wet
    p.gainLin         = 1.0f;
    p.hpfHz           = 20.0f;
    p.lpfHz           = 20000.0f;
    p.bits            = 24;
    p.adaaOrder       = 0;
    p.feedback        = 0.0f;
    p.dampHz          = 6000.0f;
    p.crossFeed       = 0.0f;
    p.loopDrive       = 1.0f;
    p.loopSatOrder    = 0;
    p.diffusion       = diffusion;
    p.diffuserSize    = size;
    p.diffModDepth    = 0.0f;
    p.diffModRateHz   = 0.5f;
    p.enableDiffuser  = enableDiff;
    return p;
}

// Run a scenario: settle, impulse at sample kSettle, capture output.
// Returns the output buffer (out[0..kSettle+kCapture)).
std::vector<float> runScenario(bool enableDiff, float diffusion, float size)
{
    Engine eng;
    eng.prepare(kFs, kBlock, 1);   // mono
    eng.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    eng.resetParams(makeParams(enableDiff, diffusion, size));

    const int total = kSettle + kCapture;
    std::vector<float> buf(static_cast<std::size_t>(total), 0.0f);
    buf[static_cast<std::size_t>(kSettle)] = 1.0f;   // impulse

    for (int off = 0; off < total; off += kBlock)
    {
        const int n = std::min(kBlock, total - off);
        float* io[1] = { buf.data() + off };
        eng.process(io, 1, n);
    }
    return buf;
}

// First sample where |out[n]| exceeds threshold (the impulse arrival).
int firstNonzero(const std::vector<float>& out, float threshold)
{
    for (int n = kSettle; n < kSettle + kCapture; ++n)
        if (std::fabs(out[static_cast<std::size_t>(n)]) > threshold)
            return n;
    return -1;
}

// Energy centroid: the center of mass of the IR energy. Robust to the
// diffuser's symmetric bloom at moderate diffusion (pre/post-arrivals cancel
// around the centroid). Returns the sample index of the energy center of mass.
int energyCentroid(const std::vector<float>& out)
{
    double totalE = 0.0, weightedSum = 0.0;
    for (int n = kSettle; n < kSettle + kCapture; ++n)
    {
        const double v = static_cast<double>(out[static_cast<std::size_t>(n)]);
        const double e = v * v;
        totalE += e;
        weightedSum += e * static_cast<double>(n);
    }
    if (totalE <= 0.0) return -1;
    return static_cast<int>(std::round(weightedSum / totalE));
}

// C7c anchoring: the comp is exactly baseTransportSamples (the cascade IR's
// energy centroid, g-invariant), so the diffused tap's energy centroid must
// land ON the diffuser-off grid at every diffusion setting. The energy
// median deliberately drifts early at high g (the IR's mass concentrates at
// t = 0 — the front-spike trade-off documented in Diffuser.h) and is NOT
// gated here; the loop-level in-sync property is gated by
// diffuser_loop_check.

// ── Test (a): diffusion = 0, onset alignment ──────────────────────────────
void testDiffusionZeroOnset()
{
    g_section = "diffusion=0 onset";
    const float size = 1.0f;   // smallest L-R skew (~0.4 ms)

    const auto outOff = runScenario(false, 0.0f, size);
    const auto outOn  = runScenario(true,  0.0f, size);

    const int onsetOff = firstNonzero(outOff, 1e-4f);
    const int onsetOn  = firstNonzero(outOn,  1e-4f);
    CHECK(onsetOff >= 0);
    CHECK(onsetOn  >= 0);

    const int diff = std::abs(onsetOn - onsetOff);
    std::printf("    diffusion=0 size=1.0: onset_off=%d onset_on=%d diff=%d (gate %d)\n",
                onsetOff, onsetOn, diff, kOnsetGate);
    CHECK(diff <= kOnsetGate);
    std::printf("diffusion=0 onset alignment (diff %d <= %d): PASS\n", diff, kOnsetGate);
}

// ── Test (b): energy centroid stays on the grid across the sweep ────────
// The C7c comp anchors the exact energy centroid (baseTransportSamples —
// g-invariant by the allpass average-group-delay identity), so the
// diffuser-on centroid must match the diffuser-off centroid within a small
// tolerance at every diffusion × size. This replaces the old median gate:
// the comp is exact at the centroid, and the median's early drift at high g
// is the documented front-spike trade-off, not a failure.
void testDiffusionCentroidSweep()
{
    g_section = "centroid sweep";
    constexpr int kCentroidGate = 150;  // 3.1 ms @48 kHz (window truncation
                                        // + measurement residual)

    const float sizes[] = { 0.5f, 1.0f, 0.0f };
    const float diffs[] = { 0.0f, 0.25f, 0.5f, 0.7f, 0.85f, 1.0f };

    for (float size : sizes)
    {
        const auto outOff = runScenario(false, 0.0f, size);
        const int cOff = energyCentroid(outOff);
        CHECK(cOff >= 0);

        for (float diff_f : diffs)
        {
            const auto outOn = runScenario(true, diff_f, size);
            const int cOn = energyCentroid(outOn);
            CHECK(cOn >= 0);

            const int d = std::abs(cOn - cOff);
            std::printf("    diffusion=%.2f size=%.1f: centroid_off=%d centroid_on=%d diff=%d (gate %d)\n",
                        static_cast<double>(diff_f), static_cast<double>(size),
                        cOff, cOn, d, kCentroidGate);
            if (d > kCentroidGate)
                FAIL("diffusion=%.2f size=%.1f centroid diff %d > %d",
                     static_cast<double>(diff_f), static_cast<double>(size), d, kCentroidGate);
        }
    }
    std::printf("centroid sweep (all <= %d): PASS\n", kCentroidGate);
}

// ── Test (c): PDC latency unchanged ───────────────────────────────────────
void testLatencyInvariant()
{
    g_section = "latency-invariant";
    CHECK(Engine::latencySamples() == kBudget);

    Engine eng;
    eng.prepare(kFs, kBlock, 1);
    eng.resetParams(makeParams(true, 0.7f, 0.5f));
    CHECK(Engine::latencySamples() == kBudget);
    eng.setParams(makeParams(false, 0.7f, 0.5f));
    CHECK(Engine::latencySamples() == kBudget);

    std::printf("PDC latency = %d (kBudget) at all diffuser states: PASS\n", kBudget);
}

} // namespace

int main()
{
    std::printf("=== Chronos diffusion_onset_check (C7c) ===\n");
    std::printf("fs=%.0f  delay=%d  settle=%d  capture=%d  onset_gate=%d samples (%.1f ms)\n\n",
                kFs, kDelay, kSettle, kCapture, kOnsetGate,
                static_cast<double>(kOnsetGate) / kFs * 1000.0);

    testDiffusionZeroOnset();
    testDiffusionCentroidSweep();
    testLatencyInvariant();

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

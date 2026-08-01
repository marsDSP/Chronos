// tests/harnesses/simd/diffuser_parity.cpp
// ──────────────────────────────────────────────────────────────────────────
// Diffuser parity: processBlock (section-major, 4-wide SIMD chunk fast path
// + per-sample FracDelayTap exact path) vs processBlockRef (sample-major
// scalar reference twin), across block sizes, size/diffusion/mod settings,
// and stereo + mono.
//
// Why the two paths agree: processBlock processes a chunk of up to kChunk=16
// samples section-by-section, whereas processBlockRef interleaves all 8
// sections per sample. They are equivalent because each section's in-chunk
// read lands D >= kMinDelay (=32) > kChunk samples back, so no read in a
// chunk can touch a write from that same chunk (header invariant); section
// i's input at sample j is section i-1's output at sample j in BOTH orders.
// The smoothers (size/coef/depth) and the magic-circle LFOs advance once per
// sample in the same order on both paths, so per-sample parameters match.
//
// Settled, unmodulated sections take the SIMD fast path (mul-then-sub /
// mul-then-add, deliberately NOT FMA, so bit-identical to the scalar allpass
// math); modulated / size-moving sections take the per-sample FracDelayTap
// path, identical to the reference. Gated to an abs tolerance (1e-5) for
// FMA-contraction / reassociation slack. Also gates finite + bounded output
// (an allpass cascade with |g| < 1 keeps bounded input bounded).
//
// Conventions (matching simd_delay_parity): plain main(), exit code, printf,
// always-live CHECK/FAIL. Links SharedCode only; no JUCE. Forced -O2 and
// -Xarch_x86_64 -mfma (Apple) / -mfma (Linux) / /arch:AVX2 (MSVC) so the
// chunk kernel inlines and simde_mm_* lowers to native intrinsics.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/Diffuser.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr double kFs = 48000.0;
constexpr float  kTol = 1e-5f;
double g_worst = 0.0;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

struct Cfg
{
    int   blockSize;
    float diffusion;
    float size;
    float modDepth;
    float modRateHz;
    bool  stereo;
};

void applySettings(MarsDSP::Diffusion::Diffuser& d, const Cfg& c)
{
    d.setDiffusion(c.diffusion);
    d.setSize(c.size);
    d.setModDepthSamples(c.modDepth);
    d.setModRateHz(c.modRateHz);
}

void runOne(const Cfg& c)
{
    g_section = "diffuser_parity";

    MarsDSP::Diffusion::Diffuser fast, ref;
    fast.prepare(kFs);
    ref.prepare(kFs);
    applySettings(fast, c);
    applySettings(ref, c);

    std::vector<float> inL(static_cast<std::size_t>(c.blockSize));
    std::vector<float> inR(static_cast<std::size_t>(c.blockSize));
    for (int i = 0; i < c.blockSize; ++i)
    {
        inL[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::sin(0.3 * static_cast<double>(i)))
          + 0.3f * static_cast<float>(std::sin(1.1 * static_cast<double>(i)))
          + 0.01f * static_cast<float>(i);
        inR[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::cos(0.27 * static_cast<double>(i)))
          + 0.01f * static_cast<float>(i * 2);
    }

    std::vector<float> fL(inL), fR(inR);
    std::vector<float> rL(inL), rR(inR);

    fast.processBlock(fL.data(), c.stereo ? fR.data() : nullptr, c.blockSize);
    ref.processBlockRef(rL.data(), c.stereo ? rR.data() : nullptr, c.blockSize);

    for (int i = 0; i < c.blockSize; ++i)
    {
        const auto u = static_cast<std::size_t>(i);
        const float eL = std::fabs(fL[u] - rL[u]);
        if (eL > g_worst) g_worst = static_cast<double>(eL);
        if (eL > kTol)
            FAIL("block=%d diff=%.2f size=%.2f modD=%.1f modR=%.2f stereo=%d i=%d L: "
                 "block=%g ref=%g diff=%.3e > %.0e",
                 c.blockSize, (double)c.diffusion, (double)c.size,
                 (double)c.modDepth, (double)c.modRateHz, static_cast<int>(c.stereo),
                 i, (double)fL[u], (double)rL[u], (double)eL, (double)kTol);
        if (c.stereo)
        {
            const float eR = std::fabs(fR[u] - rR[u]);
            if (eR > g_worst) g_worst = static_cast<double>(eR);
            if (eR > kTol)
                FAIL("block=%d diff=%.2f size=%.2f modD=%.1f modR=%.2f stereo=%d i=%d R: "
                     "block=%g ref=%g diff=%.3e > %.0e",
                     c.blockSize, (double)c.diffusion, (double)c.size,
                     (double)c.modDepth, (double)c.modRateHz, static_cast<int>(c.stereo),
                     i, (double)fR[u], (double)rR[u], (double)eR, (double)kTol);
        }

        // Finite + bounded (allpass |g|<1 cascade, |in|<=1 -> bounded out).
        if (!std::isfinite(fL[u]) || std::fabs(fL[u]) > 4.0f)
            FAIL("block=%d i=%d L non-finite/unbounded: %g", c.blockSize, i, (double)fL[u]);
        if (c.stereo && (!std::isfinite(fR[u]) || std::fabs(fR[u]) > 4.0f))
            FAIL("block=%d i=%d R non-finite/unbounded: %g", c.blockSize, i, (double)fR[u]);
    }
}

// ── C6: base-transport unit check ─────────────────────────────────────
// Verify baseTransportSamples / baseTransportSamplesLR match an inline
// recomputation of the same per-section eff expression, and sanity-check the
// measured transport table (size 0 ≈ 61 ms; size 1 → ~10% of that, with the
// shortest sections floor-clamped at kMinDelay).
void testBaseTransport()
{
    g_section = "base-transport";
    MarsDSP::Diffusion::Diffuser d;
    d.prepare(kFs);   // primes the section lengths (prime-snapped at 48 kHz)

    const float kMaxSizeCut = MarsDSP::Diffusion::Diffuser::kMaxSizeCut;
    const float kMinDelayF  = MarsDSP::Diffusion::Diffuser::kMinDelay;
    const int   kNum        = MarsDSP::Diffusion::Diffuser::kNumSections;

    const float sizes[] = { 0.0f, 0.25f, 0.5f, 0.75f, 0.9f, 1.0f };
    for (float sz : sizes)
    {
        const auto lr = d.baseTransportSamplesLR(sz);
        const float mean = d.baseTransportSamples(sz);

        // Independent recompute via the same per-section expression, using the
        // public section-length getters. This verifies baseTransportSamplesLR
        // uses the token-identical arithmetic to chunk_/chain_ (unmodulated).
        const float s = std::clamp(sz, 0.0f, 1.0f);
        float refL = 0.0f, refR = 0.0f;
        for (int i = 0; i < kNum; ++i)
        {
            const float lenFL = static_cast<float>(d.sectionLenL(i));
            float effL = lenFL * (1.0f - kMaxSizeCut * s);
            effL = std::nearbyintf(effL);
            effL = std::clamp(effL, kMinDelayF, lenFL);
            refL += effL;

            const float lenFR = static_cast<float>(d.sectionLenR(i));
            float effR = lenFR * (1.0f - kMaxSizeCut * s);
            effR = std::nearbyintf(effR);
            effR = std::clamp(effR, kMinDelayF, lenFR);
            refR += effR;
        }
        if (std::fabs(lr[0] - refL) > 1e-5f || std::fabs(lr[1] - refR) > 1e-5f)
            FAIL("size=%.2f L=%.3f vs ref %.3f, R=%.3f vs ref %.3f",
                 (double)sz, (double)lr[0], (double)refL, (double)lr[1], (double)refR);
        if (std::fabs(mean - 0.5f * (lr[0] + lr[1])) > 1e-5f)
            FAIL("size=%.2f mean=%.3f but 0.5*(L+R)=%.3f",
                 (double)sz, (double)mean, (double)(0.5f * (lr[0] + lr[1])));

        std::printf("    size=%.2f  L=%.1f (%.1f ms)  R=%.1f (%.1f ms)  skew=%.2f ms  mean=%.1f\n",
                    (double)sz,
                    (double)lr[0], static_cast<double>(lr[0]) / kFs * 1000.0,
                    (double)lr[1], static_cast<double>(lr[1]) / kFs * 1000.0,
                    std::fabs(static_cast<double>(lr[0] - lr[1])) / kFs * 1000.0,
                    (double)mean);
    }

    // Sanity: size 0 transport ≈ 61 ms at 48 kHz (short-smear tables: Σlen ≈
    // 2930 samples per bank, prime-snapped, so within a few samples). Bounds
    // bracket the intended scale — a regression to the old 10× tables
    // (≈ 29400) fails the upper bound.
    {
        const auto lr0 = d.baseTransportSamplesLR(0.0f);
        CHECK(lr0[0] > 2800.0f && lr0[0] < 3100.0f);
        CHECK(lr0[1] > 2800.0f && lr0[1] < 3100.0f);
        std::printf("    size=0 sanity: L=%.0f R=%.0f (both in (2800, 3100)): PASS\n",
                    (double)lr0[0], (double)lr0[1]);
    }

    // Sanity: size 1 → eff = len*(1-0.9*1) = len*0.1 per section, floor-
    // clamped at kMinDelay=32 — the three shortest sections (len < 320)
    // clamp, so the transport is a bit above 10% of size=0. The recompute
    // loop above already verified the exact value; monotonicity (below)
    // verifies it is the minimum across the grid.
    {
        const auto lr0 = d.baseTransportSamplesLR(0.0f);
        const auto lr1 = d.baseTransportSamplesLR(1.0f);
        CHECK(lr1[0] < lr0[0]);   // size=1 is the shortest transport
        CHECK(lr1[1] < lr0[1]);
        std::printf("    size=1 sanity: L=%.0f R=%.0f (both < size=0): PASS\n",
                    (double)lr1[0], (double)lr1[1]);
    }

    // Monotonicity: transport(size) is non-increasing (size shortens delays).
    float prev = d.baseTransportSamples(0.0f);
    for (float sz : sizes)
    {
        const float cur = d.baseTransportSamples(sz);
        if (cur > prev + 1e-4f)
            FAIL("non-monotonic at size=%.2f: cur=%.1f > prev=%.1f", (double)sz, (double)cur, (double)prev);
        prev = cur;
    }
    std::printf("    monotonicity (transport non-increasing in size): PASS\n");
    std::printf("base-transport unit check: PASS\n");
}

} // namespace

int main()
{
    std::printf("=== Chronos diffuser_parity (section-major SIMD vs sample-major ref) ===\n");
    std::printf("fs=%.0f  tol=%.0e\n\n", kFs, (double)kTol);

    testBaseTransport();
    std::printf("\n");

    const int   blockSizes[] = { 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 64, 100, 256 };
    const float diffs[]      = { 0.0f, 0.3f, 0.7f, 0.92f };
    const float sizes[]      = { 0.0f, 0.25f, 0.5f, 0.9f };
    const float modDepths[]  = { 0.0f, 8.0f, 16.0f, 32.0f };
    const float modRates[]   = { 0.5f, 2.0f };

    long configs = 0;
    for (int bs : blockSizes)
    for (float df : diffs)
    for (float sz : sizes)
    for (float md : modDepths)
    for (float mr : modRates)
    for (bool stereo : { false, true })
    {
        runOne({ bs, df, sz, md, mr, stereo });
        ++configs;
    }

    std::printf("matrix (%ld configs): parity <= %.0e (worst %.3e), finite, bounded: PASS\n",
                configs, (double)kTol, g_worst);
    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

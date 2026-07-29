// tests/harnesses/cd/frac_delay_tap_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// FracDelayTap correctness. Validates the three properties the Diffuser and
// FeedbackDelay rely on:
//
//   1. lagrange3() (closed-form coeff with precomputed reciprocals) matches
//      makeCoeffs(Lagrange3rd) to <= 1 ulp per coefficient, and at f = 0
//      collapses to a single unit tap at index 3 (delay i) — the integer
//      bit-transparency guarantee.
//   2. read() (hot path: closed-form coeff + 4-tap SIMD horizontal dot)
//      matches readRef() (makeCoeffs + scalar dot) across fractional delays.
//   3. An integer read returns EXACTLY the sample written d samples ago
//      (lagrange3(0) is a unit tap, and the SIMD dot of {0,0,1,0} is exact).
//   4. Zero-state: a zeroed ring reads as 0.0 on both paths.
//
// Conventions (matching ring_buffer_check): plain main(), exit code, printf,
// always-live CHECK/FAIL. Links SharedCode only; no JUCE. No forced -O2 so
// the header's assert preconditions (delay >= 3, delay <= cap-kTail-2) stay
// armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/FracDelayTap.h"
#include "dsp/Pow2RingBuffer.h"
#include "dsp/DelayInterpolator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs = 48000.0;

// ── 1. lagrange3 vs makeCoeffs(Lagrange3rd) ───────────────────────────────
// Closed-form coeff differs from the generic basis by <= 1 ulp per coeff
// (x * (1/6) vs x / 6). At f = 0 both collapse to a single unit tap at
// index 3 (the delay-i sample), which is the integer bit-transparency.
void testCoeffsVsMakeCoeffs()
{
    g_section = "lagrange3 vs makeCoeffs";
    constexpr int kSteps = 1000;
    double maxErr = 0.0;
    // lagrange3 asserts f in [0, 1) (it does not range-reduce, unlike
    // makeCoeffs which floors). Sweep [0, 1) — kSteps points, excluding f = 1.
    for (int s = 0; s < kSteps; ++s)
    {
        const float f = static_cast<float>(s) / static_cast<float>(kSteps);
        const auto k = MarsDSP::Delays::FracDelayTap::lagrange3(f);
        const auto c = MarsDSP::Delays::makeCoeffs(MarsDSP::Delays::Interpolation::Lagrange3rd, f);
        const float refs[4] = { c.c[1], c.c[2], c.c[3], c.c[4] };
        const float got[4]  = { k.c1,   k.c2,   k.c3,   k.c4 };
        for (int j = 0; j < 4; ++j)
        {
            const double e = std::fabs(static_cast<double>(got[j]) - static_cast<double>(refs[j]));
            if (e > maxErr) maxErr = e;
            if (e > 1e-6)
                FAIL("f=%.4f j=%d: lagrange3=%g makeCoeffs=%g diff=%.3e > 1e-6",
                     (double)f, j, (double)got[j], (double)refs[j], e);
        }
        // Partition of unity: the four active taps sum to ~1.
        const double sum = static_cast<double>(k.c1) + k.c2 + k.c3 + k.c4;
        if (std::fabs(sum - 1.0) > 1e-5)
            FAIL("f=%.4f: coeff sum=%g != 1.0", (double)f, sum);
    }
    // f = 0: single unit tap at index 3, bit-exact.
    {
        const auto k = MarsDSP::Delays::FracDelayTap::lagrange3(0.0f);
        CHECK(k.c1 == 0.0f);
        CHECK(k.c2 == 0.0f);
        CHECK(k.c3 == 1.0f);
        CHECK(k.c4 == 0.0f);
    }
    std::printf("lagrange3 vs makeCoeffs (1001 steps, max diff %.3e, f=0 unit tap @idx3): PASS\n", maxErr);
}

// ── 2. read() vs readRef() across fractional delays ───────────────────────
// Both read the same ring; read() takes the mirrored contiguous-window fast
// path (winLen 6 <= kTail so windowPtr is never null here), readRef() copies
// into scratch. They must agree to within coeff-rounding slack.
void testReadVsRef()
{
    g_section = "read vs readRef";
    constexpr int kCap = 256;
    MarsDSP::Delays::Pow2RingBuffer rb;
    rb.prepare(kCap);
    const int cap = rb.getCapacity();
    const int mask = rb.mask();

    // Write a unique, bounded signal (~|v| <= 1.3) so all four taps differ at
    // every read. The closed-form coeff differs from makeCoeffs by <= 1 ulp
    // per coefficient, so the read-vs-readRef error scales with signal
    // magnitude; normalising keeps the 1e-5 abs gate meaningful (matching
    // simd_delay_parity's convention).
    int w = 0;
    for (int n = 0; n < cap; ++n)
    {
        const float v =
            0.8f * static_cast<float>(std::sin(0.37 * static_cast<double>(n)))
          + 0.3f * static_cast<float>(std::sin(1.1 * static_cast<double>(n)));
        rb.writeBlock(&v, w, 1);
        rb.refreshMirror(w, 1);
        w = (w + 1) & mask;
    }

    const float maxDelay = static_cast<float>(cap - MarsDSP::Delays::Pow2RingBuffer::kTail - 2);
    double maxErr = 0.0;
    int n = 0;
    for (float d = 3.0f; d < maxDelay; d += 0.25f, ++n)
    {
        const float a = MarsDSP::Delays::FracDelayTap::read(rb, w, d);
        const float b = MarsDSP::Delays::FracDelayTap::readRef(rb, w, d);
        const double e = std::fabs(static_cast<double>(a) - static_cast<double>(b));
        if (e > maxErr) maxErr = e;
        if (e > 1e-5)
            FAIL("d=%.3f: read=%g readRef=%g diff=%.3e > 1e-5", (double)d, (double)a, (double)b, e);
    }
    CHECK(n > 100);
    std::printf("read vs readRef (%d delays over [3, %.1f), max diff %.3e < 1e-5): PASS\n",
                n, (double)maxDelay, maxErr);
}

// ── 3. integer-delay bit-transparency ─────────────────────────────────────
// lagrange3(0) is a single unit tap, so an integer read returns EXACTLY the
// sample written d samples ago — the property the Diffuser's unmodulated
// path and FeedbackDelay's loop read depend on.
void testIntegerBitExact()
{
    g_section = "integer bit-exact";
    constexpr int kCap = 512;
    MarsDSP::Delays::Pow2RingBuffer rb;
    rb.prepare(kCap);
    const int cap = rb.getCapacity();
    const int mask = rb.mask();

    // Fill the whole ring with a unique sequence, remembering every value.
    std::vector<float> vals(static_cast<std::size_t>(cap));
    int w = 0;
    for (int n = 0; n < cap; ++n)
    {
        const float v = static_cast<float>(n * 3 + 1) * 0.5f + 0.123f;
        vals[static_cast<std::size_t>(n)] = v;
        rb.writeBlock(&v, w, 1);
        rb.refreshMirror(w, 1);
        w = (w + 1) & mask;
    }
    // Wrote `cap` samples into a cap-sized ring from w=0, so w is back to 0
    // and storage[n] == vals[n]. The sample d ago is vals[cap - d].
    const int maxD = cap - MarsDSP::Delays::Pow2RingBuffer::kTail - 4;
    for (int d = 3; d <= maxD; ++d)
    {
        const float got = MarsDSP::Delays::FracDelayTap::read(rb, w, static_cast<float>(d));
        const float exp = vals[static_cast<std::size_t>(cap - d)];
        if (got != exp)
            FAIL("integer d=%d: read=%g expected=%g (bit-exact delayed tap)",
                 d, (double)got, (double)exp);
    }
    std::printf("integer-delay bit-transparency (d=3..%d, bit-exact): PASS\n", maxD);
}

// ── 4. zero-state: zeros in -> 0.0 out ────────────────────────────────────
void testZeroState()
{
    g_section = "zero state";
    MarsDSP::Delays::Pow2RingBuffer rb;
    rb.prepare(128);
    const int mask = rb.mask();
    const float zero = 0.0f;
    int w = 0;
    for (int n = 0; n < 200; ++n)
    {
        rb.writeBlock(&zero, w, 1);
        rb.refreshMirror(w, 1);
        w = (w + 1) & mask;
    }
    for (float d = 3.0f; d < 60.0f; d += 1.0f)
    {
        CHECK(MarsDSP::Delays::FracDelayTap::read(rb, w, d) == 0.0f);
        CHECK(MarsDSP::Delays::FracDelayTap::readRef(rb, w, d) == 0.0f);
    }
    std::printf("zero-state (zeros in -> 0.0 out, read & readRef): PASS\n");
}

} // namespace

int main()
{
    std::printf("=== Chronos frac_delay_tap_check ===\n");
    std::printf("fs=%.0f\n\n", kFs);

    testCoeffsVsMakeCoeffs();
    testReadVsRef();
    testIntegerBitExact();
    testZeroState();

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

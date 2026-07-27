// tests/harnesses/cd/short_delay_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Correctness harness for MarsDSP::Align::ShortDelay<MaxDelay>, the fixed-
// capacity integer delay used by SaturatorAlign to pad the wet path to the
// constant latency budget kBudget in the Off and ADAA2 modes (and to leave
// ADAA1's wet at 0 integer delay so the half-sample FIR supplies the rest).
//
//   1. Impulse (every d in [0, MaxDelay]) — impulse at sample 0, output
//      exactly 1.0f at sample d and 0.0f everywhere else in [0, 4*MaxDelay].
//      Bit-exact. This is the test the spec flags as catching the single
//      most likely bug in the whole alignment layer: an off-by-one in the
//      ring read/write ordering or the tap index.
//   2. Ramp (every d) — y[n] == x[n-d] bit-exact for n >= d over 1000
//      samples, wrapping the ring many times (MaxDelay=8 -> ~111 wraps).
//   3. Reset reproducibility — process a ramp, reset, reprocess, bit-exact.
//
// Tested at MaxDelay = 8 (the actual SaturatorAlign instantiation,
// kBudget = kHalfSampleTaps/2 = 16/2 = 8) and MaxDelay = 1 (the d in
// {0,1} edge, wraps fastest).
//
// Conventions (matching ring_buffer_check / halfsample_fir_check): plain
// main(), exit code, printf, always-live CHECK/FAIL (NOT assert — NDEBUG in
// Release would void every test). Links SharedCode only; no JUCE. No forced
// -O2 so the header's assert preconditions stay armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/align/ShortDelay.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

// Impulse: for every d in [0, MaxDelay], impulse at sample 0 must produce
// exactly 1.0f at sample d and 0.0f elsewhere in [0, 4*MaxDelay].
template <int MaxDelay>
void testImpulse()
{
    using MarsDSP::Align::ShortDelay;
    constexpr int kSpan = 4 * MaxDelay + 1;  // samples [0, 4*MaxDelay]
    g_section = "impulse";

    for (int d = 0; d <= MaxDelay; ++d)
    {
        ShortDelay<MaxDelay> delay;
        delay.reset();
        delay.setDelay(d);

        std::vector<float> y(static_cast<std::size_t>(kSpan));
        for (int n = 0; n < kSpan; ++n)
            y[n] = delay.process(n == 0 ? 1.0f : 0.0f);

        for (int n = 0; n < kSpan; ++n)
        {
            const float exp = (n == d) ? 1.0f : 0.0f;
            if (y[n] != exp)
                FAIL("MaxDelay=%d d=%d n=%d got=%g exp=%g (impulse)",
                     MaxDelay, d, n, (double)y[n], (double)exp);
        }
    }
    std::printf("impulse [0..%d] (MaxDelay=%d): PASS\n", MaxDelay, MaxDelay);
}

// Ramp: for every d, y[n] == x[n-d] bit-exact for n >= d over 1000 samples
// (and 0 for n < d, the zeroed ring ring-up). Wraps the ring many times.
template <int MaxDelay>
void testRamp()
{
    using MarsDSP::Align::ShortDelay;
    constexpr int kN = 1000;
    g_section = "ramp";

    for (int d = 0; d <= MaxDelay; ++d)
    {
        ShortDelay<MaxDelay> delay;
        delay.reset();
        delay.setDelay(d);

        // Unique 1-based ramp so 0 is distinguishable from uninitialised
        // ring state; a misindexed tap reads the wrong slot immediately.
        std::vector<float> x(static_cast<std::size_t>(kN));
        for (int n = 0; n < kN; ++n) x[n] = static_cast<float>(n + 1);

        for (int n = 0; n < kN; ++n)
        {
            const float got = delay.process(x[n]);
            const float exp = (n >= d) ? x[n - d] : 0.0f;
            if (got != exp)
                FAIL("MaxDelay=%d d=%d n=%d got=%g exp=%g (ramp)",
                     MaxDelay, d, n, (double)got, (double)exp);
        }
    }
    std::printf("ramp [0..%d] over %d samples (MaxDelay=%d): PASS\n",
                MaxDelay, kN, MaxDelay);
}

// Reset reproducibility: process a ramp, reset, reprocess, bit-exact match.
template <int MaxDelay>
void testReset()
{
    using MarsDSP::Align::ShortDelay;
    constexpr int kN = 200;
    g_section = "reset";

    for (int d = 0; d <= MaxDelay; ++d)
    {
        ShortDelay<MaxDelay> delay;
        std::vector<float> x(static_cast<std::size_t>(kN));
        for (int n = 0; n < kN; ++n) x[n] = static_cast<float>(n + 1);

        delay.reset();
        delay.setDelay(d);
        std::vector<float> y1(static_cast<std::size_t>(kN));
        for (int n = 0; n < kN; ++n) y1[n] = delay.process(x[n]);

        delay.reset();
        delay.setDelay(d);
        std::vector<float> y2(static_cast<std::size_t>(kN));
        for (int n = 0; n < kN; ++n) y2[n] = delay.process(x[n]);

        for (int n = 0; n < kN; ++n)
            if (y1[n] != y2[n])
                FAIL("MaxDelay=%d d=%d reset mismatch n=%d %g != %g",
                     MaxDelay, d, n, (double)y1[n], (double)y2[n]);
    }
    std::printf("reset reproducibility (MaxDelay=%d): PASS\n", MaxDelay);
}

} // namespace

int main()
{
    std::printf("=== Chronos ShortDelay correctness harness ===\n\n");

    // MaxDelay = 8 is the real SaturatorAlign instantiation
    // (kBudget = kHalfSampleTaps/2 = 16/2 = 8).
    std::printf("-- MaxDelay = 8 (kBudget) --\n");
    testImpulse<8>();
    testRamp<8>();
    testReset<8>();

    // MaxDelay = 1 exercises the d in {0,1} edge and wraps fastest.
    std::printf("\n-- MaxDelay = 1 (edge) --\n");
    testImpulse<1>();
    testRamp<1>();
    testReset<1>();

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

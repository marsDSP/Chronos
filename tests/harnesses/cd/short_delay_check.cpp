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
//   4. Power-of-two parity (C4 proof) — for every MaxDelay in [1,16], run
//      the new bit_ceil/mask ShortDelay alongside a local twin that keeps
//      the old MaxDelay+1/modulo logic, over 2000 randomized trials with
//      random 0<->nonzero setDelay changes and random reset()s. Require
//      bit-equal output on every sample. This is the test the source plans
//      asserted the change was "behaviour-preserving" without argument —
//      align_check's mode-switching test (lines 397-420) only checks
//      finiteness and would not have caught a divergence.
//
// Conventions (matching ring_buffer_check / halfsample_fir_check): plain
// main(), exit code, printf, always-live CHECK/FAIL (NOT assert — NDEBUG in
// Release would void every test). Links SharedCode only; no JUCE. No forced
// -O2 so the header's assert preconditions stay armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/align/ShortDelay.h"

#include <array>
#include <cstdint>
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

// ── 4. Power-of-two parity proof ─────────────────────────────────────────
// A local twin that keeps the OLD logic (MaxDelay+1 capacity, integer
// modulo). Compared bit-for-bit against the new bit_ceil/mask ShortDelay
template <int MaxDelay>
class ShortDelayOld {
public:
    static constexpr int kCapacity = MaxDelay + 1;
    void reset() noexcept { z_.fill(0.0f); w_ = 0; d_ = 0; }
    void setDelay(int d) noexcept { d_ = d; }
    float process(float x) noexcept {
        if (d_ == 0) return x;
        const float y = z_[(w_ - d_ + kCapacity) % kCapacity];
        z_[w_] = x;
        w_ = (w_ + 1) % kCapacity;
        return y;
    }
private:
    std::array<float, kCapacity> z_{};
    int w_{0}, d_{0};
};

// Deterministic xorshift32 so the trial sequence is reproducible.
struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed) {}
    std::uint32_t next() noexcept {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
    int range(int lo, int hi) noexcept { // inclusive
        return lo + static_cast<int>(next() % static_cast<std::uint32_t>(hi - lo + 1));
    }
    float unit() noexcept {
        return static_cast<float>(next() >> 8) * (1.0f / 8388608.0f);
    }
};

template <int MaxDelay>
void testPow2Parity()
{
    g_section = "power-of-two parity";
    constexpr int kTrials = 2000;
    Rng rng(0xC4FEED01u ^ static_cast<std::uint32_t>(MaxDelay));

    MarsDSP::Align::ShortDelay<MaxDelay> neu;
    ShortDelayOld<MaxDelay> old;
    neu.reset();
    old.reset();

    for (int t = 0; t < kTrials; ++t)
    {
        // Random delay change: 0 or a value in [1, MaxDelay].
        if (rng.next() & 1)
        {
            const int d = (rng.next() & 1) ? 0 : rng.range(1, MaxDelay);
            neu.setDelay(d);
            old.setDelay(d);
        }

        // Random reset (~10% of trials).
        if ((rng.next() % 10) == 0)
        {
            neu.reset();
            old.reset();
        }

        // Process one sample and compare bit-exactly.
        const float x = rng.unit() * 2.0f - 1.0f;
        const float yn = neu.process(x);
        const float yo = old.process(x);
        if (yn != yo)
            FAIL("MaxDelay=%d trial=%d: new=%g old=%g (bit mismatch)",
                 MaxDelay, t, (double)yn, (double)yo);
    }
    std::printf("power-of-two parity (MaxDelay=%d, %d trials): PASS\n", MaxDelay, kTrials);
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

    // Power-of-two parity proof: old (MaxDelay+1, modulo) vs new (bit_ceil,
    // mask) over MaxDelay in [1,16], 2000 randomized trials each.
    std::printf("\n-- power-of-two parity (C4 proof) --\n");
    testPow2Parity<1>();
    testPow2Parity<2>();
    testPow2Parity<3>();
    testPow2Parity<4>();
    testPow2Parity<5>();
    testPow2Parity<6>();
    testPow2Parity<7>();
    testPow2Parity<8>();
    testPow2Parity<9>();
    testPow2Parity<10>();
    testPow2Parity<11>();
    testPow2Parity<12>();
    testPow2Parity<13>();
    testPow2Parity<14>();
    testPow2Parity<15>();
    testPow2Parity<16>();

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

/**
 * Correctness harness for ShortDelay, the fixed-capacity integer delay that
 * SaturatorAlign uses to pad the wet path to the constant latency budget.
 * Plain main(), exit code, always-live CHECK/FAIL.
 */

#include "dsp/align/ShortDelay.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <print>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

/// Impulse: for every d, an impulse at sample 0 must produce 1.0f at sample d.
template <int MaxDelay>
void testImpulse()
{
    using MarsDSP::Align::ShortDelay;
    constexpr int kSpan = 4 * MaxDelay + 1;
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
                FAIL("MaxDelay={{}} d={{}} n={{}} got={{}} exp={{}} (impulse)",
                     MaxDelay, d, n, static_cast<double>(y[n]), static_cast<double>(exp));
        }
    }
    std::println("impulse [0..{}] (MaxDelay={}): PASS", MaxDelay, MaxDelay);
}

/// Ramp: for every d, y[n] equals x[n-d] bit-exact for n >= d over 1000 samples.
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

        // Unique 1-based ramp so 0 is distinguishable from uninitialised ring state.
        std::vector<float> x(static_cast<std::size_t>(kN));
        for (int n = 0; n < kN; ++n) x[n] = static_cast<float>(n + 1);

        for (int n = 0; n < kN; ++n)
        {
            const float got = delay.process(x[n]);
            const float exp = (n >= d) ? x[n - d] : 0.0f;
            if (got != exp)
                FAIL("MaxDelay={{}} d={{}} n={{}} got={{}} exp={{}} (ramp)",
                     MaxDelay, d, n, static_cast<double>(got), static_cast<double>(exp));
        }
    }
    std::println("ramp [0..{}] over {} samples (MaxDelay={}): PASS",
                MaxDelay, kN, MaxDelay);
}

/// Reset reproducibility: process a ramp, reset, reprocess, bit-exact match.
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
                FAIL("MaxDelay={{}} d={{}} reset mismatch n={{}} {{}} != {{}}",
                     MaxDelay, d, n, static_cast<double>(y1[n]), static_cast<double>(y2[n]));
    }
    std::println("reset reproducibility (MaxDelay={}): PASS", MaxDelay);
}

// A local twin that keeps the old logic (MaxDelay+1 capacity, integer modulo).
// Compared bit-for-bit against the new bit_ceil/mask ShortDelay.
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
    int w_{0};
    int d_{0};
};

/// Deterministic xorshift32 so the trial sequence is reproducible.
struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed) {}
    std::uint32_t next() noexcept {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
    int range(int lo, int hi) noexcept {
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
            FAIL("MaxDelay={{}} trial={{}}: new={{}} old={{}} (bit mismatch)",
                 MaxDelay, t, static_cast<double>(yn), static_cast<double>(yo));
    }
    std::println("power-of-two parity (MaxDelay={}, {} trials): PASS", MaxDelay, kTrials);
}

} // namespace

int main()
{
    std::println("=== Chronos ShortDelay correctness harness ===");
    // MaxDelay = 8 is the real SaturatorAlign instantiation (kBudget = 8).
    std::println("-- MaxDelay = 8 (kBudget) --");
    testImpulse<8>();
    testRamp<8>();
    testReset<8>();

    // MaxDelay = 1 exercises the d in {0,1} edge and wraps fastest.
    std::println();
    std::println("-- MaxDelay = 1 (edge) --");
    testImpulse<1>();
    testRamp<1>();
    testReset<1>();

    // Power-of-two parity proof over MaxDelay in [1,16], 2000 randomized trials each.
    std::println();
    std::println("-- power-of-two parity (C4 proof) --");
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

    std::println();
    std::println("=== ALL PROPERTIES HELD ===");
    return 0;
}

/**
 * FracDelayTap correctness. Validates the three properties the Diffuser and
 * FeedbackDelay rely on. Plain main(), exit code, always-live CHECK/FAIL.
 */

#include "dsp/FracDelayTap.h"
#include "dsp/Pow2RingBuffer.h"
#include "dsp/DelayInterpolator.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <print>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs = 48000.0;

// 1. lagrange3() vs makeCoeffs(Lagrange3rd). Closed-form coeff differs
// from the generic basis by at most 1 ulp per coeff. At f = 0 both collapse
// to a single unit tap at index 3, the integer bit-transparency.
void testCoeffsVsMakeCoeffs()
{
    g_section = "lagrange3 vs makeCoeffs";
    constexpr int kSteps = 1000;
    double maxErr = 0.0;
    // lagrange3 asserts f in [0, 1). Sweep [0, 1), kSteps points, excluding f = 1.
    for (int s = 0; s < kSteps; ++s)
    {
        const float f = static_cast<float>(s) / static_cast<float>(kSteps);
        const auto k = MarsDSP::Delays::FracDelayTap::lagrange3(f);
        const auto c = MarsDSP::Delays::makeCoeffs(MarsDSP::Delays::Interpolation::Lagrange3rd, f);
        const std::array<float, 4> refs = { c.c[1], c.c[2], c.c[3], c.c[4] };
        const std::array<float, 4> got  = { k.c1,   k.c2,   k.c3,   k.c4 };
        for (int j = 0; j < 4; ++j)
        {
            const double e = std::fabs(static_cast<double>(got[static_cast<std::size_t>(j)]) - static_cast<double>(refs[static_cast<std::size_t>(j)]));
            if (e > maxErr) maxErr = e;
            if (e > 1e-6)
                FAIL("f={{:.4f}} j={{}}: lagrange3={{}} makeCoeffs={{}} diff={{:.3e}} > 1e-6",
                     static_cast<double>(f), j, static_cast<double>(got[static_cast<std::size_t>(j)]),
                     static_cast<double>(refs[static_cast<std::size_t>(j)]), e);
        }
        // Partition of unity: the four active taps sum to approximately 1.
        const double sum = static_cast<double>(k.c1) + k.c2 + k.c3 + k.c4;
        if (std::fabs(sum - 1.0) > 1e-5)
            FAIL("f={{:.4f}}: coeff sum={{}} != 1.0", static_cast<double>(f), sum);
    }
    // f = 0: single unit tap at index 3, bit-exact.
    {
        const auto k = MarsDSP::Delays::FracDelayTap::lagrange3(0.0f);
        CHECK(k.c1 == 0.0f);
        CHECK(k.c2 == 0.0f);
        CHECK(k.c3 == 1.0f);
        CHECK(k.c4 == 0.0f);
    }
    std::println("lagrange3 vs makeCoeffs (1001 steps, max diff {:.3e}, f=0 unit tap @idx3): PASS", maxErr);
}

// 2. read() vs readRef() across fractional delays. Both read the same ring;
// read() takes the contiguous-window fast path, readRef() copies into
// scratch. They must agree to within coeff-rounding slack.
void testReadVsRef()
{
    g_section = "read vs readRef";
    constexpr int kCap = 256;
    MarsDSP::Delays::Pow2RingBuffer rb;
    rb.prepare(kCap);
    const int cap = rb.getCapacity();
    const int mask = rb.mask();

    // Write a unique, bounded signal (|v| <= 1.3) so all four taps differ at every read.
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
            FAIL("d={{:.3f}}: read={{}} readRef={{}} diff={{:.3e}} > 1e-5", static_cast<double>(d), static_cast<double>(a), static_cast<double>(b), e);
    }
    CHECK(n > 100);
    std::println("read vs readRef ({} delays over [3, {:.1f}), max diff {:.3e} < 1e-5): PASS",
                n, static_cast<double>(maxDelay), maxErr);
}

// 3. integer-delay bit-transparency. lagrange3(0) is a single unit tap, so an
// integer read returns exactly the sample written d samples ago.
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
    // Wrote cap samples into a cap-sized ring from w=0, so w is back to 0 and
    // storage[n] == vals[n]. The sample d ago is vals[cap - d].
    const int maxD = cap - MarsDSP::Delays::Pow2RingBuffer::kTail - 4;
    for (int d = 3; d <= maxD; ++d)
    {
        const float got = MarsDSP::Delays::FracDelayTap::read(rb, w, static_cast<float>(d));
        const float exp = vals[static_cast<std::size_t>(cap - d)];
        if (got != exp)
            FAIL("integer d={{}}: read={{}} expected={{}} (bit-exact delayed tap)",
                 d, static_cast<double>(got), static_cast<double>(exp));
    }
    std::println("integer-delay bit-transparency (d=3..{}, bit-exact): PASS", maxD);
}

// 4. zero-state: zeros in, 0.0 out.
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
    std::println("zero-state (zeros in, 0.0 out, read & readRef): PASS");
}

} // namespace

int main()
{
    std::println("=== Chronos frac_delay_tap_check ===");
    std::println("fs={:.0f}", kFs);
    std::println();

    testCoeffsVsMakeCoeffs();
    testReadVsRef();
    testIntegerBitExact();
    testZeroState();

    std::println();
    std::println("=== ALL PROPERTIES HELD ===");
    return 0;
}

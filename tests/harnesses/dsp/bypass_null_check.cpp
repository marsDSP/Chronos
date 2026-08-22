#include "dsp/ChronosEngine.h"
#include "dsp/align/SaturatorAlign.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <vector>

namespace {

constexpr double kFs     = 48000.0;
constexpr int    kBudget = MarsDSP::Align::SaturatorAlign::kBudget;
constexpr int    kFadeMs = 10;   // 10 ms fade
constexpr int    kFadeSamples = static_cast<int>(kFs * kFadeMs * 0.001);

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Build default engine params.
MarsDSP::ChronosEngine::Params makeParams(int adaaOrder)
{
    MarsDSP::ChronosEngine::Params p{};
    p.delaySamples = 240.0f;
    p.driveLin     = 1.0f;
    p.mix          = 100.0f;
    p.gainLin      = 1.0f;
    p.hpfHz        = 200.0f;
    p.lpfHz        = 8000.0f;
    p.bits         = 24;
    p.adaaOrder    = adaaOrder;
    return p;
}

void testBypassNull(int adaaOrder)
{
    g_section = "bypass null";
    constexpr int kBlockSize = 256;
    constexpr int kBlocks = 20;
    constexpr int kN = kBlockSize * kBlocks;

    // Generate input sine only
    std::vector<float> in(static_cast<std::size_t>(kN));
    for (int i = 0; i < kN; ++i)
        in[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::sin(0.3 * static_cast<double>(i)));

    // run bypass from the start
    std::vector<float> out(static_cast<std::size_t>(kN));
    {
        MarsDSP::ChronosEngine engine;
        engine.prepare(kFs, 128, 2);
        engine.reset();
        engine.setDitherSeeds(0x11111111u, 0x22222222u);
        engine.resetParams(makeParams(adaaOrder));
        engine.setBypass(true);
        for (int b = 0; b < kBlocks; ++b)
        {
            engine.setParams(makeParams(adaaOrder));
            for (int s = 0; s < kBlockSize; ++s)
                out[static_cast<std::size_t>(b * kBlockSize + s)] =
                    in[static_cast<std::size_t>(b * kBlockSize + s)];
            std::array<float*, 2> io{ out.data() + b * kBlockSize, out.data() + b * kBlockSize };
            engine.process(io.data(), 2, kBlockSize);
        }
    }

    // TPDF dither + quantization
    const int settle = kFadeSamples + kBudget + 64;
    const float lsb = std::ldexp(1.0f, 1 - 24);
    double maxErr = 0.0;
    for (int i = settle; i < kN; ++i)
    {
        const float exp = in[static_cast<std::size_t>(i - kBudget)];
        const float got = out[static_cast<std::size_t>(i)];
        const double e = std::fabs(static_cast<double>(got) - static_cast<double>(exp));
        if (e > maxErr) maxErr = e;
        if (e > 2.0 * static_cast<double>(lsb))
            FAIL("bypass null mode={} i={}: got={} exp={} (input delayed by {}), err={:.3} > 2*lsb={:.3}",
                 adaaOrder, i, static_cast<double>(got), static_cast<double>(exp), kBudget, e, 2.0 * static_cast<double>(lsb));
    }
    std::println("  mode={} bypass null (after {} samples): max|out - in[{}]| = {:.3} (2*lsb={:.3}): PASS",
                adaaOrder, settle, kBudget, maxErr, 2.0 * static_cast<double>(lsb));
    (void)adaaOrder;
}

// Test 2: no click across bypass toggle
void testToggleClick()
{
    g_section = "toggle click";
    constexpr int kBlockSize = 256;
    constexpr int kBlocks = 30;

    MarsDSP::ChronosEngine engine;
    engine.prepare(kFs, 128, 2);
    engine.reset();
    engine.setDitherSeeds(0x33333333u, 0x44444444u);
    engine.resetParams(makeParams(2));

    std::vector<float> in(static_cast<std::size_t>(kBlockSize * kBlocks));
    for (int i = 0; i < kBlockSize * kBlocks; ++i)
        in[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::sin(0.3 * static_cast<double>(i)));

    std::vector<float> out(static_cast<std::size_t>(kBlockSize * kBlocks));

    bool bypassed = false;
    for (int b = 0; b < kBlocks; ++b)
    {
        // Toggle bypass at block 10 and 20
        if (b == 10 || b == 20) { bypassed = !bypassed; engine.setBypass(bypassed); }
        engine.setParams(makeParams(2));
        for (int s = 0; s < kBlockSize; ++s)
            out[static_cast<std::size_t>(b * kBlockSize + s)] =
                in[static_cast<std::size_t>(b * kBlockSize + s)];
        std::array<float*, 2> io{ out.data() + b * kBlockSize, out.data() + b * kBlockSize };
        engine.process(io.data(), 2, kBlockSize);
    }

    double maxStep = 0.0;
    for (int i = 1; i < kBlockSize * kBlocks; ++i)
    {
        const double step = std::fabs(static_cast<double>(out[i]) - static_cast<double>(out[i - 1]));
        if (step > maxStep) maxStep = step;
    }

    CHECK(maxStep < 1.0);
    std::println("  toggle click (2 toggles, max sample step = {:.4} < 1.0): PASS", maxStep);
}

} // namespace

int main()
{
    std::println("=== Chronos bypass_null_check (S5) ===");
    std::println("fs={:.0}  kBudget={}  fade={} ms ({} samples)\n", kFs, kBudget, kFadeMs, kFadeSamples);

    for (int mode = 0; mode <= 2; ++mode)
        testBypassNull(mode);

    testToggleClick();

    std::println("\n=== ALL PROPERTIES HELD ===");
    return 0;
}

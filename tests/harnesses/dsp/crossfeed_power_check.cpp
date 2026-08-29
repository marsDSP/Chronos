// tests/harnesses/dsp/crossfeed_power_check.cpp
//
// Equal-power cross feed. Independent noise in L and R, feedback 0.5.
// Sweep the cross feed 0 to 1. The summed wet energy stays within 0.3 dB
// of the cross=0 baseline.
//
// The old linear mix loses 3 dB at cross 0.5 on uncorrelated material.
// The equal-power rotation preserves the total energy.
//
// Drives FeedbackDelay directly, SharedCode only, no JUCE.

#include "dsp/FeedbackDelay.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs       = 48000.0;
constexpr int    kFsInt    = 48000;
constexpr int    kBlock    = 512;
constexpr int    kMaxDelay = 262144;
constexpr int    kWarmup   = static_cast<int>(2.0 * kFs);   // 2 s build to steady state
constexpr int    kMeasure  = static_cast<int>(1.0 * kFs);   // 1 s measurement window

using MarsDSP::Delays::FeedbackDelay;

struct Xorshift32
{
    std::uint32_t s;
    explicit Xorshift32(std::uint32_t seed) : s(seed) {}
    float nextSigned() noexcept
    {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return static_cast<float>(s >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
};

// Run the loop with the given cross and return the wet RMS in dB.
double measureWetRmsDb(FeedbackDelay& fb, FeedbackDelay::Params& p, float amp)
{
    Xorshift32 rngL(0xA11CEu), rngR(0xB0B0u);
    std::vector<float> inL(static_cast<std::size_t>(kBlock));
    std::vector<float> inR(static_cast<std::size_t>(kBlock));
    std::vector<float> wetL(static_cast<std::size_t>(kBlock));
    std::vector<float> wetR(static_cast<std::size_t>(kBlock));

    double sumSq = 0.0;
    int count = 0;
    for (int pos = 0; pos < kWarmup + kMeasure; pos += kBlock)
    {
        const int n = std::min(kBlock, kWarmup + kMeasure - pos);
        for (int i = 0; i < n; ++i)
        {
            inL[static_cast<std::size_t>(i)] = amp * rngL.nextSigned();
            inR[static_cast<std::size_t>(i)] = amp * rngR.nextSigned();
        }
        fb.setParams(p);
        fb.process(inL.data(), inR.data(), wetL.data(), wetR.data(), n);
        if (pos >= kWarmup)
        {
            for (int i = 0; i < n; ++i)
            {
                const auto u = static_cast<std::size_t>(i);
                sumSq += static_cast<double>(wetL[u]) * wetL[u]
                       + static_cast<double>(wetR[u]) * wetR[u];
                ++count;
            }
        }
    }
    const double rms = std::sqrt(sumSq / static_cast<double>(count));
    return (rms > 0.0) ? 20.0 * std::log10(rms) : -999.0;
}

} // namespace

int main()
{
    std::println("=== crossfeed_power_check ===");
    std::println("fs={} feedback=0.5 delay=100ms uncorrelated noise amp=0.3\n",
                kFsInt);

    FeedbackDelay fb;
    fb.prepare(kFs, kBlock, kMaxDelay);

    FeedbackDelay::Params p;
    p.delaySamplesL   = 4800.0f;   // 100 ms
    p.delaySamplesR   = 4800.0f;
    p.feedback        = 0.5f;
    p.dampHz          = 6000.0f;
    p.loopCutHz       = 40.0f;
    p.crossFeed       = 0.0f;
    p.loopDrive       = 1.0f;
    p.satOrder        = 0;          // hard clip, no drive → no clipping
    p.enableDiffuser  = false;
    p.diffusion       = 0.7f;
    p.diffuserSize    = 0.5f;
    p.diffModDepth    = 0.0f;
    p.diffModRateHz   = 0.5f;

    const float amp = 0.3f;
    const float crossValues[] = {
        0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f
    };

    double baseDb = 0.0;
    double maxDev = 0.0;
    for (int i = 0; i < 11; ++i)
    {
        p.crossFeed = crossValues[i];
        fb.reset();
        fb.resetParams(p);
        const double db = measureWetRmsDb(fb, p, amp);
        if (i == 0) baseDb = db;
        const double dev = std::fabs(db - baseDb);
        maxDev = std::max(maxDev, dev);
        std::println("  cross={:.1}  wetRms={:.3} dB  dev={:.3} dB",
                    crossValues[i], db, dev);
    }

    std::println("\nmax deviation: {:.3} dB (gate 0.3 dB)", maxDev);
    if (maxDev > 0.3)
        FAIL("cross feed energy deviation {:.3} dB above 0.3 dB", maxDev);

    std::println("\n=== crossfeed_power_check OK ===");
    return 0;
}

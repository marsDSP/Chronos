// tests/harnesses/dsp/loop_dc_check.cpp
//
// Loop DC harness. Feeds a 0.5 DC offset with feedback 0.9 for 30 s.
// Asserts the mean output over the final second is below 1e-5.
//
// The in-loop low cut and the DC blocker sit after the saturator. They
// remove the DC offset the saturator makes so it cannot recirculate.
// The output HPF at 20 Hz removes the DC from the wet path. After 30 s
// the output mean must be near zero.
//
// Drives ChronosEngine directly, SharedCode only, no JUCE.

#include "dsp/ChronosEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...)                                                         \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs = 48000.0;
constexpr int    kFsInt = 48000;
constexpr int    kChannels = 2;
constexpr int    kBlock = 512;
constexpr double kDuration = 30.0;
constexpr int    kTotalSamples = static_cast<int>(kDuration * kFs);
constexpr int    kFinalSecondSamples = kFsInt;
constexpr double kDcOffset = 0.5;
constexpr double kMaxMean = 1e-5;

} // namespace

int main()
{
    std::printf("=== loop_dc_check ===\n");
    std::printf("fs=%d dc=%.3f feedback=0.9 loopCut=40Hz duration=%.0fs\n",
                kFsInt, kDcOffset, kDuration);

    MarsDSP::ChronosEngine engine;
    engine.prepare(kFs, kBlock, kChannels);
    engine.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    engine.setBypass(false);

    MarsDSP::ChronosEngine::Params p{};
    p.delaySamples  = 500.0f * 0.001f * static_cast<float>(kFsInt); // 500 ms
    p.driveLin      = 1.0f;
    p.mix           = 100.0f;
    p.gainLin       = 1.0f;
    p.hpfHz         = 20.0f;
    p.lpfHz         = 20000.0f;
    p.bits          = 32;
    p.adaaOrder     = 2;
    p.feedback      = 0.9f;
    p.dampHz        = 6000.0f;
    p.loopCutHz     = 40.0f;
    p.crossFeed     = 0.0f;
    p.loopDrive     = 1.0f;
    p.loopSatOrder  = 2;
    p.diffusion     = 0.0f;
    p.diffuserSize  = 0.5f;
    p.diffModDepth  = 0.0f;
    p.diffModRateHz = 0.5f;
    p.enableDiffuser = false;
    p.delaySync     = false;
    p.delayDivision = 11;
    p.delayModDepth = 0.0f;
    p.delayModRateHz = 0.35f;

    engine.reset();
    engine.resetParams(p);

    std::vector<float> bufL(static_cast<std::size_t>(kBlock), static_cast<float>(kDcOffset));
    std::vector<float> bufR(static_cast<std::size_t>(kBlock), static_cast<float>(kDcOffset));

    double sumL = 0.0, sumR = 0.0;
    int count = 0;

    for (int pos = 0; pos < kTotalSamples; pos += kBlock)
    {
        const int n = std::min(kBlock, kTotalSamples - pos);
        for (int i = 0; i < n; ++i)
        {
            bufL[static_cast<std::size_t>(i)] = static_cast<float>(kDcOffset);
            bufR[static_cast<std::size_t>(i)] = static_cast<float>(kDcOffset);
        }

        engine.setParams(p);
        float* io[2] = { bufL.data(), bufR.data() };
        engine.process(io, kChannels, n);

        if (pos + n > kTotalSamples - kFinalSecondSamples)
        {
            for (int i = 0; i < n; ++i)
            {
                sumL += static_cast<double>(bufL[static_cast<std::size_t>(i)]);
                sumR += static_cast<double>(bufR[static_cast<std::size_t>(i)]);
                ++count;
            }
        }
    }

    const double meanL = sumL / static_cast<double>(count);
    const double meanR = sumR / static_cast<double>(count);
    const double maxMean = std::max(std::fabs(meanL), std::fabs(meanR));

    std::printf("final 1 s mean: L=%+.3e R=%+.3e (gate %.0e)\n", meanL, meanR, kMaxMean);

    if (maxMean >= kMaxMean)
        FAIL("mean %.3e above %.0e", maxMean, kMaxMean);

    std::printf("\n=== loop_dc_check OK ===\n");
    return 0;
}

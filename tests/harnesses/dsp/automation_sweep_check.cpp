// tests/harnesses/dsp/automation_sweep_check.cpp
//
// Automation sweep harness. Sweeps every parameter from min to max to min
// over 10 s at block sizes 16, 64, 512, 2048. Asserts no sample-to-sample
// output delta above 0.25 and that all output is finite. The satLatency
// smoother and the damp coefficient smoother are the targets of this stage.
// Drives ChronosEngine directly, SharedCode only, no JUCE.

#include "dsp/ChronosEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...)                                                         \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs = 48000.0;
constexpr int    kFsInt = 48000;
constexpr int    kChannels = 2;
constexpr double kDuration = 10.0;
constexpr int    kTotalSamples = static_cast<int>(kDuration * kFs);
constexpr float  kMaxDelta = 0.25f;
constexpr double kSineFreq = 220.0;
constexpr float  kSineAmp = 0.05f;

float lerp(float a, float b, float t) noexcept { return a + (b - a) * t; }

// Map a sweep position in [0, 1] to the engine parameters.
// The position follows a triangle: 0 at start, 1 at midpoint, 0 at end.
MarsDSP::ChronosEngine::Params paramsAt(float pos) noexcept
{
    MarsDSP::ChronosEngine::Params p{};

    const float dly = lerp(240.0f, 24000.0f, pos);
    p.delaySamplesL = dly;
    p.delaySamplesR = dly;
    p.driveLin     = std::pow(10.0f, lerp(0.0f, 24.0f, pos) * 0.05f);
    p.mix          = lerp(0.0f, 100.0f, pos);
    p.gainLin      = std::pow(10.0f, lerp(-6.0f, 6.0f, pos) * 0.05f);
    p.hpfHz        = lerp(20.0f, 2000.0f, pos);
    p.lpfHz        = lerp(2000.0f, 20000.0f, pos);
    p.bits         = 32;
    p.adaaOrder    = (pos < 0.5f) ? 2 : 0;

    p.feedback     = lerp(0.0f, 0.9f, pos);
    p.dampHz       = lerp(200.0f, 16000.0f, pos);
    p.crossFeed    = lerp(0.0f, 1.0f, pos);
    p.loopDrive    = lerp(0.501f, 4.0f, pos);
    p.loopSatOrder = (pos < 0.5f) ? 2 : 0;
    p.diffusion    = lerp(0.0f, 1.0f, pos);
    p.diffuserSize = lerp(0.0f, 1.0f, pos);
    p.diffModDepth = lerp(0.0f, 1.5f, pos);
    p.diffModRateHz = lerp(0.1f, 4.0f, pos);
    p.enableDiffuser = (pos >= 0.5f);

    return p;
}

} // namespace

int main()
{
    std::println("=== automation_sweep_check ===");
    std::println("fs={} duration={:.0}s delta_limit={:.2}", kFsInt, kDuration, kMaxDelta);

    constexpr std::array<int, 4> blockSizes { { 16, 64, 512, 2048 } };
    const double kTwoPi = 2.0 * std::numbers::pi_v<double>;

    for (int blockSize : blockSizes)
    {
        g_section = "block-size";
        std::print("  block={} ... ", blockSize);
        std::fflush(stdout);

        MarsDSP::ChronosEngine engine;
        engine.prepare(kFs, blockSize, kChannels);
        engine.setDitherSeeds(0x12345678u, 0x9abcdef0u);
        engine.setBypass(false);

        MarsDSP::ChronosEngine::Params p0 = paramsAt(0.0f);
        engine.reset();
        engine.resetParams(p0);

        std::vector<float> bufL(static_cast<std::size_t>(blockSize));
        std::vector<float> bufR(static_cast<std::size_t>(blockSize));

        float prevL = 0.0f;
        float prevR = 0.0f;
        float maxDeltaL = 0.0f;
        float maxDeltaR = 0.0f;
        int maxDeltaSample = -1;
        bool allFinite = true;

        for (int pos = 0; pos < kTotalSamples; pos += blockSize)
        {
            const int n = std::min(blockSize, kTotalSamples - pos);
            const double tStart = static_cast<double>(pos) / kFs;
            const double tEnd = static_cast<double>(pos + n) / kFs;
            const float posStart = static_cast<float>(
                1.0 - std::abs(2.0 * tStart / kDuration - 1.0));
            const float posEnd = static_cast<float>(
                1.0 - std::abs(2.0 * tEnd / kDuration - 1.0));
            const float posMid = 0.5f * (posStart + posEnd);

            for (int i = 0; i < n; ++i)
            {
                const double t = static_cast<double>(pos + i) / kFs;
                const float v = static_cast<float>(kSineAmp * std::sin(kTwoPi * kSineFreq * t));
                bufL[static_cast<std::size_t>(i)] = v;
                bufR[static_cast<std::size_t>(i)] = v;
            }

            engine.setParams(paramsAt(posMid));
            std::array<float*, 2> io{ bufL.data(), bufR.data() };
            engine.process(io.data(), kChannels, n);

            for (int i = 0; i < n; ++i)
            {
                const float oL = bufL[static_cast<std::size_t>(i)];
                const float oR = bufR[static_cast<std::size_t>(i)];
                if (!std::isfinite(oL) || !std::isfinite(oR))
                {
                    allFinite = false;
                    FAIL("non-finite output at sample {} (L={} R={})", pos + i, static_cast<double>(oL), static_cast<double>(oR));
                }
                const float dL = std::abs(oL - prevL);
                const float dR = std::abs(oR - prevR);
                if (dL > maxDeltaL) { maxDeltaL = dL; maxDeltaSample = pos + i; }
                if (dR > maxDeltaR) { maxDeltaR = dR; maxDeltaSample = pos + i; }
                if (dL > kMaxDelta || dR > kMaxDelta)
                    FAIL("delta {:.4} at sample {} (block {})", std::max(dL, dR), pos + i, blockSize);
                prevL = oL;
                prevR = oR;
            }
        }

        CHECK(allFinite);
        std::println("ok (max delta L={:.4} R={:.4} at sample {})",
                    maxDeltaL, maxDeltaR, maxDeltaSample);
    }

    std::println("=== automation_sweep_check OK ===");
    return 0;
}

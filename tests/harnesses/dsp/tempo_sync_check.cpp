// tests/harnesses/dsp/tempo_sync_check.cpp
//
// Tempo sync harness. Verifies the division-to-milliseconds conversion at
// 120 BPM, and that a BPM ramp from 90 to 140 over 5 s produces no
// sample-to-sample output delta above 0.25. The ramp test drives the
// engine directly, computing the sync'd delay each block as the processor
// would. Links SharedCode only, no JUCE.

#include "dsp/ChronosEngine.h"
#include "utils/helpers/TempoSync.h"

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
constexpr float  kMaxDelta = 0.25f;

namespace TS = MarsDSP::Utils::Helpers::TempoSync;

} // namespace

int main()
{
    std::println("=== tempo_sync_check ===");

    // 1. Conversion checks at 120 BPM.
    g_section = "conversion";
    {
        constexpr double kBpm = 120.0;
        constexpr double kTol = 0.01;

        // 1/4 = index 11, expect 500.0 ms
        const double ms14 = TS::convertChoiceIndexToMilliseconds(11, kBpm);
        std::println("  1/4 at 120 BPM: {:.3} ms (expect 500.0)", ms14);
        CHECK(std::abs(ms14 - 500.0) < kTol);

        // 1/8. = index 10, expect 375.0 ms
        const double ms18d = TS::convertChoiceIndexToMilliseconds(10, kBpm);
        std::println("  1/8. at 120 BPM: {:.3} ms (expect 375.0)", ms18d);
        CHECK(std::abs(ms18d - 375.0) < kTol);

        // 1/4T = index 9, expect 333.333 ms
        const double ms14T = TS::convertChoiceIndexToMilliseconds(9, kBpm);
        std::println("  1/4T at 120 BPM: {:.3} ms (expect 333.333)", ms14T);
        CHECK(std::abs(ms14T - 333.333) < kTol);

        std::println("conversion at 120 BPM: PASS");
    }

    // 2. BPM ramp from 90 to 140 over 5 s, no delta above 0.25.
    g_section = "bpm-ramp";
    {
        constexpr int kBlock = 256;
        constexpr double kDuration = 5.0;
        const int kTotalSamples = static_cast<int>(kDuration * kFs);
        constexpr double kBpmStart = 90.0;
        constexpr double kBpmEnd = 140.0;
        constexpr int kDivision = 11; // 1/4

        MarsDSP::ChronosEngine engine;
        engine.prepare(kFs, kBlock, kChannels);
        engine.setDitherSeeds(0x12345678u, 0x9abcdef0u);
        engine.setBypass(false);

        MarsDSP::ChronosEngine::Params p{};
        p.driveLin = 1.0f;
        p.mix = 100.0f;
        p.gainLin = 1.0f;
        p.hpfHz = 20.0f;
        p.lpfHz = 20000.0f;
        p.bits = 32;
        p.adaaOrder = 0;
        p.feedback = 0.5f;
        p.dampHz = 6000.0f;
        p.loopCutHz = 40.0f;
        p.crossFeed = 0.0f;
        p.loopDrive = 1.0f;
        p.loopSatOrder = 0;
        p.diffusion = 0.0f;
        p.diffuserSize = 0.5f;
        p.diffModDepth = 0.0f;
        p.diffModRateHz = 0.5f;
        p.enableDiffuser = false;
        p.delaySync = true;
        p.delayDivision = kDivision;
        p.delayModDepth = 0.0f;
        p.delayModRateHz = 0.35f;

        // Snap the initial delay.
        const double ms0 = TS::convertChoiceIndexToMilliseconds(kDivision, kBpmStart);
        p.delaySamples = static_cast<float>(std::clamp(ms0, 1.0, 5000.0) * 0.001 * kFs);
        engine.reset();
        engine.resetParams(p);

        std::vector<float> bufL(static_cast<std::size_t>(kBlock));
        std::vector<float> bufR(static_cast<std::size_t>(kBlock));

        float prevL = 0.0f;
        float prevR = 0.0f;
        float maxDelta = 0.0f;
        int maxDeltaSample = -1;
        const double kTwoPi = 2.0 * std::numbers::pi_v<double>;

        for (int pos = 0; pos < kTotalSamples; pos += kBlock)
        {
            const int n = std::min(kBlock, kTotalSamples - pos);
            const double t = static_cast<double>(pos) / kFs;
            const double bpm = kBpmStart + (kBpmEnd - kBpmStart) * (t / kDuration);
            const double ms = TS::convertChoiceIndexToMilliseconds(kDivision, bpm);
            p.delaySamples = static_cast<float>(std::clamp(ms, 1.0, 5000.0) * 0.001 * kFs);

            for (int i = 0; i < n; ++i)
            {
                const double ti = static_cast<double>(pos + i) / kFs;
                const float v = 0.3f * static_cast<float>(std::sin(kTwoPi * 220.0 * ti));
                bufL[static_cast<std::size_t>(i)] = v;
                bufR[static_cast<std::size_t>(i)] = v;
            }

            engine.setParams(p);
            std::array<float*, 2> io{ bufL.data(), bufR.data() };
            engine.process(io.data(), kChannels, n);

            for (int i = 0; i < n; ++i)
            {
                const float oL = bufL[static_cast<std::size_t>(i)];
                const float oR = bufR[static_cast<std::size_t>(i)];
                const float dL = std::abs(oL - prevL);
                const float dR = std::abs(oR - prevR);
                if (dL > maxDelta) { maxDelta = dL; maxDeltaSample = pos + i; }
                if (dR > maxDelta) { maxDelta = dR; maxDeltaSample = pos + i; }
                if (dL > kMaxDelta || dR > kMaxDelta)
                    FAIL("delta {:.4} at sample {}", std::max(dL, dR), pos + i);
                prevL = oL;
                prevR = oR;
            }
        }

        std::println("BPM ramp 90->140 over 5 s: max delta {:.4} at sample {}: PASS",
                    maxDelta, maxDeltaSample);
    }

    std::println("=== tempo_sync_check OK ===");
    return 0;
}

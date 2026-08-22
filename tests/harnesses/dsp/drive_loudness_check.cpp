// tests/harnesses/dsp/drive_loudness_check.cpp
//
// Drive loudness harness. Sweeps the output drive from 0 to 24 dB on a
// 0.5-amplitude sine input at mix 100. Measures the RMS loudness at each
// step. Asserts the loudness rises monotonically and by 2.0 to 3.5 dB
// in total.
//
// The makeup gain (pow(rmsRatio, -0.7)) leaves a deliberate 2.75 dB of
// growth across the sweep for the 0.5-amplitude sine reference. The
// table is calibrated for this signal, so the measured rise lands in
// the expected band.
//
// Drives ChronosEngine directly, SharedCode only, no JUCE.

#include "dsp/ChronosEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
constexpr int    kBlock = 512;
constexpr double kDuration = 5.0;
constexpr int    kTotalSamples = static_cast<int>(kDuration * kFs);
constexpr int    kSkipSamples = static_cast<int>(1.0 * kFs); // skip 1 s for delay fill
constexpr double kSineAmp = 0.5;
constexpr double kSineFreq = 220.0;

// RMS loudness in dBFS over the measurement window.
double measureRmsDb(const std::vector<float>& l, const std::vector<float>& r,
                    int start, int len)
{
    double sumSq = 0.0;
    for (int i = 0; i < len; ++i)
    {
        const int idx = start + i;
        const double yL = static_cast<double>(l[static_cast<std::size_t>(idx)]);
        const double yR = static_cast<double>(r[static_cast<std::size_t>(idx)]);
        sumSq += yL * yL + yR * yR;
    }
    const double meanSq = sumSq / static_cast<double>(len);
    return 10.0 * std::log10(meanSq + 1e-30);
}

} // namespace

int main()
{
    std::println("=== drive_loudness_check ===");
    std::println("fs={} sine={:.3} amp {:.0} Hz mix=100 adaa=2\n", kFsInt, kSineAmp, kSineFreq);

    // Generate a 0.5-amplitude sine (the makeup table reference signal).
    std::vector<float> inL(static_cast<std::size_t>(kTotalSamples));
    std::vector<float> inR(static_cast<std::size_t>(kTotalSamples));
    {
        const double kTwoPi = 2.0 * std::numbers::pi_v<double>;
        for (int i = 0; i < kTotalSamples; ++i)
        {
            const double t = static_cast<double>(i) / kFs;
            const float v = static_cast<float>(kSineAmp * std::sin(kTwoPi * kSineFreq * t));
            inL[static_cast<std::size_t>(i)] = v;
            inR[static_cast<std::size_t>(i)] = v;
        }
    }

    const std::array<int, 9> driveSteps { { 0, 3, 6, 9, 12, 15, 18, 21, 24 } };
    constexpr int kNumSteps = static_cast<int>(sizeof(driveSteps) / sizeof(driveSteps[0]));
    std::array<double, kNumSteps> rmsDb{};

    for (int si = 0; si < kNumSteps; ++si)
    {
        g_section = "drive sweep";
        const int driveDb = driveSteps[si];

        MarsDSP::ChronosEngine engine;
        engine.prepare(kFs, kBlock, kChannels);
        engine.setDitherSeeds(0x12345678u, 0x9abcdef0u);
        engine.setBypass(false);

        MarsDSP::ChronosEngine::Params p{};
        p.delaySamples  = 10.0f * 0.001f * static_cast<float>(kFsInt); // 10 ms
        p.driveLin      = std::pow(10.0f, static_cast<float>(driveDb) / 20.0f);
        p.mix           = 100.0f;
        p.gainLin       = 1.0f;
        p.hpfHz         = 20.0f;
        p.lpfHz         = 20000.0f;
        p.bits          = 32;
        p.adaaOrder     = 2;
        p.feedback      = 0.0f;
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

        std::vector<float> outL = inL;
        std::vector<float> outR = inR;

        for (int pos = 0; pos < kTotalSamples; pos += kBlock)
        {
            const int n = std::min(kBlock, kTotalSamples - pos);
            std::array<float*, 2> io{ outL.data() + pos, outR.data() + pos };
            engine.setParams(p);
            engine.process(io.data(), kChannels, n);
        }

        rmsDb[si] = measureRmsDb(outL, outR, kSkipSamples, kTotalSamples - kSkipSamples);
        std::println("  drive {:2} dB: RMS {:.3} dBFS", driveDb, rmsDb[si]);
    }

    // Assert monotonicity.
    g_section = "monotonicity";
    for (int si = 1; si < kNumSteps; ++si)
    {
        if (rmsDb[si] < rmsDb[si - 1] - 0.001)
            FAIL("loudness not monotonic: step {} ({:.3}) < step {} ({:.3})",
                 driveSteps[si], rmsDb[si], driveSteps[si - 1], rmsDb[si - 1]);
    }
    std::println("monotonicity: PASS");

    // Assert total rise in [2.0, 3.5] dB.
    g_section = "total rise";
    const double rise = rmsDb[kNumSteps - 1] - rmsDb[0];
    std::println("total rise: {:.3} dB (gate [2.0, 3.5])", rise);
    if (rise < 2.0)
        FAIL("total rise {:.3} dB below 2.0 dB", rise);
    if (rise > 3.5)
        FAIL("total rise {:.3} dB above 3.5 dB", rise);

    std::println("\n=== drive_loudness_check OK ===");
    return 0;
}

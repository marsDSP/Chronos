// tests/harnesses/dsp/long_run_stability_check.cpp
//
// Long-run stability harness. Renders 4 hours of audio offline at 48 kHz
// with feedback 0.9 and all diffuser modulation active. The OU states
// drive the section modulation. Checks the OU state bound, the output
// RMS stability, and the output mean.
//
// Drives a FeedbackDelay directly (no engine, no ADAA, no SVF) so the
// 691 million samples finish well under the 4-minute limit. Links
// SharedCode only, no JUCE.

#include "dsp/FeedbackDelay.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...)                                                         \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs = 48000.0;
constexpr int    kFsInt = 48000;
constexpr int    kBlock = 512;
constexpr long long kTotalSamples = static_cast<long long>(4.0 * 3600.0 * kFs);
constexpr long long kOneMinSamples = static_cast<long long>(60.0 * kFs);
constexpr long long kRmsWindowSamples = static_cast<long long>(kFs);
constexpr long long kFinalMinuteStart = kTotalSamples - kOneMinSamples;

} // namespace

int main()
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    std::printf("=== long_run_stability_check ===\n");
    std::printf("fs=%d block=%d duration=4h samples=%lld\n", kFsInt, kBlock,
                static_cast<long long>(kTotalSamples));

    g_section = "prepare";
    constexpr int kMaxDelaySamp = 240000;
    const std::size_t ringFloats =
        MarsDSP::Delays::FeedbackDelay::ringStorageFloats(kFs, kBlock, kMaxDelaySamp);
    MarsDSP::Memory::BumpArena arena;
    arena.reset(ringFloats * sizeof(float));

    MarsDSP::Delays::FeedbackDelay fb;
    fb.prepare(kFs, kBlock, kMaxDelaySamp, arena);

    MarsDSP::Delays::FeedbackDelay::Params p;
    p.delaySamples = 500.0f * 0.001f * static_cast<float>(kFsInt);
    p.feedback = 0.9f;
    p.dampHz = 6000.0f;
    p.crossFeed = 0.0f;
    p.loopDrive = 1.0f;
    p.satOrder = 0;
    p.diffusion = 0.7f;
    p.diffuserSize = 0.5f;
    p.diffModDepth = 16.0f / 48.0f;
    p.diffModRateHz = 0.5f;
    p.enableDiffuser = true;
    fb.resetParams(p);

    std::vector<float> inL(static_cast<std::size_t>(kBlock));
    std::vector<float> inR(static_cast<std::size_t>(kBlock));
    std::vector<float> outL(static_cast<std::size_t>(kBlock));
    std::vector<float> outR(static_cast<std::size_t>(kBlock));

    constexpr double kSineFreq = 220.0;
    constexpr double kSineAmp = 0.001;
    const double kTwoPi = 2.0 * std::numbers::pi_v<double>;

    double rms1MinSumSq = 0.0;
    long long rms1MinCount = 0;
    bool rms1MinDone = false;

    double finalSum = 0.0;
    double finalSumSq = 0.0;
    long long finalCount = 0;

    g_section = "render";
    for (long long pos = 0; pos < kTotalSamples; pos += kBlock)
    {
        const int n = static_cast<int>(std::min(static_cast<long long>(kBlock),
                                                 kTotalSamples - pos));
        for (int i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(pos + i) / kFs;
            const float v = static_cast<float>(kSineAmp * std::sin(kTwoPi * kSineFreq * t));
            inL[static_cast<std::size_t>(i)] = v;
            inR[static_cast<std::size_t>(i)] = v;
        }

        fb.process(inL.data(), inR.data(), outL.data(), outR.data(), n);

        for (int i = 0; i < n; ++i)
        {
            const double ol = outL[static_cast<std::size_t>(i)];
            const double orr = outR[static_cast<std::size_t>(i)];
            const long long idx = pos + i;

            if (!rms1MinDone && idx >= kOneMinSamples
                && idx < kOneMinSamples + kRmsWindowSamples)
            {
                rms1MinSumSq += ol * ol + orr * orr;
                ++rms1MinCount;
            }
            if (!rms1MinDone && idx >= kOneMinSamples + kRmsWindowSamples)
                rms1MinDone = true;

            if (idx >= kFinalMinuteStart)
            {
                finalSum += ol + orr;
                finalSumSq += ol * ol + orr * orr;
                ++finalCount;
            }
        }
    }

    const auto t1 = clock::now();
    const double wallSec = std::chrono::duration<double>(t1 - t0).count();

    g_section = "OU state statistics";
    const float maxSigma = fb.ouStateMaxSigma();
    std::printf("max OU state: %.4f sigmas (gate < 6.0)\n", static_cast<double>(maxSigma));
    CHECK(maxSigma < 6.0f);

    g_section = "output RMS stability";
    CHECK(rms1MinCount > 0);
    CHECK(finalCount > 0);
    const double rms1Min = std::sqrt(rms1MinSumSq / static_cast<double>(rms1MinCount));
    const double rmsFinal = std::sqrt(finalSumSq / static_cast<double>(finalCount));
    const double dbDelta = (rms1Min > 0.0 && rmsFinal > 0.0)
        ? std::abs(20.0 * std::log10(rmsFinal / rms1Min)) : 999.0;
    std::printf("RMS at 1 min: %.6e  RMS at end: %.6e  delta: %.3f dB\n",
                rms1Min, rmsFinal, dbDelta);
    CHECK(dbDelta < 0.5);

    g_section = "output mean";
    const double meanFinal = finalSum / static_cast<double>(finalCount);
    std::printf("mean over final minute: %.3e\n", meanFinal);
    CHECK(std::abs(meanFinal) < 1e-6);

    std::printf("wall time: %.1f s\n", wallSec);
    CHECK(wallSec < 240.0);

    std::printf("=== long_run_stability_check OK ===\n");
    return 0;
}

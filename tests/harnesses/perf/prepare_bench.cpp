// tests/harnesses/perf/prepare_bench.cpp
//
// Prepare-time benchmark. Measures ChronosEngine::prepare wall time at six
// sample rates. Report only; no gate. The ceiling for the report is 100 ms.
// Links SharedCode only, no JUCE.

#include "dsp/ChronosEngine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr double kRates[] = { 44100.0, 48000.0, 88200.0, 96000.0, 176400.0, 192000.0 };
constexpr int kBlock = 512;
constexpr int kChannels = 2;
constexpr double kCeilingMs = 100.0;

} // namespace

int main()
{
    std::printf("=== prepare_bench ===\n");
    std::printf("block=%d channels=%d ceiling=%.0f ms\n\n", kBlock, kChannels, kCeilingMs);

    bool allUnderCeiling = true;
    for (double sr : kRates)
    {
        MarsDSP::ChronosEngine engine;
        const auto t0 = std::chrono::steady_clock::now();
        engine.prepare(sr, kBlock, kChannels);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const char* ok = (ms < kCeilingMs) ? "ok" : "OVER";
        if (ms >= kCeilingMs) allUnderCeiling = false;
        std::printf("  sr=%7.0f  prepare=%8.3f ms  %s\n", sr, ms, ok);
    }

    std::printf("\n%s\n", allUnderCeiling ? "=== ALL UNDER CEILING ==="
                                          : "=== SOME OVER CEILING ===");
    return 0;
}

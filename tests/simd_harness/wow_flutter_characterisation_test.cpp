// Wow and flutter characterisation harness.
//
// Sweeps rate / depth / drift across both engines and captures:
//
//   * summary stats (mean, stdev, min/max, peak abs),
//   * time-domain trace of a few seconds at a representative setting,
//   * depth sweep per engine to verify amplitude scaling is monotonic.
//
// Emits four CSVs:
//   tests/simd_harness/logs/wow_flutter_stats.csv
//   tests/simd_harness/logs/wow_flutter_depth_sweep.csv
//   tests/simd_harness/logs/wow_trace.csv
//   tests/simd_harness/logs/flutter_trace.csv

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "dsp/modulation/wow_engine.h"
#include "dsp/modulation/flutter_engine.h"

namespace fs = std::filesystem;
using namespace MarsDSP::DSP::Modulation;

namespace
{
    constexpr double kSampleRate     = 48000.0;
    constexpr int    kBlockSize      = 128;
    constexpr double kRunDuration    = 4.0;
    constexpr double kTraceDuration  = 1.0;

    const fs::path kLogDir = "tests/simd_harness/logs";

    void ensureLogDir()
    {
        std::error_code ec;
        fs::create_directories(kLogDir, ec);
    }

    struct SummaryStats
    {
        double mean;
        double stdev;
        double minValue;
        double maxValue;
        double peakAbs;
    };

    SummaryStats summarise(const std::vector<float>& samples)
    {
        double sum        = 0.0;
        double sumOfSq    = 0.0;
        double minValue   =  std::numeric_limits<double>::infinity();
        double maxValue   = -std::numeric_limits<double>::infinity();
        for (float value : samples)
        {
            const double v = static_cast<double>(value);
            sum     += v;
            sumOfSq += v * v;
            if (v < minValue) minValue = v;
            if (v > maxValue) maxValue = v;
        }
        const double count = static_cast<double>(samples.size());
        const double mean  = sum / count;
        const double var   = std::max(0.0, sumOfSq / count - mean * mean);
        return { mean, std::sqrt(var), minValue, maxValue,
                 std::max(std::fabs(minValue), std::fabs(maxValue)) };
    }

    std::vector<float> collectWowRun(float rate, float depth, float drift,
                                     double durationSeconds)
    {
        const int totalSamples = static_cast<int>(durationSeconds * kSampleRate);
        WowEngine engine;
        engine.prepare(kSampleRate, kBlockSize, 1);

        std::vector<float> output(static_cast<size_t>(totalSamples), 0.0f);
        for (int offset = 0; offset < totalSamples; offset += kBlockSize)
        {
            const int numInBlock = std::min(kBlockSize, totalSamples - offset);
            engine.prepareBlock(rate, depth, drift);
            for (int i = 0; i < numInBlock; ++i)
                output[static_cast<size_t>(offset + i)] = engine.nextSample(0);
            engine.wrapPhase(0);
        }
        return output;
    }

    std::vector<float> collectFlutterRun(float rate, float depth,
                                         double durationSeconds)
    {
        const int totalSamples = static_cast<int>(durationSeconds * kSampleRate);
        FlutterEngine engine;
        engine.prepare(kSampleRate, kBlockSize, 1);

        std::vector<float> output(static_cast<size_t>(totalSamples), 0.0f);
        for (int offset = 0; offset < totalSamples; offset += kBlockSize)
        {
            const int numInBlock = std::min(kBlockSize, totalSamples - offset);
            engine.prepareBlock(rate, depth);
            for (int i = 0; i < numInBlock; ++i)
            {
                const auto [ac, dc] = engine.nextSample(0);
                output[static_cast<size_t>(offset + i)] = ac + dc;
            }
            engine.wrapPhase(0);
        }
        return output;
    }

    void writeTrace(const fs::path& path, const std::vector<float>& samples)
    {
        std::ofstream csv(path);
        csv << "sample_index,time_ms,value\n";
        for (size_t i = 0; i < samples.size(); ++i)
        {
            const double timeMs = static_cast<double>(i) * 1000.0 / kSampleRate;
            csv << i << ',' << timeMs << ',' << samples[i] << '\n';
        }
    }

    void writeStatsRow(std::ofstream& csv, const std::string& engine,
                       float rate, float depth, float drift,
                       const SummaryStats& s)
    {
        csv << engine << ',' << rate << ',' << depth << ',' << drift
            << ',' << s.mean << ',' << s.stdev
            << ',' << s.minValue << ',' << s.maxValue
            << ',' << s.peakAbs << '\n';
    }
} // namespace

int main()
{
    ensureLogDir();

    std::ofstream statsCsv(kLogDir / "wow_flutter_stats.csv");
    statsCsv << "engine,rate,depth,drift,mean,stdev,min_value,max_value,peak_abs\n";

    std::ofstream depthSweepCsv(kLogDir / "wow_flutter_depth_sweep.csv");
    depthSweepCsv << "engine,depth,stdev,peak_abs\n";

    const std::vector<float> depthSweep{ 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

    // ------------------------------------------------------------------
    // Wow: characterise at a few (rate, drift) settings, sweep depth.
    // ------------------------------------------------------------------
    std::cout << "[wow/flutter] wow depth sweep (rate=0.5, drift=0.0)\n";
    for (const float depth : depthSweep)
    {
        const auto run = collectWowRun(0.5f, depth, 0.0f, kRunDuration);
        const auto stats = summarise(run);
        std::cout << "  depth=" << depth << "  stdev=" << stats.stdev
                  << "  peak=" << stats.peakAbs << "\n";
        writeStatsRow(statsCsv, "wow", 0.5f, depth, 0.0f, stats);
        depthSweepCsv << "wow," << depth << ',' << stats.stdev
                      << ',' << stats.peakAbs << '\n';
    }

    std::cout << "[wow/flutter] wow drift sweep (rate=0.5, depth=0.75)\n";
    for (const float drift : depthSweep)
    {
        const auto run = collectWowRun(0.5f, 0.75f, drift, kRunDuration);
        const auto stats = summarise(run);
        writeStatsRow(statsCsv, "wow_drift", 0.5f, 0.75f, drift, stats);
    }

    // ------------------------------------------------------------------
    // Flutter: sweep depth at mid-rate.
    // ------------------------------------------------------------------
    std::cout << "[wow/flutter] flutter depth sweep (rate=0.5)\n";
    for (const float depth : depthSweep)
    {
        const auto run = collectFlutterRun(0.5f, depth, kRunDuration);
        const auto stats = summarise(run);
        std::cout << "  depth=" << depth << "  stdev=" << stats.stdev
                  << "  peak=" << stats.peakAbs << "\n";
        writeStatsRow(statsCsv, "flutter", 0.5f, depth, 0.0f, stats);
        depthSweepCsv << "flutter," << depth << ',' << stats.stdev
                      << ',' << stats.peakAbs << '\n';
    }

    std::cout << "[wow/flutter] flutter rate sweep (depth=0.75)\n";
    for (const float rate : depthSweep)
    {
        const auto run = collectFlutterRun(rate, 0.75f, kRunDuration);
        const auto stats = summarise(run);
        writeStatsRow(statsCsv, "flutter_rate", rate, 0.75f, 0.0f, stats);
    }

    // ------------------------------------------------------------------
    // Traces at a representative setting so the viz can draw the
    // time-domain LFO shapes.
    // ------------------------------------------------------------------
    const auto wowTrace     = collectWowRun    (0.5f, 0.75f, 0.25f, kTraceDuration);
    const auto flutterTrace = collectFlutterRun(0.5f, 0.75f,        kTraceDuration);
    writeTrace(kLogDir / "wow_trace.csv",     wowTrace);
    writeTrace(kLogDir / "flutter_trace.csv", flutterTrace);

    statsCsv.close();
    depthSweepCsv.close();

    std::cout << "[wow/flutter] wrote "
              << (kLogDir / "wow_flutter_stats.csv") << ", "
              << (kLogDir / "wow_flutter_depth_sweep.csv") << ", "
              << (kLogDir / "wow_trace.csv") << ", "
              << (kLogDir / "flutter_trace.csv") << "\n";
    return 0;
}

// tests/harnesses/perf/bbd_bench.cpp
//
// Performance benchmark for BBD delay core (FeedbackDelay in BBD mode)
// and BrigadeLine in isolation across various delay times.

#include "bench_util.h"
#include "dsp/FeedbackDelay.h"
#include "dsp/bbd/BrigadeLine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <print>
#include <cstring>
#include <numbers>
#include <string>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;

#if defined(__clang__) || defined(__GNUC__)
    template <class T>
    inline void doNotOptimize (T const& v) noexcept
    {
        asm volatile ("" : : "r,m" (v) : "memory");
    }
#else
    template <class T>
    inline void doNotOptimize (T const& v) noexcept
    {
        volatile T sink = v;
        (void) sink;
    }
#endif

    template <class Fn>
    double benchNsPerOp (Fn fn, std::size_t ops, std::size_t reps, double& sinkOut)
    {
        double best = 1.0e30;
        double total = 0.0;
        for (std::size_t r = 0; r < reps; ++r)
        {
            const auto t0 = Clock::now();
            const double a = fn();
            const auto t1 = Clock::now();
            total += a;
            best = std::min (best, std::chrono::duration<double, std::nano> (t1 - t0).count());
        }
        sinkOut = total;
        return best / static_cast<double> (ops);
    }
} // namespace

int main (int argc, char** argv)
{
    std::string jsonPath;
    bool provisional = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp (argv[i], "--json") == 0 && i + 1 < argc) jsonPath = argv[++i];
        else if (std::strcmp (argv[i], "--provisional") == 0) provisional = true;
    }

    bench::setFtzDaz();

    constexpr double kFs = 48000.0;
    constexpr int kBlock = 256;
    constexpr std::size_t kSamples = 524288;
    constexpr std::size_t kReps = 5;
    double sink = 0.0;

    std::vector<float> inL (kSamples), inR (kSamples);
    std::vector<float> outL (kSamples), outR (kSamples);
    for (std::size_t i = 0; i < kSamples; ++i)
    {
        inL[i] = 0.5f * static_cast<float> (std::sin (2.0 * std::numbers::pi * 1000.0 * i / kFs));
        inR[i] = 0.5f * static_cast<float> (std::sin (2.0 * std::numbers::pi * 1000.0 * i / kFs));
    }

    std::vector<bench::Record> records;
    const std::array<float, 6> delaysMs { { 5.0f, 50.0f, 375.0f, 853.0f, 1500.0f, 5000.0f } };

    std::println("=== Chronos BBD Performance Benchmark ===");

    for (float dMs : delaysMs)
    {
        const float delaySamples = static_cast<float> (dMs * 0.001 * kFs);

        // 1. FeedbackDelay in BBD mode (stereo)
        {
            MarsDSP::Delays::FeedbackDelay fb;
            fb.prepare (kFs, kBlock, 262144);
            MarsDSP::Delays::FeedbackDelay::Params p;
            p.delaySamplesL = delaySamples;
            p.delaySamplesR = delaySamples;
            p.feedback = 0.42f;
            p.satOrder = 2;
            p.enableDiffuser = false;
            p.delayMode = 1; // BBD
            fb.resetParams (p);

            auto run = [&]() -> double
            {
                double acc = 0.0;
                for (std::size_t off = 0; off < kSamples; off += kBlock)
                {
                    fb.process (inL.data() + off, inR.data() + off,
                                outL.data() + off, outR.data() + off,
                                kBlock);
                    acc += outL[off];
                    doNotOptimize (acc);
                }
                return acc;
            };

            const double ns = benchNsPerOp (run, kSamples, kReps, sink);
            const std::string cfg = std::format("delay={:.0f}ms", static_cast<double> (dMs));
            records.push_back ({ "FeedbackDelay (BBD)", cfg, ns });
            std::println("  FeedbackDelay (BBD) {}: {:7.3} ns/sample", cfg, ns);
        }

        // 2. BrigadeLine in isolation
        {
            std::vector<float> mem (MarsDSP::BBD::BrigadeLine::bbdStorageFloats (1), 0.0f);
            MarsDSP::BBD::BrigadeLine line;
            line.prepare (kFs, mem.data());
            line.setDelaySeconds (static_cast<float> (dMs * 0.001));

            auto run = [&]() -> double
            {
                double acc = 0.0;
                for (std::size_t i = 0; i < kSamples; ++i)
                {
                    const float y = line.process (inL[i]);
                    acc += y;
                    doNotOptimize (acc);
                }
                return acc;
            };

            const double ns = benchNsPerOp (run, kSamples, kReps, sink);
            const std::string cfg = std::format("delay={:.0f}ms", static_cast<double> (dMs));
            records.push_back ({ "BrigadeLine::process", cfg, ns });
            std::println("  BrigadeLine::process   {}: {:7.3} ns/sample", cfg, ns);
        }
    }

    if (!jsonPath.empty())
        bench::writeJson (jsonPath, records, provisional);

    return 0;
}

// tests/harnesses/perf/compander_bench.cpp
//
// Performance benchmark for Compander cells (CompressorCell, ExpanderCell,
// and the 4-cell stereo companded pair) at 48 kHz.

#include "bench_util.h"
#include "dsp/bbd/CompanderCell.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <numbers>
#include <print>
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
    constexpr std::size_t kSamples = 524288;
    constexpr std::size_t kReps = 5;
    double sink = 0.0;

    std::vector<float> inL (kSamples), inR (kSamples);
    for (std::size_t i = 0; i < kSamples; ++i)
    {
        inL[i] = 0.5f * static_cast<float> (std::sin (2.0 * std::numbers::pi * 1000.0 * i / kFs));
        inR[i] = 0.5f * static_cast<float> (std::sin (2.0 * std::numbers::pi * 1000.0 * i / kFs));
    }

    std::vector<bench::Record> records;

    std::println("=== Chronos Compander Performance Benchmark ===");

    // 1. CompressorCell (mono)
    {
        MarsDSP::BBD::CompressorCell comp;
        comp.prepare (kFs);

        auto run = [&]() -> double
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < kSamples; ++i)
            {
                const float y = comp.processSample (inL[i]);
                acc += y;
                doNotOptimize (acc);
            }
            return acc;
        };

        const double ns = benchNsPerOp (run, kSamples, kReps, sink);
        records.push_back ({ "CompressorCell (mono)", "fs=48k", ns });
        std::println("  CompressorCell (mono):  {:7.3f} ns/sample", ns);
    }

    // 2. ExpanderCell (mono)
    {
        MarsDSP::BBD::ExpanderCell exp;
        exp.prepare (kFs);

        auto run = [&]() -> double
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < kSamples; ++i)
            {
                const float y = exp.processSample (inL[i]);
                acc += y;
                doNotOptimize (acc);
            }
            return acc;
        };

        const double ns = benchNsPerOp (run, kSamples, kReps, sink);
        records.push_back ({ "ExpanderCell (mono)", "fs=48k", ns });
        std::println("  ExpanderCell (mono):    {:7.3f} ns/sample", ns);
    }

    // 3. Compander pair (mono: 1 comp + 1 exp)
    {
        MarsDSP::BBD::CompressorCell comp;
        MarsDSP::BBD::ExpanderCell exp;
        comp.prepare (kFs);
        exp.prepare (kFs);

        auto run = [&]() -> double
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < kSamples; ++i)
            {
                const float c = comp.processSample (inL[i]);
                const float y = exp.processSample (c);
                acc += y;
                doNotOptimize (acc);
            }
            return acc;
        };

        const double ns = benchNsPerOp (run, kSamples, kReps, sink);
        records.push_back ({ "Compander pair (mono)", "fs=48k", ns });
        std::println("  Compander pair (mono):  {:7.3f} ns/sample", ns);
    }

    // 4. Stereo companded pair (2 comp + 2 exp)
    {
        MarsDSP::BBD::CompressorCell compL, compR;
        MarsDSP::BBD::ExpanderCell expL, expR;
        compL.prepare (kFs);
        compR.prepare (kFs);
        expL.prepare (kFs);
        expR.prepare (kFs);

        auto run = [&]() -> double
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < kSamples; ++i)
            {
                const float cL = compL.processSample (inL[i]);
                const float cR = compR.processSample (inR[i]);
                const float yL = expL.processSample (cL);
                const float yR = expR.processSample (cR);
                acc += yL + yR;
                doNotOptimize (acc);
            }
            return acc;
        };

        const double ns = benchNsPerOp (run, kSamples, kReps, sink);
        const double nsPerChan = ns * 0.5;
        records.push_back ({ "Compander pair (stereo)", "fs=48k", ns });
        std::println("  Compander pair (stereo):{:7.3f} ns/sample ({:.3f} ns/chan)", ns, nsPerChan);
    }

    if (!jsonPath.empty())
        bench::writeJson (jsonPath, records, provisional);

    return 0;
}

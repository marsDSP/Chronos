// tests/harnesses/perf/sallen_key_bench.cpp
//
// Performance benchmark for SallenKeyLPF, SallenKeyHPF, setParams,
// and OutputFilterStage in Digital and Analog modes.

#include "bench_util.h"
#include "dsp/SallenKeyLPF.h"
#include "dsp/SallenKeyHPF.h"
#include "dsp/OutputFilterStage.h"

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

    constexpr std::size_t kOps = 1'000'000;
    constexpr std::size_t kReps = 5;
    double sink = 0.0;

    std::vector<float> in(kOps);
    for (std::size_t i = 0; i < kOps; ++i)
        in[i] = static_cast<float> (std::sin (2.0 * std::numbers::pi * 1000.0 * i / 48000.0));

    std::vector<bench::Record> records;

    // 1. SallenKeyLPF::processSample
    {
        MarsDSP::Filters::SallenKeyLPF lpf;
        lpf.prepare (48000.0);
        lpf.setParams (1000.0f, 0.7071f);

        auto run = [&]() -> double
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < kOps; ++i)
            {
                const float y = lpf.processSample (in[i]);
                acc += y;
                doNotOptimize (acc);
            }
            return acc;
        };

        const double ns = benchNsPerOp (run, kOps, kReps, sink);
        records.push_back ({ "SallenKeyLPF::processSample", "", ns });
        std::println("  SallenKeyLPF::processSample: {:7.3} ns/sample", ns);
    }

    // 2. SallenKeyHPF::processSample
    {
        MarsDSP::Filters::SallenKeyHPF hpf;
        hpf.prepare (48000.0);
        hpf.setParams (1000.0f, 0.7071f);

        auto run = [&]() -> double
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < kOps; ++i)
            {
                const float y = hpf.processSample (in[i]);
                acc += y;
                doNotOptimize (acc);
            }
            return acc;
        };

        const double ns = benchNsPerOp (run, kOps, kReps, sink);
        records.push_back ({ "SallenKeyHPF::processSample", "", ns });
        std::println("  SallenKeyHPF::processSample: {:7.3} ns/sample", ns);
    }

    // 3. SallenKeyLPF::setParams
    {
        MarsDSP::Filters::SallenKeyLPF lpf;
        lpf.prepare (48000.0);

        constexpr std::size_t kSetOps = 100'000;
        auto run = [&]() -> double
        {
            double acc = 0.0;
            for (std::size_t i = 0; i < kSetOps; ++i)
            {
                const float f = 200.0f + static_cast<float> (i % 10000);
                lpf.setParams (f, 0.7071f);
                acc += f;
                doNotOptimize (acc);
            }
            return acc;
        };

        const double ns = benchNsPerOp (run, kSetOps, kReps, sink);
        records.push_back ({ "SallenKeyLPF::setParams", "", ns });
        std::println("  SallenKeyLPF::setParams:     {:7.3} ns/call", ns);
    }

    // 4. OutputFilterStage in Digital and Analog modes
    for (int ch : { 1, 2 })
    {
        for (int bs : { 64, 128, 256, 512 })
        {
            const std::size_t totalSamples = 524288;
            std::vector<float> bufInL (totalSamples, 0.1f);
            std::vector<float> bufInR (totalSamples, 0.1f);
            std::vector<float> bufOutL (totalSamples);
            std::vector<float> bufOutR (totalSamples);

            // Digital
            {
                MarsDSP::Filters::OutputFilterStage stage;
                stage.prepare (48000.0, ch);
                stage.setMode (MarsDSP::Filters::OutputFilterStage::Mode::Digital);
                stage.setCutoffs (20.0f, 20000.0f);

                auto run = [&]() -> double
                {
                    double acc = 0.0;
                    for (std::size_t off = 0; off < totalSamples; off += bs)
                    {
                        stage.process (bufInL.data() + off,
                                       ch > 1 ? bufInR.data() + off : nullptr,
                                       bufOutL.data() + off,
                                       ch > 1 ? bufOutR.data() + off : nullptr,
                                       bs);
                        acc += bufOutL[off];
                        doNotOptimize (acc);
                    }
                    return acc;
                };

                const double ns = benchNsPerOp (run, totalSamples, kReps, sink);
                const std::string cfg = std::format("bs={} ch={}", bs, ch);
                records.push_back ({ "OutputFilterStage (Digital)", cfg, ns });
                std::println("  OutputFilterStage (Digital) {}: {:7.3} ns/sample", cfg, ns);
            }

            // Analog
            {
                MarsDSP::Filters::OutputFilterStage stage;
                stage.prepare (48000.0, ch);
                stage.setMode (MarsDSP::Filters::OutputFilterStage::Mode::Analog);
                stage.setCutoffs (20.0f, 20000.0f);

                auto run = [&]() -> double
                {
                    double acc = 0.0;
                    for (std::size_t off = 0; off < totalSamples; off += bs)
                    {
                        stage.process (bufInL.data() + off,
                                       ch > 1 ? bufInR.data() + off : nullptr,
                                       bufOutL.data() + off,
                                       ch > 1 ? bufOutR.data() + off : nullptr,
                                       bs);
                        acc += bufOutL[off];
                        doNotOptimize (acc);
                    }
                    return acc;
                };

                const double ns = benchNsPerOp (run, totalSamples, kReps, sink);
                const std::string cfg = std::format("bs={} ch={}", bs, ch);
                records.push_back ({ "OutputFilterStage (Analog)", cfg, ns });
                std::println("  OutputFilterStage (Analog)  {}: {:7.3} ns/sample", cfg, ns);
            }
        }
    }

    if (!jsonPath.empty())
        bench::writeJson (jsonPath, records, provisional);

    return 0;
}

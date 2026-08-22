// tests/harnesses/perf/delay_line_bench.cpp
// Throughput benchmark for MarsDSP::Delays::SimdDelayLine vs the old
// juce::dsp::DelayLine<float> per-sample path that Chronos used before the
// pow2-ring port. Measures the total perf boost of the port.
//
// What it measures (stereo, 48 kHz, block 256, fractional delay 347.5):
//   1. SimdDelayLine::process()        - 4-wide SIMD kernel, Lagrange5th.
//   2. SimdDelayLine::processScalar()  - scalar dot6 kernel (A/B reference).
//   3. juce::dsp::DelayLine<float>     - per-sample setDelay + pushSample +
//      popSample × 2 ch (the exact old processBlock shape, default Linear
//      interpolation). This is the apples-to-apples old baseline; the per-sample
//      setDelay cost is included because the old code really did call it every
//      sample - the port moves delay to block-rate, so that gain is real.
//   4. Per-mode SIMD throughput (Linear / Lagrange3rd / Lagrange5th) - confirms
//      the zero-padded Linear path (6 MACs vs 2) is not a problem in practice.
//
// Timing idiom mirrors tan_bench: steady_clock, doNotOptimize compiler barriers,
// min-of-5-reps, sink accumulators keep the loops live. Forced -O2 and
// -Xarch_x86_64 -mfma (see tests/CMakeLists.txt) so the SIMD kernel inlines and
// FMADD lowers to a fused multiply-add, matching the plugin target's x86_64
// slice; arm64 has FMA unconditionally.
//
// This is the first harness to link a JUCE module (juce::juce_dsp); SharedCode
// propagates JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED etc. via INTERFACE.
//
// Exit: 0 = informational (the timing gate moved to scripts/bench_gate.py).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "bench_util.h"
#include "dsp/SimdDelayLine.h"
#include <juce_dsp/juce_dsp.h>

namespace
{
    using MarsDSP::Delays::Interpolation;
    using MarsDSP::Delays::SimdDelayLine;
    using Clock = std::chrono::steady_clock;

    constexpr double kFs = 48000.0;
    constexpr std::size_t kBlock = 256;
    constexpr std::size_t kBlocks = 4096; // 4096 × 256 = 1 048 576 samples
    constexpr std::size_t kTotal = kBlock * kBlocks;
    constexpr std::size_t kReps = 5;
    constexpr float kDelaySamples = 347.5f; // fractional, realistic
    constexpr int kMaxDelaySamples = 240000; // 5000 ms @ 48 kHz

    // Compiler barriers (Google Benchmark DoNotOptimize technique) - see tan_bench.
#if defined(__clang__) || defined(__GNUC__)
    template<class T>
    inline void doNotOptimize(T const &v) noexcept
    {
        asm volatile("" : : "r,m"(v) : "memory");
    }
#else
    template<class T>
    inline void doNotOptimize(T const &v) noexcept
    {
        volatile T sink = v;
        (void) sink;
    }
#endif

    // Run fn (which processes kTotal samples) kReps times; return best (min)
    // ns/sample and sink the accumulators so the loop body stays live.
    template<class Fn>
    double benchNsPerSample(Fn fn, double &sinkOut)
    {
        double best = std::numeric_limits<double>::infinity();
        double total = 0.0;
        for (std::size_t r = 0; r < kReps; ++r)
        {
            const auto t0 = Clock::now();
            const double a = fn();
            const auto t1 = Clock::now();
            total += a;
            best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        sinkOut = total;
        return best / static_cast<double>(kTotal);
    }

    std::vector<float> makeRampL()
    {
        std::vector<float> v(kTotal);
        for (std::size_t i = 0; i < kTotal; ++i)
            v[i] = static_cast<float>(std::sin(2.0 * std::numbers::pi * 440.0 * static_cast<double>(i) / kFs));
        return v;
    }

    std::vector<float> makeRampR()
    {
        std::vector<float> v(kTotal);
        for (std::size_t i = 0; i < kTotal; ++i)
            v[i] = static_cast<float>(std::sin(2.0 * std::numbers::pi * 330.0 * static_cast<double>(i) / kFs));
        return v;
    }

    // ---- SimdDelayLine (SIMD or scalar kernel selectable via the method) ----
    template<bool UseSimd>
    double runSimdDelayLine(const std::vector<float> &inL, const std::vector<float> &inR,
                            Interpolation mode)
    {
        SimdDelayLine dl;
        dl.prepare(kFs, kBlock, 5000.0f);
        dl.setInterpolation(mode);
        dl.reset();
        std::vector<float> wetL(kBlock);
        std::vector<float> wetR(kBlock);
        double acc = 0.0;
        for (std::size_t b = 0; b < kBlocks; ++b)
        {
            const std::size_t off = b * kBlock;
            if constexpr (UseSimd)
                dl.process(inL.data() + off, inR.data() + off,
                           wetL.data(), wetR.data(), kBlock,
                           kDelaySamples, kDelaySamples);
            else
                dl.processScalar(inL.data() + off, inR.data() + off,
                                 wetL.data(), wetR.data(), kBlock,
                                 kDelaySamples, kDelaySamples);
            for (std::size_t i = 0; i < kBlock; ++i)
            {
                acc += static_cast<double>(wetL[i] + wetR[i]);
                doNotOptimize(acc);
            }
        }
        return acc;
    }

    // ---- juce::dsp::DelayLine, per-sample, the exact old processBlock shape ----
    double runJuceDelayLine(const std::vector<float> &inL, const std::vector<float> &inR)
    {
        juce::dsp::DelayLine<float> dl;
        juce::dsp::ProcessSpec spec{};
        spec.sampleRate = kFs;
        spec.maximumBlockSize = static_cast<juce::uint32>(kBlock);
        spec.numChannels = 2;
        dl.prepare(spec);
        dl.setMaximumDelayInSamples(kMaxDelaySamples);
        dl.reset();
        double acc = 0.0;
        for (std::size_t b = 0; b < kBlocks; ++b)
        {
            const std::size_t off = b * kBlock;
            for (std::size_t i = 0; i < kBlock; ++i)
            {
                const std::size_t idx = off + i;
                dl.setDelay(kDelaySamples); // per-sample, as the old code did
                dl.pushSample(0, inL[idx]);
                dl.pushSample(1, inR[idx]);
                const float yL = dl.popSample(0);
                const float yR = dl.popSample(1);
                acc += static_cast<double>(yL + yR);
                doNotOptimize(acc);
            }
        }
        return acc;
    }

    const char *modeName(Interpolation m) noexcept
    {
        switch (m)
        {
            case Interpolation::Linear: return "Linear";
            case Interpolation::Lagrange3rd: return "Lag3";
            case Interpolation::Lagrange5th: return "Lag5";
        }
        return "?";
    }
} // namespace

int main(int argc, char **argv)
{
    std::string jsonPath;
    bool provisional = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) jsonPath = argv[++i];
        else if (std::strcmp(argv[i], "--provisional") == 0) provisional = true;
    }

    bench::setFtzDaz();

    const std::vector<float> inL = makeRampL();
    const std::vector<float> inR = makeRampR();

    std::print("=== Chronos delay-line throughput benchmark ===\n");
    std::print("sr=%.0f  block=%zu  total=%zu samples  delay=%.1f smp  reps=%zu\n", kFs, kBlock, kTotal,
               static_cast<double>(kDelaySamples), kReps);
    std::print("(min of %zu reps; sink accumulators keep loops live)\n\n", kReps);

    double sink = 0.0;

    // ---- 1/2/3. Headline: SIMD vs scalar vs juce ----
    const double nsSimd = benchNsPerSample([&] { return runSimdDelayLine<true>(inL, inR, Interpolation::Lagrange5th); },
                                           sink);
    const double nsScalar = benchNsPerSample([&]
    {
        return runSimdDelayLine<false>(inL, inR, Interpolation::Lagrange5th);
    }, sink);
    const double nsJuce = benchNsPerSample([&] { return runJuceDelayLine(inL, inR); }, sink);

    std::print("[delay] ns/sample, stereo (min of %zu reps):\n", kReps);
    std::print("       SimdDelayLine  SIMD   (Lag5) : %7.3f ns/sample\n", nsSimd);
    std::print("       SimdDelayLine  scalar (Lag5) : %7.3f ns/sample  (%.2fx vs scalar)\n", nsScalar,
               nsScalar / nsSimd);
    std::print("       juce::dsp::DelayLine per-smp : %7.3f ns/sample  (%.2fx vs juce)\n", nsJuce, nsJuce / nsSimd);
    std::print("       (timing gate moved to scripts/bench_gate.py)\n\n");

    // ---- 4. Per-mode SIMD throughput (zero-padded Linear cost check) ----
    std::print("[per-mode] SIMD ns/sample (min of %zu reps):\n", kReps);
    const Interpolation modes[] = {Interpolation::Linear, Interpolation::Lagrange3rd, Interpolation::Lagrange5th};
    double minNs = std::numeric_limits<double>::infinity();
    double maxNs = 0.0;
    std::array<double, 3> perModeNs = {{  }};
    int mi = 0;
    for (Interpolation m: modes)
    {
        const double n = benchNsPerSample([&] { return runSimdDelayLine<true>(inL, inR, m); }, sink);
        perModeNs[mi++] = n;
        std::print("       %-12s : %7.3f ns/sample\n", modeName(m), n);
        minNs = std::min(minNs, n);
        maxNs = std::max(maxNs, n);
    }
    std::print("       (max/min = %.2fx — Linear's zero-padded 6-MAC path vs Lag5)\n\n", maxNs / minNs);

    std::vector<bench::Record> records;
    records.emplace_back("SimdDelayLine", "SIMD,Lag5", nsSimd);
    records.emplace_back("SimdDelayLine", "scalar,Lag5", nsScalar);
    records.emplace_back("juce::dsp::DelayLine", "per-sample", nsJuce);
    records.emplace_back("SimdDelayLine", "SIMD,Linear", perModeNs[0]);
    records.emplace_back("SimdDelayLine", "SIMD,Lag3", perModeNs[1]);

    std::print("(sink=%f)\n", sink);

    if (!jsonPath.empty())
        bench::writeJson(jsonPath, records, provisional);

    std::print("=== DONE (informational only, gate moved to scripts/bench_gate.py) ===\n");
    return 0;
}

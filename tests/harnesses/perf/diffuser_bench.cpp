/**
 * Throughput benchmark for the Diffuser cascade on the wet path.
 * Matrix: size {0, 0.5, 1} x modDepth {0, 16} x block {64, 256, 512} x
 * ramp {settled, midRamp}. Each config also measures the scalar reference
 * twin processBlockRef. Min-of-5 reps, ns per sample.
 * Informational only: exits non-zero on NaN or Inf.
 */

#include "dsp/Diffuser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <print>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "bench_util.h"

namespace
{
    constexpr double kFs = 48000.0;
    constexpr double kPi = std::numbers::pi_v<double>;
    constexpr int kTotal = 1 << 19; // 524288 samples per rep
    constexpr int kReps = 5;
    constexpr int kWarmup = 4096; // > 50 ms (2400 smp): the smoothers settle
    constexpr float kDiffusion = 0.7f;
    constexpr float kModRateHz = 0.5f;

    using Clock = std::chrono::steady_clock;
    using MarsDSP::Diffusion::Diffuser;

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
        (void) sink; // MSVC fallback
    }
#endif

    const char *archName() noexcept
    {
#if defined(__x86_64__) || defined(_M_X64)
        return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
        return "arm64";
#else
        return "native";
#endif
    }

    struct Cfg
    {
        float size;
        float modDepth;
        int block;
        bool midRamp;
    };

    // midRamp targets are ±0.3 above and below cfg.size, clamped to [0,1].
    inline float sizeTarget(float size, bool hi) noexcept
    {
        return std::clamp(hi ? size + 0.3f : size - 0.3f, 0.0f, 1.0f);
    }

    // One timed rep: process kTotal samples in block chunks. The caller preloads
    // the work buffer. When midRamp is true, this function sets the size target
    // again every block; this automation cost is part of the measurement. The
    // function sinks the last sample of each block to keep the work live.
    template<bool UseRef>
    double runDiffuser(Diffuser &d, std::vector<float> &wL, std::vector<float> &wR,
                       const Cfg &c)
    {
        double acc = 0.0;
        bool hi = false;
        for (int off = 0; off < kTotal; off += c.block)
        {
            if (c.midRamp)
            {
                d.setSize(sizeTarget(c.size, hi));
                hi = !hi;
            }
            if constexpr (UseRef)
                d.processBlockRef(wL.data() + off, wR.data() + off, c.block);
            else
                d.processBlock(wL.data() + off, wR.data() + off, c.block);
            const int last = off + c.block - 1;
            acc += static_cast<double>(wL[static_cast<std::size_t>(last)]);
            acc += static_cast<double>(wR[static_cast<std::size_t>(last)]);
            doNotOptimize(acc);
        }
        return acc;
    }

    // Min of kReps reps, in ns per input sample. Each rep reloads the work buffer
    // from the pristine input before the clock, because processBlock works in
    // place and destroys it. Only the processBlock calls are timed.
    template<bool UseRef>
    double benchDiffuser(Diffuser &d, const std::vector<float> &inL,
                         const std::vector<float> &inR,
                         std::vector<float> &wL, std::vector<float> &wR,
                         const Cfg &c, double &sinkOut)
    {
        double best = std::numeric_limits<double>::infinity();
        double total = 0.0;
        for (int r = 0; r < kReps; ++r)
        {
            std::memcpy(wL.data(), inL.data(), sizeof(float) * static_cast<std::size_t>(kTotal));
            std::memcpy(wR.data(), inR.data(), sizeof(float) * static_cast<std::size_t>(kTotal));
            const auto t0 = Clock::now();
            const double a = runDiffuser<UseRef>(d, wL, wR, c);
            const auto t1 = Clock::now();
            total += a;
            best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        sinkOut += total;
        return best / static_cast<double>(kTotal);
    }

    // Untimed warmup: settle the smoothers (settled) or keep the size ramping
    // (midRamp), and verify that the output is finite. The function processes the
    // first kWarmup samples of the input.
    bool warmup(Diffuser &d, const std::vector<float> &inL,
                const std::vector<float> &inR,
                std::vector<float> &wL, std::vector<float> &wR, const Cfg &c)
    {
        std::memcpy(wL.data(), inL.data(), sizeof(float) * static_cast<std::size_t>(kWarmup));
        std::memcpy(wR.data(), inR.data(), sizeof(float) * static_cast<std::size_t>(kWarmup));
        bool finite = true;
        bool hi = true; // start with the high target so size=0 moves from block 1
        for (int off = 0; off < kWarmup; off += c.block)
        {
            const int m = std::min(c.block, kWarmup - off);
            if (c.midRamp)
            {
                d.setSize(sizeTarget(c.size, hi));
                hi = !hi;
            }
            d.processBlock(wL.data() + off, wR.data() + off, m);
            for (int s = 0; s < m; ++s)
            {
                if (!std::isfinite(wL[static_cast<std::size_t>(off + s)])) finite = false;
                if (!std::isfinite(wR[static_cast<std::size_t>(off + s)])) finite = false;
            }
        }
        return finite;
    }
} // namespace

int main(int argc, char **argv)
{
    std::string csvPath, jsonPath;
    bool provisional = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            csvPath = argv[++i];
        else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc)
            jsonPath = argv[++i];
        else if (std::strcmp(argv[i], "--provisional") == 0)
            provisional = true;
        else
        {
            std::println(stderr, "usage: diffuser_bench [--csv <path>] [--json <path>] [--provisional]");
            return 2;
        }
    }

    bench::setFtzDaz();

    // Pristine input: 0.5-amplitude sine pair, denormal-free.
    std::vector<float> inL(static_cast<std::size_t>(kTotal));
    std::vector<float> inR(static_cast<std::size_t>(kTotal));
    for (int i = 0; i < kTotal; ++i)
    {
        const auto u = static_cast<std::size_t>(i);
        inL[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / kFs));
        inR[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 330.0 * static_cast<double>(i) / kFs));
    }
    std::vector<float> workL(static_cast<std::size_t>(kTotal));
    std::vector<float> workR(static_cast<std::size_t>(kTotal));

    std::println("=== Chronos diffuser_bench: Diffuser throughput ===");
    std::println("fs={:.0}  total={} samples/rep  reps={} (min)  arch={}",
                kFs, kTotal, kReps, archName());
    std::println("fixed: diffusion={:.2} modRate={:.1} Hz  stereo  sine 0.5 amp\n",
                static_cast<double>(kDiffusion), static_cast<double>(kModRateHz));
    std::println("{:>6} {:>8} {:>5} {:>8} {:>7} | {:>9} {:>9} {:>7}",
                "size", "modDepth", "block", "ramp", "",
                "processBlk", "Ref", "ratio");

    const std::array<float, 3> sizes = {{ 0.0f, 0.5f, 1.0f }};
    const std::array<float, 2> modDepths = {{ 0.0f, 16.0f }};
    const std::array<int, 3> blocks = {{ 64, 256, 512 }};
    const bool ramps[2] = {false, true};

    std::string csv;
    csv += "arch,size,modDepth,block,ramp,path,ns_per_sample\n";
    std::vector<bench::Record> records;
    double grandSink = 0.0;
    bool allFinite = true;

    for (float size: sizes)
        for (float modDepth: modDepths)
            for (int block: blocks)
                for (bool midRamp: ramps)
                {
                    const Cfg c{size, modDepth, block, midRamp};

                    // Two independent instances, so processBlock and processBlockRef do not
                    // share smoother or LFO state. Both are prepared and warmed the same way.
                    Diffuser dFast, dRef;
                    dFast.prepare(kFs);
                    dRef.prepare(kFs);
                    for (Diffuser *d: {&dFast, &dRef})
                    {
                        d->setDiffusion(kDiffusion);
                        d->setModDepthSamples(modDepth);
                        d->setModRateHz(kModRateHz);
                        // settled: set the target to cfg.size one time. The smoother settles
                        // during the warmup. midRamp: the warmup sets the target again every
                        // block, so the start target only sets the ramp direction (high,
                        // through the warmup hi=true).
                        d->setSize(midRamp ? sizeTarget(size, true) : size);
                    }

                    if (!warmup(dFast, inL, inR, workL, workR, c)) allFinite = false;
                    if (!warmup(dRef, inL, inR, workL, workR, c)) allFinite = false;

                    double sink = 0.0;
                    const double nsFast = benchDiffuser<false>(dFast, inL, inR, workL, workR, c, sink);
                    const double nsRef = benchDiffuser<true>(dRef, inL, inR, workL, workR, c, sink);
                    if (!std::isfinite(sink)) allFinite = false;
                    grandSink += sink;

                    const double ratio = nsRef > 0.0 ? nsRef / nsFast : 0.0;
                    std::println("{:6.2} {:8.1} {:5} {:>8} {:>7} | {:9.3} {:9.3} {:6.2}x",
                                static_cast<double>(size), static_cast<double>(modDepth), block,
                                midRamp ? "midRamp" : "settled", "",
                                nsFast, nsRef, ratio);

                    const char *rampStr = midRamp ? "midRamp" : "settled";
                    const std::string cfg = "size=" + std::to_string(size) + ",modDepth=" +
                                            std::to_string(modDepth) + ",block=" + std::to_string(block) +
                                            ",ramp=" + rampStr;
                    for (const auto &pp: {
                             std::pair{"processBlock", nsFast},
                             std::pair{"processBlockRef", nsRef}
                         })
                    {
                        csv += archName();
                        csv += ",";
                        csv += std::to_string(size);
                        csv += ",";
                        csv += std::to_string(modDepth);
                        csv += ",";
                        csv += std::to_string(block);
                        csv += ",";
                        csv += rampStr;
                        csv += ",";
                        csv += pp.first;
                        csv += ",";
                        csv += std::to_string(pp.second);
                        csv += "\n";
                        records.push_back({pp.first, cfg, pp.second});
                    }
                }

    if (!csvPath.empty())
    {
        const std::filesystem::path p(csvPath);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());
        std::ofstream f(csvPath, std::ios::trunc);
        f << csv;
        std::println("\ncsv written to {}", csvPath.c_str());
    }

    if (!jsonPath.empty())
        bench::writeJson(jsonPath, records, provisional);

    std::println("\noutput finite: {}", allFinite ? "yes" : "NO — NaN/Inf DETECTED");
    std::println("(sink={})", grandSink);
    std::println("=== DONE (informational only, no timing gate) ===");
    return allFinite ? 0 : 1;
}

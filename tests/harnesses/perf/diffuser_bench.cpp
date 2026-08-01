// tests/harnesses/perf/diffuser_bench.cpp
// ──────────────────────────────────────────────────────────────────────────
// Throughput benchmark for MarsDSP::Diffusion::Diffuser. This is the
// 8-section Schroeder allpass on the wet path. FeedbackDelay and Diffuser are
// the only hot process paths that do not have a perf harness today. C5 to C7
// change the diffuser (toggle hygiene and base-transport compensation) and
// need a before/after number. This commit gives the C0 baseline that later
// perf claims cite.
//
// Matrix (stereo, 48 kHz, diffusion 0.7, modRate 0.5 Hz):
//   size ∈ {0, 0.5, 1}          sets the effective delay length of each
//                               section.
//   modDepth ∈ {0, 16}          0 = all-fast-path. No section is modulated, so
//                               every settled section takes the 4-wide SIMD
//                               fast path. 16 = sections A and B (indices 2
//                               and 5) run the per-sample FracDelayTap exact
//                               path.
//   block ∈ {64, 256, 512}
//   ramp ∈ {settled, midRamp}   settled: the size smoother is at rest, so
//                               unmodulated sections take the SIMD fast path.
//                               midRamp: the size target is set again every
//                               block, so the smoother never rests. The
//                               `settled` flag (sizeRamp_[0]==sizeRamp_[m-1])
//                               is false every chunk, so all sections take
//                               the per-sample exact path. This is the maximum
//                               cost.
//
// For each config the harness also measures processBlockRef. This is the
// sample-major scalar reference twin (// reference only -- do not optimize,
// do not delete). It is an always-exact baseline. This is the same as the
// SIMD-vs-scalar headline in delay_line_bench. Ref has no fast path. Its cost
// is nearly the same for settled and midRamp. It is the lower limit. The
// processBlock fast path should beat this limit when settled. It should come
// close to this limit when midRamp is true. diffuser_parity gates the
// Diffuser parity itself. This harness only measures throughput.
//
// midRamp forcing: set the size target again every block. Alternate between
// two values that are ±0.3 above and below cfg.size, clamped to [0,1]. The
// size smoother ramps over 50 ms (2400 smp @48k). A block is ≤ 512 smp (~21%
// of the ramp), and the target flips every block. The smoother can never
// reach either target. Because of this, sizeRamp always moves and settled is
// false. The warmup starts with the high target, so the smoother moves from
// the first block. Otherwise size=0 would start at its low target, which
// equals the current value, and would rest for one block.
//
// processBlock works in place. Because of this, each rep reloads the pristine
// input into a work buffer before the clock, and the harness times only the
// processBlock calls. The Diffuser is prepared one time per config, outside
// the timed region. Its 16 rings (352 KB total) would increase the measured
// time if they were allocated again. The state carries across reps. This has
// no effect on the result because the per-sample kernel cost does not depend
// on the data. An untimed warmup (4096 smp, which is more than the 50 ms
// smoother ramps) settles size, coefficient, and depth, and verifies that the
// output is finite before the min-of-5 measurement. kTotal = 1<<19 (524288).
// The diffuser rings are small (~22 KB each). They stay in the L1 and L2
// cache, so this value is sufficient for stable timing.
//
// Timing idiom (the same as delay_line_bench and chain_bench): steady_clock,
// doNotOptimize compiler barriers, min-of-5 reps, and sink accumulators.
//
// Informational only: there is no timing gate. The harness exits non-zero only
// on NaN or Inf.
//
// Build: cmake -S . -B build -DBUILD_TEST_HARNESSES=ON
//        cmake --build build --target diffuser_bench
// Run:   ./build/tests/diffuser_bench [--csv tests/logs/<arch>/diffuser_bench.csv]
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/Diffuser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "bench_util.h"

namespace {

constexpr double kFs        = 48000.0;
constexpr double kPi        = std::numbers::pi_v<double>;
constexpr int    kTotal     = 1 << 19;     // 524288 samples per rep
constexpr int    kReps      = 5;
constexpr int    kWarmup    = 4096;         // > 50 ms (2400 smp): the smoothers settle
constexpr float  kDiffusion = 0.7f;
constexpr float  kModRateHz = 0.5f;

using Clock = std::chrono::steady_clock;
using MarsDSP::Diffusion::Diffuser;

#if defined(__clang__) || defined(__GNUC__)
template <class T>
inline void doNotOptimize(T const& v) noexcept
{
    asm volatile("" : : "r,m"(v) : "memory");
}
#else
template <class T>
inline void doNotOptimize(T const& v) noexcept
{
    volatile T sink = v; (void)sink;   // MSVC fallback
}
#endif

const char* archName() noexcept
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
    int   block;
    bool  midRamp;
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
template <bool UseRef>
double runDiffuser(Diffuser& d, std::vector<float>& wL, std::vector<float>& wR,
                   const Cfg& c)
{
    double acc = 0.0;
    bool hi = false;
    for (int off = 0; off < kTotal; off += c.block)
    {
        if (c.midRamp) { d.setSize(sizeTarget(c.size, hi)); hi = !hi; }
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
template <bool UseRef>
double benchDiffuser(Diffuser& d, const std::vector<float>& inL,
                     const std::vector<float>& inR,
                     std::vector<float>& wL, std::vector<float>& wR,
                     const Cfg& c, double& sinkOut)
{
    double best  = std::numeric_limits<double>::infinity();
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
bool warmup(Diffuser& d, const std::vector<float>& inL,
            const std::vector<float>& inR,
            std::vector<float>& wL, std::vector<float>& wR, const Cfg& c)
{
    std::memcpy(wL.data(), inL.data(), sizeof(float) * static_cast<std::size_t>(kWarmup));
    std::memcpy(wR.data(), inR.data(), sizeof(float) * static_cast<std::size_t>(kWarmup));
    bool finite = true;
    bool hi = true;   // start with the high target so size=0 moves from block 1
    for (int off = 0; off < kWarmup; off += c.block)
    {
        const int m = std::min(c.block, kWarmup - off);
        if (c.midRamp) { d.setSize(sizeTarget(c.size, hi)); hi = !hi; }
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

int main(int argc, char** argv)
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
            std::fprintf(stderr, "usage: diffuser_bench [--csv <path>] [--json <path>] [--provisional]\n");
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

    std::printf("=== Chronos diffuser_bench: Diffuser throughput ===\n");
    std::printf("fs=%.0f  total=%d samples/rep  reps=%d (min)  arch=%s\n",
                kFs, kTotal, kReps, archName());
    std::printf("fixed: diffusion=%.2f modRate=%.1f Hz  stereo  sine 0.5 amp\n\n",
                static_cast<double>(kDiffusion), static_cast<double>(kModRateHz));
    std::printf("%6s %8s %5s %8s %7s | %9s %9s %7s\n",
                "size", "modDepth", "block", "ramp", "",
                "processBlk", "Ref", "ratio");

    const float sizes[3]      = { 0.0f, 0.5f, 1.0f };
    const float modDepths[2]  = { 0.0f, 16.0f };
    const int   blocks[3]     = { 64, 256, 512 };
    const bool  ramps[2]      = { false, true };

    std::string csv;
    csv += "arch,size,modDepth,block,ramp,path,ns_per_sample\n";
    std::vector<bench::Record> records;
    double grandSink = 0.0;
    bool allFinite = true;

    for (float size : sizes)
    for (float modDepth : modDepths)
    for (int block : blocks)
    for (bool midRamp : ramps)
    {
        const Cfg c { size, modDepth, block, midRamp };

        // Two independent instances, so processBlock and processBlockRef do not
        // share smoother or LFO state. Both are prepared and warmed the same way.
        Diffuser dFast, dRef;
        dFast.prepare(kFs);
        dRef.prepare(kFs);
        for (Diffuser* d : { &dFast, &dRef })
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
        if (!warmup(dRef,  inL, inR, workL, workR, c)) allFinite = false;

        double sink = 0.0;
        const double nsFast = benchDiffuser<false>(dFast, inL, inR, workL, workR, c, sink);
        const double nsRef  = benchDiffuser<true>(dRef,  inL, inR, workL, workR, c, sink);
        if (!std::isfinite(sink)) allFinite = false;
        grandSink += sink;

        const double ratio = nsRef > 0.0 ? nsRef / nsFast : 0.0;
        std::printf("%6.2f %8.1f %5d %8s %7s | %9.3f %9.3f %6.2fx\n",
                    static_cast<double>(size), static_cast<double>(modDepth), block,
                    midRamp ? "midRamp" : "settled", "",
                    nsFast, nsRef, ratio);

        const char* rampStr = midRamp ? "midRamp" : "settled";
        const std::string cfg = "size=" + std::to_string(size) + ",modDepth=" +
            std::to_string(modDepth) + ",block=" + std::to_string(block) +
            ",ramp=" + rampStr;
        for (const auto& pp : { std::pair<const char*, double>{ "processBlock", nsFast },
                                std::pair<const char*, double>{ "processBlockRef", nsRef } })
        {
            csv += archName(); csv += ",";
            csv += std::to_string(size); csv += ",";
            csv += std::to_string(modDepth); csv += ",";
            csv += std::to_string(block); csv += ",";
            csv += rampStr; csv += ",";
            csv += pp.first; csv += ",";
            csv += std::to_string(pp.second); csv += "\n";
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
        std::printf("\ncsv written to %s\n", csvPath.c_str());
    }

    if (!jsonPath.empty())
        bench::writeJson(jsonPath, records, provisional);

    std::printf("\noutput finite: %s\n", allFinite ? "yes" : "NO — NaN/Inf DETECTED");
    std::printf("(sink=%f)\n", grandSink);
    std::printf("=== DONE (informational only, no timing gate) ===\n");
    return allFinite ? 0 : 1;
}

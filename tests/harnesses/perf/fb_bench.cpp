/**
 * Throughput benchmark for FeedbackDelay, the feedback-loop path.
 * Matrix: delay {48, 480, 4800, 96000, 235000} x feedback {0.5, 0.95} x
 * satOrder {0, 1, 2} x block {64, 256, 512} x channels {1, 2}.
 * Min-of-5 reps, ns per sample. The ring is prepared outside the timed
 * region. Informational only: exits non-zero on NaN or Inf.
 */

#include "dsp/FeedbackDelay.h"
#include "dsp/SimdDelayLine.h"

#include <algorithm>
#include <bit>
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
    constexpr int kMaxBlock = 512;
    constexpr int kTotal = 1 << 19; // 524288 samples per rep
    constexpr int kReps = 5;
    constexpr int kWarmup = 4096; // untimed: verify finite output, fill the caches

    // Fixed values (not matrix axes).
    constexpr float kDampHz = 6000.0f;
    constexpr float kCrossFeed = 0.37f;
    constexpr float kLoopDriveDb = 12.0f;

    using Clock = std::chrono::steady_clock;
    using MarsDSP::Delays::FeedbackDelay;
    using MarsDSP::Delays::SimdDelayLine;

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
        int delay; // samples
        float feedback;
        int satOrder; // 0,1,2
        int block;
        int channels; // 1=mono, 2=stereo
    };

    // One rep: process kTotal samples in block-sized chunks. The function sinks
    // the wet output so the loop body stays live. The caller prepares fb.
    double runFb(FeedbackDelay &fb, const std::vector<float> &inL,
                 const std::vector<float> &inR, const Cfg &c,
                 std::vector<float> &wetL, std::vector<float> &wetR)
    {
        double acc = 0.0;
        const float *rR = (c.channels > 1) ? inR.data() : nullptr;
        float *wR = (c.channels > 1) ? wetR.data() : nullptr;
        for (int off = 0; off < kTotal; off += c.block)
        {
            fb.process(inL.data() + off, rR, wetL.data(), wR, c.block);
            acc += static_cast<double>(wetL[static_cast<std::size_t>(c.block - 1)]);
            if (c.channels > 1)
                acc += static_cast<double>(wetR[static_cast<std::size_t>(c.block - 1)]);
            doNotOptimize(acc);
        }
        return acc;
    }

    // Min of kReps reps, in ns per input sample. The fb state carries across reps.
    double benchFb(FeedbackDelay &fb, const std::vector<float> &inL,
                   const std::vector<float> &inR, const Cfg &c,
                   std::vector<float> &wetL, std::vector<float> &wetR, double &sinkOut)
    {
        double best = std::numeric_limits<double>::infinity();
        double total = 0.0;
        for (int r = 0; r < kReps; ++r)
        {
            const auto t0 = Clock::now();
            const double a = runFb(fb, inL, inR, c, wetL, wetR);
            const auto t1 = Clock::now();
            total += a;
            best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        sinkOut += total;
        return best / static_cast<double>(kTotal);
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
            std::println(stderr, "usage: fb_bench [--csv <path>] [--json <path>] [--provisional]");
            return 2;
        }
    }

    bench::setFtzDaz();

    // Input: 0.5-amplitude sine pair, denormal-free.
    std::vector<float> inL(static_cast<std::size_t>(kTotal));
    std::vector<float> inR(static_cast<std::size_t>(kTotal));
    for (int i = 0; i < kTotal; ++i)
    {
        const auto u = static_cast<std::size_t>(i);
        inL[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / kFs));
        inR[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 330.0 * static_cast<double>(i) / kFs));
    }

    // This matches ChronosEngine::prepare: 5000 ms max delay,
    // getMaxDelaySamples() (the contractual max, 240000) → FeedbackDelay's
    // maxDelaySamples. For the matrix blocks {64,256,512} this gives a
    // 262144-float ring = 1.0 MB/channel (see the header note).
    SimdDelayLine dl;
    dl.prepare(kFs, kMaxBlock, 5000.0f);
    const int fbMaxDelay = dl.getMaxDelaySamples();
    // The fb ring capacity = bit_ceil(maxDelaySamples + maxBlockSize + kTail +
    // 8). Compute it here for the headline (all matrix blocks round to the
    // same value at 48 kHz).
    const unsigned fbMinCap = static_cast<unsigned>(
        fbMaxDelay + kMaxBlock + MarsDSP::Delays::Pow2RingBuffer::kTail + 8);
    const int fbRingCap = static_cast<int>(std::bit_ceil(fbMinCap));

    const float loopDriveLin = std::pow(10.0f, kLoopDriveDb / 20.0f);

    std::println("=== Chronos fb_bench: FeedbackDelay throughput ===");
    std::println("fs={:.0}  total={} samples/rep  reps={} (min)  arch={}",
                kFs, kTotal, kReps, archName());
    std::println("fb maxDelaySamples={} (contractual max) -> ring capacity {} floats = {:.2} MB/channel",
                fbMaxDelay, fbRingCap, static_cast<double>(fbRingCap) * 4.0 / 1048576.0);
    std::println("fixed: dampHz={:.0} crossFeed={:.2} loopDrive={:.0}dB(lin {:.3})  sine 0.5 amp\n",
                static_cast<double>(kDampHz), static_cast<double>(kCrossFeed),
                static_cast<double>(kLoopDriveDb), static_cast<double>(loopDriveLin));

    std::println("{:>7} {:>6} {:>7} {:>5} {:>3} | {:>9}",
                "delay", "fb", "sat", "block", "ch", "ns/sample");

    const std::array<int, 5> delays = {{ 48, 480, 4800, 96000, 235000 }};
    const std::array<float, 2> fbs = {{ 0.5f, 0.95f }};
    const std::array<int, 3> sats = {{ 0, 1, 2 }};
    const std::array<int, 3> blocks = {{ 64, 256, 512 }};
    const std::array<int, 2> chans = {{ 1, 2 }};

    std::string csv;
    csv += "arch,delay,feedback,satOrder,block,channels,ns_per_sample\n";
    std::vector<bench::Record> records;
    double grandSink = 0.0;
    bool allFinite = true;

    for (int delay: delays)
        for (float fbk: fbs)
            for (int sat: sats)
                for (int block: blocks)
                    for (int ch: chans)
                    {
                        const Cfg c{delay, fbk, sat, block, ch};

                        FeedbackDelay fb;
                        fb.prepare(kFs, block, fbMaxDelay);
                        FeedbackDelay::Params p;
                        p.delaySamplesL = static_cast<float>(delay);
                        p.delaySamplesR = static_cast<float>(delay);
                        p.feedback = fbk;
                        p.dampHz = kDampHz;
                        p.crossFeed = kCrossFeed;
                        p.loopDrive = loopDriveLin;
                        p.satOrder = sat;
                        fb.resetParams(p); // snap the smoothers to their targets (settled steady state)

                        std::vector<float> wetL(static_cast<std::size_t>(block));
                        std::vector<float> wetR(static_cast<std::size_t>(block));

                        // Untimed warmup: verify that the output is finite. The pass also fills
                        // the caches and trains the branch predictor.
                        {
                            const float *rR = (ch > 1) ? inR.data() : nullptr;
                            float *wR = (ch > 1) ? wetR.data() : nullptr;
                            int done = 0;
                            bool finite = true;
                            while (done < kWarmup)
                            {
                                const int n = std::min(block, kWarmup - done);
                                fb.process(inL.data() + done, (ch > 1) ? inR.data() + done : nullptr,
                                           wetL.data(), wR, n);
                                for (int s = 0; s < n; ++s)
                                {
                                    if (!std::isfinite(wetL[static_cast<std::size_t>(s)])) finite = false;
                                    if (ch > 1 && !std::isfinite(wetR[static_cast<std::size_t>(s)])) finite = false;
                                }
                                done += n;
                            }
                            if (!finite) allFinite = false;
                        }
                        // The smoothers stay settled because resetParams snapped them. The ring
                        // state carries into the timed reps. This has no effect on the
                        // throughput because the kernel cost does not depend on the data. Do
                        // not re-prepare or re-reset inside the timed region.

                        double sink = 0.0;
                        const double ns = benchFb(fb, inL, inR, c, wetL, wetR, sink);
                        if (!std::isfinite(sink)) allFinite = false;
                        grandSink += sink;

                        std::println("{:7} {:6.2} {:7} {:5} {:3} | {:9.3}",
                                    delay, static_cast<double>(fbk), sat, block, ch, ns);

                        csv += archName();
                        csv += ",";
                        csv += std::to_string(delay);
                        csv += ",";
                        csv += std::to_string(fbk);
                        csv += ",";
                        csv += std::to_string(sat);
                        csv += ",";
                        csv += std::to_string(block);
                        csv += ",";
                        csv += std::to_string(ch);
                        csv += ",";
                        csv += std::to_string(ns);
                        csv += "\n";

                        const std::string cfg = "delay=" + std::to_string(delay) + ",fb=" +
                                                std::to_string(fbk) + ",sat=" + std::to_string(sat) + ",block=" +
                                                std::to_string(block) + ",ch=" + std::to_string(ch);
                        records.push_back({"FeedbackDelay", cfg, ns});
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

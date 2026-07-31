// tests/harnesses/perf/fb_bench.cpp
// ──────────────────────────────────────────────────────────────────────────
// Throughput benchmark for MarsDSP::Delays::FeedbackDelay. This is the
// feedback-loop path that ChronosEngine::process uses when feedback > 0:
//   delay tap → damp one-pole → DC blocker → cross-mix → drive·tanh(ADAA) →
//   makeup → ring write.
// FeedbackDelay and Diffuser are the only hot process paths that do not have
// a perf harness today. C1 halves the rings. C3 adds chunked block
// processing. Both change this path and need a before/after number.
// This commit gives the C0 baseline that later perf claims cite.
//
// Matrix (stereo and mono, 48 kHz):
//   delay ∈ {48, 480, 4800, 96000, 235000}  short, typical, and long.
//   235000 tests the tap reads in the cold region of the 1 MB ring. The fb
//     ring is sized from the SimdDelayLine contractual max (see the prepare note).
//   feedback ∈ {0.5, 0.95}
//   satOrder ∈ {0, 1, 2}                     hard clamp, ADAA1, ADAA2
//   block ∈ {64, 256, 512}
//   channels ∈ {1, 2}
// Fixed values (not matrix axes): dampHz 6000, crossFeed 0.37,
//   loopDrive 12 dB linear (≈3.98). This value drives the tanh ceiling so
//   ADAA1 and ADAA2 do real work. The input is a sine at 0.5 amplitude. It has
//   no denormals.
//
// Ring sizing — this matches ChronosEngine::prepare (post-C1):
//   ChronosEngine prepares the 5000 ms SimdDelayLine and passes
//   delayLine_.getMaxDelaySamples() to fbDelay_.prepare(...). This is the
//   contractual max delay, not the pow2 capacity. This harness does the same.
//   It builds a SimdDelayLine, calls prepare(48k, 512, 5000ms), and gets
//   maxDelaySamples 240000. It passes this value as FeedbackDelay's
//   maxDelaySamples. FeedbackDelay adds (maxDelaySamples + maxBlockSize +
//   kTail + 8) and rounds the sum up to the next power of two. The result is
//   262144 = 1 MB/channel. Before C1, the engine passed getCapacity() =
//   262144, which double-rounded to 524288 = 2 MB/channel. C1 halves the fb
//   rings: 6 MB total becomes 4 MB total.
//
// Timing idiom (the same as delay_line_bench and chain_bench): steady_clock,
// doNotOptimize compiler barriers, min-of-5 reps, and sink accumulators that
// keep the loop body live. The FeedbackDelay is prepared one time per config,
// outside the timed region. The allocation and the memset of its 2 MB ring
// would increase the measured time. The state carries across reps. This has
// no effect on the result because the per-sample kernel cost does not depend
// on the data: the operation sequence per sample is fixed, and the ADAA
// fallback branches depend on the data but not on the time. A short untimed
// pass runs first. It verifies that the output is finite. It also runs data
// through the path before the timed measurement. This fills the caches and
// trains the branch predictor. kTotal = 1<<19 (524288, ~11 s @48k) is greater
// than the 235000 long delay, so the tap reads real recirculated data. The
// value is also small enough that the 180-cell matrix runs quickly.
//
// Informational only: there is no timing gate, because a timing gate fires on
// machine noise. The harness exits non-zero only when the output has NaN or
// Inf. This means that the chain was assembled incorrectly.
//
// Build: cmake -S . -B build -DBUILD_TEST_HARNESSES=ON
//        cmake --build build --target fb_bench
// Run:   ./build/tests/fb_bench [--csv tests/logs/<arch>/fb_bench.csv]
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/FeedbackDelay.h"
#include "dsp/SimdDelayLine.h"

#include <algorithm>
#include <bit>
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

namespace {

constexpr double kFs          = 48000.0;
constexpr double kPi          = std::numbers::pi_v<double>;
constexpr int    kMaxBlock    = 512;
constexpr int    kTotal       = 1 << 19;   // 524288 samples per rep
constexpr int    kReps        = 5;
constexpr int    kWarmup      = 4096;       // untimed: verify finite output, fill the caches

// Fixed values (not matrix axes).
constexpr float  kDampHz      = 6000.0f;
constexpr float  kCrossFeed   = 0.37f;
constexpr float  kLoopDriveDb = 12.0f;

using Clock = std::chrono::steady_clock;
using MarsDSP::Delays::FeedbackDelay;
using MarsDSP::Delays::SimdDelayLine;

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
    int   delay;     // samples
    float feedback;
    int   satOrder;  // 0,1,2
    int   block;
    int   channels;  // 1=mono, 2=stereo
};

// One rep: process kTotal samples in block-sized chunks. The function sinks
// the wet output so the loop body stays live. The caller prepares fb.
double runFb(FeedbackDelay& fb, const std::vector<float>& inL,
             const std::vector<float>& inR, const Cfg& c,
             std::vector<float>& wetL, std::vector<float>& wetR)
{
    double acc = 0.0;
    const float* rR = (c.channels > 1) ? inR.data() : nullptr;
    float*       wR = (c.channels > 1) ? wetR.data() : nullptr;
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
double benchFb(FeedbackDelay& fb, const std::vector<float>& inL,
               const std::vector<float>& inR, const Cfg& c,
               std::vector<float>& wetL, std::vector<float>& wetR, double& sinkOut)
{
    double best  = std::numeric_limits<double>::infinity();
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

int main(int argc, char** argv)
{
    std::string csvPath;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            csvPath = argv[++i];
        else
        {
            std::fprintf(stderr, "usage: fb_bench [--csv <path>]\n");
            return 2;
        }
    }

    // Input: 0.5-amplitude sine pair, denormal-free.
    std::vector<float> inL(static_cast<std::size_t>(kTotal));
    std::vector<float> inR(static_cast<std::size_t>(kTotal));
    for (int i = 0; i < kTotal; ++i)
    {
        const auto u = static_cast<std::size_t>(i);
        inL[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / kFs));
        inR[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 330.0 * static_cast<double>(i) / kFs));
    }

    // This matches ChronosEngine::prepare after C1: 5000 ms SimdDelayLine →
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

    std::printf("=== Chronos fb_bench: FeedbackDelay throughput ===\n");
    std::printf("fs=%.0f  total=%d samples/rep  reps=%d (min)  arch=%s\n",
                kFs, kTotal, kReps, archName());
    std::printf("fb maxDelaySamples=%d (contractual max) -> ring capacity %d floats = %.2f MB/channel\n",
                fbMaxDelay, fbRingCap, static_cast<double>(fbRingCap) * 4.0 / 1048576.0);
    std::printf("fixed: dampHz=%.0f crossFeed=%.2f loopDrive=%.0fdB(lin %.3f)  sine 0.5 amp\n\n",
                static_cast<double>(kDampHz), static_cast<double>(kCrossFeed),
                static_cast<double>(kLoopDriveDb), static_cast<double>(loopDriveLin));

    std::printf("%7s %6s %7s %5s %3s | %9s\n",
                "delay", "fb", "sat", "block", "ch", "ns/sample");

    const int   delays[5] = { 48, 480, 4800, 96000, 235000 };
    const float fbs[2]    = { 0.5f, 0.95f };
    const int   sats[3]   = { 0, 1, 2 };
    const int   blocks[3] = { 64, 256, 512 };
    const int   chans[2]  = { 1, 2 };

    std::string csv;
    csv += "arch,delay,feedback,satOrder,block,channels,ns_per_sample\n";
    double grandSink = 0.0;
    bool allFinite = true;

    for (int delay : delays)
    for (float fbk : fbs)
    for (int sat : sats)
    for (int block : blocks)
    for (int ch : chans)
    {
        const Cfg c { delay, fbk, sat, block, ch };

        FeedbackDelay fb;
        fb.prepare(kFs, block, fbMaxDelay);
        FeedbackDelay::Params p;
        p.delaySamples = static_cast<float>(delay);
        p.feedback     = fbk;
        p.dampHz       = kDampHz;
        p.crossFeed    = kCrossFeed;
        p.loopDrive    = loopDriveLin;
        p.satOrder     = sat;
        fb.resetParams(p);   // snap the smoothers to their targets (settled steady state)

        std::vector<float> wetL(static_cast<std::size_t>(block));
        std::vector<float> wetR(static_cast<std::size_t>(block));

        // Untimed warmup: verify that the output is finite. The pass also fills
        // the caches and trains the branch predictor.
        {
            const float* rR = (ch > 1) ? inR.data() : nullptr;
            float*       wR = (ch > 1) ? wetR.data() : nullptr;
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

        std::printf("%7d %6.2f %7d %5d %3d | %9.3f\n",
                    delay, static_cast<double>(fbk), sat, block, ch, ns);

        csv += archName(); csv += ",";
        csv += std::to_string(delay); csv += ",";
        csv += std::to_string(fbk); csv += ",";
        csv += std::to_string(sat); csv += ",";
        csv += std::to_string(block); csv += ",";
        csv += std::to_string(ch); csv += ",";
        csv += std::to_string(ns); csv += "\n";
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

    std::printf("\noutput finite: %s\n", allFinite ? "yes" : "NO — NaN/Inf DETECTED");
    std::printf("(sink=%f)\n", grandSink);
    std::printf("=== DONE (informational only, no timing gate) ===\n");
    return allFinite ? 0 : 1;
}

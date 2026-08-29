// tests/harnesses/dsp/glide_rate_check.cpp
//
// Glide rate harness. Steps the delay from 100 ms to 4000 ms and asserts the
// tap velocity never exceeds 4.0 samples per sample and the move completes
// within 21 s. Drives FeedbackDelay directly, SharedCode only, no JUCE.

#include "dsp/FeedbackDelay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...)                                                         \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs = 48000.0;
constexpr int    kFsInt = 48000;
constexpr int    kBlock = 512;
constexpr int    kMaxDelay = 262144;   // matches fb_parity / the engine fb ring capacity
constexpr float  kMaxStep = 4.0f;
constexpr float  kVelocityTol = 1e-3f;  // float rounding on step = dist / round(dist / maxStep)
constexpr float  kCompleteTol = 0.5f;   // samples: the tap within half a sample of the target
constexpr double kMaxCompleteSeconds = 21.0;

using MarsDSP::Delays::FeedbackDelay;

// Step the delay from startMs to endMs and verify the glide rate limit.
void runGlide(float startMs, float endMs)
{
    FeedbackDelay fb;
    fb.prepare(kFs, kBlock, kMaxDelay);

    FeedbackDelay::Params p;
    const float dlyStart = static_cast<float>(startMs * 0.001f * static_cast<float>(kFsInt));
    p.delaySamplesL  = dlyStart;
    p.delaySamplesR  = dlyStart;
    p.feedback       = 0.0f;   // plain delay, no recursion
    p.dampHz         = 6000.0f;
    p.crossFeed      = 0.0f;
    p.loopDrive      = 1.0f;
    p.satOrder       = 0;
    p.enableDiffuser = false;
    p.diffusion      = 0.7f;
    p.diffuserSize   = 0.5f;
    p.diffModDepth   = 0.0f;
    p.diffModRateHz  = 0.5f;
    fb.resetParams(p);

    const float target = static_cast<float>(endMs * 0.001f * static_cast<float>(kFsInt));
    p.delaySamplesL = target;
    p.delaySamplesR = target;

    std::vector<float> inL(static_cast<std::size_t>(kBlock), 0.0f);
    std::vector<float> inR(static_cast<std::size_t>(kBlock), 0.0f);
    std::vector<float> outL(static_cast<std::size_t>(kBlock));
    std::vector<float> outR(static_cast<std::size_t>(kBlock));

    float maxVelocity = 0.0f;
    int maxVelocityBlock = -1;
    bool completed = false;
    double completeSeconds = 0.0;

    const int maxSamples = static_cast<int>(kMaxCompleteSeconds * kFs) + kBlock;
    int pos = 0;
    while (pos < maxSamples)
    {
        fb.setParams(p);
        const float before = fb.currentDelaySamples();
        fb.process(inL.data(), inR.data(), outL.data(), outR.data(), kBlock);
        const float after = fb.currentDelaySamples();

        const float velocity = std::abs(after - before) / static_cast<float>(kBlock);
        if (velocity > maxVelocity) { maxVelocity = velocity; maxVelocityBlock = pos; }
        if (velocity > kMaxStep + kVelocityTol)
            FAIL("velocity {:.6} samples/sample at pos {} exceeds {:.3}", velocity, pos, kMaxStep);

        pos += kBlock;
        if (std::abs(after - target) <= kCompleteTol)
        {
            completed = true;
            completeSeconds = static_cast<double>(pos) / kFs;
            break;
        }
    }

    CHECK(completed);
    if (completeSeconds > kMaxCompleteSeconds)
        FAIL("completed at {:.3} s, exceeds {:.1} s", completeSeconds, kMaxCompleteSeconds);

    std::println("  {:6.1} ms -> {:6.1} ms: max velocity {:.5} samples/sample at block {}, completed at {:.3} s: PASS",
                static_cast<double>(startMs), static_cast<double>(endMs),
                static_cast<double>(maxVelocity), maxVelocityBlock / kBlock,
                completeSeconds);
}

} // namespace

int main()
{
    std::println("=== glide_rate_check ===");
    std::println("fs={} block={} max_step={:.1} samples/sample complete_within={:.0}s",
                kFsInt, kBlock, kMaxStep, kMaxCompleteSeconds);

    g_section = "100ms->4000ms";
    runGlide(100.0f, 4000.0f);

    std::println("=== glide_rate_check OK ===");
    return 0;
}

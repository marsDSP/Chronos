// tests/harnesses/dsp/delay_mod_check.cpp
//
// Delay modulation harness. Verifies the OU delay modulation depth
// calibration. A 1 kHz tone through the modulated delay shows an RMS pitch
// deviation equal to the depth in cents. A depth of zero gives zero
// deviation. The measurement reads the instantaneous frequency from the
// positive-going zero crossings of the output, with linear interpolation.
//
// Conventions (matching latency_null_check / chain_parity): plain main(),
// exit code, printf, always-live CHECK/FAIL. Links SharedCode only; no JUCE.

#include "dsp/FeedbackDelay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <vector>

namespace {

constexpr double kFs       = 48000.0;
constexpr int    kBlock    = 256;
constexpr int    kMaxDelay = 262144;   // matches the engine's fb ring capacity
constexpr double kToneHz   = 1000.0;
constexpr int    kRunSec   = 60;
constexpr int    kSkipSec  = 2;        // smoother ramp + delay fill-in

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

using MarsDSP::Delays::FeedbackDelay;

// Render a 1 kHz tone through the modulated delay and measure the RMS
// pitch deviation in cents. Also returns the largest |cents| sample.
double measureRmsCents(float depthCents, float rateHz, double& maxAbsCents)
{
    FeedbackDelay fb;
    fb.prepare(kFs, kBlock, kMaxDelay);

    FeedbackDelay::Params p;
    p.delaySamplesL  = 24000.0f;   // 500 ms; the calibration is delay-free
    p.delaySamplesR  = 24000.0f;
    p.feedback       = 0.0f;
    p.dampHz         = 20000.0f;
    p.loopCutHz      = 20.0f;
    p.crossFeed      = 0.0f;
    p.loopDrive      = 1.0f;
    p.satOrder       = 0;
    p.enableDiffuser = false;
    p.delayModDepth  = depthCents;
    p.delayModRateHz = rateHz;
    fb.resetParams(p);

    const int total = static_cast<int>(kRunSec * kFs);
    std::vector<float> in(static_cast<std::size_t>(total));
    std::vector<float> out(static_cast<std::size_t>(total), 0.0f);
    for (int i = 0; i < total; ++i)
        in[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * kToneHz
                                               * static_cast<double>(i) / kFs));

    for (int pos = 0; pos < total; pos += kBlock)
    {
        const int n = std::min(kBlock, total - pos);
        fb.setParams(p);
        fb.process(in.data() + pos, nullptr, out.data() + pos, nullptr, n);
    }

    const int skip = static_cast<int>(kSkipSec * kFs);
    double prev = out[static_cast<std::size_t>(skip - 1)];
    double lastCross = -1.0;
    double sumSq = 0.0;
    long count = 0;
    maxAbsCents = 0.0;

    for (int i = skip; i < total; ++i)
    {
        const double cur = out[static_cast<std::size_t>(i)];
        if (prev < 0.0 && cur >= 0.0)
        {
            const double frac = prev / (prev - cur);
            const double crossPos = static_cast<double>(i - 1) + frac;
            if (lastCross > 0.0)
            {
                const double period = crossPos - lastCross;
                // Glitch guard: a period far from 48 samples is not the tone.
                CHECK(period > 40.0 && period < 56.0);
                const double fInst = kFs / period;
                const double cents = 1200.0 * std::log2(fInst / kToneHz);
                sumSq += cents * cents;
                maxAbsCents = std::max(maxAbsCents, std::fabs(cents));
                ++count;
            }
            lastCross = crossPos;
        }
        prev = cur;
    }

    CHECK(count > 1000);
    return std::sqrt(sumSq / static_cast<double>(count));
}

} // namespace

int main()
{
    std::println("=== Chronos delay_mod_check (OU depth calibration) ===");
    std::println("fs={:.0}  tone={:.0} Hz  run={} s  skip={} s\n",
                kFs, kToneHz, kRunSec, kSkipSec);

    g_section = "depth-50-cent-calibration";
    double maxAbs = 0.0;
    const double rms50 = measureRmsCents(50.0f, 1.0f, maxAbs);
    std::println("depth 50 cents @ 1 Hz: RMS deviation {:.3} cents, max {:.3} cents",
                rms50, maxAbs);
    CHECK(rms50 >= 47.0 && rms50 <= 53.0);

    g_section = "depth-zero";
    double maxAbs0 = 0.0;
    const double rms0 = measureRmsCents(0.0f, 1.0f, maxAbs0);
    std::println("depth 0 cents:         RMS deviation {:.6} cents, max {:.6} cents",
                rms0, maxAbs0);
    CHECK(rms0 < 0.01);
    CHECK(maxAbs0 < 0.05);

    std::println("\n=== ALL PROPERTIES HELD ===");
    return 0;
}

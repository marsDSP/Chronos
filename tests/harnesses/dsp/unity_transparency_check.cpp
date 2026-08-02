// tests/harnesses/dsp/unity_transparency_check.cpp
//
// Unity transparency harness. At mix 0, bits 32, gain 0 dB, drive 0, bypass
// off, the output equals the input delayed by kBudget exactly. The quantiser
// bypass at full resolution makes this bit-exact. Drives ChronosEngine
// directly, SharedCode only, no JUCE.

#include "dsp/ChronosEngine.h"
#include "dsp/align/SaturatorAlign.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...)                                                         \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs = 48000.0;
constexpr int    kFsInt = 48000;
constexpr int    kBlock = 256;
constexpr int    kChannels = 2;
constexpr int    kBudget = MarsDSP::Align::SaturatorAlign::kBudget;
constexpr int    kRenderSamples = 48000;

} // namespace

int main()
{
    std::printf("=== unity_transparency_check ===\n");
    std::printf("fs=%d block=%d kBudget=%d\n", kFsInt, kBlock, kBudget);

    g_section = "prepare";
    MarsDSP::ChronosEngine engine;
    engine.prepare(kFs, kBlock, kChannels);
    engine.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    engine.setBypass(false);

    MarsDSP::ChronosEngine::Params p{};
    p.delaySamples = 500.0f * 0.001f * static_cast<float>(kFsInt);
    p.driveLin = 1.0f;       // drive 0 dB
    p.mix = 0.0f;            // dry only
    p.gainLin = 1.0f;        // gain 0 dB
    p.hpfHz = 20.0f;
    p.lpfHz = 20000.0f;
    p.bits = 32;             // quantiser bypass
    p.adaaOrder = 0;
    p.feedback = 0.0f;
    p.dampHz = 6000.0f;
    p.crossFeed = 0.0f;
    p.loopDrive = 1.0f;
    p.loopSatOrder = 0;
    p.diffusion = 0.0f;
    p.diffuserSize = 0.5f;
    p.diffModDepth = 0.0f;
    p.diffModRateHz = 0.5f;
    p.enableDiffuser = false;
    engine.reset();
    engine.resetParams(p);

    // Build a deterministic input: a mix of sine and an impulse.
    std::vector<float> inL(static_cast<std::size_t>(kRenderSamples));
    std::vector<float> inR(static_cast<std::size_t>(kRenderSamples));
    for (int i = 0; i < kRenderSamples; ++i)
    {
        const double t = static_cast<double>(i) / kFs;
        const float v = 0.3f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 220.0 * t));
        inL[static_cast<std::size_t>(i)] = v;
        inR[static_cast<std::size_t>(i)] = -v;
    }
    inL[100] = 0.9f;
    inR[100] = -0.9f;

    std::vector<float> outL = inL;
    std::vector<float> outR = inR;

    g_section = "render";
    for (int pos = 0; pos < kRenderSamples; pos += kBlock)
    {
        const int n = std::min(kBlock, kRenderSamples - pos);
        engine.setParams(p);
        float* io[2] = { outL.data() + pos, outR.data() + pos };
        engine.process(io, kChannels, n);
    }

    g_section = "transparency";
    // The output must equal the input delayed by kBudget, sample for sample.
    int mismatches = 0;
    for (int i = 0; i < kRenderSamples; ++i)
    {
        const float expectedL = (i >= kBudget)
            ? inL[static_cast<std::size_t>(i - kBudget)]
            : 0.0f;
        const float expectedR = (i >= kBudget)
            ? inR[static_cast<std::size_t>(i - kBudget)]
            : 0.0f;

        const float gotL = outL[static_cast<std::size_t>(i)];
        const float gotR = outR[static_cast<std::size_t>(i)];

        if (gotL != expectedL || gotR != expectedR)
        {
            ++mismatches;
            if (mismatches <= 10)
                std::printf("  mismatch at %d: L got=%g exp=%g  R got=%g exp=%g\n",
                            i, (double)gotL, (double)expectedL,
                               (double)gotR, (double)expectedR);
        }
    }

    if (mismatches > 0)
        FAIL("%d sample mismatches (expected exact equality, delayed by kBudget=%d)",
             mismatches, kBudget);

    std::printf("exact transparency: %d samples, output == input delayed by %d: PASS\n",
                kRenderSamples, kBudget);
    std::printf("=== unity_transparency_check OK ===\n");
    return 0;
}

// tests/harnesses/rt/rt_stack_check.cpp
//
// Real-time stack guard. Paint a 256 kB region of the stack with 0xA5 in a
// child frame that returns, so the stack space is free again. Then call
// ChronosEngine::process in another child frame that reuses the same stack
// region. After 1000 blocks, scan the region from the bottom up for the first
// byte that is not 0xA5. The distance from the top to that byte is the stack
// high-water mark. Fail above 16 kB.
//
// The configs cycle through the chunked feedback path with the diffuser on,
// the per-sample fallback at a tiny delay, and the modulated exact path. This
// reaches every fallback in the audio path.
//
// SharedCode only, no JUCE.

#include "dsp/ChronosEngine.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <print>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// Portable compiler barrier and noinline. The GCC/clang asm clobber becomes
// _ReadWriteBarrier on MSVC. The noinline keeps the canary frame and the
// process frame distinct so process reuses the canary's freed stack space.
#if defined(__clang__) || defined(__GNUC__)
#define CHRONOS_COMPILER_BARRIER() asm volatile("" ::: "memory")
#define CHRONOS_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define CHRONOS_COMPILER_BARRIER() _ReadWriteBarrier()
#define CHRONOS_NOINLINE __declspec(noinline)
#else
#define CHRONOS_COMPILER_BARRIER() (void)0
#define CHRONOS_NOINLINE
#endif

namespace
{
    constexpr double kFs = 48000.0;
    constexpr int kChannels = 2;
    constexpr int kBlock = 512;
    constexpr int kBlocks = 1000;
    constexpr int kCanaryBytes = 262144; // 256 kB
    constexpr int kGateBytes = 16 * 1024; // 16 kB

    // Canary boundaries. Set by paintCanaryFrame_, read by the scan.
    std::uintptr_t g_canaryLow = 0;
    std::uintptr_t g_canaryHigh = 0;

    // Paint the canary in a child frame. The frame holds a 256 kB array. When the
    // function returns, the stack pointer moves back up and the array space is
    // free for the next child frame to reuse.
    CHRONOS_NOINLINE
    void paintCanaryFrame_() noexcept
    {
        unsigned char canary[static_cast<std::size_t>(kCanaryBytes)];
        std::memset(canary, 0xA5, static_cast<std::size_t>(kCanaryBytes));
        g_canaryLow = reinterpret_cast<std::uintptr_t>(&canary[0]);
        g_canaryHigh = reinterpret_cast<std::uintptr_t>(&canary[0]) + static_cast<std::uintptr_t>(kCanaryBytes);
        CHRONOS_COMPILER_BARRIER();
    }

    // Call process in a child frame. This frame reuses the canary stack space, so
    // the process call chain writes over the 0xA5 pattern from the top down.
    CHRONOS_NOINLINE
    void processFrame_(MarsDSP::ChronosEngine &eng, float *ioL, float *ioR, int n) noexcept
    {
        std::array<float*, 2> io{ioL, ioR};
        eng.process(io.data(), kChannels, n);
    }

    // Call prepare in a child frame so the canary captures the prepare stack.
    // The section length prime scan moved off the stack in S12, so this stays
    // well under the gate.
    CHRONOS_NOINLINE
    void prepareFrame_(MarsDSP::ChronosEngine &eng, double sr) noexcept
    {
        eng.prepare(sr, kBlock, kChannels);
    }

    struct RtConfig
    {
        float delaySamples;
        float feedback;
        float dampHz;
        float crossFeed;
        float loopDrive;
        int loopSatOrder;
        float driveLin;
        int adaaOrder;
        float mix;
        bool enableDiffuser;
        float diffusion;
        float diffuserSize;
        float diffModDepth;
        float diffModRateHz;
    };

    MarsDSP::ChronosEngine::Params toParams(const RtConfig &c) noexcept
    {
        MarsDSP::ChronosEngine::Params p{};
        p.delaySamples = c.delaySamples;
        p.driveLin = c.driveLin;
        p.mix = c.mix;
        p.gainLin = 1.0f;
        p.hpfHz = 20.0f;
        p.lpfHz = 20000.0f;
        p.bits = 32;
        p.adaaOrder = c.adaaOrder;
        p.feedback = c.feedback;
        p.dampHz = c.dampHz;
        p.crossFeed = c.crossFeed;
        p.loopDrive = c.loopDrive;
        p.loopSatOrder = c.loopSatOrder;
        p.diffusion = c.diffusion;
        p.diffuserSize = c.diffuserSize;
        p.diffModDepth = c.diffModDepth;
        p.diffModRateHz = c.diffModRateHz;
        p.enableDiffuser = c.enableDiffuser;
        return p;
    }

    // Four configs that hit the chunked diffuser-on path, the per-sample fallback
    // at a tiny delay, the modulated exact path, and a high-drive saturation path.
    const std::array<RtConfig, 4> &rtConfigs()
    {
        static const std::array<RtConfig, 4> kConfigs{
            {
                {
                    24000.0f, 0.70f, 6000.0f, 0.0f, 4.0f, 2, std::pow(10.0f, 12.0f / 20.0f), 2, 100.0f, true, 0.7f,
                    0.5f, 16.0f, 1.0f
                },
                {6.0f, 0.50f, 6000.0f, 0.0f, 1.0f, 0, 1.0f, 0, 100.0f, false, 0.0f, 0.5f, 0.0f, 0.0f},
                {
                    2400.0f, 0.90f, 4000.0f, 0.3f, 8.0f, 2, std::pow(10.0f, 24.0f / 20.0f), 2, 100.0f, true, 0.8f, 0.3f,
                    32.0f, 1.5f
                },
                {
                    2400.0f, 0.60f, 8000.0f, 0.0f, 2.0f, 1, std::pow(10.0f, 18.0f / 20.0f), 1, 80.0f, true, 0.6f, 0.1f,
                    24.0f, 2.0f
                },
            }
        };
        return kConfigs;
    }
} // namespace

int main()
{
    std::println("=== Chronos rt_stack_check ===");
    std::println("fs={:.0} stereo  block={}  blocks={}  canary={} kB  gate={} kB\n",
                kFs, kBlock, kBlocks, kCanaryBytes / 1024, kGateBytes / 1024);

    // Prepare the engine and buffers.
    std::vector<float> ioL(static_cast<std::size_t>(kBlock), 0.0f);
    std::vector<float> ioR(static_cast<std::size_t>(kBlock), 0.0f);
    for (int i = 0; i < kBlock; ++i)
    {
        ioL[static_cast<std::size_t>(i)] = 0.5f * static_cast<float>(std::sin(
                                               2.0 * 3.14159265358979323846 * 440.0 * i / kFs));
        ioR[static_cast<std::size_t>(i)] = 0.5f * static_cast<float>(std::sin(
                                               2.0 * 3.14159265358979323846 * 330.0 * i / kFs));
    }

    MarsDSP::ChronosEngine engine;
    engine.prepare(kFs, kBlock, kChannels);
    engine.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    engine.setBypass(false);
    engine.resetParams(toParams(rtConfigs()[0]));

    // Paint the canary once. Each process call overwrites the top of the
    // canary down to its own depth. A deeper call leaves its mark lower; a
    // shallower call after it does not restore the lower bytes. The final
    // scan therefore reports the maximum depth across all calls.
    paintCanaryFrame_();

    for (int i = 0; i < kBlocks; ++i)
    {
        const RtConfig &c = rtConfigs()[static_cast<std::size_t>((i / 250) % 4)];
        MarsDSP::ChronosEngine::Params p = toParams(c);
        // Sweep the diffuser size within each config so the exact path runs.
        const float sizeSweep = 0.2f + 0.6f * static_cast<float>(0.5 + 0.5 * std::sin(
                                                                     2.0 * 3.14159265358979323846 * i / 97.0));
        p.diffuserSize = (c.enableDiffuser) ? sizeSweep : c.diffuserSize;
        engine.setParams(p);

        // Re-prepare at a different rate every 250 blocks so the canary
        // captures the prepare path (the prime scan and the arena setup).
        if (i % 250 == 0 && i > 0)
            prepareFrame_(engine, kFs);

        processFrame_(engine, ioL.data(), ioR.data(), kBlock);
    }

    // Scan from the bottom (low address) up for the first byte that is not
    // 0xA5. No function calls run here, so the scan does not clobber the
    // canary. The high-water mark is the distance from the top to that byte.
    const auto *base = reinterpret_cast<const volatile unsigned char *>(g_canaryLow);
    std::size_t preserved = 0;
    for (; preserved < static_cast<std::size_t>(kCanaryBytes); ++preserved)
    {
        if (base[preserved] != 0xA5)
            break;
    }
    const std::size_t highWater = static_cast<std::size_t>(kCanaryBytes) - preserved;

    std::println("stack high-water mark: {} bytes ({:.2} kB)", highWater, static_cast<double>(highWater) / 1024.0);
    std::println("gate: {} kB", kGateBytes / 1024);

    if (highWater > static_cast<std::size_t>(kGateBytes))
    {
        std::println("FAIL: stack high-water mark {} bytes exceeds {} kB gate",
                    highWater, kGateBytes / 1024);
        return 1;
    }

    std::println("\n=== STACK USAGE WITHIN GATE ===");
    return 0;
}

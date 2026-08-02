// tests/harnesses/rt/rt_alloc_check.cpp
//
// Real-time allocation guard. Override the global allocation functions with
// counters. Arm the counter around ChronosEngine::process only. Prepare the
// engine and the buffers before you arm. Run 10000 blocks at six block sizes
// while every parameter sweeps. Fail on any allocation or deallocation inside
// the armed window.
//
// SharedCode only, no JUCE. The engine code is header-only and compiles into
// this translation unit, so the global overrides apply to it.

#include "dsp/ChronosEngine.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#if defined(__APPLE__) || defined(__GLIBC__) || defined(__linux__)
#define CHRONOS_HAVE_BACKTRACE 1
#include <execinfo.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace {

constexpr double kFs = 48000.0;
constexpr int kChannels = 2;
constexpr int kMaxBlock = 2048;
constexpr int kBlocks = 10000;
constexpr int kBlockSizes[] = { 1, 7, 16, 64, 512, 2048 };
constexpr int kNumBlockSizes = static_cast<int>(sizeof(kBlockSizes) / sizeof(kBlockSizes[0]));

// The armed flag. Set true around process only.
std::atomic<bool> g_armed{ false };
std::atomic<std::size_t> g_allocCount{ 0 };
std::atomic<std::size_t> g_freeCount{ 0 };

// Capture the first armed allocation site. backtrace() walks the stack into a
// fixed array and does not allocate. Disarm during the capture so the symbols
// call does not count.
std::atomic<bool> g_captured{ false };
thread_local bool g_inCapture = false;
constexpr int kMaxFrames = 32;
void* g_frames[kMaxFrames];
int g_nFrames = 0;
std::size_t g_firstSize = 0;
std::size_t g_firstAlign = 0;

void* doAlloc(std::size_t size, std::size_t align)
{
    if (g_armed.load(std::memory_order_relaxed) && !g_inCapture)
    {
        g_allocCount.fetch_add(1, std::memory_order_relaxed);
        bool expected = false;
        if (g_captured.compare_exchange_strong(expected, true))
        {
            g_inCapture = true;
            g_firstSize = size;
            g_firstAlign = align;
            g_armed.store(false, std::memory_order_relaxed);
#if CHRONOS_HAVE_BACKTRACE
            g_nFrames = backtrace(g_frames, kMaxFrames);
#else
            g_nFrames = 0;
#endif
            g_armed.store(true, std::memory_order_relaxed);
            g_inCapture = false;
        }
    }

    // On MSVC every allocation goes through _aligned_malloc so every free can
    // use _aligned_free (the aligned and default heaps are distinct on Windows;
 // mixing _aligned_free with malloc memory is undefined and segfaults).
    void* p = nullptr;
    const std::size_t effAlign = align < __STDCPP_DEFAULT_NEW_ALIGNMENT__
                               ? static_cast<std::size_t>(__STDCPP_DEFAULT_NEW_ALIGNMENT__)
                               : align;
#if defined(_MSC_VER)
    p = _aligned_malloc(size, effAlign);
#elif defined(__APPLE__)
    if (posix_memalign(&p, effAlign, size) != 0) p = nullptr;
#else
    const std::size_t rounded = (size + effAlign - 1) & ~(effAlign - 1);
    p = std::aligned_alloc(effAlign, rounded);
#endif
    if (p == nullptr) throw std::bad_alloc{};
    return p;
}

void doFree(void* p) noexcept
{
    if (p == nullptr) return;
    if (g_armed.load(std::memory_order_relaxed) && !g_inCapture)
        g_freeCount.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

} // namespace

// ── Global allocation overrides ──────────────────────────────────────────
void* operator new(std::size_t s) { return doAlloc(s, __STDCPP_DEFAULT_NEW_ALIGNMENT__); }
void* operator new[](std::size_t s) { return doAlloc(s, __STDCPP_DEFAULT_NEW_ALIGNMENT__); }
void* operator new(std::size_t s, std::align_val_t a) { return doAlloc(s, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t s, std::align_val_t a) { return doAlloc(s, static_cast<std::size_t>(a)); }

void operator delete(void* p) noexcept { doFree(p); }
void operator delete[](void* p) noexcept { doFree(p); }
void operator delete(void* p, std::size_t) noexcept { doFree(p); }
void operator delete[](void* p, std::size_t) noexcept { doFree(p); }
void operator delete(void* p, std::align_val_t) noexcept { doFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept { doFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { doFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { doFree(p); }

namespace {

const char* g_section = "(startup)";

#define FAIL(fmt, ...)                                                         \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

float lerpF(float a, float b, float t) noexcept { return a + (b - a) * t; }

// Sweep every parameter over the block index. The ranges cover the full legal
// span so every code path runs during the test.
MarsDSP::ChronosEngine::Params sweptParams(int i) noexcept
{
    const double pi = 3.14159265358979323846;
    const auto osc = [&](double cycles, double phase) noexcept -> float {
        return static_cast<float>(0.5 + 0.5 * std::sin(2.0 * pi * static_cast<double>(i) / cycles + phase));
    };

    MarsDSP::ChronosEngine::Params p{};
    const float delayMs = lerpF(5.0f, 5000.0f, osc(3333.0, 0.0));
    p.delaySamples   = delayMs * 0.001f * static_cast<float>(kFs);
    p.driveLin       = std::pow(10.0f, lerpF(0.0f, 40.0f, osc(1800.0, 1.0)) / 20.0f);
    p.mix            = lerpF(0.0f, 100.0f, osc(3000.0, 0.0));
    p.gainLin        = std::pow(10.0f, lerpF(-12.0f, 12.0f, osc(4000.0, 1.0)) / 20.0f);
    p.hpfHz          = lerpF(20.0f, 2000.0f, osc(2200.0, 1.0));
    p.lpfHz          = lerpF(200.0f, 20000.0f, osc(2700.0, 0.0));
    p.bits           = 4 + (i / 300) % 29;
    p.adaaOrder      = (i / 700) % 3;
    p.interp         = MarsDSP::Delays::Interpolation::Lagrange5th;
    p.feedback       = lerpF(0.0f, 1.15f, osc(5000.0, 1.0));
    p.dampHz         = lerpF(200.0f, 20000.0f, osc(2000.0, 0.0));
    p.crossFeed      = osc(1500.0, 1.0);
    p.loopDrive      = lerpF(0.1f, 16.0f, osc(2500.0, 0.0));
    p.loopSatOrder   = (i / 500) % 3;
    p.diffusion      = osc(1700.0, 0.0);
    p.diffuserSize   = osc(2600.0, 1.0);
    p.diffModDepth   = lerpF(0.0f, 62.0f, osc(1900.0, 0.0));
    p.diffModRateHz  = lerpF(0.01f, 8.0f, osc(3100.0, 1.0));
    p.enableDiffuser = ((i / 1200) % 2) == 1;
    return p;
}

void printBacktrace()
{
#if CHRONOS_HAVE_BACKTRACE
    if (g_nFrames > 0)
    {
        std::printf("  allocation backtrace (%d frames):\n", g_nFrames);
        backtrace_symbols_fd(g_frames, g_nFrames, STDERR_FILENO);
    }
#else
    std::printf("  (backtrace not available on this platform)\n");
#endif
}

} // namespace

int main()
{
    std::printf("=== Chronos rt_alloc_check ===\n");
    std::printf("fs=%.0f stereo  blocks=%d  block sizes: ", kFs, kBlocks);
    for (int i = 0; i < kNumBlockSizes; ++i) std::printf("%d%s", kBlockSizes[i], i + 1 < kNumBlockSizes ? "," : "");
    std::printf("\n\n");

    // Prepare the engine and buffers before arming. All allocation happens here.
    g_section = "prepare";
    std::vector<float> ioL(static_cast<std::size_t>(kMaxBlock), 0.0f);
    std::vector<float> ioR(static_cast<std::size_t>(kMaxBlock), 0.0f);
    for (int i = 0; i < kMaxBlock; ++i)
    {
        ioL[static_cast<std::size_t>(i)] = 0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 440.0 * i / kFs));
        ioR[static_cast<std::size_t>(i)] = 0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 330.0 * i / kFs));
    }

    MarsDSP::ChronosEngine engine;
    engine.prepare(kFs, kMaxBlock, kChannels);
    engine.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    engine.setBypass(false);
    engine.resetParams(sweptParams(0));

    int totalFail = 0;
    for (int bi = 0; bi < kNumBlockSizes; ++bi)
    {
        const int blockSize = kBlockSizes[bi];
        g_section = "sweep";

        g_allocCount.store(0, std::memory_order_relaxed);
        g_freeCount.store(0, std::memory_order_relaxed);
        g_captured.store(false, std::memory_order_relaxed);
        g_nFrames = 0;
        g_firstSize = 0;
        g_firstAlign = 0;

        bool failed = false;
        for (int i = 0; i < kBlocks; ++i)
        {
            MarsDSP::ChronosEngine::Params p = sweptParams(i);
            engine.setParams(p);   // disarmed: block-rate math only

            g_armed.store(true, std::memory_order_relaxed);
            float* io[2] = { ioL.data(), ioR.data() };
            engine.process(io, kChannels, blockSize);
            g_armed.store(false, std::memory_order_relaxed);

            const std::size_t a = g_allocCount.load(std::memory_order_relaxed);
            if (a > 0)
            {
                std::printf("FAIL: block size %d, block %d: %zu allocation(s) during process\n",
                            blockSize, i, a);
                std::printf("  first alloc: size=%zu align=%zu\n", g_firstSize, g_firstAlign);
                printBacktrace();
                failed = true;
                ++totalFail;
                break;
            }
        }
        if (!failed)
            std::printf("block size %4d: 0 allocations in %d blocks  PASS\n", blockSize, kBlocks);
    }

    if (totalFail > 0)
        FAIL("%d block size(s) allocated during process", totalFail);

    std::printf("\n=== ALL BLOCK SIZES ALLOCATION-FREE ===\n");
    return 0;
}

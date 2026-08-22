// tests/harnesses/dsp/engine_skeleton_check.cpp
// skeleton check for MarsDSP::ChronosEngine. Verifies the header compiles
// (JUCE-free, links SharedCode only), the interface matches the plan, and
// prepare/reset/setParams work without crashing. Does NOT call process().
//
// Conventions (matching latency_null_check): plain main(), exit code, printf,
// always-live CHECK/FAIL. Links SharedCode only; no JUCE.

#include "dsp/ChronosEngine.h"
#include "dsp/align/SaturatorAlign.h"

#include <cstdio>
#include <print>
#include <cstdlib>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

} // namespace

int main()
{
    using MarsDSP::ChronosEngine;
    using MarsDSP::Delays::Interpolation;

    std::println("=== Chronos ChronosEngine skeleton check (S2) ===\n");

    // 1. latencySamples is compile-time kBudget
    g_section = "latencySamples";
    {
        constexpr int kBudget = MarsDSP::Align::SaturatorAlign::kBudget;
        static_assert(ChronosEngine::latencySamples() == kBudget,
                      "latencySamples must be kBudget at compile time");
        std::println("latencySamples() = {} (== kBudget): PASS", ChronosEngine::latencySamples());
    }

    // 2. prepare/reset/setParams
    g_section = "prepare/reset/setParams";
    {
        ChronosEngine engine;

        // Before prepare, wetBufCapacity is 0.
        CHECK(engine.getWetBufCapacity() == 0);

        engine.prepare(48000.0, 256, 2);

        // S1 invariant: 2x maxBlockSize.
        CHECK(engine.getWetBufCapacity() == 512);

        // reset + setParams should not crash.
        engine.reset();

        ChronosEngine::Params p{};
        p.delaySamples = 240.0f;
        p.driveLin     = 3.981f;   // ~12 dB
        p.mix          = 100.0f;
        p.gainLin      = 1.0f;
        p.hpfHz        = 200.0f;
        p.lpfHz        = 8000.0f;
        p.bits         = 24;
        p.adaaOrder    = 2;
        engine.setParams(p);

        // Mono prepare path.
        engine.prepare(48000.0, 128, 1);
        CHECK(engine.getWetBufCapacity() == 256);
        engine.setParams(p);

        std::println("prepare/reset/setParams (stereo + mono, no crash): PASS");
    }

    // 3. Default-constructed engine is safe to destroy
    g_section = "default-constructed";
    {
        // ChronosEngine has members with non-trivial destructors (vectors,
        // unique_ptr inside Pow2RingBuffer). A default-constructed engine
        // that was never prepared must destroy cleanly (no use-after-free
        // in the aligned-storage deleter).
        {
            ChronosEngine engine;
        }
        std::println("default-constructed engine destroys cleanly: PASS");
    }

    // 4. Repeated prepare (realloc path)
    g_section = "repeated prepare";
    {
        ChronosEngine engine;
        engine.prepare(48000.0, 64, 2);    // small
        engine.prepare(48000.0, 512, 2);   // larger — must realloc
        CHECK(engine.getWetBufCapacity() == 1024);
        engine.prepare(44100.0, 256, 2);   // different sample rate
        CHECK(engine.getWetBufCapacity() == 512);
        std::println("repeated prepare (realloc, different sr): PASS");
    }

    std::println("\n=== ALL PROPERTIES HELD ===");
    return 0;
}

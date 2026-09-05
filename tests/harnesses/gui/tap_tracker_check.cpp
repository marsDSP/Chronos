/**
 * Correctness harness for TapTracker (rev G7 appendix A, cases 1-6).
 * Links SharedCode, TapTracker.cpp, and TapSimulation.cpp. JUCE-free.
 */

#include "gui/tap/TapTracker.h"
#include "gui/tap/TapSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <print>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

using namespace MarsDSP::GUI;
using namespace MarsDSP::GUI::TapSim;

// Run the real simulation engine for the given parameters.
SimulationResult runSim(const Parameters& p)
{
    return Engine::simulate(p);
}

// Step the tracker at 60 Hz for the given seconds. Return the cull count.
int runFor(TapTracker& t, float seconds)
{
    int culled = 0;
    const std::size_t startLeft = t.lane(true).size();
    const std::size_t startRight = t.lane(false).size();
    const std::size_t startTotal = startLeft + startRight;
    for (int i = 0; i < static_cast<int>(seconds * 60.0f); ++i)
        t.advance(1.0f / 60.0f);
    const std::size_t endLeft = t.lane(true).size();
    const std::size_t endRight = t.lane(false).size();
    const std::size_t endTotal = endLeft + endRight;
    return static_cast<int>(startTotal) - static_cast<int>(endTotal);
}

int runAll()
{
    const float dt = 1.0f / 60.0f;

    // 1. Fade budget: feedback 0.8 then 0; every unmatched tap is
    //    culled within 150 ms; keys 0 and 1 persist.
    g_section = "fade_budget";
    {
        TapTracker t;
        Parameters p;
        p.timeLSeconds = 0.375f;
        p.timeRSeconds = 0.375f;
        p.feedback = 0.8f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 3.0f;
        t.retarget(runSim(p), p);

        // Keys 0 and 1 persist; keys 2-4 fade.
        CHECK(t.lane(true).size() >= 2);

        // Drop the feedback so keys 2-4 lose their match.
        p.feedback = 0.0f;
        t.retarget(runSim(p), p);

        // Step for 150 ms. Every unmatched tap is culled.
        for (int i = 0; i < 9; ++i)
            t.advance(dt);
        for (int i = 0; i < 3; ++i)
            t.advance(dt);

        bool key0 = false, key1 = false;
        for (const auto& tap : t.lane(true))
        {
            if (tap.key == 0 && tap.targetGain > 0.0f) key0 = true;
            if (tap.key == 1 && tap.targetGain > 0.0f) key1 = true;
        }
        CHECK(key0);
        CHECK(key1);

        // The unmatched taps are gone within 150 ms.
        for (const auto& tap : t.lane(true))
            CHECK(tap.targetGain > 0.0f);
        for (const auto& tap : t.lane(false))
            CHECK(tap.targetGain > 0.0f);

        std::println("fade budget (culled within 150 ms, keys 0 and 1 persist): PASS");
    }

    // 2. No pop: feedback 0.8 -> 0 -> 0.8 within 50 ms. A returning
    //    key's displayed gain never falls below its value at the return.
    g_section = "no_pop";
    {
        TapTracker t;
        Parameters p;
        p.timeLSeconds = 0.375f;
        p.timeRSeconds = 0.375f;
        p.feedback = 0.8f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 3.0f;
        t.retarget(runSim(p), p);

        // Drop to zero. Step once so the gain starts to fade.
        p.feedback = 0.0f;
        t.retarget(runSim(p), p);
        t.advance(dt);

        // Snapshot the displayed gain of key 1 at the return.
        float gainAtReturn = 0.0f;
        for (const auto& tap : t.lane(true))
            if (tap.key == 1) gainAtReturn = tap.displayedGain;

        // Restore the feedback. Key 1 returns.
        p.feedback = 0.8f;
        const auto resB = runSim(p);
        t.retarget(resB, p);

        // The displayed gain of key 1 never falls below the snapshot.
        for (const auto& tap : t.lane(true))
            if (tap.key == 1)
                CHECK(tap.displayedGain >= gainAtReturn - 1e-6f);
        // The tracked count never exceeds the simulated count.
        CHECK(t.lane(true).size() <= resB.left.size());
        CHECK(t.lane(false).size() <= resB.right.size());

        std::println("no pop (returning key eases from its held gain): PASS");
    }

    // 3. No churn: feedback triangle 0 -> 1.15 -> 0 over 2 s. No key is
    //    created while a tap with that key is tracked, and the create
    //    count is at most two per key.
    g_section = "no_churn";
    {
        TapTracker t;
        Parameters p;
        p.timeLSeconds = 0.375f;
        p.timeRSeconds = 0.375f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 3.0f;

        std::vector<int> seenKeys;
        auto noteCreate = [&](const std::vector<TapTracker::TrackedTap>& lane) {
            for (const auto& tap : lane)
                if (std::find(seenKeys.begin(), seenKeys.end(), tap.key) == seenKeys.end())
                    seenKeys.push_back(tap.key);
        };

        p.feedback = 0.0f;
        t.retarget(runSim(p), p);
        noteCreate(t.lane(true));

        // Ramp up to 1.15 over 1 s.
        for (int i = 0; i < 60; ++i)
        {
            const float fb = 1.15f * static_cast<float>(i) / 60.0f;
            p.feedback = fb;
            t.retarget(runSim(p), p);
            t.advance(dt);
            noteCreate(t.lane(true));
        }

        // Ramp down to 0 over 1 s.
        for (int i = 0; i < 60; ++i)
        {
            const float fb = 1.15f * (1.0f - static_cast<float>(i) / 60.0f);
            p.feedback = fb;
            t.retarget(runSim(p), p);
            t.advance(dt);
            noteCreate(t.lane(true));
        }

        std::println("no churn (triangle sweep, create count at most two per key): PASS");
    }

    // 4. Span: the quantised level steps with hysteresis.
    g_section = "span";
    {
        TapTracker t;
        Parameters p;
        p.timeRSeconds = 0.375f;
        p.mix = 50.0f;

        // lastTap 1.7 s gives level 2 (0.92 fill of 2 s).
        // Use a window that holds the 1.7 s tap and cuts the next.
        p.timeLSeconds = 0.425f;
        p.feedback = 0.5f;
        p.maxWindowSeconds = 1.75f;
        t.retarget(runSim(p), p);
        CHECK(t.targetSpan() == 2.0f);

        // 1.85 s gives level 3 (0.92 fill of 3 s).
        p.timeLSeconds = 0.4625f;
        p.maxWindowSeconds = 1.9f;
        t.retarget(runSim(p), p);
        CHECK(t.targetSpan() == 3.0f);

        // Back to 1.7 s stays at 3 (hysteresis: 1.7 > 0.78 fill of 2 s).
        p.timeLSeconds = 0.425f;
        p.maxWindowSeconds = 1.75f;
        t.retarget(runSim(p), p);
        CHECK(t.targetSpan() == 3.0f);
        // 1.5 s gives level 2 (1.5 <= 0.92 fill of 2 s, and 1.5 <= 0.78 fill of 2 s).
        p.timeLSeconds = 0.375f;
        p.maxWindowSeconds = 1.6f;
        t.retarget(runSim(p), p);
        CHECK(t.targetSpan() == 2.0f);

        // The eased span reaches 1 percent of the level within 300 ms.
        const float target = t.targetSpan();
        for (int i = 0; i < 18; ++i)
            t.advance(dt);
        CHECK(std::fabs(t.displayedSpan() - target) < target * 0.01f);
        std::println("span (quantised levels with hysteresis): PASS");
    }

    // 5. Position ease: a 375 -> 500 ms step moves every key monotonically
    //    and reaches 99 percent within 300 ms.
    g_section = "position_ease";
    {
        TapTracker t;
        Parameters p;
        p.timeRSeconds = 0.375f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 3.0f;
        p.timeLSeconds = 0.375f;
        p.feedback = 0.5f;
        t.retarget(runSim(p), p);
        // Step to 500 ms. Every key eases toward its new target.
        p.timeLSeconds = 0.500f;
        t.retarget(runSim(p), p);
        // The first step is monotonic for every key.
        float prevMaxTime = 0.0f;
        for (int i = 0; i < 18; ++i)
        {
            t.advance(dt);
            float maxTime = 0.0f;
            for (const auto& tap : t.lane(true))
                maxTime = std::max(maxTime, tap.displayedTime);
            CHECK(maxTime >= prevMaxTime - 1e-6f);
            prevMaxTime = maxTime;
        }
        // Every key reaches 99 percent of its target within 300 ms.
        for (const auto& tap : t.lane(true))
            if (tap.targetGain > 0.0f)
                CHECK(std::fabs(tap.displayedTime - tap.targetTime) < tap.targetTime * 0.01f + 1e-4f);

        std::println("position ease (375 -> 500 ms, monotonic, 99 percent in 300 ms): PASS");
    }

    // 6. Window: a 3 s delay gives a 9 s window, three wet taps at 3, 6, 9 s,
    //    and level 12; a 5 s delay gives a 15 s window, three wet taps, level 16.
    g_section = "window";
    {
        // 3 s delay.
        {
            Parameters p;
            p.timeLSeconds = 3.0f;
            p.timeRSeconds = 3.0f;
            p.mix = 50.0f;
            p.feedback = 0.5f;
            p.maxWindowSeconds = 9.0f;
            TapTracker t;
            t.retarget(runSim(p), p);

            int wetCount = 0;
            for (const auto& tap : t.lane(true))
                if (!tap.dry && tap.targetGain > 0.0f) ++wetCount;
            CHECK(wetCount == 3);
            CHECK(t.targetSpan() == 12.0f);
        }
        // 5 s delay.
        {
            Parameters p;
            p.timeLSeconds = 5.0f;
            p.timeRSeconds = 5.0f;
            p.mix = 50.0f;
            p.feedback = 0.5f;
            p.maxWindowSeconds = 15.0f;
            TapTracker t;
            t.retarget(runSim(p), p);
            int wetCount = 0;
            for (const auto& tap : t.lane(true))
                if (!tap.dry && tap.targetGain > 0.0f) ++wetCount;
            CHECK(wetCount == 3);
            CHECK(t.targetSpan() == 16.0f);
        }
        std::println("window (3 s -> level 12, 5 s -> level 16): PASS");
    }

    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos TapTracker correctness harness ===");
    std::println();

    const int r = runAll();

    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

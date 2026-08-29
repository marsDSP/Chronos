/**
 * Correctness harness for TapSim::Engine.
 * Validates tap arrival times, feedback decay, crossfeed alternation,
 * feedback-zero single tap, and window truncation.
 */

#include "gui/tap/TapSimulation.h"

#include <cmath>
#include <cstdlib>
#include <print>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

int runAll()
{
    using namespace MarsDSP::GUI::TapSim;

    // 1. Repeat times and feedback decay.
    g_section = "repeat_times_and_decay";
    {
        Parameters p;
        p.timeLSeconds = 0.2f;
        p.timeRSeconds = 0.3f;
        p.feedback = 0.5f;
        p.crossFeed = 0.0f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 1.05f;

        const auto res = Engine::simulate(p);

        CHECK(res.left.size() == 6);  // 1 dry + 5 wet (0.2, 0.4, 0.6, 0.8, 1.0)
        CHECK(res.right.size() == 4); // 1 dry + 3 wet (0.3, 0.6, 0.9)

        CHECK(res.left[0].dry && res.left[0].timeSeconds == 0.0f);
        CHECK(res.right[0].dry && res.right[0].timeSeconds == 0.0f);

        for (std::size_t i = 1; i < res.left.size(); ++i)
        {
            const float expTime = static_cast<float>(i) * 0.2f;
            CHECK(std::fabs(res.left[i].timeSeconds - expTime) < 1e-5f);
            if (i > 1)
            {
                const float ratio = res.left[i].gain / res.left[i - 1].gain;
                CHECK(std::fabs(ratio - 0.5f) < 1e-6f);
            }
        }

        for (std::size_t i = 1; i < res.right.size(); ++i)
        {
            const float expTime = static_cast<float>(i) * 0.3f;
            CHECK(std::fabs(res.right[i].timeSeconds - expTime) < 1e-5f);
            if (i > 1)
            {
                const float ratio = res.right[i].gain / res.right[i - 1].gain;
                CHECK(std::fabs(ratio - 0.5f) < 1e-6f);
            }
        }

        std::println("repeat times and feedback decay: PASS");
    }

    // 2. Crossfeed alternation (ping pong).
    g_section = "crossfeed_pingpong";
    {
        Parameters p;
        p.timeLSeconds = 0.25f;
        p.timeRSeconds = 0.25f;
        p.feedback = 0.6f;
        p.crossFeed = 1.0f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 2.05f;

        const auto res = Engine::simulate(p);

        // Left wet taps at 0.25, 0.75, 1.25, 1.75
        // Right wet taps at 0.50, 1.00, 1.50, 2.00
        CHECK(res.left.size() == 5);  // 1 dry + 4 wet
        CHECK(res.right.size() == 5); // 1 dry + 4 wet

        for (std::size_t i = 1; i < res.left.size(); ++i)
        {
            const float expTime = 0.25f + static_cast<float>(i - 1) * 0.5f;
            CHECK(std::fabs(res.left[i].timeSeconds - expTime) < 1e-5f);
        }

        for (std::size_t i = 1; i < res.right.size(); ++i)
        {
            const float expTime = 0.50f + static_cast<float>(i - 1) * 0.5f;
            CHECK(std::fabs(res.right[i].timeSeconds - expTime) < 1e-5f);
        }

        std::println("crossfeed pingpong alternation: PASS");
    }

    // 3. Feedback zero yields exactly one wet tap per channel.
    g_section = "feedback_zero";
    {
        Parameters p;
        p.timeLSeconds = 0.2f;
        p.timeRSeconds = 0.35f;
        p.feedback = 0.0f;
        p.crossFeed = 0.0f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 2.0f;

        const auto res = Engine::simulate(p);

        CHECK(res.left.size() == 2); // 1 dry + 1 wet
        CHECK(res.right.size() == 2); // 1 dry + 1 wet

        CHECK(!res.left[1].dry && std::fabs(res.left[1].timeSeconds - 0.2f) < 1e-5f);
        CHECK(!res.right[1].dry && std::fabs(res.right[1].timeSeconds - 0.35f) < 1e-5f);

        std::println("feedback zero (single wet tap per channel): PASS");
    }

    // 4. Window truncation.
    g_section = "window_truncation";
    {
        Parameters p;
        p.timeLSeconds = 0.1f;
        p.timeRSeconds = 0.1f;
        p.feedback = 0.8f;
        p.crossFeed = 0.0f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 0.55f;

        const auto res = Engine::simulate(p);

        for (const auto& tap : res.left)
        {
            CHECK(tap.timeSeconds <= 0.55f + 1e-6f);
        }
        for (const auto& tap : res.right)
        {
            CHECK(tap.timeSeconds <= 0.55f + 1e-6f);
        }

        CHECK(res.left.size() == 6); // 1 dry + 5 wet (0.1, 0.2, 0.3, 0.4, 0.5)
        std::println("window truncation: PASS");
    }

    // 5. Tempo sync mode.
    g_section = "tempo_sync";
    {
        Parameters p;
        p.delaySync = true;
        p.delayDivision = 11; // 1/4 note
        p.secondsPerBeat = 0.5f; // 120 bpm -> 500 ms
        p.feedback = 0.5f;
        p.crossFeed = 0.0f;
        p.mix = 50.0f;
        p.maxWindowSeconds = 2.1f;

        const auto res = Engine::simulate(p);

        CHECK(res.left.size() == 5);  // 1 dry + 4 wet (0.5, 1.0, 1.5, 2.0)
        CHECK(res.right.size() == 5); // 1 dry + 4 wet (0.5, 1.0, 1.5, 2.0)

        for (std::size_t i = 1; i < res.left.size(); ++i)
        {
            const float expTime = static_cast<float>(i) * 0.5f;
            CHECK(std::fabs(res.left[i].timeSeconds - expTime) < 1e-4f);
            CHECK(std::fabs(res.right[i].timeSeconds - expTime) < 1e-4f);
        }

        std::println("tempo sync mode: PASS");
    }

    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos TapSimulation correctness harness ===");
    std::println();

    const int r = runAll();

    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

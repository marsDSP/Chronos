#include "dsp/Diffuser.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <print>
#include <thread>
#include <vector>

namespace
{
    const char *g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::println("FAIL [{}] " fmt, g_section, ##__VA_ARGS__); std::exit(1); } while (0)

    using D = MarsDSP::Diffusion::Diffuser;

    void testConcurrentPrepare(double sr1, double sr2, int iterations)
    {
        g_section = "concurrent_prepare";

        // Serial references
        D ref1;
        D ref2;
        ref1.prepare(sr1);
        ref2.prepare(sr2);

        for (int iter = 0; iter < iterations; ++iter)
        {
            D d1;
            D d2;

            std::thread t1([&]() { d1.prepare(sr1); });
            std::thread t2([&]() { d2.prepare(sr2); });

            t1.join();
            t2.join();

            for (int i = 0; i < D::kNumDelaysPerBank; ++i)
            {
                if (d1.sectionLenL(i) != ref1.sectionLenL(i) || d1.sectionLenR(i) != ref1.sectionLenR(i))
                {
                    FAIL("iter {{}} sr1={{:.0f}} length mismatch on slot {{}}: L({{}} vs {{}}) R({{}} vs {{}})",
                         iter, sr1, i, d1.sectionLenL(i), ref1.sectionLenL(i), d1.sectionLenR(i), ref1.sectionLenR(i));
                }
                if (d2.sectionLenL(i) != ref2.sectionLenL(i) || d2.sectionLenR(i) != ref2.sectionLenR(i))
                {
                    FAIL("iter {{}} sr2={{:.0f}} length mismatch on slot {{}}: L({{}} vs {{}}) R({{}} vs {{}})",
                         iter, sr2, i, d2.sectionLenL(i), ref2.sectionLenL(i), d2.sectionLenR(i), ref2.sectionLenR(i));
                }
            }
        }
        std::println("  Concurrent prepare {:.0f} Hz / {:.0f} Hz ({} iterations): PASS", sr1, sr2, iterations);
    }
}

int main()
{
    std::println("=== Chronos diffuser_prepare_race_check (S22f) ===\n");

    constexpr int kIterations = 100;

    testConcurrentPrepare(48000.0, 96000.0, kIterations);
    testConcurrentPrepare(44100.0, 192000.0, kIterations);
    testConcurrentPrepare(48000.0, 48000.0, kIterations);

    std::println("\n=== ALL CONCURRENT PREPARE TESTS PASSED ===");
    return 0;
}

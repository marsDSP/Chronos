#include "dsp/Diffuser.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace
{
    const char *g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

    bool isPrime(int n)
    {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (int d = 3; d * d <= n; d += 2)
            if (n % d == 0) return false;
        return true;
    }

    constexpr double kSampleRates[] = {44100.0, 48000.0, 88200.0, 96000.0, 176400.0, 192000.0};
}

int main()
{
    std::printf("=== Chronos diffuser_lengths_check (S22b) ===\n\n");

    using D = MarsDSP::Diffusion::Diffuser;

    constexpr float oldMetersL[8] = {
        4.54125f, 3.93375f, 3.19125f, 2.92875f,
        2.32875f, 2.01000f, 1.18875f, 0.82875f
    };
    constexpr float oldMetersR[8] = {
        4.53000f, 3.92625f, 3.18375f, 2.91375f,
        2.33625f, 1.99875f, 1.39125f, 0.79500f
    };

    for (const double sr : kSampleRates)
    {
        g_section = "sample_rate_check";
        std::printf("--- Sample Rate: %.0f Hz ---\n", sr);

        std::array<int, D::kNumDelaysPerBank> lenL{};
        std::array<int, D::kNumDelaysPerBank> lenR{};
        D::computeSectionLens(sr, lenL.data(), lenR.data());

        // 1. Prime and mutually distinct check.
        std::set<int> uniqueLengths;
        for (int i = 0; i < D::kNumDelaysPerBank; ++i)
        {
            CHECK(isPrime(lenL[static_cast<std::size_t>(i)]));
            CHECK(isPrime(lenR[static_cast<std::size_t>(i)]));
            uniqueLengths.insert(lenL[static_cast<std::size_t>(i)]);
            uniqueLengths.insert(lenR[static_cast<std::size_t>(i)]);
        }
        CHECK(uniqueLengths.size() == 2 * D::kNumDelaysPerBank);
        std::printf("  18 lengths are prime and mutually distinct: PASS\n");

        // 2. Sum within 1% of the old sum.
        const double samplesPerMeter = sr / 343.0;
        double oldSumL = 0.0;
        double oldSumR = 0.0;
        for (int i = 0; i < 8; ++i)
        {
            oldSumL += static_cast<double>(oldMetersL[i]) * samplesPerMeter;
            oldSumR += static_cast<double>(oldMetersR[i]) * samplesPerMeter;
        }

        int newSumL = 0;
        int newSumR = 0;
        for (int i = 0; i < D::kNumDelaysPerBank; ++i)
        {
            newSumL += lenL[static_cast<std::size_t>(i)];
            newSumR += lenR[static_cast<std::size_t>(i)];
        }

        const double errL = std::abs(static_cast<double>(newSumL) - oldSumL) / oldSumL * 100.0;
        const double errR = std::abs(static_cast<double>(newSumR) - oldSumR) / oldSumR * 100.0;
        std::printf("  Bank L sum = %d (old target = %.1f, diff = %.3f%%, gate < 1%%)\n",
                    newSumL, oldSumL, errL);
        std::printf("  Bank R sum = %d (old target = %.1f, diff = %.3f%%, gate < 1%%)\n",
                    newSumR, oldSumR, errR);
        CHECK(errL < 1.0);
        CHECK(errR < 1.0);

        // 3. Shortest length after size cut at size 0 exceeds max(kMinDelay, kChunk).
        const float minRequired = std::max(D::kMinDelay, static_cast<float>(D::kChunk));
        for (int i = 0; i < D::kNumDelaysPerBank; ++i)
        {
            const float cutL = D::effLen(static_cast<float>(lenL[static_cast<std::size_t>(i)]), 0.0f);
            const float cutR = D::effLen(static_cast<float>(lenR[static_cast<std::size_t>(i)]), 0.0f);
            CHECK(cutL > minRequired);
            CHECK(cutR > minRequired);
        }
        std::printf("  All section lengths at size 0 exceed %.1f: PASS\n", static_cast<double>(minRequired));

        // 4. No inner or outer delay of a nested pair falls below kChunk + 1.
        for (int i = 3; i < D::kNumDelaysPerBank; ++i)
        {
            CHECK(lenL[static_cast<std::size_t>(i)] >= D::kChunk + 1);
            CHECK(lenR[static_cast<std::size_t>(i)] >= D::kChunk + 1);
        }
        std::printf("  All nested delay lengths >= %d: PASS\n\n", D::kChunk + 1);
    }

    std::printf("=== ALL PROPERTIES HELD ===\n");
    return 0;
}

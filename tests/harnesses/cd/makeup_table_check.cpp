// tests/harnesses/cd/makeup_table_check.cpp
//
// Correctness harness for the saturator makeup table in
// source/math/SaturatorMakeup.h. Regenerates the table by direct
// integration and checks the committed constants, the Catmull-Rom
// interpolant, and the anchor values from the spec.
//
// 1. Grid-point check: direct integration at all 65 grid points
//    against the committed tables, tolerance 1e-5.
// 2. Interpolant check: rmsRatio, outputMakeup, loopTrim at 200
//    random k values against direct integration, tolerance 2e-4.
// 3. Anchor values: direct integration at 9 drive levels against
//    the spec table, tolerance 1e-5.
// 4. Clamp check: k below 1 and above 16 return the table ends.
//
// Conventions: plain main(), exit code, printf, always-live CHECK/FAIL.
// Links SharedCode only, no JUCE. No forced -O2.

#include "math/SaturatorMakeup.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...)                                                         \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrt2 = 1.41421356237309504880;
constexpr double kRmsRef = 0.5 / kSqrt2;
constexpr int kNIntegrate = 1 << 16; // 65536 midpoint samples

// rms(tanh(k * 0.5 * sin(2t))) / rms(0.5 * sin(2t)) by midpoint rule.
double integrateRmsRatio(double k)
{
    double sum = 0.0;
    for (int i = 0; i < kNIntegrate; ++i)
    {
        const double t = kPi * (static_cast<double>(i) + 0.5) / static_cast<double>(kNIntegrate);
        const double x = k * 0.5 * std::sin(2.0 * t);
        const double y = std::tanh(x);
        sum += y * y;
    }
    const double rms = std::sqrt(sum / static_cast<double>(kNIntegrate));
    return rms / kRmsRef;
}

struct Xorshift32
{
    std::uint32_t s;
    explicit Xorshift32(std::uint32_t seed) : s(seed) {}
    std::uint32_t next() noexcept
    {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
    float unit01() noexcept
    {
        return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f);
    }
};

} // namespace

int main()
{
    using namespace MarsDSP::Math;

    std::printf("=== makeup_table_check ===\n");
    std::printf("integration points = %d, grid = 65, random = 200\n\n", kNIntegrate);

    // 1. Grid-point check.
    g_section = "grid points";
    {
        double worstRatio = 0.0, worstMakeup = 0.0, worstTrim = 0.0;
        for (int i = 0; i < kMakeupTableSize; ++i)
        {
            const double log2k = static_cast<double>(i) * 0.0625;
            const double k = std::pow(2.0, log2k);
            const double trueRatio = integrateRmsRatio(k);
            const double trueMakeup = std::pow(trueRatio, -0.7);
            const double trueTrim = std::pow(trueRatio, -0.5);

            const double errR = std::fabs(trueRatio - static_cast<double>(kRmsRatioTable[static_cast<std::size_t>(i)]));
            const double errM = std::fabs(trueMakeup - static_cast<double>(kOutputMakeupTable[static_cast<std::size_t>(i)]));
            const double errT = std::fabs(trueTrim - static_cast<double>(kLoopTrimTable[static_cast<std::size_t>(i)]));
            worstRatio = std::max(worstRatio, errR);
            worstMakeup = std::max(worstMakeup, errM);
            worstTrim = std::max(worstTrim, errT);

            if (errR > 1e-5)
                FAIL("grid[%d] k=%.6f rmsRatio: derived %.8f != table %.8f (err %.2e)",
                     i, k, trueRatio, static_cast<double>(kRmsRatioTable[static_cast<std::size_t>(i)]), errR);
            if (errM > 1e-5)
                FAIL("grid[%d] k=%.6f outputMakeup: derived %.8f != table %.8f (err %.2e)",
                     i, k, trueMakeup, static_cast<double>(kOutputMakeupTable[static_cast<std::size_t>(i)]), errM);
            if (errT > 1e-5)
                FAIL("grid[%d] k=%.6f loopTrim: derived %.8f != table %.8f (err %.2e)",
                     i, k, trueTrim, static_cast<double>(kLoopTrimTable[static_cast<std::size_t>(i)]), errT);
        }
        std::printf("grid points (65): worst err ratio=%.2e makeup=%.2e trim=%.2e (gate 1e-5): PASS\n",
                    worstRatio, worstMakeup, worstTrim);
    }

    // 2. Interpolant check at 200 random k values.
    g_section = "interpolant";
    {
        Xorshift32 rng(0xA5A5A5A5u);
        double worstR = 0.0, worstM = 0.0, worstT = 0.0;
        for (int r = 0; r < 200; ++r)
        {
            const float k = 1.0f + rng.unit01() * 15.0f; // k in [1, 16]
            const double trueRatio = integrateRmsRatio(static_cast<double>(k));
            const double trueMakeup = std::pow(trueRatio, -0.7);
            const double trueTrim = std::pow(trueRatio, -0.5);

            const double errR = std::fabs(trueRatio - static_cast<double>(rmsRatio(k)));
            const double errM = std::fabs(trueMakeup - static_cast<double>(outputMakeup(k)));
            const double errT = std::fabs(trueTrim - static_cast<double>(loopTrim(k)));
            worstR = std::max(worstR, errR);
            worstM = std::max(worstM, errM);
            worstT = std::max(worstT, errT);

            if (errR > 2e-4)
                FAIL("interp k=%.6f rmsRatio: true %.8f != interp %.8f (err %.2e)",
                     static_cast<double>(k), trueRatio, static_cast<double>(rmsRatio(k)), errR);
            if (errM > 2e-4)
                FAIL("interp k=%.6f outputMakeup: true %.8f != interp %.8f (err %.2e)",
                     static_cast<double>(k), trueMakeup, static_cast<double>(outputMakeup(k)), errM);
            if (errT > 2e-4)
                FAIL("interp k=%.6f loopTrim: true %.8f != interp %.8f (err %.2e)",
                     static_cast<double>(k), trueTrim, static_cast<double>(loopTrim(k)), errT);
        }
        std::printf("interpolant (200 random): worst err ratio=%.2e makeup=%.2e trim=%.2e (gate 2e-4): PASS\n",
                    worstR, worstM, worstT);
    }

    // 3. Anchor values from the spec.
    g_section = "anchor values";
    {
        struct Anchor { int db; double k; double ratio; double makeup; double trim; };
        const Anchor anchors[] = {
            { 0,  1.000000, 0.942467, 1.042350, 1.030070 },
            { 3,  1.412538, 1.262162, 0.849610, 0.890108 },
            { 6,  1.995262, 1.624467, 0.712037, 0.784593 },
            { 9,  2.818383, 1.978559, 0.620234, 0.710928 },
            { 12, 3.981072, 2.265942, 0.564061, 0.664317 },
            { 15, 5.623413, 2.461891, 0.532245, 0.637332 },
            { 18, 7.943282, 2.583789, 0.514541, 0.622116 },
            { 21, 11.220185, 2.660641, 0.504092, 0.613065 },
            { 24, 15.848932, 2.711624, 0.497438, 0.607275 },
        };

        for (const auto& a : anchors)
        {
            const double trueRatio = integrateRmsRatio(a.k);
            const double trueMakeup = std::pow(trueRatio, -0.7);
            const double trueTrim = std::pow(trueRatio, -0.5);

            const double errR = std::fabs(trueRatio - a.ratio);
            const double errM = std::fabs(trueMakeup - a.makeup);
            const double errT = std::fabs(trueTrim - a.trim);
            if (errR > 1e-5)
                FAIL("anchor %d dB rmsRatio: derived %.8f != spec %.8f (err %.2e)",
                     a.db, trueRatio, a.ratio, errR);
            if (errM > 1e-5)
                FAIL("anchor %d dB outputMakeup: derived %.8f != spec %.8f (err %.2e)",
                     a.db, trueMakeup, a.makeup, errM);
            if (errT > 1e-5)
                FAIL("anchor %d dB loopTrim: derived %.8f != spec %.8f (err %.2e)",
                     a.db, trueTrim, a.trim, errT);
        }
        std::printf("anchor values (9 drive levels, gate 1e-5): PASS\n");
    }

    // 4. Clamp check: k outside [1, 16] returns the table ends.
    g_section = "clamp";
    {
        CHECK(std::fabs(rmsRatio(0.5f) - kRmsRatioTable[0]) < 1e-7f);
        CHECK(std::fabs(rmsRatio(20.0f) - kRmsRatioTable[kMakeupTableSize - 1]) < 1e-7f);
        CHECK(std::fabs(outputMakeup(0.5f) - kOutputMakeupTable[0]) < 1e-7f);
        CHECK(std::fabs(loopTrim(20.0f) - kLoopTrimTable[kMakeupTableSize - 1]) < 1e-7f);
        std::printf("clamp (k<1 and k>16 return table ends): PASS\n");
    }

    std::printf("\n=== makeup_table_check OK ===\n");
    return 0;
}

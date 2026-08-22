/**
 * Correctness harness for SimdDelayLine, the scalar block delay line.
 * Validates write-before-read geometry, the dual-read sub-block crossfade,
 * the one-pole delay-position smoother, and the three interpolation modes.
 * Plain main(), exit code, always-live CHECK/FAIL.
 */

#include "dsp/SimdDelayLine.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <print>
#include <vector>

namespace
{
    using MarsDSP::Delays::Interpolation;
    using MarsDSP::Delays::SimdDelayLine;

    constexpr double kSr = 48000.0;
    constexpr int kBlock = 256;
    constexpr int kSmallBlock = 64;
    constexpr float kMaxDelayMs = 5000.0f;

    const char *g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

    // Degree-1/3/5 polynomials for the centroid test. Coefficients stay modest
    // so float32 magnitude stays O(1..1e3) over the checked range.
    double poly1(double x) noexcept { return 0.1 * x + 0.5; }
    double poly3(double x) noexcept { return 1e-4 * x * x * x - 1e-3 * x * x + 0.05 * x + 0.3; }

    double poly5(double x) noexcept
    {
        return 1e-7 * x * x * x * x * x - 1e-4 * x * x * x + 1e-3 * x * x - 0.05 * x + 0.2;
    }

    /// Feed one block of p(integer x) at a constant fractional delay pos = D + 0.5.
    /// Check the interior reproduces p(t - pos). Return the max relative error.
    double polyRepro(Interpolation mode, double (*p)(double), const char *name, float tol)
    {
        SimdDelayLine dl;
        dl.prepare(kSr, kBlock, 100.0f);
        dl.setInterpolation(mode);
        dl.reset();

        std::vector<float> in(static_cast<std::size_t>(kBlock));
        std::vector<float> wet(static_cast<std::size_t>(kBlock), 0.0f);
        for (int x = 0; x < kBlock; ++x)
            in[static_cast<std::size_t>(x)] = static_cast<float>(p(static_cast<double>(x)));

        const float pos = 20.5f;
        dl.process(in.data(), nullptr, wet.data(), nullptr, kBlock, pos, pos);

        // Safe interior: the 6 taps for output t span input indices [t-pos-3, t-pos+2].
        double maxRel = 0.0;
        for (int t = 25; t <= 120; ++t)
        {
            const double oracle = p(static_cast<double>(t) - static_cast<double>(pos));
            const double denom = std::max(1.0, std::fabs(oracle));
            const double rel = std::fabs(static_cast<double>(wet[static_cast<std::size_t>(t)]) - oracle) / denom;
            if (rel > maxRel) maxRel = rel;
            if (rel > static_cast<double>(tol))
                FAIL("{{}} t={{}} wet={{:.9g}} oracle={{:.9g}} rel={{:.3e}} tol={{:.0e}}",
                 name, t, static_cast<double>(wet[static_cast<std::size_t>(t)]), oracle, rel, static_cast<double>(tol));
        }
        std::println("  {:<14} max rel err = {:.3e} (tol {:.0e})", name, maxRel, static_cast<double>(tol));
        return maxRel;
    }

    int runAll()
    {
        // 1. Integer impulse, all 3 modes.
        g_section = "integer impulse";
        for (int mode = 0; mode < 3; ++mode)
        {
            SimdDelayLine dl;
            dl.prepare(kSr, kBlock, kMaxDelayMs);
            dl.setInterpolation(static_cast<Interpolation>(mode));
            dl.reset();

            std::vector<float> inL(static_cast<std::size_t>(kBlock), 0.0f);
            std::vector<float> inR(static_cast<std::size_t>(kBlock), 0.0f);
            std::vector<float> wetL(static_cast<std::size_t>(kBlock), 0.0f);
            std::vector<float> wetR(static_cast<std::size_t>(kBlock), 0.0f);
            inL[0] = 1.0f;
            inR[0] = 1.0f;

            const float D = 50.0f;
            dl.process(inL.data(), inR.data(), wetL.data(), wetR.data(), kBlock, D, D);

            for (int t = 0; t < kBlock; ++t)
            {
                const float exp = (t == 50) ? 1.0f : 0.0f;
                if (std::fabs(wetL[static_cast<std::size_t>(t)] - exp) > 1e-5f)
                    FAIL("mode {{}} t={{}} wetL={{}} exp={{}}", mode, t, static_cast<double>(wetL[static_cast<std::size_t>(t)]),
                     static_cast<double>(exp));
                if (std::fabs(wetR[static_cast<std::size_t>(t)] - exp) > 1e-5f)
                    FAIL("mode {{}} t={{}} wetR={{}} exp={{}}", mode, t, static_cast<double>(wetR[static_cast<std::size_t>(t)]),
                     static_cast<double>(exp));
            }
        }
        std::println("integer impulse (3 modes, D=50): PASS");

        // 2. Polynomial reproduction / centroid, all 3 modes.
        g_section = "polynomial reproduction";
        polyRepro(Interpolation::Linear, poly1, "Linear", 1e-4f);
        polyRepro(Interpolation::Lagrange3rd, poly3, "Lagrange3rd", 1e-3f);
        polyRepro(Interpolation::Lagrange5th, poly5, "Lagrange5th", 1e-2f);
        std::println("polynomial reproduction (3 modes, pos=20.5): PASS");

        // 3. Stereo independence.
        g_section = "stereo independence"; {
            SimdDelayLine dl;
            dl.prepare(kSr, kBlock, kMaxDelayMs);
            dl.setInterpolation(Interpolation::Linear);
            dl.reset();

            std::vector<float> inL(static_cast<std::size_t>(kBlock));
            std::vector<float> inR(static_cast<std::size_t>(kBlock));
            std::vector<float> wetL(static_cast<std::size_t>(kBlock), 0.0f);
            std::vector<float> wetR(static_cast<std::size_t>(kBlock), 0.0f);
            for (int x = 0; x < kBlock; ++x)
            {
                inL[static_cast<std::size_t>(x)] = static_cast<float>(x) * 0.01f;
                inR[static_cast<std::size_t>(x)] = static_cast<float>(x) * 0.01f + 1000.0f;
            }

            const float D = 40.0f;
            dl.process(inL.data(), inR.data(), wetL.data(), wetR.data(), kBlock, D, D);

            for (int t = 0; t < kBlock; ++t)
            {
                const float expL = (t >= 40) ? inL[static_cast<std::size_t>(t - 40)] : 0.0f;
                const float expR = (t >= 40) ? inR[static_cast<std::size_t>(t - 40)] : 0.0f;
                if (std::fabs(wetL[static_cast<std::size_t>(t)] - expL) > 1e-4f)
                    FAIL("t={{}} wetL={{}} exp={{}}", t, static_cast<double>(wetL[static_cast<std::size_t>(t)]),
                     static_cast<double>(expL));
                if (std::fabs(wetR[static_cast<std::size_t>(t)] - expR) > 1e-4f)
                    FAIL("t={{}} wetR={{}} exp={{}}", t, static_cast<double>(wetR[static_cast<std::size_t>(t)]),
                     static_cast<double>(expR));
                // The constant 1000.0 channel offset must survive the delay line intact.
                if (t >= 40)
                    if (std::fabs((wetR[static_cast<std::size_t>(t)] - wetL[static_cast<std::size_t>(t)]) - 1000.0f) >
                        1e-2f)
                        FAIL("t={{}} channel offset broke: L={{{{}}}} R={{{{}}}}", t,
                         static_cast<double>(wetL[static_cast<std::size_t>(t)]),
                         static_cast<double>(wetR[static_cast<std::size_t>(t)]));
            }
        }
        std::println("stereo independence (D=40): PASS");

        // 4. Multi-block delay (D > blockSize).
        g_section = "multi-block delay"; {
            SimdDelayLine dl;
            dl.prepare(kSr, kSmallBlock, kMaxDelayMs);
            dl.setInterpolation(Interpolation::Lagrange5th);
            dl.reset();

            const int blk = kSmallBlock;
            const int D = 100;
            std::vector<float> in(static_cast<std::size_t>(blk));
            std::vector<float> wet(static_cast<std::size_t>(blk));

            // Block 0: impulse at index 5 (global sample 5). Output global 105 lands
            // in block 1 at index 41, so block 0 output must be silent.
            std::fill(in.begin(), in.end(), 0.0f);
            in[5] = 1.0f;
            dl.process(in.data(), nullptr, wet.data(), nullptr, blk, static_cast<float>(D), static_cast<float>(D));
            for (int t = 0; t < blk; ++t)
                if (std::fabs(wet[static_cast<std::size_t>(t)]) > 1e-5f)
                    FAIL("block 0 t={{}} wet={{}} (expected 0)", t, static_cast<double>(wet[static_cast<std::size_t>(t)]));

            // Block 1: zero input, impulse out at index 41.
            std::fill(in.begin(), in.end(), 0.0f);
            dl.process(in.data(), nullptr, wet.data(), nullptr, blk, static_cast<float>(D), static_cast<float>(D));
            for (int t = 0; t < blk; ++t)
            {
                const float exp = (t == 41) ? 1.0f : 0.0f;
                if (std::fabs(wet[static_cast<std::size_t>(t)] - exp) > 1e-5f)
                    FAIL("block 1 t={{}} wet={{}} exp={{}}", t, static_cast<double>(wet[static_cast<std::size_t>(t)]),
                     static_cast<double>(exp));
            }
        }
        std::println("multi-block delay (blk=64, D=100): PASS");

        // 5. Delay-move settling.
        g_section = "delay-move settling"; {
            SimdDelayLine dl;
            dl.prepare(kSr, kSmallBlock, kMaxDelayMs);
            dl.setInterpolation(Interpolation::Linear);
            dl.reset();

            const int blk = kSmallBlock;
            std::vector<float> in(static_cast<std::size_t>(blk));
            std::vector<float> wet(static_cast<std::size_t>(blk));

            // Globally-continuous ramp input[g] = g * 0.001, with g a persistent global
            // sample index across both phases. The counter must not restart between
            // phases or the ramp discontinuity is reproduced D samples later.
            int g = 0;
            const auto runBlocks = [&](int D, int nBlocks)
            {
                for (int b = 0; b < nBlocks; ++b)
                {
                    for (int x = 0; x < blk; ++x, ++g)
                        in[static_cast<std::size_t>(x)] = static_cast<float>(g * 0.001);
                    dl.process(in.data(), nullptr, wet.data(), nullptr, blk, static_cast<float>(D),
                               static_cast<float>(D));
                }
            };

            // Settle at D1 = 30 (10 blocks; firstBlock snaps, the rest hold).
            const int gAfterD1 = 10 * blk;
            runBlocks(30, 10); {
                const int blockStartG = gAfterD1 - blk;
                for (int t = 0; t < blk; ++t)
                {
                    const int globalT = blockStartG + t;
                    const float exp = (globalT >= 30) ? static_cast<float>((globalT - 30) * 0.001) : 0.0f;
                    if (std::fabs(wet[static_cast<std::size_t>(t)] - exp) > 1e-4f)
                        FAIL("D1 settle t={{}} wet={{}} exp={{}}", t, static_cast<double>(wet[static_cast<std::size_t>(t)]),
                         static_cast<double>(exp));
                }
            }

            // Move to D2 = 60. The one-pole smoother (20 ms tau) is asymptotic;
            // run 250 blocks so the position is indistinguishable from 60 at float precision.
            const int d2Blocks = 250;
            runBlocks(60, d2Blocks); {
                const int blockStartG = gAfterD1 + (d2Blocks - 1) * blk;
                double maxErr = 0.0;
                for (int t = 0; t < blk; ++t)
                {
                    const int globalT = blockStartG + t;
                    const float exp = static_cast<float>((globalT - 60) * 0.001);
                    const double err = std::fabs(
                        static_cast<double>(wet[static_cast<std::size_t>(t)]) - static_cast<double>(exp));
                    if (err > maxErr) maxErr = err;
                    if (err > 1e-4)
                        FAIL("D2 settle t={{}} wet={{}} exp={{}} err={{}}", t,
                         static_cast<double>(wet[static_cast<std::size_t>(t)]), static_cast<double>(exp), err);
                }
                std::println("  D2=60 settle max abs err = {:.3e}", maxErr);
            }
        }
        std::println("delay-move settling (D1=30 -> D2=60): PASS");

        // 6. Ring-wrap endurance.
        g_section = "ring-wrap endurance"; {
            SimdDelayLine dl;
            dl.prepare(kSr, kSmallBlock, 10.0f); // small capacity (~1 wrap / 16 blocks)
            dl.setInterpolation(Interpolation::Lagrange5th);
            dl.reset();

            const int blk = kSmallBlock;
            const int D = 30;
            std::vector<float> in(static_cast<std::size_t>(blk));
            std::vector<float> wet(static_cast<std::size_t>(blk));

            const int cap = dl.getCapacity();
            CHECK(cap == 1024);

            // Period-997 pattern (997 coprime to blk=64): every sample is uniquely
            // checkable and the oracle is a closed form.
            const int totalBlocks = 4000;
            double maxErr = 0.0;
            int worstB = 0;
            int worstT = 0;
            for (int b = 0; b < totalBlocks; ++b)
            {
                for (int x = 0; x < blk; ++x)
                    in[static_cast<std::size_t>(x)] = static_cast<float>(((b * blk + x) % 997) * 0.001);

                dl.process(in.data(), nullptr, wet.data(), nullptr, blk, static_cast<float>(D), static_cast<float>(D));

                for (int t = 0; t < blk; ++t)
                {
                    const int globalT = b * blk + t;
                    const float exp = (globalT < D)
                                          ? 0.0f
                                          : static_cast<float>(((globalT - D) % 997) * 0.001);
                    const double err = std::fabs(
                        static_cast<double>(wet[static_cast<std::size_t>(t)]) - static_cast<double>(exp));
                    if (err > maxErr)
                    {
                        maxErr = err;
                        worstB = b;
                        worstT = t;
                    }
                    if (err > 1e-5)
                        FAIL("b={{}} t={{}} globalT={{}} wet={{}} exp={{}}", b, t, globalT,
                         static_cast<double>(wet[static_cast<std::size_t>(t)]), static_cast<double>(exp));
                }
            }
            const int wraps = totalBlocks * blk / cap;
            std::println("ring-wrap endurance ({} blocks, ~{} wraps): PASS (max err {:.3e} at b={} t={})",
                         totalBlocks, wraps, maxErr, worstB, worstT);
        }

        // 7. Zero-in, zero-out.
        g_section = "zero-in zero-out"; {
            SimdDelayLine dl;
            dl.prepare(kSr, kBlock, kMaxDelayMs);
            dl.setInterpolation(Interpolation::Lagrange5th);
            dl.reset();

            std::vector<float> in(static_cast<std::size_t>(kBlock), 0.0f);
            std::vector<float> wet(static_cast<std::size_t>(kBlock), 1.0f);
            dl.process(in.data(), nullptr, wet.data(), nullptr, kBlock, 137.0f, 137.0f);
            for (int t = 0; t < kBlock; ++t)
                if (wet[static_cast<std::size_t>(t)] != 0.0f)
                    FAIL("t={{}} wet={{}} (expected exactly 0)", t, static_cast<double>(wet[static_cast<std::size_t>(t)]));
        }
        std::println("zero-in zero-out: PASS");

        // 8. Mono path (inR/wetR null).
        g_section = "mono path"; {
            SimdDelayLine dl;
            dl.prepare(kSr, kBlock, kMaxDelayMs);
            dl.setInterpolation(Interpolation::Lagrange5th);
            dl.reset();

            std::vector<float> in(static_cast<std::size_t>(kBlock));
            std::vector<float> wet(static_cast<std::size_t>(kBlock), 0.0f);
            for (int x = 0; x < kBlock; ++x)
                in[static_cast<std::size_t>(x)] = static_cast<float>(x) * 0.01f;

            dl.process(in.data(), nullptr, wet.data(), nullptr, kBlock, 50.0f, 50.0f);
            for (int t = 0; t < kBlock; ++t)
            {
                const float exp = (t >= 50) ? in[static_cast<std::size_t>(t - 50)] : 0.0f;
                if (std::fabs(wet[static_cast<std::size_t>(t)] - exp) > 1e-4f)
                    FAIL("t={{}} wet={{}} exp={{}}", t, static_cast<double>(wet[static_cast<std::size_t>(t)]),
                     static_cast<double>(exp));
            }
        }
        std::println("mono path (inR/wetR null, D=50): PASS");

        return 0;
    }
} // namespace

int main()
{
    std::println("=== Chronos SimdDelayLine correctness harness ===");
    std::println("sr={:.0f}  block={}  smallBlock={}  modes={{Linear,Lagrange3rd,Lagrange5th}}",
                 kSr, kBlock, kSmallBlock);
    std::println();
    const int r = runAll();
    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

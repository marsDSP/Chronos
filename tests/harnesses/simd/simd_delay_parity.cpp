// tests/harnesses/simd/simd_delay_parity.cpp
// Parity harness: SimdDelayLine SIMD 4-wide kernel (process()) vs the scalar
// dot6 kernel (processScalar()), run on two independent but identically
// prepared instances in lockstep. The spec's Step 6 requires "parity to a
// tolerance, not bit-exact, because the MAC tree reassociates."
//
// Both paths share the write-before-read / mirror / one-pole-smoother / sub-
// block dual-read pipeline; only the per-sample evaluation differs. Because
// the two instances are fed identical input and identical per-block delay,
// their writeIdx and smoother state evolve identically - the only divergence
// is the SIMD FMA accumulation vs the scalar dot6, which this harness bounds.
//
// Sections:
//   1. Block-size sweep × 3 modes, integer delay (impulse + ramp).
//      Covers subN = 1..16: blk=1 (scalar tail only), 4 (one SIMD group),
//      7 (1 group + 3-tail), 12/16 (full groups), 17/23/24/64/100/256
//      (mixed sub-block boundaries).
//   2. Fractional delays (30.25 / 30.5 / 30.75 / 99.1), Lagrange5th, ramp.
//   3. Stereo independence (distinct L/R ramps).
//   4. Mono path (inR/wetR null).
//   5. Ring-wrap endurance (small capacity, 4000 blocks, period-997 pattern).
//   6. Delay-move crossfade region (D1=30 -> D2=60); the per-sample alpha
//      ramp is where SIMD vs scalar diverge most.
//
// Conventions (matching simd_delay_check.cpp): plain main(), exit code,
// printf, always-live CHECK/FAIL. Links SharedCode only; no JUCE. Forced
// -O2 and -Xarch_x86_64 -mfma (see tests/CMakeLists.txt) so the SIMD kernel
// inlines and simde_mm_fmadd_ps lowers to a fused multiply-add, matching the
// plugin target's x86_64 slice; arm64 has FMA unconditionally.

#include "dsp/SimdDelayLine.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <vector>

namespace
{
    using MarsDSP::Delays::Interpolation;
    using MarsDSP::Delays::SimdDelayLine;

    constexpr double kSr = 48000.0;
    constexpr float kTol = 1e-5f; // abs; covers float32 MAC reassociation on O(1) signals

    const char *g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

    struct Result
    {
        std::vector<float> wetL;
        std::vector<float> wetR;
    };

    // Run one evaluation path (simd=true → process(), false → processScalar()) on a
    // full input split into nBlocks of blk samples, with a per-block delay. The two
    // paths are run on separate, identically prepared instances.
    Result runPath(bool simd, int blk, float maxDelayMs,
                   const std::vector<float> &delays,
                   Interpolation mode,
                   const std::vector<float> &inL,
                   const std::vector<float> &inR,
                   bool stereo)
    {
        const auto nBlocks = static_cast<int>(delays.size());
        SimdDelayLine dl;
        dl.prepare(kSr, blk, maxDelayMs);
        dl.setInterpolation(mode);
        dl.reset();

        Result r;
        r.wetL.resize(static_cast<std::size_t>(nBlocks) * blk);
        if (stereo) r.wetR.resize(static_cast<std::size_t>(nBlocks) * blk);

        std::vector<float> inBlk(static_cast<std::size_t>(blk));
        std::vector<float> wetBlk(static_cast<std::size_t>(blk));
        std::vector<float> inBlkR;
        std::vector<float> wetBlkR;
        if (stereo)
        {
            inBlkR.resize(static_cast<std::size_t>(blk));
            wetBlkR.resize(static_cast<std::size_t>(blk));
        }

        for (int b = 0; b < nBlocks; ++b)
        {
            for (int x = 0; x < blk; ++x)
                inBlk[static_cast<std::size_t>(x)] = inL[static_cast<std::size_t>(b * blk + x)];
            if (stereo)
                for (int x = 0; x < blk; ++x)
                    inBlkR[static_cast<std::size_t>(x)] = inR[static_cast<std::size_t>(b * blk + x)];

            const float d = delays[static_cast<std::size_t>(b)];
            const float *inRp = stereo ? inBlkR.data() : nullptr;
            float *wetRp = stereo ? wetBlkR.data() : nullptr;

            if (simd)
                dl.process(inBlk.data(), inRp, wetBlk.data(), wetRp, blk, d, d);
            else
                dl.processScalar(inBlk.data(), inRp, wetBlk.data(), wetRp, blk, d, d);

            for (int x = 0; x < blk; ++x)
                r.wetL[static_cast<std::size_t>(b * blk + x)] = wetBlk[static_cast<std::size_t>(x)];
            if (stereo)
                for (int x = 0; x < blk; ++x)
                    r.wetR[static_cast<std::size_t>(b * blk + x)] = wetBlkR[static_cast<std::size_t>(x)];
        }
        return r;
    }

    // Compare two results, return max abs diff, FAIL on tolerance breach.
    double compare(const Result &a, const Result &b, bool stereo, const char *section)
    {
        g_section = section;
        CHECK(a.wetL.size() == b.wetL.size());
        double maxErr = 0.0;
        for (std::size_t i = 0; i < a.wetL.size(); ++i)
        {
            const double e = std::fabs(static_cast<double>(a.wetL[i]) - static_cast<double>(b.wetL[i]));
            if (e > maxErr) maxErr = e;
            if (e > static_cast<double>(kTol))
                FAIL("L i={} simd={:.9} scalar={:.9} e={:.3} tol={:.0}",
                 i, static_cast<double>(a.wetL[i]), static_cast<double>(b.wetL[i]), e, static_cast<double>(kTol));
        }
        if (stereo)
        {
            CHECK(a.wetR.size() == b.wetR.size());
            for (std::size_t i = 0; i < a.wetR.size(); ++i)
            {
                const double e = std::fabs(static_cast<double>(a.wetR[i]) - static_cast<double>(b.wetR[i]));
                if (e > maxErr) maxErr = e;
                if (e > static_cast<double>(kTol))
                    FAIL("R i={} simd={:.9} scalar={:.9} e={:.3} tol={:.0}",
                     i, static_cast<double>(a.wetR[i]), static_cast<double>(b.wetR[i]), e, static_cast<double>(kTol));
            }
        }
        return maxErr;
    }

    const char *modeName(Interpolation m) noexcept
    {
        switch (m)
        {
            case Interpolation::Linear: return "Linear";
            case Interpolation::Lagrange3rd: return "Lag3";
            case Interpolation::Lagrange5th: return "Lag5";
        }
        return "?";
    }

    int runAll()
    {
        const std::array<int, 11> blks { { 1, 4, 7, 12, 16, 17, 23, 24, 64, 100, 256 } };
        const Interpolation modes[] = {Interpolation::Linear, Interpolation::Lagrange3rd, Interpolation::Lagrange5th};

        // 1. Block-size sweep × 3 modes, integer delay
        g_section = "block-size sweep (impulse)";
        for (int blk: blks)
        {
            for (Interpolation m: modes)
            {
                const int nBlocks = 4;
                std::vector<float> inL(static_cast<std::size_t>(nBlocks * blk), 0.0f);
                inL[0] = 1.0f; // impulse at global sample 0
                std::vector<float> delays(static_cast<std::size_t>(nBlocks), 30.0f);
                const Result rs = runPath(true, blk, 100.0f, delays, m, inL, {}, false);
                const Result rv = runPath(false, blk, 100.0f, delays, m, inL, {}, false);
                const double e = compare(rs, rv, false, "block-size sweep (impulse)");
                if (blk == 256 || blk == 1 || blk == 7)
                    std::println("  blk={:<3} {:<4} impulse: max abs diff = {:.3}", blk, modeName(m), e);
            }
        }
        std::println("block-size sweep × 3 modes (impulse, D=30): PASS");

        // 2. Fractional delays, Lagrange5th, ramp
        g_section = "fractional delays"; {
            const int blk = 256;
            const int nBlocks = 4;
            std::vector<float> inL(static_cast<std::size_t>(nBlocks * blk));
            for (int i = 0; i < nBlocks * blk; ++i)
                inL[static_cast<std::size_t>(i)] = static_cast<float>(i) * 0.01f;
            const std::array<float, 4> fracDelays { { 30.25f, 30.5f, 30.75f, 99.1f } };
            for (float d: fracDelays)
            {
                std::vector<float> delays(static_cast<std::size_t>(nBlocks), d);
                const Result rs = runPath(true, blk, 100.0f, delays, Interpolation::Lagrange5th, inL, {}, false);
                const Result rv = runPath(false, blk, 100.0f, delays, Interpolation::Lagrange5th, inL, {}, false);
                const double e = compare(rs, rv, false, "fractional delays");
                std::println("  D={:.2} ramp: max abs diff = {:.3}", static_cast<double>(d), e);
            }
        }
        std::println("fractional delays (Lag5, ramp): PASS");

        // 3. Stereo independence
        g_section = "stereo"; {
            const int blk = 256;
            const int nBlocks = 4;
            std::vector<float> inL(static_cast<std::size_t>(nBlocks * blk));
            std::vector<float> inR(static_cast<std::size_t>(nBlocks * blk));
            for (int i = 0; i < nBlocks * blk; ++i)
            {
                inL[static_cast<std::size_t>(i)] = static_cast<float>(i) * 0.01f;
                inR[static_cast<std::size_t>(i)] = static_cast<float>(i) * 0.01f + 1.0f; // distinct channel
            }
            std::vector<float> delays(static_cast<std::size_t>(nBlocks), 40.5f);
            const Result rs = runPath(true, blk, 100.0f, delays, Interpolation::Lagrange5th, inL, inR, true);
            const Result rv = runPath(false, blk, 100.0f, delays, Interpolation::Lagrange5th, inL, inR, true);
            const double e = compare(rs, rv, true, "stereo");
            std::println("  stereo D=40.5: max abs diff = {:.3}", e);
        }
        std::println("stereo independence (D=40.5): PASS");

        // 4. Mono path (inR/wetR null)
        g_section = "mono"; {
            const int blk = 256;
            const int nBlocks = 4;
            std::vector<float> inL(static_cast<std::size_t>(nBlocks * blk));
            for (int i = 0; i < nBlocks * blk; ++i)
                inL[static_cast<std::size_t>(i)] = static_cast<float>(i) * 0.01f;
            std::vector<float> delays(static_cast<std::size_t>(nBlocks), 50.5f);
            const Result rs = runPath(true, blk, 100.0f, delays, Interpolation::Lagrange5th, inL, {}, false);
            const Result rv = runPath(false, blk, 100.0f, delays, Interpolation::Lagrange5th, inL, {}, false);
            const double e = compare(rs, rv, false, "mono");
            std::println("  mono D=50.5: max abs diff = {:.3}", e);
        }
        std::println("mono path (inR/wetR null): PASS");

        // 5. Ring-wrap endurance
        g_section = "ring-wrap"; {
            const int blk = 64;
            const int nBlocks = 4000;
            std::vector<float> inL(static_cast<std::size_t>(nBlocks * blk));
            for (int i = 0; i < nBlocks * blk; ++i)
                inL[static_cast<std::size_t>(i)] = static_cast<float>((i % 997) * 0.001); // period-997, O(1)
            std::vector<float> delays(static_cast<std::size_t>(nBlocks), 30.0f);
            const Result rs = runPath(true, blk, 10.0f, delays, Interpolation::Lagrange5th, inL, {}, false);
            const Result rv = runPath(false, blk, 10.0f, delays, Interpolation::Lagrange5th, inL, {}, false);
            const double e = compare(rs, rv, false, "ring-wrap");
            std::println("  ring-wrap (cap=512, 4000 blocks, ~500 wraps): max abs diff = {:.3}", e);
        }
        std::println("ring-wrap endurance: PASS");

        // 6. Delay-move crossfade region
        g_section = "delay-move"; {
            const int blk = 64;
            const int settle = 10, move = 250;
            const int nBlocks = settle + move;
            std::vector<float> inL(static_cast<std::size_t>(nBlocks * blk));
            for (int i = 0; i < nBlocks * blk; ++i)
                inL[static_cast<std::size_t>(i)] = static_cast<float>(i) * 0.001f; // globally-continuous ramp
            std::vector<float> delays(static_cast<std::size_t>(nBlocks));
            for (int b = 0; b < settle; ++b) delays[static_cast<std::size_t>(b)] = 30.0f;
            for (int b = settle; b < nBlocks; ++b) delays[static_cast<std::size_t>(b)] = 60.0f;
            const Result rs = runPath(true, blk, 5000.0f, delays, Interpolation::Lagrange5th, inL, {}, false);
            const Result rv = runPath(false, blk, 5000.0f, delays, Interpolation::Lagrange5th, inL, {}, false);
            const double e = compare(rs, rv, false, "delay-move");
            std::println("  delay-move (D1=30 -> D2=60, {} blocks): max abs diff = {:.3}", nBlocks, e);
        }
        std::println("delay-move crossfade region: PASS");

        return 0;
    }
} // namespace

int main()
{
    std::println("=== Chronos SimdDelayLine SIMD-vs-scalar parity harness ===");
    std::println("sr={:.0}  tol={:.0}  (process = SIMD, processScalar = reference)\n", kSr, static_cast<double>(kTol));

    int r = runAll();

    std::println("\n=== {} ===", r == 0 ? "PARITY HELD (within tol)" : "PARITY FAILED");
    return r;
}

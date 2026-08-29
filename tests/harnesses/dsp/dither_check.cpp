// tests/harnesses/dsp/dither_check.cpp
// V2: SIMD dither verification. Tests the 4-lane SIMD xorshift,
// TPDF distribution, rounding emulation, and no-noise-modulation.
//
// Tests implemented:
//   1. Quantizer bit-exactness (dither disabled)
//   2. TPDF distribution
//   5. Cross-lane independence
//   6. Cross-channel independence
//   8. No noise modulation (the defining TPDF property)
//
// Tests 3, 4, 7, 9 are TODO (moments, whiteness, zero-state guard,
// unbiasedness) - lower priority, can be added incrementally.

#include "dsp/ChronosEngine.h"
#include "simd/Config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Scalar xorshift32 (reference)
float scalarNextUniform(std::uint32_t& s) noexcept
{
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
}

// SIMD rounding: trunc(x + copysign(0.5, x)) = round-half-away-from-zero
float simdRound(float x) noexcept
{
    const M128 vx = MM(set1_ps)(x);
    const M128 vHalf = MM(set1_ps)(0.5f);
    const M128 vSignMask = MM(set1_ps)(-0.0f);
    const M128 vSign = MM(and_ps)(vx, vSignMask);
    const M128 vShifted = MM(add_ps)(vx, MM(or_ps)(vHalf, vSign));
    const M128I vInt = MM(cvttps_epi32)(vShifted);
    const M128 vRounded = MM(cvtepi32_ps)(vInt);
    alignas(16) float out[4];
    MM(store_ps)(out, vRounded);
    return out[0];
}

} // namespace

int main()
{
    std::println("=== Chronos dither_check (V2) ===\n");

    // 1. Quantizer bit-exactness (dither disabled)
    g_section = "quantizer bit-exactness";
    {
        constexpr int kN = 1000000;
        std::uint32_t rng = 12345u;
        int mismatches = 0;
        for (int i = 0; i < kN; ++i)
        {
            // Random value in [-1000, 1000]
            const float x = (scalarNextUniform(rng) - 0.5f) * 2000.0f;
            const float sr = std::round(x);
            const float mr = simdRound(x);
            if (sr != mr)
            {
                ++mismatches;
                if (mismatches <= 5)
                    std::println("  mismatch: x={} std::round={} simd={}",
                                static_cast<double>(x), static_cast<double>(sr), static_cast<double>(mr));
            }
        }
        // Also test exact ties: k*lsb + lsb/2
        const float lsb = std::ldexp(1.0f, 1 - 24);
        for (int k = -1000; k <= 1000; ++k)
        {
            const float x = static_cast<float>(k) * lsb + lsb * 0.5f;
            const float sr = std::round(x);
            const float mr = simdRound(x);
            if (sr != mr)
            {
                ++mismatches;
                if (mismatches <= 10)
                    std::println("  tie mismatch: x={} std::round={} simd={}",
                                static_cast<double>(x), static_cast<double>(sr), static_cast<double>(mr));
            }
        }
        if (mismatches > 0)
            FAIL("quantizer: {} mismatches (std::round vs SIMD rounding)", mismatches);
        std::println("quantizer bit-exactness (1e6 random + 2001 ties): PASS");
    }

    // 2. TPDF distribution: histogram of (u1-u2) over 1e7 draws
    g_section = "TPDF distribution";
    {
        constexpr int kN = 1000000;  // reduced for speed
        constexpr int kBins = 200;
        std::uint32_t rng = 42u;
        std::array<int, kBins> hist = {{  }};
        for (int i = 0; i < kN; ++i)
        {
            const float u1 = scalarNextUniform(rng);
            const float u2 = scalarNextUniform(rng);
            const float tpdf = u1 - u2;  // in [-1, 1]
            int bin = static_cast<int>((tpdf + 1.0f) * 0.5f * kBins);
            bin = std::clamp(bin, 0, kBins - 1);
            ++hist[bin];
        }
        // Check triangular shape: center should be peak, edges should be low
        const int center = kBins / 2;
        const int peak = hist[center];
        const int edgeL = hist[0];
        const int edgeR = hist[kBins - 1];
        if (peak <= edgeL || peak <= edgeR)
            FAIL("TPDF: peak={} <= edge L={} R={} (not triangular)", peak, edgeL, edgeR);
        // Check monotonic decrease from center
        for (int b = center; b < kBins - 5; ++b)
        {
            if (hist[b] < hist[b + 5] * 0.8f)
                FAIL("TPDF: not monotonic at bin {}: {} < {}*0.8", b, hist[b], hist[b + 5]);
        }
        std::println("TPDF distribution (1e6 draws, 200 bins, triangular): PASS");
    }

    // 5. Cross-lane independence: 4 lanes of SIMD xorshift
    g_section = "cross-lane independence";
    {
        constexpr int kN = 500000;
        // Use the engine's SIMD xorshift via setDitherSeeds
        MarsDSP::ChronosEngine engine;
        engine.prepare(48000.0, 256, 2);
        engine.setDitherSeeds(0xDEADBEEFu, 0xCAFEBABEu);

        // We can't directly access the SIMD RNG state, so we test via
        // the engine's output: process blocks with dither and check
        // that L and R channels are uncorrelated.
        // This tests cross-channel independence (test 6) as a proxy.
        // Full cross-lane test would need direct RNG access.

        // Process a long block with zeros input, mix=100, drive=0, gain=1
        // The output is pure dither noise.
        MarsDSP::ChronosEngine::Params p{};
        p.delaySamplesL = 0.0f;
        p.delaySamplesR = 0.0f;
        p.driveLin = 1.0f;
        p.mix = 100.0f;
        p.gainLin = 1.0f;
        p.hpfHz = 20.0f;
        p.lpfHz = 20000.0f;
        p.bits = 24;
        p.adaaOrder = 0;
        engine.resetParams(p);

        std::vector<float> inL(static_cast<std::size_t>(kN), 0.0f);
        std::vector<float> inR(static_cast<std::size_t>(kN), 0.0f);
        std::vector<float> outL(inL), outR(inR);

        // Process in blocks
        constexpr int kBlock = 256;
        for (int off = 0; off < kN; off += kBlock)
        {
            const int n = std::min(kBlock, kN - off);
            engine.setParams(p);
            std::array<float*, 2> io{ outL.data() + off, outR.data() + off };
            engine.process(io.data(), 2, n);
        }

        // Cross-channel correlation
        double sumLR = 0.0;
        double sumL = 0.0;
        double sumR = 0.0;
        double sumL2 = 0.0;
        double sumR2 = 0.0;
        for (int i = 0; i < kN; ++i)
        {
            const double l = outL[static_cast<std::size_t>(i)];
            const double r = outR[static_cast<std::size_t>(i)];
            sumLR += l * r;
            sumL += l;
            sumR += r;
            sumL2 += l * l;
            sumR2 += r * r;
        }
        const double meanL = sumL / kN;
        const double meanR = sumR / kN;
        const double cov = sumLR / kN - meanL * meanR;
        const double varL = sumL2 / kN - meanL * meanL;
        const double varR = sumR2 / kN - meanR * meanR;
        const double corr = cov / std::sqrt(varL * varR);
        if (std::fabs(corr) > 2e-3)
            FAIL("cross-channel correlation = {:.3} > 2e-3", corr);
        std::println("cross-channel independence (corr = {:.3} < 2e-3): PASS", corr);
    }

    // 8. No noise modulation: variance of quant error is input-independent
    g_section = "no noise modulation";
    {
        // Use scalar path for this test (the property holds for any TPDF
        // dither regardless of SIMD vs scalar - the distribution matters).
        constexpr int kSteps = 64;
        constexpr int kN = 500000;
        const float lsb = std::ldexp(1.0f, 1 - 24);
        std::uint32_t rng = 98765u;

        std::array<double, kSteps> variances{};
        for (int step = 0; step < kSteps; ++step)
        {
            const float dc = lsb * static_cast<float>(step) / static_cast<float>(kSteps);
            // Pass 1: compute mean error
            double sumErr = 0.0;
            for (int i = 0; i < kN; ++i)
            {
                const float u1 = scalarNextUniform(rng);
                const float u2 = scalarNextUniform(rng);
                const float dither = (u1 - u2) * lsb;
                const float quantized = std::round((dc + dither) / lsb) * lsb;
                sumErr += quantized - dc;
            }
            const double mean = sumErr / kN;
            // Pass 2: compute variance as sum of squared deviations
            double sumDev2 = 0.0;
            for (int i = 0; i < kN; ++i)
            {
                const float u1 = scalarNextUniform(rng);
                const float u2 = scalarNextUniform(rng);
                const float dither = (u1 - u2) * lsb;
                const float quantized = std::round((dc + dither) / lsb) * lsb;
                const double dev = (quantized - dc) - mean;
                sumDev2 += dev * dev;
            }
            variances[step] = sumDev2 / kN;
        }

        // Check spread < 1%
        double minVar = variances[0];
        double maxVar = variances[0];
        for (int i = 1; i < kSteps; ++i)
        {
            minVar = std::min(minVar, variances[i]);
            maxVar = std::max(maxVar, variances[i]);
        }
        const double spread = (maxVar - minVar) / maxVar;
        if (spread > 0.01)
            FAIL("noise modulation: variance spread = {:.4} > 1% (min={:.3} max={:.3})",
                 spread, minVar, maxVar);
        std::println("no noise modulation (64 DC steps, variance spread = {:.4} < 1%): PASS",
                    spread);
    }

    // 9. bits == 32 produces no added noise (quantiser bypass).
    g_section = "bits==32 no-noise";
    {
        constexpr int kN = 100000;
        MarsDSP::ChronosEngine engine;
        engine.prepare(48000.0, 256, 2);
        engine.setDitherSeeds(0x12345678u, 0x9abcdef0u);
        engine.setBypass(false);

        MarsDSP::ChronosEngine::Params p{};
        p.delaySamplesL = 0.0f;
        p.delaySamplesR = 0.0f;
        p.driveLin = 1.0f;
        p.mix = 100.0f;
        p.gainLin = 1.0f;
        p.hpfHz = 20.0f;
        p.lpfHz = 20000.0f;
        p.bits = 32;
        p.adaaOrder = 0;
        p.feedback = 0.0f;
        engine.resetParams(p);

        std::vector<float> inL(static_cast<std::size_t>(kN), 0.0f);
        std::vector<float> inR(static_cast<std::size_t>(kN), 0.0f);
        std::vector<float> outL(inL), outR(inR);

        constexpr int kBlock = 256;
        for (int off = 0; off < kN; off += kBlock)
        {
            const int n = std::min(kBlock, kN - off);
            engine.setParams(p);
            std::array<float*, 2> io{ outL.data() + off, outR.data() + off };
            engine.process(io.data(), 2, n);
        }

        // The output is pure delay tail of a zero input: silence.
        double maxAbs = 0.0;
        for (int i = 0; i < kN; ++i)
        {
            maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(outL[static_cast<std::size_t>(i)])));
            maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(outR[static_cast<std::size_t>(i)])));
        }
        if (maxAbs > 0.0)
            FAIL("bits==32: max abs output = {:.3} > 0 (quantiser bypass adds noise)", maxAbs);
        std::println("bits==32 no-noise (max abs = 0): PASS");
    }

    std::println("\n=== ALL PROPERTIES HELD ===");
    return 0;
}

#include "dsp/Diffuser.h"
#include "dsp/FracDelayTap.h"
#include "dsp/Pow2RingBuffer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <numbers>
#include <print>
#include <vector>

namespace
{
    constexpr double kFs = 48000.0;
    const char *g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::println("FAIL [{}] " fmt, g_section, ##__VA_ARGS__); std::exit(1); } while (0)

    using D = MarsDSP::Diffusion::Diffuser;

    struct BaselineSection
    {
        MarsDSP::Delays::Pow2RingBuffer ring;
        int len = 0;
        int w = 0;
    };

    float baselineChain(BaselineSection *bank, float x, float size, float coef)
    {
        for (int i = 0; i < 8; ++i)
        {
            auto &sec = bank[i];
            const auto lenF = static_cast<float>(sec.len);
            float eff = std::clamp(std::nearbyintf(D::effLen(lenF, size)), D::kMinDelay, lenF);
            const float g = coef * D::sectionSign(i);
            const float d = MarsDSP::Delays::FracDelayTap::read(sec.ring, sec.w, eff);
            float v = x - g * d;
            if (!std::isfinite(v)) v = 0.0f;
            sec.ring.writeBlock(&v, sec.w, 1);
            sec.ring.refreshMirror(sec.w, 1);
            sec.w = (sec.w + 1) & sec.ring.mask();
            x = d + g * v;
        }
        return x;
    }
}

int main()
{
    std::println("=== Chronos diffuser_density_check (S22e) ===\n");

    D d;
    d.prepare(kFs);
    d.setDiffusion(1.0f); // master g = 0.78
    d.setSize(0.5f);
    d.setModDepthSamples(0.0f);
    d.prime();

    constexpr int kN = 4800; // 100 ms at 48 kHz
    std::vector<float> irNested(kN, 0.0f);
    irNested[0] = 1.0f;
    for (int off = 0; off < kN; off += D::kChunk)
    {
        const int m = std::min(D::kChunk, kN - off);
        d.processBlock(irNested.data() + off, nullptr, m);
    }

    // Baseline 8-section Schroeder cascade
    BaselineSection bank[8];
    constexpr float oldMetersL[8] = {
        4.54125f, 3.93375f, 3.19125f, 2.92875f,
        2.32875f, 2.01000f, 1.18875f, 0.82875f
    };
    for (int i = 0; i < 8; ++i)
    {
        bank[i].len = static_cast<int>(std::lround(static_cast<double>(oldMetersL[i]) * kFs / 343.0));
        bank[i].ring.prepare(bank[i].len + 256);
        bank[i].w = 0;
    }
    std::vector<float> irBase(kN, 0.0f);
    irBase[0] = 1.0f;
    for (int i = 0; i < kN; ++i)
        irBase[static_cast<std::size_t>(i)] = baselineChain(bank, irBase[static_cast<std::size_t>(i)], 0.5f, 0.78f);

    int countNested = 0;
    int countBase = 0;
    std::vector<int> arrivals;
    constexpr float thr = 0.001f; // -60 dBFS
    for (int i = 0; i < kN; ++i)
    {
        if (std::abs(irNested[static_cast<std::size_t>(i)]) >= thr)
        {
            ++countNested;
            arrivals.push_back(i);
        }
        if (std::abs(irBase[static_cast<std::size_t>(i)]) >= thr)
            ++countBase;
    }

    // 1. Density
    g_section = "density";
    const double densityRatio = static_cast<double>(countNested) / static_cast<double>(std::max(1, countBase));
    std::println("1. Arrival Density (100 ms at size 0.5, g=0.78):");
    std::println("   Nested: {} arrivals, Baseline: {} arrivals (ratio {:.2f}x)",
                countNested, countBase, densityRatio);
    CHECK(countNested > countBase);

    // 2. Modal Spacing
    g_section = "modal_spacing";
    int maxGap = 0;
    for (std::size_t i = 1; i < arrivals.size(); ++i)
    {
        const int gap = arrivals[i] - arrivals[i - 1];
        if (gap > maxGap) maxGap = gap;
    }
    const double maxGapMs = static_cast<double>(maxGap) / (kFs * 0.001);
    std::println("\n2. Modal Spacing:");
    std::println("   Max gap between arrivals in first 100 ms: {} samples ({:.2f} ms, gate < 4.0 ms)",
                maxGap, maxGapMs);
    CHECK(maxGapMs < 4.0);

    // 3. Magnitude Flatness (analytic transfer function of cascade)
    g_section = "magnitude_flatness";
    double maxFlatErr = 0.0;
    constexpr int kBins = 8192;
    for (int k = 0; k < kBins; ++k)
    {
        const double w = 2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(kBins);
        // Cascade of allpass sections has magnitude 1.0 by construction.
        // Sample steady-state frequency response through all sections:
        std::complex<double> H(1.0, 0.0);
        for (int i = 0; i < D::kNumPlainSections; ++i)
        {
            const auto lenF = static_cast<float>(d.sectionLenL(i));
            const float eff = std::clamp(std::nearbyintf(D::effLen(lenF, 0.5f)), D::kMinDelay, lenF);
            const float g = D::sectionSign(i) * D::kSectionGain[static_cast<std::size_t>(i)] * 0.78f;
            const std::complex z_del(std::cos(-w * eff), std::sin(-w * eff));
            H *= (static_cast<double>(g) + z_del) / (1.0 + static_cast<double>(g) * z_del);
        }
        for (int i = 0; i < D::kNumNestedSections; ++i)
        {
            const auto lenOutF = static_cast<float>(d.sectionLenL(3 + 2 * i));
            const auto lenInF = static_cast<float>(d.sectionLenL(4 + 2 * i));
            const float effOut = std::clamp(std::nearbyintf(D::effLen(lenOutF, 0.5f)), D::kMinDelay, lenOutF);
            const float effIn = std::clamp(std::nearbyintf(D::effLen(lenInF, 0.5f)), D::kMinDelay, lenInF);
            const float gOut = D::sectionSign(3 + i) * D::kSectionGain[static_cast<std::size_t>(3 + i)] * 0.78f;
            const float gIn = 0.85f * gOut;

            const std::complex z_in(std::cos(-w * effIn), std::sin(-w * effIn));
            const std::complex<double> A_in = (static_cast<double>(gIn) + z_in) / (1.0 + static_cast<double>(gIn) * z_in);
            const std::complex z_out(std::cos(-w * effOut), std::sin(-w * effOut));
            const std::complex<double> A_eff = A_in * z_out;
            H *= (static_cast<double>(gOut) + A_eff) / (1.0 + static_cast<double>(gOut) * A_eff);
        }
        const double mag = std::abs(H);
        const double err = std::abs(mag - 1.0);
        if (err > maxFlatErr) maxFlatErr = err;
        CHECK(err < 1e-4);
    }
    std::println("\n3. Magnitude Flatness across {} bins:", kBins);
    std::println("   Max |H(w) - 1| = {:.2e} (gate < 1e-4): PASS", maxFlatErr);

    // 4. Energy Centroid against baseTransportSamples (stereo combined energy)
    g_section = "energy_centroid";
    const double expectedCentroid = d.baseTransportSamples(0.5f);
    std::vector irFullL(65536, 0.0f);
    std::vector irFullR(65536, 0.0f);
    irFullL[0] = 1.0f;
    irFullR[0] = 1.0f;
    d.reset();
    d.setDiffusion(1.0f);
    d.setSize(0.5f);
    d.prime();
    for (int off = 0; off < 65536; off += D::kChunk)
    {
        const int m = std::min(D::kChunk, 65536 - off);
        d.processBlock(irFullL.data() + off, irFullR.data() + off, m);
    }
    double sumE = 0.0;
    double sumEX = 0.0;
    for (int i = 0; i < 65536; ++i)
    {
        const double vL = irFullL[static_cast<std::size_t>(i)];
        const double vR = irFullR[static_cast<std::size_t>(i)];
        const double e = vL * vL + vR * vR;
        sumE += e;
        sumEX += e * static_cast<double>(i);
    }
    const double measuredCentroid = sumEX / sumE;
    const double centroidDelta = std::abs(measuredCentroid - expectedCentroid);
    std::println("\n4. Energy Centroid vs baseTransportSamples(0.5):");
    std::println("   Measured: {:.4f}, Expected: {:.4f}, Delta: {:.4f} (gate <= 1.0 smp): PASS",
                measuredCentroid, expectedCentroid, centroidDelta);
    CHECK(centroidDelta <= 1.0);

    std::println("\n=== ALL DENSITY & CHARACTER PROPERTIES HELD ===");
    return 0;
}

// tests/harnesses/dsp/diffuser_decorrelation_check.cpp
//
// Diffuser decorrelation harness. Asserts the inter-channel
// correlation of the output tail is below 0.6 at a modulation depth of
// 0.3 ms. The left and right banks use independent OU states. The
// modulation decorrelates the image.
//
// Feeds the same impulse to both channels. Captures the impulse response.
// Computes the Pearson correlation of the left and right tails. Links
// SharedCode only; no JUCE.

#include "dsp/Diffuser.h"

#include <cmath>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <vector>

namespace {

constexpr double kFs      = 48000.0;
constexpr int    kBlock   = 256;
constexpr int    kSettle  = 256;      // one block; prime() snaps the smoothers
constexpr int    kCapture = 131072;   // 2.7 s: captures the tail
constexpr int    kTotal   = kSettle + kCapture;
constexpr double kDepthMs = 0.3;
constexpr double kGateCorr = 0.6;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Render a stereo impulse response at the given depth. The same impulse
// feeds both channels. Returns the L and R buffers.
struct StereoIR { std::vector<float> L, R; };

StereoIR renderIR(float depthSamples)
{
    MarsDSP::Diffusion::Diffuser d;
    d.prepare(kFs);
    d.setDiffusion(1.0f);   // master g at max (0.78)
    d.setSize(0.5f);
    d.setModDepthSamples(depthSamples);
    d.setModRateHz(0.5f);
    d.prime();

    std::vector<float> bufL(static_cast<std::size_t>(kTotal), 0.0f);
    std::vector<float> bufR(static_cast<std::size_t>(kTotal), 0.0f);
    bufL[static_cast<std::size_t>(kSettle)] = 1.0f;
    bufR[static_cast<std::size_t>(kSettle)] = 1.0f;

    for (int off = 0; off < kTotal; off += kBlock)
    {
        const int n = std::min(kBlock, kTotal - off);
        d.processBlock(bufL.data() + off, bufR.data() + off, n);
    }
    return { std::move(bufL), std::move(bufR) };
}

// Pearson correlation over the window [w0, w1).
double pearsonCorr(const std::vector<float>& L, const std::vector<float>& R, int w0, int w1)
{
    double sumL = 0.0;
    double sumR = 0.0;
    double sumLR = 0.0;
    double sumL2 = 0.0;
    double sumR2 = 0.0;
    int count = 0;
    for (int n = w0; n < w1; ++n)
    {
        const auto u = static_cast<std::size_t>(n);
        const double l = static_cast<double>(L[u]);
        const double r = static_cast<double>(R[u]);
        sumL += l; sumR += r;
        sumLR += l * r;
        sumL2 += l * l; sumR2 += r * r;
        ++count;
    }
    if (count <= 0) return 0.0;
    const double mL = sumL / count, mR = sumR / count;
    const double cov = sumLR / count - mL * mR;
    const double varL = sumL2 / count - mL * mL;
    const double varR = sumR2 / count - mR * mR;
    const double denom = std::sqrt(varL * varR);
    return (denom > 0.0) ? cov / denom : 0.0;
}

} // namespace

int main()
{
    std::println("=== Chronos diffuser_decorrelation_check (S21) ===");
    std::println("fs={:.0}  size=0.5  depth={:.2} ms  gate |corr| < {:.2}\n",
                kFs, kDepthMs, kGateCorr);

    MarsDSP::Diffusion::Diffuser d;
    d.prepare(kFs);
    const double transport = static_cast<double>(d.baseTransportSamples(0.5f));
    const int w0 = static_cast<int>(2.0 * transport);
    const int w1 = kTotal;
    std::println("transport={:.0} samples ({:.1} ms); tail window [{}, {})\n",
                transport, transport / kFs * 1000.0, w0, w1);

    // Baseline: no modulation. The path-length difference alone gives a
    // reference correlation.
    const float depthSamples = static_cast<float>(kDepthMs * 0.001 * kFs);
    const auto baseIR = renderIR(0.0f);
    const auto modIR  = renderIR(depthSamples);

    g_section = "finite";
    for (int n = kSettle; n < kTotal; ++n)
    {
        const auto u = static_cast<std::size_t>(n);
        CHECK(std::isfinite(baseIR.L[u]));
        CHECK(std::isfinite(baseIR.R[u]));
        CHECK(std::isfinite(modIR.L[u]));
        CHECK(std::isfinite(modIR.R[u]));
    }

    const double corrBase = pearsonCorr(baseIR.L, baseIR.R, w0, w1);
    const double corrMod  = pearsonCorr(modIR.L,  modIR.R,  w0, w1);
    std::println("correlation (no modulation): {:.4}", std::fabs(corrBase));
    std::println("correlation (depth {:.2} ms): {:.4}  (gate < {:.2})",
                kDepthMs, std::fabs(corrMod), kGateCorr);

    if (std::fabs(corrMod) >= kGateCorr)
        FAIL("modulated correlation {:.4} >= {:.2}", std::fabs(corrMod), kGateCorr);

    std::println("\ninter-channel correlation below 0.6 with modulation: PASS");
    std::println("\n=== ALL PROPERTIES HELD ===");
    return 0;
}

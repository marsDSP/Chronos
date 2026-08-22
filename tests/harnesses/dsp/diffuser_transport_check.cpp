// tests/harnesses/dsp/diffuser_transport_check.cpp
//
// Diffuser size-direction + transport-centroid harness (S4).
//
// For each size in {0, 0.25, 0.5, 0.75, 1} and each allpass coefficient in
// {0, 0.4, 0.78}, measures the energy centroid of the 8-section cascade
// impulse response and compares it against Diffuser::baseTransportSamples().
// The average group delay of an allpass of order N is exactly N, so the
// cascade's energy centroid is the sum of the section delays at every g.
//
// Two gates:
//   (1) |measured_centroid - baseTransportSamples(size)| <= 1 sample, at
//       every size x coefficient.
//   (2) the transport is strictly increasing in size (locks the direction:
//       size 0 = shortest path, size 1 = full path).
//
// The centroid is measured over the combined L+R energy. Both banks are
// allpass with equal impulse energy, so the combined-energy centroid is the
// mean of the two bank centroids, which is exactly baseTransportSamples
// (the mean of baseTransportSamplesLR). Links SharedCode only; no JUCE.

#include "dsp/Diffuser.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <vector>

namespace {

constexpr double kFs       = 48000.0;
constexpr int    kBlock    = 256;
constexpr int    kSettle   = 256;     // one block; prime() snaps the smoothers
constexpr int    kCapture  = 65536;   // >> max transport (~2933 smp): captures the IR tail
constexpr int    kTotal    = kSettle + kCapture;

const std::array<float, 5> kSizes = {{ 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }};
const std::array<float, 3> kCoeffs = {{ 0.0f, 0.4f, 0.78f }};
constexpr int kNumSizes = static_cast<int>(sizeof(kSizes) / sizeof(kSizes[0]));
constexpr int kNumCoefs = static_cast<int>(sizeof(kCoeffs) / sizeof(kCoeffs[0]));

const char* g_section = "(startup)";

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Run one impulse response and return the energy centroid relative to the
// impulse position (in samples). Both banks get the impulse; the centroid is
// over the combined L+R energy so it equals the mean of the bank transports.
double measuredCentroid(MarsDSP::Diffusion::Diffuser& d, float size, float coef)
{
    // setDiffusion takes a 0..1 amount; kMaxCoefficient scales it to the
    // allpass g. Convert the target coefficient back to the amount so the
    // actual allpass g is exactly {0, 0.4, 0.78}.
    const float amount = coef / MarsDSP::Diffusion::Diffuser::kMaxCoefficient;
    d.setDiffusion(amount);
    d.setSize(size);
    d.setModDepthSamples(0.0f);   // no LFO: deterministic, fast path
    d.setModRateHz(0.5f);
    d.prime();                     // snap the size/coef smoothers to their targets

    std::vector<float> bufL(static_cast<std::size_t>(kTotal), 0.0f);
    std::vector<float> bufR(static_cast<std::size_t>(kTotal), 0.0f);
    bufL[static_cast<std::size_t>(kSettle)] = 1.0f;
    bufR[static_cast<std::size_t>(kSettle)] = 1.0f;

    for (int off = 0; off < kTotal; off += kBlock)
    {
        const int n = std::min(kBlock, kTotal - off);
        d.processBlock(bufL.data() + off, bufR.data() + off, n);
    }

    double sumE = 0.0;
    double sumEX = 0.0;
    for (int n = kSettle; n < kTotal; ++n)
    {
        const auto u = static_cast<std::size_t>(n);
        const double vL = static_cast<double>(bufL[u]);
        const double vR = static_cast<double>(bufR[u]);
        const double e = vL * vL + vR * vR;
        sumE += e;
        sumEX += e * static_cast<double>(n - kSettle);
    }
    if (sumE <= 0.0) FAIL("zero energy at size={:.2} coef={:.2}", size, coef);
    return sumEX / sumE;
}

} // namespace

int main()
{
    std::println("=== Chronos diffuser_transport_check ===");
    std::println("fs={:.0}  sizes={{0,0.25,0.5,0.75,1}}  coefs={{0,0.4,0.78}}  gate=1 sample\n", kFs);

    MarsDSP::Diffusion::Diffuser d;
    d.prepare(kFs);

    // (1) centroid vs baseTransportSamples at every size x coef.
    std::println("{:>6} {:>6} | {:>10} {:>10} {:>9} | {}",
                "size", "coef", "measured", "predicted", "|d|", "pass");
    std::array<double, kNumCoefs> prevCentroidPerCoef = {{  }};
    bool firstSize = true;
    double worstErr = 0.0;
    for (int si = 0; si < kNumSizes; ++si)
    {
        const float size = kSizes[si];
        for (int ci = 0; ci < kNumCoefs; ++ci)
        {
            const float coef = kCoeffs[ci];
            g_section = "centroid";
            const double measured = measuredCentroid(d, size, coef);
            const double predicted = static_cast<double>(d.baseTransportSamples(size));
            const double err = std::fabs(measured - predicted);
            worstErr = std::max(worstErr, err);
            const bool ok = err <= 1.0;
            std::println("{:6.2} {:6.2} | {:10.3} {:10.3} {:9.4} | {}",
                        static_cast<double>(size), static_cast<double>(coef),
                        measured, predicted, err, ok ? "PASS" : "FAIL");
            if (!ok)
                FAIL("size={:.2} coef={:.2}: centroid {:.4} vs predicted {:.4} (|d|={:.4} > 1.0)",
                     size, coef, measured, predicted, err);

            // (2) strictly increasing in size, per coefficient.
            if (!firstSize)
            {
                g_section = "strictly increasing";
                if (!(measured > prevCentroidPerCoef[ci] + 1e-6))
                    FAIL("size {:.2} -> {:.2} coef {:.2}: transport not strictly increasing ({:.4} -> {:.4})",
                         kSizes[si - 1], size, coef, prevCentroidPerCoef[ci], measured);
            }
            prevCentroidPerCoef[ci] = measured;
        }
        firstSize = false;
    }

    std::println("\nworst |measured - predicted| over all cells: {:.4} samples (gate 1.0)", worstErr);
    std::println("transport strictly increasing in size at every coefficient: PASS");
    std::println("\n=== ALL TRANSPORT GATES HELD ===");
    return 0;
}

// tests/harnesses/dsp/bbd_response_check.cpp
//
// Verification harness for the BrigadeLine response family.
// Measures the steady-state sine response against the analytic model
// |H_in(f)| * ZOH(f) * |H_out(f)|. The register takes one write per two
// clock ticks, so the hold period is 2 / f_clk.
// A broadband impulse cannot measure the undersampled line: the bucket
// sampler catches the impulse at one phase only. Each probe is a settled
// sine and the response is read with a coherent Goertzel.

#include "dsp/bbd/BrigadeLine.h"
#include "dsp/bbd/PoleBank.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <print>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace
{
    const char* g_section = "(startup)";

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

    constexpr int kN = 32768;

    // Analytic 4th-order Butterworth magnitude of one bank.
    double bankMagnitude (double f, double fc)
    {
        using namespace MarsDSP::BBD;
        const double w0 = 2.0 * std::numbers::pi * fc;
        const std::complex<double> s (0.0, 2.0 * std::numbers::pi * f);
        const auto h1 = s * s + s * (w0 / kButterworthQ1) + w0 * w0;
        const auto h2 = s * s + s * (w0 / kButterworthQ2) + w0 * w0;
        return (w0 * w0 * w0 * w0) / std::abs (h1 * h2);
    }

    // Model: input bank, the bucket zero-order hold, output bank.
    // The hold period is 2 / f_clk (one write per two clock ticks).
    double analyticModelDb (double f, double fClk)
    {
        using namespace MarsDSP::BBD;
        const double x = 2.0 * std::numbers::pi * f / fClk;
        const double zoh = (std::fabs (x) < 1.0e-9) ? 1.0 : std::fabs (std::sin (x) / x);
        const double mag = bankMagnitude (f, static_cast<double> (kInputCutoffHz))
                         * zoh
                         * bankMagnitude (f, static_cast<double> (kOutputCutoffHz));
        return 20.0 * std::log10 (std::max (mag, 1.0e-30));
    }

    // Steady-state amplitude at a bin-exact probe frequency, in dB.
    double measureToneDb (double fs, float clk, double f, float* storage)
    {
        MarsDSP::BBD::BrigadeLine line;
        line.prepare (fs, storage);
        line.reset();
        line.setClockHz (clk);

        const double transport = (2.0 * MarsDSP::BBD::BrigadeLine::kStages + 0.5)
                               * fs / static_cast<double> (clk);
        const int warm = static_cast<int> (transport) + 8192;
        const double w = 2.0 * std::numbers::pi * f / fs;

        for (int i = 0; i < warm; ++i)
            line.process (static_cast<float> (std::sin (w * static_cast<double> (i))));

        const double kG = 2.0 * std::cos (w);
        double s1 = 0.0;
        double s2 = 0.0;
        for (int i = 0; i < kN; ++i)
        {
            const double in = std::sin (w * static_cast<double> (warm + i));
            const double y = static_cast<double> (line.process (static_cast<float> (in)));
            const double s0 = y + kG * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        const double p = (s1 * s1 + s2 * s2 - kG * s1 * s2)
                       * (2.0 / (static_cast<double> (kN) * static_cast<double> (kN)));
        const double amp = std::sqrt (std::max (2.0 * p, 1.0e-60));
        return 20.0 * std::log10 (amp);
    }
} // namespace

int main()
{
    using MarsDSP::BBD::BrigadeLine;

    std::vector<float> storage (BrigadeLine::bbdStorageFloats (1), 0.0f);
    const std::array<double, 3> sampleRates { { 44100.0, 48000.0, 96000.0 } };
    const std::array<float, 5> clocks { { 5000.0f, 10000.0f, 20000.0f, 40000.0f, 96000.0f } };

    g_section = "response_check";
    std::println("=== BBD Darkening Response Check (steady-state sine probes) ===");

    for (double fs : sampleRates)
    {
        for (float clk : clocks)
        {
            // Above ~0.17*fs the matched-Z banks deviate from the analytic
            // prototype; the tolerance widens to the bank stopband gate.
            const double fMax = std::min ({ 0.40 * static_cast<double> (clk), 20000.0, 0.42 * fs });
            double worst = 0.0;
            double worstF = 0.0;
            for (double f = 20.0; f <= fMax; f *= 1.3)
            {
                const int k = std::max (1, static_cast<int> (std::lround (f * kN / fs)));
                const double fProbe = static_cast<double> (k) * fs / static_cast<double> (kN);
                const double mMeas = measureToneDb (fs, clk, fProbe, storage.data());
                const double mModel = analyticModelDb (fProbe, static_cast<double> (clk));
                const double diff = std::fabs (mMeas - mModel);
                const double tol = (fProbe <= 0.17 * fs) ? 0.5 : 3.0;
                if (diff > worst) { worst = diff; worstF = fProbe; }
                if (diff >= tol)
                    std::println("FAIL cell: fs={:.0f} clk={:.0f} f={:.1f} meas={:.3f} model={:.3f} diff={:.3f}",
                                 fs, static_cast<double> (clk), fProbe, mMeas, mModel, diff);
                CHECK (diff < tol);
            }
            std::println("fs={:.0f} clk={:.0f}: worst |meas-model| = {:.3f} dB at {:.1f} Hz",
                         fs, static_cast<double> (clk), worst, worstF);
        }
    }

    std::println("=== bbd_response_check OK ===");
    return 0;
}

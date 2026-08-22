// tests/harnesses/cd/bbd_pole_bank_check.cpp
//
// Verification harness for BBD analytic Sallen-Key pole banks:
// closed-form self-consistency, DC unity, magnitude/phase response against
// analytic Butterworth prototype, and cross-check against cascaded WDF Sallen-Key sections.

#include "dsp/bbd/PoleBank.h"
#include "dsp/SallenKeyLPF.h"

#include <algorithm>
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

    constexpr int kIRLength = 65536;

    std::complex<double> dtft (const std::vector<float>& ir, double freqHz, double sampleRate)
    {
        const double omega = 2.0 * std::numbers::pi * freqHz / sampleRate;
        std::complex<double> sum (0.0, 0.0);
        for (std::size_t n = 0; n < ir.size(); ++n)
        {
            const double angle = -omega * static_cast<double> (n);
            sum += static_cast<double> (ir[n]) * std::complex<double> (std::cos (angle), std::sin (angle));
        }
        return sum;
    }

    double magnitudeDb (const std::vector<float>& ir, double freqHz, double sampleRate)
    {
        const auto H = dtft (ir, freqHz, sampleRate);
        const double mag = std::abs (H);
        return mag > 1.0e-30 ? 20.0 * std::log10 (mag) : -600.0;
    }

    double phaseDeg (const std::vector<float>& ir, double freqHz, double sampleRate)
    {
        const auto H = dtft (ir, freqHz, sampleRate);
        return std::arg (H) * 180.0 / std::numbers::pi;
    }

    // Analytic 4th-order Butterworth response: H(s) = w0^4 / ((s^2 + s*w0/Q1 + w0^2)(s^2 + s*w0/Q2 + w0^2))
    std::complex<double> analyticButterworth4 (double freqHz, double fc)
    {
        const double w0 = 2.0 * std::numbers::pi * fc;
        const std::complex<double> s (0.0, 2.0 * std::numbers::pi * freqHz);
        const double q1 = MarsDSP::BBD::kButterworthQ1;
        const double q2 = MarsDSP::BBD::kButterworthQ2;

        const auto sec1 = s * s + s * (w0 / q1) + w0 * w0;
        const auto sec2 = s * s + s * (w0 / q2) + w0 * w0;
        return (w0 * w0 * w0 * w0) / (sec1 * sec2);
    }
} // namespace

int main()
{
    using namespace MarsDSP::BBD;

    const std::array<double, 4> testRates { { 44100.0, 48000.0, 96000.0, 192000.0 } };
    const std::array<float, 4> testCutoffs { { 1000.0f, 4000.0f, 9900.0f, 19800.0f } };

    // 1. Closed-form self-consistency & DC Unity
    g_section = "closed_form_and_dc_unity";
    {
        for (double rate : testRates)
        {
            const float Ts = static_cast<float> (1.0 / rate);
            for (float fc : testCutoffs)
            {
                OutputPoleBank outBank (Ts);
                outBank.set_freq (fc);
                const float h0 = outBank.calcH0();
                CHECK (std::fabs (h0 - 1.0f) < 1.0e-6f);

                const auto pr = computeButterworthPolesAndResidues (static_cast<double> (fc));
                // Verify residues sum to 0 (for 4th order, sum of residues is 0)
                std::complex<double> sumRes (0.0, 0.0);
                for (int m = 0; m < 4; ++m)
                    sumRes += pr.residues[m];
                CHECK (std::abs (sumRes) < 1.0e-9);

                // Verify -sum(r_m / p_m) == 1.0
                std::complex<double> sumH0 (0.0, 0.0);
                for (int m = 0; m < 4; ++m)
                    sumH0 -= pr.residues[m] / pr.poles[m];
                CHECK (std::abs (sumH0 - 1.0) < 1.0e-9);
            }
        }
    }

    // 2. Magnitude & Phase Response against Analytic Prototype
    g_section = "magnitude_and_phase";
    {
        for (double rate : testRates)
        {
            const float Ts = static_cast<float> (1.0 / rate);
            for (float fc : { 9900.0f, 9500.0f })
            {
                InputPoleBank inBank (Ts);
                inBank.set_freq (fc);

                // Generate discrete impulse response
                std::vector<float> ir (kIRLength, 0.0f);
                inBank.reset();
                const auto pr = computeButterworthPolesAndResidues (static_cast<double> (fc));
                std::complex<float> gCoef[4];
                for (int m = 0; m < 4; ++m)
                    gCoef[m] = static_cast<std::complex<float>> (pr.residues[m] * static_cast<double> (Ts));

                for (int n = 0; n < kIRLength; ++n)
                {
                    const float inSample = (n == 0) ? 1.0f : 0.0f;
                    inBank.process (inSample);
                    float y = 0.0f;
                    for (int m = 0; m < 4; ++m)
                        y += (pr.residues[m] * static_cast<std::complex<double>>(inBank.x[m])).real();
                    ir[n] = y;
                }

                // Check magnitude across passband within 0.35 dB, and stopband within 3.0 dB
                for (double f = 20.0; f <= std::min (20000.0, rate * 0.45); f *= 1.5)
                {
                    const double mMeas = magnitudeDb (ir, f, rate);
                    const auto HExact = analyticButterworth4 (f, static_cast<double> (fc));
                    const double mExact = 20.0 * std::log10 (std::abs (HExact));
                    const double delta = std::fabs (mMeas - mExact);
                    if (f <= static_cast<double> (fc) * 0.8)
                        CHECK (delta < 0.35);
                    else
                        CHECK (delta < 3.0);
                }

                // Phase at fc: 4th-order matched-Z phase tracks prototype with high-frequency warping
                const double pFc = phaseDeg (ir, static_cast<double> (fc), rate);
                const double pDiff = std::fabs (std::abs (pFc) - 180.0);
                CHECK (pDiff < 45.0);
            }
        }
    }

    // 3. Cross-implementation check against cascaded WDF SallenKeyLPF
    g_section = "cross_implementation_wdf";
    {
        constexpr double fs = 48000.0;
        constexpr float fc = 9900.0f;

        // Cascade two WDF SallenKeyLPF instances with Q1 and Q2
        MarsDSP::Filters::SallenKeyLPF sec1;
        MarsDSP::Filters::SallenKeyLPF sec2;
        sec1.prepare (fs);
        sec1.setParams (fc, static_cast<float> (kButterworthQ1));
        sec2.prepare (fs);
        sec2.setParams (fc, static_cast<float> (kButterworthQ2));

        std::vector<float> irWdf (kIRLength, 0.0f);
        irWdf[0] = sec2.processSample (sec1.processSample (1.0f));
        for (int n = 1; n < kIRLength; ++n)
            irWdf[n] = sec2.processSample (sec1.processSample (0.0f));

        // Pole bank IR
        const float Ts = static_cast<float> (1.0 / fs);
        InputPoleBank inBank (Ts);
        inBank.set_freq (fc);
        inBank.reset();
        const auto pr = computeButterworthPolesAndResidues (static_cast<double> (fc));
        std::complex<float> gCoef[4];
        for (int m = 0; m < 4; ++m)
            gCoef[m] = static_cast<std::complex<float>> (pr.residues[m] * static_cast<double> (Ts));

        std::vector<float> irBank (kIRLength, 0.0f);
        for (int n = 0; n < kIRLength; ++n)
        {
            inBank.process (n == 0 ? 1.0f : 0.0f);
            float y = 0.0f;
            for (int m = 0; m < 4; ++m)
                y += (pr.residues[m] * static_cast<std::complex<double>>(inBank.x[m])).real();
            irBank[n] = y;
        }

        for (double f = 20.0; f <= fs * 0.40; f *= 1.4)
        {
            const double mWdf = magnitudeDb (irWdf, f, fs);
            const double mBank = magnitudeDb (irBank, f, fs);
            const double diff = std::fabs (mWdf - mBank);
            std::println("f={:.1} mWdf={:.3} mBank={:.3} diff={:.3}", f, mWdf, mBank, diff);
            if (f <= static_cast<double> (fc) * 0.8)
                CHECK (diff < 0.5);
        }
    }

    std::println("=== bbd_pole_bank_check OK ===");
    return 0;
}

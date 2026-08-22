// tests/harnesses/dsp/sallen_key_response_check.cpp
//
// Verification harness for Sallen-Key low-pass and high-pass filter responses,
// prewarping, stopband slopes, and cross-mode agreement with SimdSVF.

#include "dsp/SallenKeyLPF.h"
#include "dsp/SallenKeyHPF.h"
#include "dsp/StateVariable.h"
#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/Nonlinearities.h"

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

    template <typename Filter>
    std::vector<float> measureIR (Filter& filter, double sampleRate, float f0, float q)
    {
        filter.prepare (sampleRate);
        filter.setParams (f0, q);
        std::vector<float> ir (kIRLength, 0.0f);
        ir[0] = filter.processSample (1.0f);
        for (int i = 1; i < kIRLength; ++i)
            ir[i] = filter.processSample (0.0f);
        return ir;
    }

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

    double findCutoffMinus3dB (const std::vector<float>& ir, double sampleRate, bool isLowpass)
    {
        // Binary search for -3 dB frequency
        double fLow = 10.0;
        double fHigh = sampleRate * 0.49;
        for (int it = 0; it < 30; ++it)
        {
            const double fMid = 0.5 * (fLow + fHigh);
            const double db = magnitudeDb (ir, fMid, sampleRate);
            if (isLowpass)
            {
                if (db < -3.0103)
                    fHigh = fMid;
                else
                    fLow = fMid;
            }
            else
            {
                if (db > -3.0103)
                    fHigh = fMid;
                else
                    fLow = fMid;
            }
        }
        return 0.5 * (fLow + fHigh);
    }
} // namespace

int main()
{
    using MarsDSP::Filters::SallenKeyLPF;
    using MarsDSP::Filters::SallenKeyHPF;
    using MarsDSP::Filters::SimdSVF;

    // 1. Low-pass response table (fs = 96 kHz)
    g_section = "lpf_table_96k";
    {
        constexpr double fs = 96000.0;
        struct Row { float f0; float q; double expected20Hz; double expected2kHz; double expected20kHz; };
        const Row table[] = {
            { 200.0f,  0.7071f,   0.00, -40.03, -82.76 },
            { 1000.0f, 0.7071f,   0.00, -12.35, -54.80 },
            { 1000.0f, 2.0f,      0.00, -10.07, -54.79 },
            { 1000.0f, 8.0f,      0.00,  -9.67, -54.79 },
            { 5000.0f, 0.7071f,   0.00,  -0.12, -26.86 },
            { 5000.0f, 4.0f,      0.00,  +1.44, -26.46 }
        };

        SallenKeyLPF lpf;
        for (const auto& row : table)
        {
            const auto ir = measureIR (lpf, fs, row.f0, row.q);
            const double m20  = magnitudeDb (ir, 20.0, fs);
            const double m2k  = magnitudeDb (ir, 2000.0, fs);
            const double m20k = magnitudeDb (ir, 20000.0, fs);

            CHECK (std::fabs (m20 - row.expected20Hz) < 0.5);
            CHECK (std::fabs (m2k - row.expected2kHz) < 0.5);
            CHECK (std::fabs (m20k - row.expected20kHz) < 0.5);
        }
    }

    // 2. High-pass response table (fs = 96 kHz)
    g_section = "hpf_table_96k";
    {
        constexpr double fs = 96000.0;
        struct Row { float f0; float q; double at20Hz; double atHalfF0; double atF0; double at2F0; double at15kHz; double phaseAtF0; };
        const Row table[] = {
            { 100.0f,  0.7071f, -27.97, -12.35, -3.10, -0.30, -0.00, +90.0 },
            { 500.0f,  0.7071f, -55.92, -12.35, -3.10, -0.30, -0.00, +90.0 },
            { 500.0f,  2.0f,    -55.91, -10.07, +5.36, +1.97, +0.01, +90.0 },
            { 1000.0f, 0.7071f, -67.96, -12.34, -3.09, -0.30, -0.00, +90.0 },
            { 5000.0f, 0.7071f, -95.89, -12.31, -3.02, -0.27, -0.05, +89.3 }
        };

        SallenKeyHPF hpf;
        for (const auto& row : table)
        {
            const auto ir = measureIR (hpf, fs, row.f0, row.q);
            const double m20    = magnitudeDb (ir, 20.0, fs);
            const double mHalf  = magnitudeDb (ir, row.f0 * 0.5, fs);
            const double mF0    = magnitudeDb (ir, row.f0, fs);
            const double m2F0   = magnitudeDb (ir, row.f0 * 2.0, fs);
            const double m15k   = magnitudeDb (ir, 15000.0, fs);
            const double pF0    = phaseDeg (ir, row.f0, fs);

            CHECK (std::fabs (m20 - row.at20Hz) < 0.3);
            CHECK (std::fabs (mHalf - row.atHalfF0) < 0.3);
            CHECK (std::fabs (mF0 - row.atF0) < 0.3);
            CHECK (std::fabs (m2F0 - row.at2F0) < 0.3);
            CHECK (std::fabs (m15k - row.at15kHz) < 0.3);
            CHECK (std::fabs (pF0 - row.phaseAtF0) < 2.0);
        }
    }

    // 3. Prewarp accuracy
    g_section = "prewarp";
    {
        for (double fs : { 44100.0, 48000.0 })
        {
            SallenKeyLPF lpf;
            for (float f0 : { 200.0f, 1000.0f, 5000.0f, 12000.0f, 20000.0f })
            {
                if (f0 >= fs * 0.49f) continue;
                const auto ir = measureIR (lpf, fs, f0, 0.7071f);
                const double measuredF0 = findCutoffMinus3dB (ir, fs, true);
                const double relErr = std::fabs (measuredF0 - f0) / f0;
                CHECK (relErr < 0.01);
            }

            SallenKeyHPF hpf;
            for (float f0 : { 20.0f, 100.0f, 500.0f, 2000.0f })
            {
                const auto ir = measureIR (hpf, fs, f0, 0.7071f);
                const double measuredF0 = findCutoffMinus3dB (ir, fs, false);
                const double relErr = std::fabs (measuredF0 - f0) / f0;
                std::println("HPF fs={:.0} f0={:.1} measuredF0={:.3f} relErr={:.4f}", fs, f0, measuredF0, relErr);
                CHECK (relErr < 0.015);
            }
        }
    }

    // 4. High-pass stopband slope
    g_section = "hpf_stopband_slope";
    {
        constexpr double fs = 96000.0;
        SallenKeyHPF hpf;
        const auto ir = measureIR (hpf, fs, 100.0f, 0.7071f);
        const double m20 = magnitudeDb (ir, 20.0, fs);
        CHECK (std::fabs (m20 - (-27.96)) < 0.1);
    }

    // 5. DC gain and Nyquist-band gain
    g_section = "dc_and_passband_gain";
    {
        constexpr double fs = 48000.0;
        SallenKeyLPF lpf;
        for (float q : { 0.05f, 0.7071f, 2.0f, 8.0f, 10.9999f })
        {
            lpf.prepare (fs);
            lpf.setParams (1000.0f, q);
            float y = 0.0f;
            for (int i = 0; i < 20000; ++i)
                y = lpf.processSample (1.0f);
            CHECK (std::fabs (y - 1.0f) < 1.0e-4f);
        }

        SallenKeyHPF hpf;
        const auto ir = measureIR (hpf, fs, 100.0f, 0.7071f);
        const double m15k = magnitudeDb (ir, 15000.0, fs);
        CHECK (std::fabs (m15k) < 1.0e-4);
    }

    // 6. Realised Q monotonicity
    g_section = "q_monotonicity";
    {
        constexpr double fs = 96000.0;
        const std::array<float, 7> qVals { { 0.05f, 0.2f, 0.7071f, 2.0f, 5.0f, 8.0f, 10.9999f } };
        SallenKeyLPF lpf;
        double prevPeak = -100.0;
        for (float q : qVals)
        {
            const auto ir = measureIR (lpf, fs, 1000.0f, q);
            // Search peak magnitude around 1000 Hz
            double maxMag = -100.0;
            for (int fi = 500; fi <= 2000; fi += 10)
                maxMag = std::max (maxMag, magnitudeDb (ir, static_cast<double> (fi), fs));

            if (q <= 8.0f)
                CHECK (maxMag >= prevPeak - 1.0e-5);
            prevPeak = maxMag;
        }
    }

    // 7. Cross-mode agreement (SimdSVF vs Sallen-Key)
    g_section = "cross_mode_agreement";
    {
        constexpr double fs = 48000.0;
        const std::array<float, 5> cutoffs { { 50.0f, 200.0f, 1000.0f, 5000.0f, 15000.0f } };

        SallenKeyLPF lpf;
        SimdSVF svfLpf;
        svfLpf.reset();

        for (float f0 : cutoffs)
        {
            const auto irSk = measureIR (lpf, fs, f0, 0.7071f);

            // Measure SVF impulse response
            svfLpf.reset();
            svfLpf.setCoeffForBlock (SimdSVF::SVFType::LowPass, fs, f0, 0.7071, 0.0, 1);
            std::vector<float> irSvf (kIRLength, 0.0f);
            {
                const M128 in0 = MM(set_ps) (0.0f, 0.0f, 0.0f, 1.0f);
                const M128 out0 = svfLpf.processBlockStep (in0);
                alignas(16) float tmp[4];
                MM(store_ps) (tmp, out0);
                irSvf[0] = tmp[0];
                const M128 inZero = MM(setzero_ps)();
                for (int i = 1; i < kIRLength; ++i)
                {
                    const M128 out = svfLpf.processBlockStep (inZero);
                    MM(store_ps) (tmp, out);
                    irSvf[i] = tmp[0];
                }
            }

            for (double mult : { 0.25, 0.5, 1.0, 2.0, 4.0 })
            {
                const double fProbe = f0 * mult;
                if (fProbe < 20.0 || fProbe > fs * 0.45) continue;
                const double mSk = magnitudeDb (irSk, fProbe, fs);
                const double mSvf = magnitudeDb (irSvf, fProbe, fs);
                CHECK (std::fabs (mSk - mSvf) < 1.5);
            }
        }
    }

    // 8. S42 Transparency
    g_section = "transparency_s42";
    {
        constexpr double fs = 48000.0;
        constexpr double f0 = 1000.0;
        constexpr double level = 0.1; // -20 dBFS
        constexpr int N = 48000;

        SallenKeyLPF lpf;
        lpf.prepare (fs);
        lpf.setParams (20000.0f, 0.7071f);
        MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa;
        adaa.reset();

        constexpr float kRail = 4.0f;
        constexpr float invRail = 1.0f / kRail;

        double sumTotal = 0.0;
        double sumFund = 0.0;

        // Warm up
        for (int i = 0; i < 4800; ++i)
        {
            const float in = static_cast<float> (level * std::sin (2.0 * std::numbers::pi * f0 * i / fs));
            float y = lpf.processSample (in);
            y = static_cast<float> (adaa.process (static_cast<double> (y * invRail))) * kRail;
        }

        // Measure Goertzel at 1 kHz fundamental and total power
        double s_prev1 = 0.0;
        double s_prev2 = 0.0;
        const double kGoertzel = 2.0 * std::cos (2.0 * std::numbers::pi * f0 / fs);

        for (int i = 0; i < N; ++i)
        {
            const float in = static_cast<float> (level * std::sin (2.0 * std::numbers::pi * f0 * i / fs));
            float y = lpf.processSample (in);
            y = static_cast<float> (adaa.process (static_cast<double> (y * invRail))) * kRail;

            sumTotal += y * y;
            const double s = y + kGoertzel * s_prev1 - s_prev2;
            s_prev2 = s_prev1;
            s_prev1 = s;
        }

        const double fundPower = (s_prev1 * s_prev1 + s_prev2 * s_prev2 - kGoertzel * s_prev1 * s_prev2) * (2.0 / (static_cast<double>(N) * N));
        const double totalPower = sumTotal / N;
        const double thdPower = std::max (0.0, totalPower - fundPower);
        const double thdDb = 10.0 * std::log10 (std::max (thdPower / fundPower, 1.0e-30));

        CHECK (thdDb < -60.0);
    }

    // 9. S42 Aliasing
    g_section = "aliasing_s42";
    {
        constexpr double fs = 48000.0;
        constexpr int N = 16384;
        constexpr int kWarmup = 2048;
        constexpr int k0 = 1025; // odd bin index ~ 3003 Hz
        const double f0 = static_cast<double>(k0) * fs / static_cast<double>(N);
        constexpr double level = 4.0; // 0 dB relative to saturation rail (4.0f)
        constexpr float kRail = 4.0f;
        constexpr float invRail = 1.0f / kRail;

        MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa;
        adaa.reset();

        for (int i = 0; i < kWarmup; ++i)
        {
            const float in = static_cast<float> (level * std::sin (2.0 * std::numbers::pi * f0 * i / fs));
            adaa.process (static_cast<double> (in * invRail));
        }

        std::vector<double> yAdaa(N);
        std::vector<double> yPlain(N);
        for (int i = 0; i < N; ++i)
        {
            const float in = static_cast<float> (level * std::sin (2.0 * std::numbers::pi * f0 * (i + kWarmup) / fs));
            yAdaa[i] = adaa.process (static_cast<double> (in * invRail)) * kRail;
            yPlain[i] = std::tanh (static_cast<double> (in * invRail)) * kRail;
        }

        auto analyzeSignal = [](const std::vector<double>& y, int bin0)
        {
            const int len = static_cast<int>(y.size());
            double total = 0.0;
            for (double v : y) total += v * v;
            const double invLen = 1.0 / static_cast<double>(len);

            auto goertzel = [](const double* x, int nSamp, int k)
            {
                const double w = 2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(nSamp);
                const double coeff = 2.0 * std::cos(w);
                double s1 = 0.0, s2 = 0.0;
                for (int i = 0; i < nSamp; ++i)
                {
                    const double s0 = x[i] + coeff * s1 - s2;
                    s2 = s1;
                    s1 = s0;
                }
                return s1 * s1 + s2 * s2 - coeff * s1 * s2;
            };

            double harm = goertzel(y.data(), len, 0) * invLen;
            for (int j = 1; j * bin0 < len / 2; ++j)
                harm += 2.0 * goertzel(y.data(), len, j * bin0) * invLen;

            const double aliasE = std::max(1.0e-30, total - harm);
            const double fundE = 2.0 * goertzel(y.data(), len, bin0) * invLen;
            return 10.0 * std::log10(aliasE / fundE);
        };

        const double aliasAdaaDbc = analyzeSignal(yAdaa, k0);
        const double aliasPlainDbc = analyzeSignal(yPlain, k0);
        const double diffDb = aliasAdaaDbc - aliasPlainDbc;
        std::println("ADAA1 alias={:.2f} dBc, Plain={:.2f} dBc, diff={:.2f} dB",
                     aliasAdaaDbc, aliasPlainDbc, diffDb);

        // ADAA1 aliasing is at least 6 dB below plain tanh
        CHECK (diffDb <= -6.0);
    }

    std::println("=== sallen_key_response_check OK ===");
    return 0;
}

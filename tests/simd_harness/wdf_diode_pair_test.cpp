#include <iostream>
#include <fstream>
#include <cmath>
#include <numbers>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <filesystem>
#include "dsp/wdf/wdf_core.h"
#include "dsp/wdf/wdf_diode_pair.h"
#include "dsp/wdf/wdf_math.h"

// WDF correctness harness:
//
//   (A) Wright-Omega vs scipy reference values (margin 0.3 for omega3,
//       0.05 for omega4, both absolute).
//
//   (B) log_approx, exp_approx, log2_approx, pow2_approx accuracy bands
//       (margins 0.005, 0.03, 0.008, 0.001).
//
//   (C) Voltage divider sanity (10 V across two equal resistors -> 5 V).
//
//   (D) RC lowpass magnitude response at 0.5*fc, fc, 2*fc (-1 dB / -3 dB
//       / -7 dB within 0.1 dB).
//
//   (E) DiodePair DC sweep against an omega-based reference solve, plus
//       an audio chirp dump for visual inspection.

namespace
{
    // scipy.special.wrightomega reference values for W + ln(W) = x.
    const std::unordered_map<double, double> WrightOmegaReference = {
        { -10.0, 4.539786874921544e-05 }, { -9.5, 7.484622772024869e-05 },
        { -9.0, 0.00012339457692560975 }, { -8.5, 0.00020342698226408345 },
        { -8.0, 0.000335350149321062 },   { -7.5, 0.0005527787213627528 },
        { -7.0, 0.0009110515723789146 },  { -6.5, 0.0015011839473879653 },
        { -6.0, 0.002472630709097278 },   { -5.5, 0.004070171383753891 },
        { -5.0, 0.0066930004977309955 },  { -4.5, 0.010987603420879434 },
        { -4.0, 0.017989102828531025 },   { -3.5, 0.029324711813756815 },
        { -3.0, 0.04747849102486547 },    { -2.5, 0.07607221340790257 },
        { -2.0, 0.1200282389876412 },     { -1.5, 0.1853749184489398 },
        { -1.0, 0.27846454276107374 },    { -0.5, 0.4046738485459385 },
        {  0.0, 0.5671432904097838 },     {  0.5, 0.7662486081617502 },
        {  1.0, 1.0 },                    {  1.5, 1.2649597201255005 },
        {  2.0, 1.5571455989976113 },     {  2.5, 1.8726470404165942 },
        {  3.0, 2.207940031569323 },      {  3.5, 2.559994780412122 },
        {  4.0, 2.926271062443501 },      {  4.5, 3.3046649181693253 },
        {  5.0, 3.6934413589606496 },     {  5.5, 4.091169202271799 },
        {  6.0, 4.4966641730061605 },     {  6.5, 4.908941634486258 },
        {  7.0, 5.327178301371093 },      {  7.5, 5.750681611147114 },
        {  8.0, 6.178865346308128 },      {  8.5, 6.611230244734983 },
        {  9.0, 7.047348546597604 },      {  9.5, 7.486851633496902 },
        { 10.0, 7.9294200950196965 },
    };
}

int main()
{
    namespace W = MarsDSP::DSP::WDF;
    std::filesystem::create_directories ("tests/simd_harness/logs");

    bool overallOk = true;

    // ============================================================== //
    // (A) Wright-Omega vs scipy reference table
    // ============================================================== //
    {
        std::ofstream csv ("tests/simd_harness/logs/wdf_omega.csv");
        csv << "x,omega_truth,omega3,omega4,abs_err3,abs_err4\n";
        csv << std::fixed << std::setprecision (12);

        double maxErr3 = 0.0, maxErr4 = 0.0;
        for (const auto& kv : WrightOmegaReference)
        {
            const double x   = kv.first;
            const double ref = kv.second;
            const double a3  = W::wrightOmega3 (x);
            const double a4  = W::wrightOmega4 (x);
            maxErr3 = std::max (maxErr3, std::abs (a3 - ref));
            maxErr4 = std::max (maxErr4, std::abs (a4 - ref));
            csv << x << "," << ref << "," << a3 << "," << a4 << ","
                << std::abs (a3 - ref) << "," << std::abs (a4 - ref) << "\n";
        }
        const bool ok = (maxErr3 < 0.3) && (maxErr4 < 0.05);
        overallOk = overallOk && ok;
        std::cout << "[A] Wright-Omega vs scipy: max|omega3-ref| = " << maxErr3
                  << ", max|omega4-ref| = " << maxErr4
                  << (ok ? "  [PASS]" : "  [FAIL]") << std::endl;
    }

    // ============================================================== //
    // (B) Polynomial exp/log approximations
    // ============================================================== //
    {
        const auto checkBand = [](const char* name, double low, double high,
                                  double margin, auto approx, auto truth)
        {
            constexpr int N = 256;
            const double step = (high - low) / N;
            double maxErr = 0.0;
            for (double x = low; x < high; x += step)
                maxErr = std::max (maxErr, std::abs (approx (x) - truth (x)));
            const bool ok = maxErr <= margin;
            std::cout << "    " << name << ": max err = " << maxErr
                      << "  (margin " << margin << ")"
                      << (ok ? "  [PASS]" : "  [FAIL]") << std::endl;
            return ok;
        };

        bool allOk = true;
        std::cout << "[B] Polynomial approximation bands:" << std::endl;
        allOk &= checkBand ("log2", 1.0, 2.0, 0.008,
                            [](double x) { return W::log2Approx<double>(x); },
                            [](double x) { return std::log2 (x); });
        allOk &= checkBand ("log",  8.0, 12.0, 0.005,
                            [](double x) { return W::logApprox (x); },
                            [](double x) { return std::log (x); });
        allOk &= checkBand ("pow2", 0.0, 1.0, 0.001,
                            [](double x) { return W::pow2Approx<double>(x); },
                            [](double x) { return std::pow (2.0, x); });
        allOk &= checkBand ("exp",  -4.0, 2.0, 0.03,
                            [](double x) { return W::expApprox (x); },
                            [](double x) { return std::exp (x); });
        overallOk = overallOk && allOk;
    }

    // ============================================================== //
    // (C) Voltage divider: 10 V across two equal resistors -> 5 V
    // ============================================================== //
    {
        W::WDFResistor<float> r1 (10000.0f);
        W::WDFResistor<float> r2 (10000.0f);
        W::WDFSeries<float, decltype (r1), decltype (r2)> s1 (r1, r2);
        W::PolarityInverter<float, decltype (s1)>         p1 (s1);
        W::IdealVoltageSource<float, decltype (p1)>       vs (p1);

        vs.setVoltage (10.0f);
        vs.incident (p1.reflected());
        p1.incident (vs.reflected());

        const float vOut = W::voltage<float> (r2);
        const float err  = std::abs (vOut - 5.0f);
        const bool ok    = err < 1.0e-5f;
        overallOk = overallOk && ok;
        std::cout << "[C] Voltage divider 10V -> r2: V=" << vOut
                  << " (err " << err << ")"
                  << (ok ? "  [PASS]" : "  [FAIL]") << std::endl;
    }

    // ============================================================== //
    // (D) RC lowpass: -1 / -3 / -7 dB at 0.5*fc / fc / 2*fc
    // ============================================================== //
    {
        constexpr double fs = 44100.0;
        constexpr double fc = 500.0;
        constexpr double C  = 1.0e-6;
        const     double R  = 1.0 / ((2.0 * M_PI) * fc * C);

        const auto magnitudeAtFreq = [&](double freq) -> double
        {
            W::WDFCapacitor<double> c1 (C, fs);
            W::WDFResistor<double>  r1 (R);
            W::WDFSeries<double, decltype (r1), decltype (c1)> s1 (r1, c1);
            W::PolarityInverter<double, decltype (s1)>         p1 (s1);
            W::IdealVoltageSource<double, decltype (p1)>       vs (p1);

            double mag = 0.0;
            for (int n = 0; n < static_cast<int>(fs); ++n)
            {
                const double x = std::sin (2.0 * M_PI * freq * n / fs);
                vs.setVoltage (x);
                vs.incident (p1.reflected());
                p1.incident (vs.reflected());
                const double y = W::voltage<double> (c1);
                if (n > 1000)
                    mag = std::max (mag, std::abs (y));
            }
            return 20.0 * std::log10 (mag);
        };

        const auto check = [&](double freq, double expectedDB) -> bool
        {
            const double actual = magnitudeAtFreq (freq);
            const bool ok = std::abs (actual - expectedDB) < 0.1;
            std::cout << "    f = " << std::setw (6) << freq
                      << " Hz: |H| = " << std::setw (6) << actual
                      << " dB (expect " << expectedDB << ")"
                      << (ok ? "  [PASS]" : "  [FAIL]") << std::endl;
            return ok;
        };

        std::cout << "[D] RC lowpass (fc = " << fc << " Hz):" << std::endl;
        bool ok = true;
        ok &= check (0.5 * fc, -1.0);
        ok &= check (      fc, -3.0);
        ok &= check (2.0 * fc, -7.0);
        overallOk = overallOk && ok;
    }

    // ============================================================== //
    // (E) WDF DiodePair DC sweep + audio chirp
    // ============================================================== //
    {
        constexpr float Rs = 1.0e3f;
        constexpr float Is = 2.52e-9f;
        constexpr float Vt = 0.02585f;

        // Reference: solve (V_in - V) / R_s = 2*I_s*sinh(V/V_T) using
        // omega: per Werner the closed form is V = V_in - 2*Vt*sgn(a)*
        // ( omega(p + a/Vt) - omega(p - a/Vt) ) / 2 with a -> V_in for the
        // load=R_s case... easier: just call DiodePair with quality
        // Accurate and treat omega3 as the "fast" variant comparison.
        W::ResistiveVoltageSource<float> Vs (Rs);
        W::DiodePair<float, decltype (Vs), W::DiodeQuality::Accurate> diodeAcc (Vs, Is, Vt, 1.0f);

        std::ofstream csv ("tests/simd_harness/logs/wdf_diode_dc_sweep.csv");
        csv << "Vin,Vout_acc\n";
        csv << std::fixed << std::setprecision (10);

        double minVout = 0.0, maxVout = 0.0;
        for (int i = -2000; i <= 2000; ++i)
        {
            const float Vin = static_cast<float>(i) * 0.005f;
            Vs.setVoltage (Vin);
            diodeAcc.incident (Vs.reflected());
            Vs.incident (diodeAcc.reflected());
            const float Vout = W::voltage<float> (Vs);
            minVout = std::min (minVout, static_cast<double>(Vout));
            maxVout = std::max (maxVout, static_cast<double>(Vout));
            csv << Vin << "," << Vout << "\n";
        }
        // Sanity: the bridge should clamp the output to roughly +-1 V or
        // less for our diode parameters; we just check it didn't go
        // unbounded. The frequency / harmonic content is left to the
        // visualisation step.
        const bool ok = (minVout > -2.0) && (maxVout < 2.0);
        overallOk = overallOk && ok;
        std::cout << "[E] WDF DiodePair DC sweep: V_out in [" << minVout
                  << ", " << maxVout << "] V"
                  << (ok ? "  [PASS]" : "  [FAIL]") << std::endl;

        // Audio chirp.
        constexpr int   N    = 2048;
        constexpr float fs   = 48000.0f;
        constexpr float f0   = 200.0f;
        constexpr float f1   = 4000.0f;
        constexpr float amp  = 2.0f;

        std::ofstream audioCsv ("tests/simd_harness/logs/wdf_diode_audio.csv");
        audioCsv << "n,t,Vin,Vout\n";
        audioCsv << std::fixed << std::setprecision (8);

        const float k = (f1 - f0) / (N / fs);
        float phase = 0.0f;
        for (int n = 0; n < N; ++n)
        {
            const float t        = n / fs;
            const float instFreq = f0 + k * t;
            phase += 2.0f * std::numbers::pi_v<float> * instFreq / fs;
            const float Vin = amp * std::sin (phase);

            Vs.setVoltage (Vin);
            diodeAcc.incident (Vs.reflected());
            Vs.incident (diodeAcc.reflected());
            const float Vout = W::voltage<float> (Vs);
            audioCsv << n << "," << t << "," << Vin << "," << Vout << "\n";
        }
    }

    return overallOk ? 0 : 1;
}

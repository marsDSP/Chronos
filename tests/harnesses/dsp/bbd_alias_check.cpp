// tests/harnesses/dsp/bbd_alias_check.cpp
//
// Aliasing measurement harness for BrigadeLine and FeedbackDelay (BBD mode).
// Coherent sampling, Goertzel harmonic masking, Parseval total energy.
// The primary gate is the prediction match: the measured inharmonic energy
// must agree with the analytic folding prediction of the tone images at
// k*f_w +- f0, where f_w = f_clk/2 is the bucket write rate.
// The comparative gate uses a probe above the input cutoff and compares
// absolute alias power: for a single in-band tone the input bank scales the
// fundamental and its images equally, so a dBc gate cannot see the filter.

#include "dsp/bbd/BrigadeLine.h"
#include "dsp/FeedbackDelay.h"

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

    using MarsDSP::BBD::BrigadeLine;
    using MarsDSP::Delays::FeedbackDelay;

    constexpr double kFs = 48000.0;
    constexpr int N = 32768; // power of 2

    // Compute the coherent frequency closest to nominal f0 with an odd bin index.
    double getCoherentF0 (double targetF0, double fs, int Npts)
    {
        int k0 = static_cast<int> (std::round (targetF0 * Npts / fs));
        if (k0 % 2 == 0) k0 += 1;
        return static_cast<double> (k0) * fs / static_cast<double> (Npts);
    }

    struct AliasResult
    {
        double dbc;         // alias power relative to the fundamental
        double aliasPowDb;  // absolute alias power in dB
        double fundPow;
    };

    AliasResult measureAlias (const std::vector<float>& signal, double f0, double fs)
    {
        double totalPower = 0.0;
        for (int i = 0; i < N; ++i)
            totalPower += static_cast<double> (signal[i]) * static_cast<double> (signal[i]);
        totalPower /= N;

        double harmPower = 0.0;
        double fundPower = 0.0;

        for (int k = 1; k * f0 < fs * 0.5; ++k)
        {
            const double fk = k * f0;
            const double kG = 2.0 * std::cos (2.0 * std::numbers::pi * fk / fs);
            double s1 = 0.0;
            double s2 = 0.0;

            for (int i = 0; i < N; ++i)
            {
                const double s = static_cast<double> (signal[i]) + kG * s1 - s2;
                s2 = s1;
                s1 = s;
            }

            const double pK = (s1 * s1 + s2 * s2 - kG * s1 * s2)
                            * (2.0 / (static_cast<double> (N) * static_cast<double> (N)));
            harmPower += pK;
            if (k == 1) fundPower = pK;
        }

        const double aliasPower = std::max (1.0e-30, totalPower - harmPower);
        AliasResult r;
        r.fundPow = fundPower;
        r.aliasPowDb = 10.0 * std::log10 (aliasPower);
        r.dbc = 10.0 * std::log10 (aliasPower / std::max (fundPower, 1.0e-30));
        return r;
    }

    double bankMagnitude (double f, double fc)
    {
        using namespace MarsDSP::BBD;
        const double w0 = 2.0 * std::numbers::pi * fc;
        const std::complex<double> s (0.0, 2.0 * std::numbers::pi * f);
        const auto h1 = s * s + s * (w0 / kButterworthQ1) + w0 * w0;
        const auto h2 = s * s + s * (w0 / kButterworthQ2) + w0 * w0;
        return (w0 * w0 * w0 * w0) / std::abs (h1 * h2);
    }

    struct Prediction
    {
        double dbc;
        bool   subsonicImage;   // an image below 30 Hz: the window cannot resolve it
        bool   nulledFund;      // the fundamental sits on a hold null
    };

    // Analytic folding prediction of the inharmonic energy for one tone.
    // Every image carries the same input-bank factor as the fundamental,
    // so the input bank cancels from the dBc ratio. The harmonic mask
    // captures only the leakage fraction near each harmonic bin, so each
    // image contributes its residual (1 - sinc^2) energy.
    Prediction predictAliasDbc (double f0, double clk, double fs)
    {
        const double fw = 0.5 * clk;
        const double binW = fs / N;

        auto env = [&](double f)
        {
            const double x = std::numbers::pi * f / fw;
            return (std::fabs (x) < 1.0e-12) ? 1.0 : std::fabs (std::sin (x) / x);
        };

        const double fundAmp = env (f0) * bankMagnitude (f0, MarsDSP::BBD::kOutputCutoffHz);

        Prediction pr;
        pr.subsonicImage = false;
        pr.nulledFund = (env (f0) < 0.02);

        double aliasPow = 0.0;
        const int kMaxIm = static_cast<int> (fs / fw) + 3;
        for (int k = 1; k <= kMaxIm; ++k)
        {
            for (const double fi : { k * fw - f0, k * fw + f0 })
            {
                const double fa = std::fabs (fi);
                if (fa >= 0.5 * fs) continue;
                if (fa < 30.0) pr.subsonicImage = true;

                // The fraction the harmonic-bin Goertzel comb removes.
                const double m = std::round (fa / f0);
                double captured = 0.0;
                if (m >= 0.0 && m * f0 < fs * 0.5 + 1.0)
                {
                    const double deltaBins = (fa - m * f0) / binW;
                    const double xd = std::numbers::pi * deltaBins;
                    const double sincd = (std::fabs (xd) < 1.0e-12) ? 1.0 : (std::sin (xd) / xd);
                    captured = sincd * sincd;
                }

                const double a = env (fa) * bankMagnitude (fa, MarsDSP::BBD::kOutputCutoffHz);
                aliasPow += (1.0 - captured) * a * a;
            }
        }

        pr.dbc = 10.0 * std::log10 (std::max (aliasPow, 1.0e-12)
                                  / std::max (fundAmp * fundAmp, 1.0e-12));
        return pr;
    }

    // Render one single-pass tone through the line with a transport-aware warmup.
    std::vector<float> renderLine (float clk, const std::vector<float>& in, float* storage,
                                   float inputCutoffHz)
    {
        BrigadeLine line;
        line.prepare (kFs, storage);
        line.setClockHz (clk);
        if (inputCutoffHz > 0.0f)
            line.setInputFilterFreq (inputCutoffHz);

        const int transport = static_cast<int> ((2.0 * BrigadeLine::kStages + 0.5)
                                                * kFs / static_cast<double> (clk));
        const int warm = transport + 8192;
        for (int i = 0; i < warm; ++i)
            line.process (in[static_cast<std::size_t> (i % N)]);

        // The warmup length is a multiple of nothing: keep phase continuity
        // by indexing the periodic input with the running sample index.
        std::vector<float> out (N);
        for (int i = 0; i < N; ++i)
            out[static_cast<std::size_t> (i)] = line.process (in[static_cast<std::size_t> ((warm + i) % N)]);
        return out;
    }
} // namespace

int main()
{
    std::vector<float> storage (BrigadeLine::bbdStorageFloats (1), 0.0f);
    const std::array<float, 5> clocks { { 2000.0f, 4000.0f, 8000.0f, 16000.0f, 40000.0f } };
    const std::array<double, 4> testF0s { { 110.0, 1000.0, 5000.0, 10000.0 } };

    g_section = "prediction_match";
    std::println("=== BBD Alias Character Check ===");

    for (float clk : clocks)
    {
        for (double targetF0 : testF0s)
        {
            const double f0 = getCoherentF0 (targetF0, kFs, N);
            std::vector<float> in (N);
            for (int i = 0; i < N; ++i)
                in[static_cast<std::size_t> (i)] =
                    0.5f * static_cast<float> (std::sin (2.0 * std::numbers::pi * f0 * i / kFs));

            const auto out = renderLine (clk, in, storage.data(), 0.0f);
            const auto meas = measureAlias (out, f0, kFs);
            const auto pred = predictAliasDbc (f0, static_cast<double> (clk), kFs);

            const char* skip = pred.nulledFund ? " (nulled fund, sanity only)"
                             : pred.subsonicImage ? " (subsonic image, sanity only)" : "";
            std::println("clk={:.0f} f0={:.1f}: meas={:.1f} dBc pred={:.1f} dBc{}",
                         static_cast<double> (clk), f0, meas.dbc, pred.dbc, skip);

            CHECK (std::isfinite (meas.dbc));
            if (pred.nulledFund || pred.subsonicImage)
                continue;

            if (pred.dbc >= -50.0)
            {
                CHECK (std::fabs (meas.dbc - pred.dbc) <= 3.5);
            }
            else
            {
                CHECK (meas.dbc <= -46.0);
            }
        }
    }

    // Comparative gate: a probe above the input cutoff. Widening the input
    // bank by 4x raises the probe level into the sampler and every folded
    // image with it. Absolute alias power must rise by at least 12 dB.
    g_section = "comparative_widened_input";
    {
        const double fHi = getCoherentF0 (18500.0, kFs, N);
        std::vector<float> in (N);
        for (int i = 0; i < N; ++i)
            in[static_cast<std::size_t> (i)] =
                0.5f * static_cast<float> (std::sin (2.0 * std::numbers::pi * fHi * i / kFs));

        for (float clk : { 8000.0f, 16000.0f })
        {
            const auto outNormal = renderLine (clk, in, storage.data(), 0.0f);
            const auto outWide = renderLine (clk, in, storage.data(),
                                             MarsDSP::BBD::kInputCutoffHz * 4.0f);

            const auto mN = measureAlias (outNormal, fHi, kFs);
            const auto mW = measureAlias (outWide, fHi, kFs);
            std::println("comparative clk={:.0f}: normal={:.1f} dB widened={:.1f} dB rise={:.1f} dB",
                         static_cast<double> (clk), mN.aliasPowDb, mW.aliasPowDb,
                         mW.aliasPowDb - mN.aliasPowDb);
            CHECK (mW.aliasPowDb >= mN.aliasPowDb + 12.0);
        }
    }

    // In-loop accumulation
    g_section = "in_loop_accumulation";
    {
        const double f0 = getCoherentF0 (1000.0, kFs, N);
        std::vector<float> in (N);
        for (int i = 0; i < N; ++i)
            in[static_cast<std::size_t> (i)] =
                0.1f * static_cast<float> (std::sin (2.0 * std::numbers::pi * f0 * i / kFs));

        // Single pass
        double aliasSingle = 0.0;
        {
            FeedbackDelay fb;
            fb.prepare (kFs, 256, 262144);
            FeedbackDelay::Params p;
            p.delaySamples = 4800.0f; // 100 ms
            p.feedback = 0.0f;
            p.dampHz = 20000.0f;
            p.loopCutHz = 20.0f;
            p.satOrder = 0;
            p.enableDiffuser = false;
            p.delayMode = 1;
            fb.resetParams (p);

            std::vector<float> out (N);
            for (int r = 0; r < 2; ++r)
                for (int pos = 0; pos < N; pos += 256)
                    fb.process (in.data() + pos, nullptr, out.data() + pos, nullptr, 256);

            aliasSingle = measureAlias (out, f0, kFs).dbc;
        }

        // In-loop with feedback = 0.9
        double aliasLoop = 0.0;
        {
            FeedbackDelay fb;
            fb.prepare (kFs, 256, 262144);
            FeedbackDelay::Params p;
            p.delaySamples = 4800.0f;
            p.feedback = 0.9f;
            p.dampHz = 20000.0f;
            p.loopCutHz = 20.0f;
            p.satOrder = 0;
            p.enableDiffuser = false;
            p.delayMode = 1;
            fb.resetParams (p);

            std::vector<float> out (N);
            for (int r = 0; r < 5; ++r)
                for (int pos = 0; pos < N; pos += 256)
                    fb.process (in.data() + pos, nullptr, out.data() + pos, nullptr, 256);

            aliasLoop = measureAlias (out, f0, kFs).dbc;
        }

        std::println("in-loop: single={:.1f} dBc loop={:.1f} dBc (bound single+23)",
                     aliasSingle, aliasLoop);
        CHECK (aliasLoop <= aliasSingle + 23.0);
    }

    std::println("=== bbd_alias_check OK ===");
    return 0;
}

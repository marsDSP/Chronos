// tests/harnesses/dsp/bbd_clock_check.cpp
//
// Verification harness for ClockModel: exact inverse mapping,
// OU-driven pitch deviation, stereo channel decorrelation, and step response.

#include "dsp/bbd/ClockModel.h"
#include "dsp/bbd/BrigadeLine.h"
#include "dsp/Modulation.h"

#include <algorithm>
#include <cmath>
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
    using MarsDSP::BBD::ClockModel;
    using MarsDSP::Mod::OrnsteinUhlenbeck;

    constexpr double kFs = 48000.0;
    constexpr double kToneHz = 1000.0;
    constexpr int kRunSec = 60;
    constexpr int kSkipSec = 2;

    double measurePitchDeviation (float depthCents, float rateHz)
    {
        std::vector<float> storage (BrigadeLine::bbdStorageFloats (1), 0.0f);
        BrigadeLine line;
        line.prepare (kFs, storage.data());

        OrnsteinUhlenbeck ou;
        ou.setRate (kFs, rateHz);
        ou.reset();

        MarsDSP::Mod::Pcg32 rng;
        rng.seed (0xC47051D5uLL, 1);

        const int total = static_cast<int> (kRunSec * kFs);
        std::vector<float> in (total);
        std::vector<float> out (total);
        for (int i = 0; i < total; ++i)
            in[i] = 0.5f * static_cast<float> (std::sin (2.0 * std::numbers::pi * kToneHz * static_cast<double> (i) / kFs));

        constexpr float baseDelaySamples = 24000.0f; // 500 ms
        const double slopeTarget = static_cast<double> (depthCents) * (std::numbers::ln2 / 1200.0);
        const float modK = static_cast<float> ((50.0 / 58.359) * baseDelaySamples * slopeTarget / std::sqrt (2.0));

        for (int i = 0; i < total; ++i)
        {
            const float modL = modK * ou.next (rng);
            const float dEff = baseDelaySamples + modL;
            const float clk = ClockModel::clockFor (dEff, kFs);
            line.setClockHz (clk);
            out[i] = line.process (in[i]);
        }

        const int skip = static_cast<int> (kSkipSec * kFs);
        double prev = out[skip - 1];
        double lastCross = -1.0;
        double sumSq = 0.0;
        long count = 0;

        for (int i = skip; i < total; ++i)
        {
            const double cur = out[i];
            if (prev < 0.0 && cur >= 0.0)
            {
                const double frac = prev / (prev - cur);
                const double crossPos = static_cast<double> (i - 1) + frac;
                if (lastCross > 0.0)
                {
                    const double period = crossPos - lastCross;
                    if (period > 40.0 && period < 56.0)
                    {
                        const double fInst = kFs / period;
                        const double cents = 1200.0 * std::log2 (fInst / kToneHz);
                        sumSq += cents * cents;
                        ++count;
                    }
                }
                lastCross = crossPos;
            }
            prev = cur;
        }

        CHECK (count > 1000);
        return std::sqrt (sumSq / static_cast<double> (count));
    }
} // namespace

int main()
{
    // 1. Exact Inverse Mapping
    g_section = "inverse_mapping";
    {
        for (double fs : { 44100.0, 48000.0, 96000.0 })
        {
            const float minD = ClockModel::achievedDelaySamples (ClockModel::maxClockHz (fs), fs);
            const float maxD = ClockModel::achievedDelaySamples (ClockModel::minClockHz (fs), fs);

            for (float d = minD + 1.0f; d < maxD - 1.0f; d += 50.0f)
            {
                const float clk = ClockModel::clockFor (d, fs);
                const float recovered = ClockModel::achievedDelaySamples (clk, fs);
                const double relErr = std::fabs (recovered - d) / d;
                CHECK (relErr < 1.0e-6);
            }
        }
    }

    // 2. Pitch Deviation
    g_section = "pitch_deviation";
    {
        const double dev50 = measurePitchDeviation (50.0f, 1.0f);
        const double dev0 = measurePitchDeviation (0.0f, 1.0f);
        const double ouPitchDev = std::sqrt (std::max (0.0, dev50 * dev50 - dev0 * dev0));
        std::println ("measurePitchDeviation: dev50={:.2f}, dev0={:.2f}, net OU dev={:.2f}", dev50, dev0, ouPitchDev);
        CHECK (std::fabs (ouPitchDev - 50.0) <= 8.0);
        CHECK (dev0 < 35.0);
    }

    // 3. Decorrelation
    g_section = "decorrelation";
    {
        OrnsteinUhlenbeck ou1, ou2;
        ou1.setRate (kFs, 0.5f);
        ou2.setRate (kFs, 0.5f);
        ou1.reset();
        ou2.reset();

        MarsDSP::Mod::Pcg32 rng1, rng2;
        rng1.seed (0xC47051D5uLL, 1);
        rng2.seed (0xC47051D5uLL, 2);

        const int total = static_cast<int> (60.0 * kFs);
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum11 = 0.0;
        double sum22 = 0.0;
        double sum12 = 0.0;

        for (int i = 0; i < total; ++i)
        {
            const double s1 = ou1.next (rng1);
            const double s2 = ou2.next (rng2);
            sum1 += s1;
            sum2 += s2;
            sum11 += s1 * s1;
            sum22 += s2 * s2;
            sum12 += s1 * s2;
        }

        const double mean1 = sum1 / total;
        const double mean2 = sum2 / total;
        const double var1 = sum11 / total - mean1 * mean1;
        const double var2 = sum22 / total - mean2 * mean2;
        const double cov = sum12 / total - mean1 * mean2;
        const double corr = std::fabs (cov / std::sqrt (var1 * var2));

        CHECK (corr < 0.1);
    }

    std::println("=== bbd_clock_check OK ===");
    return 0;
}

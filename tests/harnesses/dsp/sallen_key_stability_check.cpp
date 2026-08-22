// tests/harnesses/dsp/sallen_key_stability_check.cpp
//
// Stability, ring-out, denormal hygiene, sweep robustness, non-finite recovery,
// and reset determinism harness for SallenKeyLPF and SallenKeyHPF.

#include "dsp/SallenKeyLPF.h"
#include "dsp/SallenKeyHPF.h"
#include "../perf/bench_util.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <print>
#include <cstdlib>
#include <limits>
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

    using MarsDSP::Filters::SallenKeyLPF;
    using MarsDSP::Filters::SallenKeyHPF;

    template <typename Filter>
    double runRingOut (Filter& filter, double fs, float f0, float q)
    {
        filter.prepare (fs);
        filter.setParams (f0, q);

        for (int i = 0; i < 64; ++i)
        {
            const float in = (i % 2 == 0) ? 1.0f : -1.0f;
            const float y = filter.processSample (in);
            CHECK (std::isfinite (y));
        }

        double maxTail = 0.0;
        for (int i = 0; i < 400000; ++i)
        {
            const float y = filter.processSample (0.0f);
            CHECK (std::isfinite (y));
            if (i >= 200000)
                maxTail = std::max (maxTail, static_cast<double> (std::fabs (y)));
        }
        return maxTail;
    }
} // namespace

int main()
{
    const std::array<double, 6> sampleRates { { 44100.0, 48000.0, 88200.0, 96000.0, 176400.0, 192000.0 } };
    constexpr int numCutoffs = 61;

    // 1. Ring-out
    g_section = "ring_out";
    {
        bench::setFtzDaz();
        double worstTail = 0.0;
        for (double fs : sampleRates)
        {
            for (int fi = 0; fi < numCutoffs; ++fi)
            {
                const float f0 = static_cast<float> (20.0 * std::pow (1000.0, static_cast<double> (fi) / (numCutoffs - 1)));
                if (f0 >= fs * 0.49f) continue;

                SallenKeyLPF lpf;
                const double tailLpf = runRingOut (lpf, fs, f0, 0.7071f);
                worstTail = std::max (worstTail, tailLpf);
                CHECK (tailLpf < 1.0e-30);

                SallenKeyHPF hpf;
                const double tailHpf = runRingOut (hpf, fs, f0, 0.7071f);
                worstTail = std::max (worstTail, tailHpf);
                CHECK (tailHpf < 1.0e-30);
            }
        }
    }

    // 2. Parameter sweep under audio
    g_section = "sweep_under_audio";
    {
        for (double fs : { 44100.0, 48000.0, 96000.0, 192000.0 })
        {
            const int totalSamples = static_cast<int> (fs * 10.0);
            SallenKeyLPF lpf;
            lpf.prepare (fs);

            SallenKeyHPF hpf;
            hpf.prepare (fs);

            float maxOutLpf = 0.0f;
            float maxOutHpf = 0.0f;

            for (int i = 0; i < totalSamples; ++i)
            {
                // Triangle sweep on [0, 1]
                const double phase = static_cast<double> (i) / totalSamples;
                const double tri = (phase < 0.5) ? 2.0 * phase : 2.0 * (1.0 - phase);
                const float f = static_cast<float> (10.0 * std::pow (fs * 0.49 / 10.0, tri));

                if (i % 32 == 0)
                {
                    lpf.setParams (f, 2.0f);
                    hpf.setParams (f, 2.0f);
                }

                const float in = std::sin (static_cast<float> (2.0 * std::numbers::pi * 1000.0 * i / fs));
                const float yL = lpf.processSample (in);
                const float yH = hpf.processSample (in);

                CHECK (std::isfinite (yL));
                CHECK (std::isfinite (yH));

                maxOutLpf = std::max (maxOutLpf, std::fabs (yL));
                maxOutHpf = std::max (maxOutHpf, std::fabs (yH));
            }

            CHECK (maxOutLpf <= 8.0f);
            CHECK (maxOutHpf <= 8.0f);
        }
    }

    // 3. Non-finite recovery
    g_section = "non_finite_recovery";
    {
        constexpr double fs = 48000.0;
        const float badInputs[] = {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()
        };

        for (float bad : badInputs)
        {
            SallenKeyLPF lpf;
            lpf.prepare (fs);
            lpf.setParams (1000.0f, 0.7071f);

            lpf.processSample (bad);
            lpf.reset();

            // Check recovery within 64 samples
            for (int i = 0; i < 64; ++i)
            {
                const float y = lpf.processSample (0.1f);
                CHECK (std::isfinite (y));
            }

            SallenKeyHPF hpf;
            hpf.prepare (fs);
            hpf.setParams (1000.0f, 0.7071f);

            hpf.processSample (bad);
            hpf.reset();

            for (int i = 0; i < 64; ++i)
            {
                const float y = hpf.processSample (0.1f);
                CHECK (std::isfinite (y));
            }
        }
    }

    // 4. Reset determinism
    g_section = "reset_determinism";
    {
        constexpr double fs = 48000.0;
        SallenKeyLPF lpf;
        lpf.prepare (fs);
        lpf.setParams (1000.0f, 0.7071f);

        std::vector<float> run1 (1000), run2 (1000);
        for (int i = 0; i < 1000; ++i)
            run1[i] = lpf.processSample (std::sin (0.1f * i));

        lpf.reset();

        for (int i = 0; i < 1000; ++i)
            run2[i] = lpf.processSample (std::sin (0.1f * i));

        for (int i = 0; i < 1000; ++i)
            CHECK (run1[i] == run2[i]);
    }

    // 5. Extreme cutoffs
    g_section = "extreme_cutoffs";
    {
        constexpr double fs = 48000.0;
        SallenKeyLPF lpf;
        lpf.prepare (fs);

        lpf.setParams (0.0f, 0.7071f);
        float y0 = lpf.processSample (1.0f);
        CHECK (std::isfinite (y0));

        lpf.setParams (static_cast<float> (10.0 * fs), 0.7071f);
        float yInf = lpf.processSample (1.0f);
        CHECK (std::isfinite (yInf));
    }

    std::println("=== sallen_key_stability_check OK ===");
    return 0;
}

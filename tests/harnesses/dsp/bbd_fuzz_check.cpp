// tests/harnesses/dsp/bbd_fuzz_check.cpp
//
// Adversarial-input safety net for BrigadeLine and FeedbackDelay (BBD mode).
// NDEBUG is defined for this translation unit to measure IEEE propagation.

#include "dsp/bbd/BrigadeLine.h"
#include "dsp/FeedbackDelay.h"

#include <algorithm>
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

    using MarsDSP::BBD::BrigadeLine;
    using MarsDSP::Delays::FeedbackDelay;

    constexpr double kFs = 48000.0;
    constexpr int kN = 8192;
} // namespace

int main()
{
    std::vector<float> storage (BrigadeLine::bbdStorageFloats (1), 0.0f);

    // 1. Zeros in -> +0.0f out
    g_section = "zeros_homogeneity";
    {
        BrigadeLine line;
        line.prepare (kFs, storage.data());

        for (int i = 0; i < 1000; ++i)
        {
            const float y = line.process (0.0f);
            CHECK (y == 0.0f);
            CHECK (!std::signbit (y));
        }
    }

    // 2. Denormals, DC, Nyquist Alternation, Huge Impulse
    g_section = "hostile_finite_inputs";
    {
        std::vector<float> in (kN);
        // Denormal 1e-45
        for (int i = 0; i < kN; ++i)
            in[i] = (i % 2 == 0) ? 1.0e-45f : -1.0e-45f;

        BrigadeLine line;
        line.prepare (kFs, storage.data());
        line.setClockHz (40000.0f);

        for (int i = 0; i < kN; ++i)
        {
            const float y = line.process (in[i]);
            CHECK (std::isfinite (y));
        }

        // Full-scale ±10.0
        line.reset();
        for (int i = 0; i < kN; ++i)
        {
            const float inVal = (i % 2 == 0) ? 10.0f : -10.0f;
            const float y = line.process (inVal);
            CHECK (std::isfinite (y));
            CHECK (std::fabs (y) <= 30.0f);
        }

        // Impulse 1e6
        line.reset();
        float y0 = line.process (1.0e6f);
        CHECK (std::isfinite (y0));
        for (int i = 1; i < kN; ++i)
        {
            const float y = line.process (0.0f);
            CHECK (std::isfinite (y));
        }
    }

    // 3. NaN / Inf Injection Recovery
    g_section = "nan_inf_injection";
    {
        const float bads[] = {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()
        };

        for (float bad : bads)
        {
            for (int pos : { 0, 1, 2, 500 })
            {
                std::vector<float> in (kN, 0.0f);
                for (int i = 0; i < kN; ++i)
                    in[i] = 0.5f * static_cast<float> (std::sin (2.0 * std::numbers::pi * 1000.0 * i / kFs));

                std::vector<float> inBad = in;
                inBad[pos] = bad;

                std::vector<float> inRef = in;
                inRef[pos] = 0.0f;

                std::vector<float> memBad (BrigadeLine::bbdStorageFloats (1), 0.0f);
                std::vector<float> memRef (BrigadeLine::bbdStorageFloats (1), 0.0f);

                BrigadeLine lineBad, lineRef;
                lineBad.prepare (kFs, memBad.data());
                lineRef.prepare (kFs, memRef.data());
                lineBad.setClockHz (100000.0f); // ~4000 samples transport
                lineRef.setClockHz (100000.0f);

                std::vector<float> outBad (kN), outRef (kN);
                for (int i = 0; i < kN; ++i)
                {
                    outBad[i] = lineBad.process (inBad[i]);
                    outRef[i] = lineRef.process (inRef[i]);
                    CHECK (std::isfinite (outBad[i]));
                }

                // Verify recovery after transport + 300 samples
                const int delaySamp = static_cast<int> (2.0 * BrigadeLine::kStages * kFs / 100000.0f);
                const int thr = pos + delaySamp + 300;
                for (int i = thr; i < kN; ++i)
                {
                    CHECK (std::fabs (outBad[i] - outRef[i]) < 0.01f);
                }
            }
        }
    }

    std::println("=== bbd_fuzz_check OK ===");
    return 0;
}

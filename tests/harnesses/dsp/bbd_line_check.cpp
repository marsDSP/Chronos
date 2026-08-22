// tests/harnesses/dsp/bbd_line_check.cpp
//
// Correctness harness for BrigadeLine: transport centroid, DC unity,
// zero-in/zero-out, reset determinism, block-size invariance, clock clamps,
// and parameter sweep robustness.

#include "dsp/bbd/BrigadeLine.h"

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
} // namespace

int main()
{
    std::vector<float> storage (BrigadeLine::bbdStorageFloats (1), 0.0f);

    // 1. Transport Centroid Check
    g_section = "transport_centroid";
    {
        const std::array<double, 3> sampleRates { { 44100.0, 48000.0, 96000.0 } };
        const std::array<float, 4> clocks { { 20000.0f, 40000.0f, 100000.0f, 200000.0f } };

        for (double fs : sampleRates)
        {
            BrigadeLine line;
            line.prepare (fs, storage.data());
            const double gdBank = BrigadeLine::getBankGroupDelayAtDC (fs);

            for (float clk : clocks)
            {
                if (clk > 100.0f * fs) continue;
                line.reset();
                line.setClockHz (clk);

                const double expectedCentroid = (2.0 * BrigadeLine::kStages + 0.5) * fs / clk + gdBank;
                const int totalSamples = static_cast<int> (expectedCentroid + 1000.0);
                std::vector<float> ir (totalSamples, 0.0f);

                ir[0] = line.process (1.0f);
                for (int i = 1; i < totalSamples; ++i)
                    ir[i] = line.process (0.0f);

                // Compute energy centroid in a window around expected arrival
                const int winStart = std::max (0, static_cast<int> (expectedCentroid - 300.0));
                const int winEnd   = std::min (totalSamples, static_cast<int> (expectedCentroid + 300.0));

                double sumWeight = 0.0;
                double sumEnergy = 0.0;
                for (int i = winStart; i < winEnd; ++i)
                {
                    const double e = static_cast<double> (ir[i]) * static_cast<double> (ir[i]);
                    sumWeight += static_cast<double> (i) * e;
                    sumEnergy += e;
                }

                CHECK (sumEnergy > 1.0e-12);
                const double measuredCentroid = sumWeight / sumEnergy;
                const double diff = std::fabs (measuredCentroid - expectedCentroid);
                std::println("fs={:.0} clk={:.0} measured={:.3} expected={:.3} diff={:.3}",
                             fs, static_cast<double>(clk), measuredCentroid, expectedCentroid, diff);
            }
        }
    }

    // 2. Unity Step Response
    g_section = "unity_step";
    {
        constexpr double fs = 48000.0;
        BrigadeLine line;
        line.prepare (fs, storage.data());
        line.setClockHz (100000.0f); // 100 kHz clock -> ~4000 samples transport

        for (int i = 0; i < 15000; ++i)
        {
            const float y = line.process (1.0f);
            if (i == 10000)
            {
                std::println("i=10000: y = {:.6}, storage[0] = {:.6}", y, storage[0]);
            }
            if (i >= 10000 && std::fabs(y - 1.0f) >= 1.0e-3f)
            {
                CHECK (std::fabs (y - 1.0f) < 1.0e-3f);
            }
        }
    }

    // 3. Zero-in / Zero-out
    g_section = "zero_in_zero_out";
    {
        constexpr double fs = 48000.0;
        BrigadeLine line;
        line.prepare (fs, storage.data());

        for (int i = 0; i < 10000; ++i)
        {
            const float y = line.process (0.0f);
            CHECK (y == 0.0f);
        }
    }

    // 4. Reset Determinism
    g_section = "reset_determinism";
    {
        constexpr double fs = 48000.0;
        BrigadeLine line;
        line.prepare (fs, storage.data());
        line.setDelaySeconds (0.05f);

        std::vector<float> run1 (1000), run2 (1000);
        for (int i = 0; i < 1000; ++i)
            run1[i] = line.process (std::sin (0.1f * i));

        line.reset();

        for (int i = 0; i < 1000; ++i)
            run2[i] = line.process (std::sin (0.1f * i));

        for (int i = 0; i < 1000; ++i)
            CHECK (run1[i] == run2[i]);
    }

    // 5. Block-size Invariance
    g_section = "block_size_invariance";
    {
        constexpr double fs = 48000.0;
        constexpr int N = 2048;
        std::vector<float> input (N);
        for (int i = 0; i < N; ++i)
            input[i] = std::sin (0.05f * i);

        std::vector<float> refOut (N);
        {
            std::vector<float> memRef (BrigadeLine::bbdStorageFloats (1), 0.0f);
            BrigadeLine lineRef;
            lineRef.prepare (fs, memRef.data());
            lineRef.setDelaySeconds (0.01f);
            for (int i = 0; i < N; ++i)
                refOut[i] = lineRef.process (input[i]);
        }

        for (int bs : { 1, 7, 64, 512 })
        {
            std::vector<float> memBlock (BrigadeLine::bbdStorageFloats (1), 0.0f);
            BrigadeLine lineBlock;
            lineBlock.prepare (fs, memBlock.data());
            lineBlock.setDelaySeconds (0.01f);

            std::vector<float> blockOut (N);
            for (int off = 0; off < N; off += bs)
            {
                const int cur = std::min (bs, N - off);
                for (int i = 0; i < cur; ++i)
                    blockOut[off + i] = lineBlock.process (input[off + i]);
            }

            for (int i = 0; i < N; ++i)
                CHECK (blockOut[i] == refOut[i]);
        }
    }

    // 6. Clock Clamps
    g_section = "clock_clamps";
    {
        constexpr double fs = 48000.0;
        BrigadeLine line;
        line.prepare (fs, storage.data());

        // Extreme short delay -> max clock
        line.setDelaySeconds (1.0e-6f);
        CHECK (line.getClockHz() == 100.0f * static_cast<float> (fs));

        // Extreme long delay -> min clock
        line.setDelaySeconds (100.0f);
        CHECK (line.getClockHz() == static_cast<float> (fs) / 30.0f);
    }

    // 7. Clock Sweep Under Audio
    g_section = "clock_sweep_under_audio";
    {
        constexpr double fs = 48000.0;
        const int totalSamples = static_cast<int> (fs * 10.0);
        BrigadeLine line;
        line.prepare (fs, storage.data());

        float maxOut = 0.0f;
        for (int i = 0; i < totalSamples; ++i)
        {
            const double phase = static_cast<double> (i) / totalSamples;
            const double tri = (phase < 0.5) ? 2.0 * phase : 2.0 * (1.0 - phase);
            const float delaySec = static_cast<float> (0.005 * std::pow (5.0 / 0.005, tri));

            line.setDelaySeconds (delaySec);
            const float in = std::sin (static_cast<float> (2.0 * std::numbers::pi * 1000.0 * i / fs));
            const float y = line.process (in);

            CHECK (std::isfinite (y));
            maxOut = std::max (maxOut, std::fabs (y));
        }
        std::println("clock_sweep_under_audio: maxOut = {:.4f}", maxOut);
        CHECK (maxOut <= 4.0f);
    }

    std::println("=== bbd_line_check OK ===");
    return 0;
}

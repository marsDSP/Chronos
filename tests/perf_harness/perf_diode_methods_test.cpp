#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <numbers>
#include <iomanip>
#include <filesystem>
#include "dsp/math/fastermath.h"
#include "dsp/wdf/wdf_core.h"
#include "dsp/wdf/wdf_diode_pair.h"

// Per-block timing of the two WDF DiodePair quality settings (omega3 / omega4)
// and the two SIMD ADAA sinh variants (slow div_ps / fast rcp_ps + Newton
// refinement).

int main()
{
    namespace W = MarsDSP::DSP::WDF;
    std::filesystem::create_directories ("tests/perf_harness/logs");

    constexpr int    blockSize  = 512;
    constexpr int    iterations = 20000;
    constexpr float  Rs         = 1.0e3f;
    constexpr float  Is         = 2.52e-9f;
    constexpr float  Vt         = 0.02585f;

    std::vector<float> input  (blockSize);
    std::vector<float> output (blockSize);
    for (int i = 0; i < blockSize; ++i)
        input[i] = 1.5f * std::sin (2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / 64.0f);

    auto micros = [](auto a, auto b)
    {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    // ---------- WDF DiodePair (Accurate / Wright-Omega 4) ----------
    double timeWdfAcc = 0.0;
    {
        W::ResistiveVoltageSource<float> Vs (Rs);
        W::DiodePair<float, decltype (Vs), W::DiodeQuality::Accurate> diode (Vs, Is, Vt, 1.0f);

        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it)
        {
            for (int i = 0; i < blockSize; ++i)
            {
                Vs.setVoltage (input[i]);
                diode.incident (Vs.reflected());
                Vs.incident (diode.reflected());
                output[i] = W::voltage<float> (Vs);
            }
            if (output[0] > 1e9f) std::cout << "x";
        }
        timeWdfAcc = micros (t0, std::chrono::high_resolution_clock::now()) / iterations;
    }

    // ---------- WDF DiodePair (Fast / Wright-Omega 3) ----------
    double timeWdfFast = 0.0;
    {
        W::ResistiveVoltageSource<float> Vs (Rs);
        W::DiodePair<float, decltype (Vs), W::DiodeQuality::Fast> diode (Vs, Is, Vt, 1.0f);

        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it)
        {
            for (int i = 0; i < blockSize; ++i)
            {
                Vs.setVoltage (input[i]);
                diode.incident (Vs.reflected());
                Vs.incident (diode.reflected());
                output[i] = W::voltage<float> (Vs);
            }
            if (output[0] > 1e9f) std::cout << "x";
        }
        timeWdfFast = micros (t0, std::chrono::high_resolution_clock::now()) / iterations;
    }

    // ---------- ADAA sinh slow path (existing div_ps) ----------
    double timeAdaaSinhSlow = 0.0;
    {
        float carryX = 0.0f, carryF = 0.0f;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it)
        {
            for (int i = 0; i < blockSize; i += 4)
            {
                const auto vIn  = SIMD_MM(loadu_ps)(input.data() + i);
                const auto vOut = MarsDSP::FasterMath::fasterSinhADAA (vIn, carryX, carryF);
                SIMD_MM(storeu_ps)(output.data() + i, vOut);
            }
            if (output[0] > 1e9f) std::cout << "x";
        }
        timeAdaaSinhSlow = micros (t0, std::chrono::high_resolution_clock::now()) / iterations;
    }

    // ---------- ADAA sinh fast path (rcp_ps + Newton refine) ----------
    double timeAdaaSinhFast = 0.0;
    {
        float carryX = 0.0f, carryF = 0.0f;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it)
        {
            for (int i = 0; i < blockSize; i += 4)
            {
                const auto vIn  = SIMD_MM(loadu_ps)(input.data() + i);
                const auto vOut = MarsDSP::FasterMath::fasterSinhADAA_fast (vIn, carryX, carryF);
                SIMD_MM(storeu_ps)(output.data() + i, vOut);
            }
            if (output[0] > 1e9f) std::cout << "x";
        }
        timeAdaaSinhFast = micros (t0, std::chrono::high_resolution_clock::now()) / iterations;
    }

    std::ofstream csv ("tests/perf_harness/logs/perf_diode_methods.csv");
    csv << "method,avg_time_us,relative_speedup_vs_wdf_accurate\n";
    csv << std::fixed << std::setprecision (4);
    csv << "WDF DiodePair (omega4)," << timeWdfAcc       << "," << 1.0                             << "\n";
    csv << "WDF DiodePair (omega3)," << timeWdfFast      << "," << (timeWdfAcc / timeWdfFast)      << "\n";
    csv << "ADAA sinh (slow),"       << timeAdaaSinhSlow << "," << (timeWdfAcc / timeAdaaSinhSlow) << "\n";
    csv << "ADAA sinh (fast),"       << timeAdaaSinhFast << "," << (timeWdfAcc / timeAdaaSinhFast) << "\n";

    std::cout << "\nResults (us per " << blockSize << "-sample block, "
              << iterations << " iterations):\n";
    std::cout << std::fixed << std::setprecision (3);
    std::cout << "  WDF DiodePair (omega4): " << std::setw (8) << timeWdfAcc       << " us  (ref)\n";
    std::cout << "  WDF DiodePair (omega3): " << std::setw (8) << timeWdfFast      << " us  ("
              << (timeWdfAcc / timeWdfFast) << "x faster)\n";
    std::cout << "  ADAA sinh (slow div):   " << std::setw (8) << timeAdaaSinhSlow << " us\n";
    std::cout << "  ADAA sinh (fast rcp):   " << std::setw (8) << timeAdaaSinhFast << " us  ("
              << (timeAdaaSinhSlow / timeAdaaSinhFast) << "x vs slow)\n";

    return 0;
}

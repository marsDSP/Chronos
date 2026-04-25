#include <iostream>
#include <fstream>
#include <cmath>
#include <numbers>
#include <vector>
#include <iomanip>
#include <filesystem>
#include "dsp/math/fastermath.h"

// Two CSVs:
//   1) simd_cosh_results.csv      static correctness vs std::cosh on [-π, π]
//   2) simd_cosh_adaa_results.csv ADAA on a swept-frequency sine input,
//                                 compared to direct cosh on the same signal.

int main()
{
    std::filesystem::create_directories("tests/simd_harness/logs");

    // ============================================================ //
    // 1. Static correctness: linear sweep through [-π, π]
    // ============================================================ //
    {
        const float start = -std::numbers::pi_v<float>;
        const float end   =  std::numbers::pi_v<float>;
        const int   steps = 4096;
        const float step_size = (end - start) / steps;

        std::ofstream csv("tests/simd_harness/logs/simd_cosh_results.csv");
        if (!csv.is_open())
        {
            std::cerr << "Failed to open tests/simd_harness/logs/simd_cosh_results.csv" << std::endl;
            return 1;
        }

        csv << "x,std_cosh,pade_scalar,pade_simd,abs_err_scalar,abs_err_simd,diff_simd_scalar\n";
        csv << std::fixed << std::setprecision(10);

        for (int i = 0; i <= steps; i += 4)
        {
            float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int j = 0; j < 4; ++j)
            {
                x_vals[j] = (i + j <= steps)
                          ? start + static_cast<float>(i + j) * step_size
                          : 0.0f;
            }

            float std_results[4];
            float scalar_results[4];
            for (int j = 0; j < 4; ++j)
            {
                std_results[j]    = std::cosh(x_vals[j]);
                scalar_results[j] = MarsDSP::padeCoshApprox(x_vals[j]);
            }

            SIMD_M128 vx   = SIMD_MM(set_ps)(x_vals[3], x_vals[2], x_vals[1], x_vals[0]);
            SIMD_M128 vres = MarsDSP::fasterCosh(vx);

            float simd_results[4];
            SIMD_MM(storeu_ps)(simd_results, vres);

            for (int j = 0; j < 4; ++j)
            {
                if (i + j > steps) break;

                const float val_std    = std_results[j];
                const float val_scalar = scalar_results[j];
                const float val_simd   = simd_results[j];

                csv << x_vals[j]                            << ","
                    << val_std                              << ","
                    << val_scalar                           << ","
                    << val_simd                             << ","
                    << std::abs(val_std    - val_scalar)    << ","
                    << std::abs(val_std    - val_simd)      << ","
                    << std::abs(val_scalar - val_simd)      << "\n";
            }
        }
        csv.close();
        std::cout << "Generated tests/simd_harness/logs/simd_cosh_results.csv ("
                  << (steps + 1) << " points)." << std::endl;
    }

    // ============================================================ //
    // 2. ADAA correctness on a chirp signal.
    // ============================================================ //
    {
        const float sampleRate = 48000.0f;
        const int   numSamples = 4096;
        const float fStart     = 100.0f;
        const float fEnd       = 8000.0f;
        const float amplitude  = 1.5f;

        std::ofstream csv("tests/simd_harness/logs/simd_cosh_adaa_results.csv");
        if (!csv.is_open())
        {
            std::cerr << "Failed to open tests/simd_harness/logs/simd_cosh_adaa_results.csv" << std::endl;
            return 1;
        }

        csv << "n,x,direct,adaa,diff\n";
        csv << std::fixed << std::setprecision(10);

        std::vector<float> input(numSamples);
        const float k = (fEnd - fStart) / (numSamples / sampleRate);
        float phase = 0.0f;
        for (int n = 0; n < numSamples; ++n)
        {
            const float t = n / sampleRate;
            const float instFreq = fStart + k * t;
            phase += 2.0f * std::numbers::pi_v<float> * instFreq / sampleRate;
            input[n] = amplitude * std::sin(phase);
        }

        std::vector<float> direct(numSamples);
        std::vector<float> adaa(numSamples);

        for (int i = 0; i < numSamples; i += 4)
        {
            SIMD_M128 vx   = SIMD_MM(loadu_ps)(&input[i]);
            SIMD_M128 vres = MarsDSP::fasterCosh(vx);
            SIMD_MM(storeu_ps)(&direct[i], vres);
        }

        {
            float carryX = 0.0f;
            float carryF = 0.0f;
            for (int i = 0; i < numSamples; i += 4)
            {
                SIMD_M128 vx   = SIMD_MM(loadu_ps)(&input[i]);
                SIMD_M128 vres = MarsDSP::fasterCoshADAA(vx, carryX, carryF);
                SIMD_MM(storeu_ps)(&adaa[i], vres);
            }
        }

        for (int n = 0; n < numSamples; ++n)
        {
            csv << n           << ","
                << input[n]    << ","
                << direct[n]   << ","
                << adaa[n]     << ","
                << std::abs(direct[n] - adaa[n]) << "\n";
        }
        csv.close();
        std::cout << "Generated tests/simd_harness/logs/simd_cosh_adaa_results.csv ("
                  << numSamples << " samples)." << std::endl;
    }

    return 0;
}

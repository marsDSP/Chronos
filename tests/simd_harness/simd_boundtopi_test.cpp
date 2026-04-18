#include <iostream>
#include <fstream>
#include <cmath>
#include <numbers>
#include <vector>
#include <iomanip>
#include "dsp/math/fastermath.h"

int main()
{
    const float start = -10.0f * M_PI;
    const float end = 10.0f * M_PI;
    const int steps = 4096;
    const float step_size = (end - start) / steps;

    std::ofstream csv("tests/simd_harness/logs/simd_boundtopi_results.csv");
    if (!csv.is_open())
    {
        std::cerr << "Failed to open tests/simd_harness/logs/simd_boundtopi_results.csv" << std::endl;
        return 1;
    }

    csv << "x,scalar_res,simd_res,abs_diff\n";
    csv << std::fixed << std::setprecision(10);

    for (int i = 0; i <= steps; i += 4)
    {
        float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < 4; ++j)
        {
            if (i + j <= steps)
                x_vals[j] = start + static_cast<float>(i + j) * step_size;
            else
                x_vals[j] = 0.0f; // pad with zeros
        }

        // 1. Calculate scalar boundToPi
        float scalar_results[4];
        for (int j = 0; j < 4; ++j)
        {
            scalar_results[j] = MarsDSP::boundToPi(x_vals[j]);
        }

        // 2. Calculate SIMD boundToPi
        SIMD_M128 vx = SIMD_MM(set_ps)(x_vals[3], x_vals[2], x_vals[1], x_vals[0]);
        SIMD_M128 vres = MarsDSP::boundToPiSIMD(vx);

        float simd_results[4];
        SIMD_MM(storeu_ps)(simd_results, vres);

        // 3. Output results
        for (int j = 0; j < 4; ++j)
        {
            if (i + j > steps) break;

            float x = x_vals[j];
            float val_scalar = scalar_results[j];
            float val_simd = simd_results[j];

            float diff = std::abs(val_scalar - val_simd);

            csv << x << ","
                << val_scalar << ","
                << val_simd << ","
                << diff << "\n";
        }
    }

    csv.close();
    std::cout << "Successfully generated tests/simd_harness/logs/simd_boundtopi_results.csv with " << (steps + 1) << " data points." << std::endl;

    return 0;
}

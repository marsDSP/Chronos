#include <iostream>
#include <fstream>
#include <cmath>
#include <numbers>
#include <vector>
#include <iomanip>
#include "dsp/math/fastermath.h"

int main()
{
    const float start = -5.0f;
    const float end = 5.0f;
    const int steps = 4096;
    const float step_size = (end - start) / steps;

    std::ofstream csv("tests/simd_harness/logs/simd_tanh_results.csv");
    if (!csv.is_open())
    {
        std::cerr << "Failed to open tests/simd_harness/logs/simd_tanh_results.csv" << std::endl;
        return 1;
    }

    csv << "x,std_tanh,pade_scalar,pade_simd,pade_simd_bounded,abs_err_scalar,abs_err_simd,abs_err_simd_bounded\n";
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

        // 1. Calculate std::tanh and scalar Pade
        float std_results[4];
        float scalar_results[4];
        for (int j = 0; j < 4; ++j)
        {
            std_results[j] = std::tanh(x_vals[j]);
            scalar_results[j] = MarsDSP::padeTanhApprox(x_vals[j]);
        }

        // 2. Calculate SIMD Pade
        SIMD_M128 vx = SIMD_MM(set_ps)(x_vals[3], x_vals[2], x_vals[1], x_vals[0]);
        SIMD_M128 vres = MarsDSP::fasterTanh(vx);
        SIMD_M128 vres_bounded = MarsDSP::fasterTanhBounded(vx);

        float simd_results[4];
        float simd_bounded_results[4];
        SIMD_MM(storeu_ps)(simd_results, vres);
        SIMD_MM(storeu_ps)(simd_bounded_results, vres_bounded);

        // 3. Output results
        for (int j = 0; j < 4; ++j)
        {
            if (i + j > steps) break;

            float x = x_vals[j];
            float val_std = std_results[j];
            float val_scalar = scalar_results[j];
            float val_simd = simd_results[j];
            float val_simd_bounded = simd_bounded_results[j];

            float err_scalar = std::abs(val_std - val_scalar);
            float err_simd = std::abs(val_std - val_simd);
            float err_simd_bounded = std::abs(val_std - val_simd_bounded);

            csv << x << ","
                << val_std << ","
                << val_scalar << ","
                << val_simd << ","
                << val_simd_bounded << ","
                << err_scalar << ","
                << err_simd << ","
                << err_simd_bounded << "\n";
        }
    }

    csv.close();
    std::cout << "Successfully generated tests/simd_harness/logs/simd_tanh_results.csv with " << (steps + 1) << " data points." << std::endl;

    return 0;
}

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <numbers>
#include <iomanip>
#include <filesystem>
#include "dsp/math/fastermath.h"

int main()
{
    const int blockSize = 512;
    const int iterations = 1000000;

    // Ensure the logs directory exists
    std::filesystem::create_directories("tests/perf_harness/logs");

    std::vector<float> input(blockSize);
    for (int i = 0; i < blockSize; ++i)
    {
        // sinh grows exponentially - keep input bounded to [-π, π]
        // (the design domain of the Padé). Same range as sin perf for parity.
        input[i] = -std::numbers::pi_v<float>
                 + static_cast<float>(i) / blockSize * 2.0f * std::numbers::pi_v<float>;
    }

    std::vector<float> output(blockSize);

    std::cout << "Benchmarking hyperbolic sine implementations (Block Size: " << blockSize
              << ", Iterations: " << iterations << ")..." << std::endl;

    // 1. Benchmark std::sinh (Scalar Baseline)
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        for (int i = 0; i < blockSize; ++i)
        {
            output[i] = std::sinh(input[i]);
        }
        if (output[0] > 1e9f) std::cout << "Never happens";
    }
    auto end = std::chrono::high_resolution_clock::now();
    double timeStd = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

    // 2. Benchmark Pade (Scalar)
    start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        for (int i = 0; i < blockSize; ++i)
        {
            output[i] = MarsDSP::padeSinhApprox(input[i]);
        }
        if (output[0] > 1e9f) std::cout << "Never happens";
    }
    end = std::chrono::high_resolution_clock::now();
    double timeScalar = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

    // 3. Benchmark Pade (SIMD)
    start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        for (int i = 0; i < blockSize; i += 4)
        {
            SIMD_M128 vx = SIMD_MM(loadu_ps)(&input[i]);
            SIMD_M128 vres = MarsDSP::fasterSinh(vx);
            SIMD_MM(storeu_ps)(&output[i], vres);
        }
        if (output[0] > 1e9f) std::cout << "Never happens";
    }
    end = std::chrono::high_resolution_clock::now();
    double timeSimd = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

    // 4. Benchmark ADAA (SIMD)
    start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        float carryX = 0.0f;
        float carryF = 0.0f;
        for (int i = 0; i < blockSize; i += 4)
        {
            SIMD_M128 vx = SIMD_MM(loadu_ps)(&input[i]);
            SIMD_M128 vres = MarsDSP::fasterSinhADAA(vx, carryX, carryF);
            SIMD_MM(storeu_ps)(&output[i], vres);
        }
        if (output[0] > 1e9f) std::cout << "Never happens";
    }
    end = std::chrono::high_resolution_clock::now();
    double timeAdaa = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

    // Output to CSV
    std::ofstream csv("tests/perf_harness/logs/perf_sinh_results.csv");
    if (!csv.is_open())
    {
        std::cerr << "Failed to open tests/perf_harness/logs/perf_sinh_results.csv" << std::endl;
        return 1;
    }

    csv << "algorithm,avg_time_us,speedup\n";
    csv << std::fixed << std::setprecision(6);
    csv << "std::sinh," << timeStd << ",1.0\n";
    csv << "Pade (Scalar)," << timeScalar << "," << (timeStd / timeScalar) << "\n";
    csv << "Pade (SIMD),"   << timeSimd   << "," << (timeStd / timeSimd)   << "\n";
    csv << "ADAA (SIMD),"   << timeAdaa   << "," << (timeStd / timeAdaa)   << "\n";
    csv.close();

    std::cout << "\nResults (Average time per block of " << blockSize << " samples):" << std::endl;
    std::cout << "  std::sinh:       " << std::setw(8) << timeStd    << " us"                                                << std::endl;
    std::cout << "  Pade (Scalar):   " << std::setw(8) << timeScalar << " us (" << (timeStd / timeScalar) << "x faster)"     << std::endl;
    std::cout << "  Pade (SIMD):     " << std::setw(8) << timeSimd   << " us (" << (timeStd / timeSimd)   << "x faster)"     << std::endl;
    std::cout << "  ADAA (SIMD):     " << std::setw(8) << timeAdaa   << " us (" << (timeStd / timeAdaa)   << "x faster)"     << std::endl;

    return 0;
}

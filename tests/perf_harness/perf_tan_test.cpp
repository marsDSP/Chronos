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
        // Use a safe range for tan [-1.5, 1.5]
        input[i] = -1.5f + static_cast<float>(i) / blockSize * 3.0f;
    }

    std::vector<float> output(blockSize);

    std::cout << "Benchmarking tangent implementations (Block Size: " << blockSize << ", Iterations: " << iterations << ")..." << std::endl;

    // 1. Benchmark std::tan (Scalar Baseline)
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        for (int i = 0; i < blockSize; ++i)
        {
            output[i] = std::tan(input[i]);
        }
        // Use the output to prevent compiler optimization
        if (output[0] > 1000.0f) std::cout << "Never happens";
    }
    auto end = std::chrono::high_resolution_clock::now();
    double timeStd = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

    // 2. Benchmark Pade (Scalar)
    start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        for (int i = 0; i < blockSize; ++i)
        {
            output[i] = MarsDSP::padeTanApprox(input[i]);
        }
        if (output[0] > 1000.0f) std::cout << "Never happens";
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
            SIMD_M128 vres = MarsDSP::fasterTan(vx);
            SIMD_MM(storeu_ps)(&output[i], vres);
        }
        if (output[0] > 1000.0f) std::cout << "Never happens";
    }
    end = std::chrono::high_resolution_clock::now();
    double timeSimd = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

    // Output to CSV
    std::ofstream csv("tests/perf_harness/logs/perf_tan_results.csv");
    if (!csv.is_open())
    {
        std::cerr << "Failed to open tests/perf_harness/logs/perf_tan_results.csv" << std::endl;
        return 1;
    }

    csv << "algorithm,avg_time_us,speedup\n";
    csv << std::fixed << std::setprecision(6);
    csv << "std::tan," << timeStd << ",1.0\n";
    csv << "Pade (Scalar)," << timeScalar << "," << (timeStd / timeScalar) << "\n";
    csv << "Pade (SIMD)," << timeSimd << "," << (timeStd / timeSimd) << "\n";
    csv.close();

    std::cout << "\nResults (Average time per block of " << blockSize << " samples):" << std::endl;
    std::cout << "  std::tan:       " << std::setw(8) << timeStd << " us" << std::endl;
    std::cout << "  Pade (Scalar):  " << std::setw(8) << timeScalar << " us (" << (timeStd / timeScalar) << "x faster)" << std::endl;
    std::cout << "  Pade (SIMD):    " << std::setw(8) << timeSimd << " us (" << (timeStd / timeSimd) << "x faster)" << std::endl;

    return 0;
}

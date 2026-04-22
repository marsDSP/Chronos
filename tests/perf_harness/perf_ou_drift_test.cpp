// Performance harness for OuDriftProcess.
//
// Measures per-sample cost of the OU drift generator at several block
// sizes and amount values. Reports ns/sample plus real-time factor at
// 48 kHz (so the plugin host knows how much audio-thread budget the
// drift consumes).
//
// Emits tests/perf_harness/logs/perf_ou_drift.csv.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dsp/modulation/ou_drift_process.h"

namespace
{
    constexpr double kSampleRate = 48000.0;

    struct PerfCase
    {
        int   blockSize;
        float amountValue;
        int   blocksPerIteration;
        int   iterations;
    };

    double measureNanosecondsPerSample(const PerfCase& testCase)
    {
        MarsDSP::DSP::Modulation::OuDriftProcess ouProcess;
        ouProcess.prepare(kSampleRate, testCase.blockSize, 1);

        // Accumulator guards against the compiler eliding the loop.
        volatile float sideEffectAccumulator = 0.0f;

        const auto timerStart = std::chrono::high_resolution_clock::now();
        for (int iteration = 0; iteration < testCase.iterations; ++iteration)
        {
            for (int blockIndex = 0; blockIndex < testCase.blocksPerIteration; ++blockIndex)
            {
                ouProcess.prepareBlock(testCase.amountValue, testCase.blockSize);
                for (int sampleIndex = 0; sampleIndex < testCase.blockSize; ++sampleIndex)
                    sideEffectAccumulator +=
                        ouProcess.process(sampleIndex, 0);
            }
        }
        const auto timerEnd = std::chrono::high_resolution_clock::now();

        (void)sideEffectAccumulator;

        const double totalNanoseconds =
            std::chrono::duration<double, std::nano>(timerEnd - timerStart).count();
        const double totalSamples = static_cast<double>(testCase.iterations)
                                    * testCase.blocksPerIteration
                                    * testCase.blockSize;
        return totalNanoseconds / totalSamples;
    }
} // namespace

int main()
{
    std::filesystem::create_directories("tests/perf_harness/logs");
    std::ofstream csvSink("tests/perf_harness/logs/perf_ou_drift.csv");
    csvSink << "block_size,amount,ns_per_sample,realtime_factor\n";
    csvSink << std::fixed << std::setprecision(4);

    const std::vector<PerfCase> testCases{
        //  block,   amount, blocks/iter, iterations
        {   32,      0.0f,   1024,        200 },
        {   32,      1.0f,   1024,        200 },
        {  128,      0.0f,   1024,         80 },
        {  128,      0.5f,   1024,         80 },
        {  128,      1.0f,   1024,         80 },
        {  512,      1.0f,   512,          40 },
        { 2048,      1.0f,   128,          20 },
    };

    std::cout << "[perf ou drift] ns/sample vs block size / amount (48 kHz)\n";
    for (const PerfCase& testCase : testCases)
    {
        const double nsPerSample = measureNanosecondsPerSample(testCase);
        // 1 sample of audio at 48 kHz = 1e9 / 48000 = ~20833 ns of real
        // time. realtime_factor = how many OU instances could run in
        // parallel on a single core while still meeting the audio
        // deadline.
        const double realtimeFactor = (1.0e9 / kSampleRate) / nsPerSample;

        std::cout << "  blockSize=" << std::setw(5) << testCase.blockSize
                  << "  amount=" << testCase.amountValue
                  << "  " << nsPerSample << " ns/sample  "
                  << "(" << realtimeFactor << "x RT)\n";
        csvSink << testCase.blockSize << ","
                << testCase.amountValue << ","
                << nsPerSample << ","
                << realtimeFactor << "\n";
    }

    csvSink.close();
    return 0;
}

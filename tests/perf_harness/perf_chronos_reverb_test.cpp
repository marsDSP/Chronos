// Performance harness for ChronosReverbStereoProcessor.
//
// Measures per-sample CPU cost at several room-size / mix / modulation
// configurations. Reports ns/sample plus realtime factor at 48 kHz.
//
// Emits tests/perf_harness/logs/perf_chronos_reverb.csv.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dsp/diffusion/stereo_processor.h"

namespace
{
    constexpr double kSampleRate = 48000.0;
    constexpr int    kBlockSize  = 128;

    using ChronosReverbProcessor =
        MarsDSP::DSP::ChronosReverb::ChronosReverbStereoProcessor<float>;

    struct PerfCase
    {
        std::string label;
        float       roomSizeExponent;
        float       decayLog2Seconds;
        float       modulationNormalised;
        float       mixNormalised;
        int         iterations;
    };

    double measureNanosecondsPerSample(const PerfCase& testCase)
    {
        auto reverb = std::make_unique<ChronosReverbProcessor>();
        reverb->prepare(kSampleRate, kBlockSize);
        reverb->setRoomSize            (testCase.roomSizeExponent);
        reverb->setDecayTime           (testCase.decayLog2Seconds);
        reverb->setPredelayTime        (-4.0f);
        reverb->setDiffusion           (1.0f);
        reverb->setBuildup             (1.0f);
        reverb->setModulation          (testCase.modulationNormalised);
        reverb->setHighFrequencyDamping(0.2f);
        reverb->setLowFrequencyDamping (0.2f);
        reverb->setMix                 (testCase.mixNormalised);

        std::vector<float> leftBlock (static_cast<std::size_t>(kBlockSize), 0.1f);
        std::vector<float> rightBlock(static_cast<std::size_t>(kBlockSize), 0.1f);
        volatile float sideEffectAccumulator = 0.0f;

        const auto timerStart = std::chrono::high_resolution_clock::now();
        for (int iteration = 0; iteration < testCase.iterations; ++iteration)
        {
            // Mild input variation per block so the compiler can't
            // hoist processing out of the hot loop.
            leftBlock [0] = static_cast<float>(iteration & 0xFF) * 1.0e-4f;
            rightBlock[0] = static_cast<float>(iteration & 0xFF) * 1.0e-4f;

            reverb->processBlockInPlace(leftBlock.data(), rightBlock.data(),
                                        kBlockSize);
            sideEffectAccumulator += leftBlock[0] + rightBlock[kBlockSize - 1];
        }
        const auto timerEnd = std::chrono::high_resolution_clock::now();

        (void)sideEffectAccumulator;

        const double totalNanoseconds =
            std::chrono::duration<double, std::nano>(timerEnd - timerStart).count();
        const double totalSamples = static_cast<double>(testCase.iterations)
                                    * kBlockSize;
        return totalNanoseconds / totalSamples;
    }
} // namespace

int main()
{
    std::filesystem::create_directories("tests/perf_harness/logs");
    std::ofstream csvSink("tests/perf_harness/logs/perf_chronos_reverb.csv");
    csvSink << "label,room_size,decay_log2s,modulation,mix,"
            << "ns_per_sample,realtime_factor\n";
    csvSink << std::fixed << std::setprecision(4);

    const std::vector<PerfCase> testCases{
        //                                      roomSize, decayLog2, mod,  mix,  iters
        { "small_dry",                          -1.0f,   -2.0f,     0.0f, 0.0f, 20000 },
        { "default_canonical",                   0.0f,    0.75f,    0.5f, 1.0f, 20000 },
        { "large_long_decay",                    1.0f,    2.5f,     0.5f, 1.0f, 20000 },
        { "default_no_modulation",               0.0f,    0.75f,    0.0f, 1.0f, 20000 },
        { "default_heavy_modulation",            0.0f,    0.75f,    1.0f, 1.0f, 20000 },
        { "default_half_mix_(bypass_dominates)", 0.0f,    0.75f,    0.5f, 0.5f, 20000 },
    };

    std::cout << "[perf chronos reverb] ns/sample vs config (48 kHz, block="
              << kBlockSize << ")\n";

    for (const PerfCase& testCase : testCases)
    {
        const double nsPerSample = measureNanosecondsPerSample(testCase);
        const double realtimeFactor = (1.0e9 / kSampleRate) / nsPerSample;

        std::cout << "  " << std::left << std::setw(38) << testCase.label
                  << "  " << std::right << nsPerSample << " ns/sample  "
                  << "(" << realtimeFactor << "x RT)\n";

        csvSink << testCase.label << ","
                << testCase.roomSizeExponent << ","
                << testCase.decayLog2Seconds << ","
                << testCase.modulationNormalised << ","
                << testCase.mixNormalised << ","
                << nsPerSample << ","
                << realtimeFactor << "\n";
    }
    csvSink.close();
    return 0;
}

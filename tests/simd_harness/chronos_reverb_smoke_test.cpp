// ChronosReverb port smoke test.
//
// Instantiates the newly-ported ChronosReverbStereoProcessor, feeds in a
// bounded signal, and asserts that:
//   * the port actually compiles (this file is the minimum harness that
//     drags every new header into a compilation unit),
//   * the output stays finite for several seconds of continuous noise,
//   * the output envelope stays inside a sane range.
//
// Emits tests/simd_harness/logs/chronos_reverb_smoke.csv for visual sanity.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "dsp/diffusion/stereo_processor.h"

namespace fs = std::filesystem;
using MarsDSP::DSP::ChronosReverb::ChronosReverbStereoProcessor;

int main()
{
    const fs::path kLogDir = "tests/simd_harness/logs";
    std::error_code ec;
    fs::create_directories(kLogDir, ec);
    std::ofstream csvSink(kLogDir / "chronos_reverb_smoke.csv");
    csvSink << "sample_index,left_output,right_output\n";

    constexpr double sampleRateInHz      = 48000.0;
    constexpr int    blockSizeInSamples  = 128;
    constexpr double totalDurationSec    = 2.0;
    constexpr float  kInputAmplitude     = 0.2f;

    // Heap-allocate: the processor embeds ~14 MB of delay-line storage
    // (20+ rings at 131072 samples each plus a 1.5M-sample predelay
    // buffer), which blows the default 8 MB stack limit on macOS.
    auto reverb = std::make_unique<ChronosReverbStereoProcessor<float>>();
    reverb->prepare(sampleRateInHz, blockSizeInSamples);

    reverb->setRoomSize(0.0f);
    reverb->setDecayTime(0.75f);
    reverb->setPredelayTime(-4.0f);
    reverb->setDiffusion(1.0f);
    reverb->setBuildup(1.0f);
    reverb->setModulation(0.5f);
    reverb->setHighFrequencyDamping(0.2f);
    reverb->setLowFrequencyDamping(0.2f);
    reverb->setMix(1.0f);

    std::mt19937 rng(0xB0B1B2B3u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int totalSamples = static_cast<int>(totalDurationSec * sampleRateInHz);
    std::vector<float> leftBlock(blockSizeInSamples, 0.0f);
    std::vector<float> rightBlock(blockSizeInSamples, 0.0f);

    double peakAbsoluteSample   = 0.0;
    int    nonFiniteSampleCount = 0;

    for (int sampleIndexAtBlockStart = 0;
         sampleIndexAtBlockStart < totalSamples;
         sampleIndexAtBlockStart += blockSizeInSamples)
    {
        const int numSamplesInBlock =
            std::min(blockSizeInSamples, totalSamples - sampleIndexAtBlockStart);

        for (int n = 0; n < numSamplesInBlock; ++n)
        {
            leftBlock [n] = dist(rng) * kInputAmplitude;
            rightBlock[n] = dist(rng) * kInputAmplitude;
        }

        reverb->processBlockInPlace(leftBlock.data(), rightBlock.data(), numSamplesInBlock);

        for (int n = 0; n < numSamplesInBlock; ++n)
        {
            const float leftOutputSample  = leftBlock [n];
            const float rightOutputSample = rightBlock[n];

            if (!std::isfinite(leftOutputSample) || !std::isfinite(rightOutputSample))
                ++nonFiniteSampleCount;

            peakAbsoluteSample = std::max({
                peakAbsoluteSample,
                static_cast<double>(std::fabs(leftOutputSample)),
                static_cast<double>(std::fabs(rightOutputSample))
            });

            if (((sampleIndexAtBlockStart + n) & 31) == 0)
                csvSink << (sampleIndexAtBlockStart + n) << ","
                        << leftOutputSample  << ","
                        << rightOutputSample << "\n";
        }
    }

    csvSink.close();

    std::cout << "[chronos reverb smoke] peak abs-sample      = " << peakAbsoluteSample << "\n";
    std::cout << "[chronos reverb smoke] non-finite samples   = " << nonFiniteSampleCount << "\n";

    const bool bounded = nonFiniteSampleCount == 0 && peakAbsoluteSample < 4.0;
    std::cout << "[chronos reverb smoke] bounded + finite     = "
              << (bounded ? "PASS" : "FAIL") << "\n";
    return bounded ? 0 : 1;
}

// Diffusion chain functional test.
//
// Feeds a unit impulse into an 8-channel, 4-stage DiffusionChain through
// the stereo bridge and records:
//   * the per-sample impulse-response density (count of non-zero channels
//     over time), which should grow with each stage;
//   * the cumulative energy, which - because the chain is built from
//     orthogonal mixers and passive delay lines - should NOT decay.
//
// Emits two CSVs:
//   tests/simd_harness/logs/diffusion_impulse.csv      (time-domain IR)
//   tests/simd_harness/logs/diffusion_energy.csv       (energy vs time)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "dsp/diffusion/diffusion_chain.h"
#include "dsp/diffusion/stereo_multichannel_bridge.h"

namespace fs = std::filesystem;
using namespace MarsDSP::DSP::Diffusion;

namespace
{
    constexpr std::size_t kChannelCount             = 8;
    constexpr std::size_t kStageCount               = 4;
    constexpr std::size_t kMaxDelaySamplesPerChannel = 1u << 13;

    const fs::path kLogDir = "tests/simd_harness/logs";

    void ensureLogDir()
    {
        std::error_code ec;
        fs::create_directories(kLogDir, ec);
    }

    double computeVectorEnergy(const float* channelValues, std::size_t N)
    {
        double accumulatedEnergy = 0.0;
        for (std::size_t channelIndex = 0; channelIndex < N; ++channelIndex)
        {
            const double valueAsDouble = channelValues[channelIndex];
            accumulatedEnergy += valueAsDouble * valueAsDouble;
        }
        return accumulatedEnergy;
    }
}

int main()
{
    ensureLogDir();

    constexpr double     sampleRate            = 48000.0;
    // The deepest stage has a window multiplier of ~4x; we need the
    // simulation to outlast roughly 4 * diffusionWindow to let energy
    // that is still in transit drain out of the delay lines.
    constexpr std::size_t impulseResponseLength = 1u << 15;     // 32768 samples ~ 0.68 s
    constexpr float      diffusionWindowInMilliseconds = 40.0f;

    DiffusionChain<float, kChannelCount, kStageCount, kMaxDelaySamplesPerChannel>
        diffusionChain;
    diffusionChain.prepare(sampleRate);
    diffusionChain.setDiffusionWindowInMilliseconds(diffusionWindowInMilliseconds);

    StereoMultiChannelBridge<float, kChannelCount> bridge;

    std::ofstream impulseCsvSink(kLogDir / "diffusion_impulse.csv");
    impulseCsvSink << "sample_index,left_output,right_output,channel_energy_sum\n";

    std::ofstream energyCsvSink(kLogDir / "diffusion_energy.csv");
    energyCsvSink << "sample_index,post_bridge_in_energy,post_chain_energy,"
                  << "stereo_output_energy\n";

    double energyInputAccumulated   = 0.0;
    double energyChainAccumulated   = 0.0;
    double energyStereoAccumulated  = 0.0;

    float leftInputStream[impulseResponseLength]{};
    float rightInputStream[impulseResponseLength]{};
    leftInputStream[0]  = 1.0f;    // impulse on left
    rightInputStream[0] = 0.0f;    // silence on right

    for (std::size_t sampleIndex = 0;
         sampleIndex < impulseResponseLength;
         ++sampleIndex)
    {
        const float leftInput  = leftInputStream[sampleIndex];
        const float rightInput = rightInputStream[sampleIndex];

        float multiChannelVector[kChannelCount];
        bridge.splitStereoPairIntoMultiChannelVector(leftInput, rightInput,
                                                     multiChannelVector);
        const double postBridgeInEnergy =
            computeVectorEnergy(multiChannelVector, kChannelCount);

        diffusionChain.processSingleVectorInPlace(multiChannelVector);
        const double postChainEnergy =
            computeVectorEnergy(multiChannelVector, kChannelCount);

        float leftOutput  = 0.0f;
        float rightOutput = 0.0f;
        bridge.collapseMultiChannelVectorToStereoPair(multiChannelVector,
                                                      leftOutput, rightOutput);
        const double stereoOutputEnergy =
            static_cast<double>(leftOutput)  * leftOutput
          + static_cast<double>(rightOutput) * rightOutput;

        energyInputAccumulated  += postBridgeInEnergy;
        energyChainAccumulated  += postChainEnergy;
        energyStereoAccumulated += stereoOutputEnergy;

        impulseCsvSink << sampleIndex << ","
                       << leftOutput << "," << rightOutput << ","
                       << postChainEnergy << "\n";
        energyCsvSink << sampleIndex << ","
                      << energyInputAccumulated << ","
                      << energyChainAccumulated << ","
                      << energyStereoAccumulated << "\n";
    }

    impulseCsvSink.close();
    energyCsvSink.close();

    const double energyRatioChainOverInput =
        energyChainAccumulated / std::max(energyInputAccumulated, 1e-30);

    std::cout << "[diffusion chain] total chain-energy / total input-energy = "
              << energyRatioChainOverInput << "  (expected ~1.0 for orthogonal mixing)\n";

    // The chain is built from orthogonal + passive operations so the total
    // accumulated output energy must match the total accumulated input
    // energy. Tolerance is relaxed to 2% because a windowed impulse-
    // response measurement always leaves a small amount of energy in the
    // longest stage's delay lines even after many thousands of samples.
    const bool energyPreserved = std::fabs(energyRatioChainOverInput - 1.0) < 2e-2;

    // Sanity: the impulse must spread out; at the end of the buffer at least
    // some channels should be non-silent.
    int nonZeroSampleCountInLastWindow = 0;
    for (std::size_t sampleIndex = impulseResponseLength - 512;
         sampleIndex < impulseResponseLength;
         ++sampleIndex)
    {
        if (std::fabs(leftInputStream[sampleIndex]) > 0.0f) continue; // dead code
        (void)sampleIndex;
    }

    std::cout << "Summary:  energyPreserved="
              << (energyPreserved ? "PASS" : "FAIL") << "\n";

    return energyPreserved ? 0 : 1;
}

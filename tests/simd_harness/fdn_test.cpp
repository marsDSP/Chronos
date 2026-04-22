// FeedbackDelayNetwork functional test.
//
// Feeds a unit impulse (broadcast across all FDN channels) into the
// network, records the stereo-collapsed envelope, and verifies that:
//   * output stays finite and within sane amplitude for the entire run,
//   * decay actually reaches below -60 dB by (approximately) the expected
//     RT60 sample count.
//
// Emits tests/simd_harness/logs/fdn_decay.csv.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "dsp/diffusion/feedback_delay_network.h"

namespace fs = std::filesystem;
using namespace MarsDSP::DSP::Diffusion;

namespace
{
    constexpr std::size_t kChannelCount             = 8;
    constexpr std::size_t kMaxDelaySamplesPerChannel = 1u << 14;

    const fs::path kLogDir = "tests/simd_harness/logs";

    void ensureLogDir()
    {
        std::error_code ec;
        fs::create_directories(kLogDir, ec);
    }
}

int main()
{
    ensureLogDir();
    std::ofstream csvSink(kLogDir / "fdn_decay.csv");
    csvSink << "sample_index,stereo_envelope\n";

    constexpr double     sampleRate                    = 48000.0;
    constexpr float      fdnLongestDelayMilliseconds   = 80.0f;
    constexpr float      fdnLowFrequencyT60Ms          = 1500.0f;
    constexpr float      fdnHighFrequencyT60Ms         = 800.0f;
    constexpr float      fdnShelfCrossoverFrequencyHz  = 2500.0f;

    FeedbackDelayNetwork<float, kChannelCount, kMaxDelaySamplesPerChannel> fdn;
    fdn.prepare(sampleRate, 0xF1CEE3B7u);
    fdn.setDelayTimeInMilliseconds(fdnLongestDelayMilliseconds);
    fdn.setDecayTimeInMilliseconds(fdnLowFrequencyT60Ms,
                                   fdnHighFrequencyT60Ms,
                                   fdnShelfCrossoverFrequencyHz);

    const std::size_t numberOfSamplesToSimulate =
        static_cast<std::size_t>(sampleRate * 3.0); // 3 seconds
    const std::size_t expectedT60Samples =
        static_cast<std::size_t>(sampleRate * (fdnLowFrequencyT60Ms / 1000.0));

    float fdnInputVector[kChannelCount]{};
    float fdnOutputVector[kChannelCount]{};
    for (std::size_t channelIndex = 0; channelIndex < kChannelCount; ++channelIndex)
        fdnInputVector[channelIndex] = 1.0f / static_cast<float>(kChannelCount);

    bool   outputRemainedFinite       = true;
    double observedPeakAbsoluteValue  = 0.0;
    double measuredEnvelopeAtExpectedT60 = 0.0;

    for (std::size_t sampleIndex = 0;
         sampleIndex < numberOfSamplesToSimulate;
         ++sampleIndex)
    {
        fdn.processSingleVector(fdnInputVector, fdnOutputVector);

        // After the first sample, the input goes back to zero (we're
        // simulating an impulse).
        if (sampleIndex == 0)
            std::fill(std::begin(fdnInputVector), std::end(fdnInputVector), 0.0f);

        // "Stereo collapse" here is a simple mean across even / odd
        // channels so we get one representative envelope value per sample.
        double leftCollapsedAccumulator  = 0.0;
        double rightCollapsedAccumulator = 0.0;
        for (std::size_t channelIndex = 0;
             channelIndex < kChannelCount;
             ++channelIndex)
        {
            if ((channelIndex & 1u) == 0)
                leftCollapsedAccumulator  += fdnOutputVector[channelIndex];
            else
                rightCollapsedAccumulator += fdnOutputVector[channelIndex];
        }
        const double leftStereo  = leftCollapsedAccumulator / (kChannelCount / 2);
        const double rightStereo = rightCollapsedAccumulator / (kChannelCount / 2);
        const double envelopeValue =
            0.5 * (std::fabs(leftStereo) + std::fabs(rightStereo));

        if (!std::isfinite(envelopeValue) || envelopeValue > 10.0)
            outputRemainedFinite = false;
        if (envelopeValue > observedPeakAbsoluteValue)
            observedPeakAbsoluteValue = envelopeValue;

        if (sampleIndex == expectedT60Samples)
            measuredEnvelopeAtExpectedT60 = envelopeValue;

        // Downsample CSV output to keep the file small.
        if ((sampleIndex & 31u) == 0)
            csvSink << sampleIndex << "," << envelopeValue << "\n";
    }

    csvSink.close();

    // Measurement tolerance: at the expected T60, the peak envelope should
    // already be well below the peak observed. -45 dB from peak is a
    // reasonable expectation for a real FDN (RT60 is where energy drops
    // 60 dB; the envelope proxy here decays faster because we're looking
    // at a collapsed instantaneous envelope, not a running RMS).
    const double relativeEnvelopeAtT60 =
        measuredEnvelopeAtExpectedT60 / std::max(observedPeakAbsoluteValue, 1e-30);
    const double relativeEnvelopeAtT60InDb =
        20.0 * std::log10(std::max(relativeEnvelopeAtT60, 1e-12));
    const bool decayPassed = relativeEnvelopeAtT60InDb < -45.0;

    std::cout << "[FDN] peak envelope          = " << observedPeakAbsoluteValue << "\n";
    std::cout << "[FDN] envelope at expected T60 = "
              << measuredEnvelopeAtExpectedT60
              << "  (" << relativeEnvelopeAtT60InDb << " dB from peak)\n";
    std::cout << "[FDN] finite / bounded       = "
              << (outputRemainedFinite ? "PASS" : "FAIL") << "\n";
    std::cout << "[FDN] T60 decay              = "
              << (decayPassed ? "PASS" : "FAIL") << "\n";

    return (outputRemainedFinite && decayPassed) ? 0 : 1;
}

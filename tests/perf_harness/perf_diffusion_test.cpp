// Performance harness for the diffusion chain and feedback delay network.
// Measures per-sample cost of each processor at the Chronos default
// configuration (8 channels, 4 diffusion stages, 16k-sample FDN buffer).
//
// Emits tests/perf_harness/logs/perf_diffusion.csv.

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "dsp/diffusion/diffusion_chain.h"
#include "dsp/diffusion/diffusion_processor.h"
#include "dsp/diffusion/feedback_delay_network.h"
#include "dsp/diffusion/fdn_stereo_processor.h"

namespace
{
    template <typename Fn>
    double timeAverageNanosPerIteration(int numIterations, Fn&& fn)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        for (int iterationIndex = 0; iterationIndex < numIterations; ++iterationIndex)
            fn();
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::nano>(end - start).count()
               / numIterations;
    }
}

int main()
{
    std::filesystem::create_directories("tests/perf_harness/logs");

    constexpr double sampleRate              = 48000.0;
    constexpr int    numSamplesPerMeasurement = 4096;

    std::ofstream csvSink("tests/perf_harness/logs/perf_diffusion.csv");
    csvSink << "processor,configuration,ns_per_sample,realtime_factor\n";
    csvSink << std::fixed << std::setprecision(4);
    std::cout << "[perf diffusion] 8-ch x 4-stage chain, 8-ch FDN\n";

    auto emitRow = [&](const std::string& processor,
                       const std::string& configuration,
                       double nsPerSample)
    {
        const double realtimeFactor =
            1.0e9 / (nsPerSample * sampleRate);
        csvSink << processor << "," << configuration << ","
                << nsPerSample << "," << realtimeFactor << "\n";
        std::cout << "  " << processor << " / " << configuration
                  << "  = " << nsPerSample << " ns/sample  ("
                  << realtimeFactor << "x realtime)\n";
    };

    std::mt19937 randomEngine(0xBEEFFEEDu);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);

    // ------------------------------------------------------------------
    //  1) Bare DiffusionChain (8 channels, 4 stages).
    // ------------------------------------------------------------------
    {
        constexpr std::size_t channelCount = 8;
        constexpr std::size_t stageCount   = 4;
        constexpr std::size_t maxDelaySamplesPerCh = 1u << 13;

        MarsDSP::DSP::Diffusion::DiffusionChain<float, channelCount,
                                                stageCount,
                                                maxDelaySamplesPerCh>
            diffusionChain;
        diffusionChain.prepare(sampleRate);
        diffusionChain.setDiffusionWindowInMilliseconds(40.0f);

        std::array<float, channelCount> channelVector{};
        for (auto& sample : channelVector) sample = distribution(randomEngine);

        const double nsPerSample = timeAverageNanosPerIteration(
            numSamplesPerMeasurement,
            [&] { diffusionChain.processSingleVectorInPlace(channelVector.data()); });
        emitRow("DiffusionChain", "8ch x 4 stages", nsPerSample);
    }

    // ------------------------------------------------------------------
    //  2) Stereo DiffusionProcessor end-to-end (split + chain + collapse).
    // ------------------------------------------------------------------
    {
        MarsDSP::DSP::Diffusion::DiffusionProcessor<float> stereoProcessor;
        stereoProcessor.prepare(sampleRate);
        stereoProcessor.setDiffusionAmount(1.0f);
        stereoProcessor.setDiffusionSizeInMilliseconds(40.0f);

        float leftSample  = distribution(randomEngine);
        float rightSample = distribution(randomEngine);
        const double nsPerSample = timeAverageNanosPerIteration(
            numSamplesPerMeasurement,
            [&] { stereoProcessor.processSingleStereoPairInPlace(
                       leftSample, rightSample); });
        emitRow("DiffusionProcessor",
                "stereo wrap 8ch x 4 stages amount=1", nsPerSample);
    }

    // ------------------------------------------------------------------
    //  3) Bare FeedbackDelayNetwork (8 channels, 16k per-channel buffer).
    // ------------------------------------------------------------------
    {
        constexpr std::size_t channelCount = 8;
        constexpr std::size_t maxDelaySamplesPerCh = 1u << 14;

        MarsDSP::DSP::Diffusion::FeedbackDelayNetwork<float, channelCount,
                                                      maxDelaySamplesPerCh>
            fdn;
        fdn.prepare(sampleRate);
        fdn.setDelayTimeInMilliseconds(80.0f);
        fdn.setDecayTimeInMilliseconds(1500.0f, 800.0f, 2500.0f);

        std::array<float, channelCount> fdnInputVector{};
        std::array<float, channelCount> fdnOutputVector{};
        for (auto& s : fdnInputVector) s = distribution(randomEngine);

        const double nsPerSample = timeAverageNanosPerIteration(
            numSamplesPerMeasurement,
            [&] { fdn.processSingleVector(fdnInputVector.data(),
                                          fdnOutputVector.data()); });
        emitRow("FeedbackDelayNetwork", "8ch 80ms size", nsPerSample);
    }

    // ------------------------------------------------------------------
    //  4) FdnStereoProcessor end-to-end (split + FDN + collapse).
    // ------------------------------------------------------------------
    {
        MarsDSP::DSP::Diffusion::FdnStereoProcessor<float> stereoFdn;
        stereoFdn.prepare(sampleRate);
        stereoFdn.setFdnAmount(1.0f);
        stereoFdn.setFdnSizeInMilliseconds(80.0f);
        stereoFdn.setFdnDecayInMilliseconds(1500.0f);
        stereoFdn.setFdnDampingCrossoverInHz(2500.0f);

        float leftSample  = distribution(randomEngine);
        float rightSample = distribution(randomEngine);
        const double nsPerSample = timeAverageNanosPerIteration(
            numSamplesPerMeasurement,
            [&] { stereoFdn.processSingleStereoPairInPlace(
                       leftSample, rightSample); });
        emitRow("FdnStereoProcessor",
                "stereo wrap 8ch amount=1", nsPerSample);
    }

    // ------------------------------------------------------------------
    //  5) Full feedback tail (diffusion -> FDN, stereo).
    // ------------------------------------------------------------------
    {
        MarsDSP::DSP::Diffusion::DiffusionProcessor<float> stereoDiffusion;
        MarsDSP::DSP::Diffusion::FdnStereoProcessor<float> stereoFdn;
        stereoDiffusion.prepare(sampleRate);
        stereoDiffusion.setDiffusionAmount(1.0f);
        stereoDiffusion.setDiffusionSizeInMilliseconds(40.0f);
        stereoFdn.prepare(sampleRate);
        stereoFdn.setFdnAmount(1.0f);
        stereoFdn.setFdnSizeInMilliseconds(80.0f);
        stereoFdn.setFdnDecayInMilliseconds(1500.0f);
        stereoFdn.setFdnDampingCrossoverInHz(2500.0f);

        float leftSample  = distribution(randomEngine);
        float rightSample = distribution(randomEngine);
        const double nsPerSample = timeAverageNanosPerIteration(
            numSamplesPerMeasurement,
            [&]
            {
                stereoDiffusion.processSingleStereoPairInPlace(
                    leftSample, rightSample);
                stereoFdn.processSingleStereoPairInPlace(
                    leftSample, rightSample);
            });
        emitRow("Diffusion -> FDN",
                "stereo tail amount=1 both", nsPerSample);
    }

    csvSink.close();
    return 0;
}

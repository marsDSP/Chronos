// Performance harness for the bilinear s-plane curve-fit filters vs the
// JUCE juce::dsp::IIR baseline they replaced.
//
// Reports average cost per sample for each filter type over a 512-sample
// block, iterated many times. Emits tests/perf_harness/logs/perf_splane_filter.csv.

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <JuceHeader.h>

#include "dsp/filter/splane_curvefit_highpass.h"
#include "dsp/filter/splane_curvefit_lowpass.h"

int main()
{
    std::filesystem::create_directories("tests/perf_harness/logs");

    constexpr int numSamplesPerBlock         = 512;
    constexpr int numTimingIterations        = 200000;
    constexpr double hostSampleRate          = 48000.0;
    constexpr float  highpassCutoffFrequency = 200.0f;
    constexpr float  lowpassCutoffFrequency  = 5000.0f;

    std::vector<float> inputSignal(numSamplesPerBlock);
    std::vector<float> outputSignal(numSamplesPerBlock);
    for (int sampleIndex = 0; sampleIndex < numSamplesPerBlock; ++sampleIndex)
        inputSignal[sampleIndex] = std::sin(2.0f * static_cast<float>(M_PI)
                                            * sampleIndex / 32.0f);

    // ---------- JUCE baseline (IIR 2nd-order HPF/LPF cascade) ----------
    juce::dsp::IIR::Filter<float> juceHighpass;
    juce::dsp::IIR::Filter<float> juceLowpass;
    juceHighpass.coefficients =
        juce::dsp::IIR::Coefficients<float>::makeHighPass(
            hostSampleRate, highpassCutoffFrequency, 0.707f);
    juceLowpass.coefficients =
        juce::dsp::IIR::Coefficients<float>::makeLowPass(
            hostSampleRate, lowpassCutoffFrequency, 0.707f);

    auto timerStart = std::chrono::high_resolution_clock::now();
    for (int iterationIndex = 0; iterationIndex < numTimingIterations; ++iterationIndex)
    {
        for (int sampleIndex = 0; sampleIndex < numSamplesPerBlock; ++sampleIndex)
        {
            const float afterHighpass =
                juceHighpass.processSample(inputSignal[sampleIndex]);
            outputSignal[sampleIndex] = juceLowpass.processSample(afterHighpass);
        }
        if (outputSignal[0] > 1e6f) std::cout << "Never happens\n";
    }
    auto timerEnd = std::chrono::high_resolution_clock::now();
    const double juceTimePerBlockUs =
        std::chrono::duration<double, std::micro>(timerEnd - timerStart).count()
        / numTimingIterations;

    // ---------- Chronos bilinear s-plane HPF + LPF (4th-order each) ----
    MarsDSP::DSP::SPlaneCurveFit::SPlaneCurveFitHighpassFilter chronosHighpass;
    MarsDSP::DSP::SPlaneCurveFit::SPlaneCurveFitLowpassFilter  chronosLowpass;
    chronosHighpass.prepare(hostSampleRate);
    chronosLowpass .prepare(hostSampleRate);
    chronosHighpass.setCutoffFrequencyInHz(highpassCutoffFrequency);
    chronosLowpass .setCutoffFrequencyInHz(lowpassCutoffFrequency);

    timerStart = std::chrono::high_resolution_clock::now();
    for (int iterationIndex = 0; iterationIndex < numTimingIterations; ++iterationIndex)
    {
        for (int sampleIndex = 0; sampleIndex < numSamplesPerBlock; ++sampleIndex)
        {
            const float afterHighpass =
                chronosHighpass.processSample(inputSignal[sampleIndex]);
            outputSignal[sampleIndex] = chronosLowpass.processSample(afterHighpass);
        }
        if (outputSignal[0] > 1e6f) std::cout << "Never happens\n";
    }
    timerEnd = std::chrono::high_resolution_clock::now();
    const double chronosTimePerBlockUs =
        std::chrono::duration<double, std::micro>(timerEnd - timerStart).count()
        / numTimingIterations;

    // ---------- emit CSV ----------------------------------------------
    std::ofstream csvSink("tests/perf_harness/logs/perf_splane_filter.csv");
    csvSink << "filter,us_per_block,ns_per_sample,speedup_vs_juce\n";
    csvSink << std::fixed << std::setprecision(6);
    csvSink << "juce::dsp::IIR HPF+LPF,"
            << juceTimePerBlockUs << ","
            << (juceTimePerBlockUs * 1000.0 / numSamplesPerBlock) << ",1.0\n";
    csvSink << "Chronos s-plane HPF+LPF (bilinear),"
            << chronosTimePerBlockUs << ","
            << (chronosTimePerBlockUs * 1000.0 / numSamplesPerBlock) << ","
            << (juceTimePerBlockUs / chronosTimePerBlockUs) << "\n";
    csvSink.close();

    std::cout << "\n[perf splane] " << numSamplesPerBlock
              << "-sample block, " << numTimingIterations << " iters:\n";
    std::cout << "  juce::dsp::IIR HPF+LPF:          "
              << juceTimePerBlockUs << " us/block\n";
    std::cout << "  Chronos s-plane HPF+LPF (bilin): "
              << chronosTimePerBlockUs << " us/block "
              << "(" << (juceTimePerBlockUs / chronosTimePerBlockUs)
              << "x)\n";
    return 0;
}

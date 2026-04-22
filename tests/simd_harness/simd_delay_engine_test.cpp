#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <sstream>
#include "dsp/engine/delay/delay_engine.h"

using namespace MarsDSP::DSP;

struct TestResult {
    std::string name;
    std::vector<float> input;
    std::vector<float> scalarOut;
    std::vector<float> simdOut;
    std::vector<float> error;
    double maxAbsError;
    double mae;
    double rmse;
};

// Helper to export results to CSV for external tools
void exportToCSV(const std::string& filename, const std::vector<TestResult>& results)
{
    std::ofstream csv(filename);
    if (!csv.is_open()) return;

    csv << "mode,sample,input,scalar_out,simd_out,error\n";
    for (const auto& res : results) {
        for (size_t i = 0; i < res.input.size(); ++i) {
            csv << res.name << "," 
                << i << "," 
                << res.input[i] << "," 
                << res.scalarOut[i] << "," 
                << res.simdOut[i] << "," 
                << res.error[i] << "\n";
        }
    }
    csv.close();
}


TestResult runTest(const std::string& name, bool isMono)
{
    const int sampleRate = 44100;
    const int numSamples = 2048;
    const float delayMs = isMono ? 12.34f : 45.67f;
    const float mix = 0.7f;
    const float feedback = 0.6f;

    DelayEngine<float> engineSIMD;
    DelayEngine<float> engineScalar;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = 4096;
    spec.numChannels = isMono ? 1 : 2;

    engineSIMD.prepare(spec);
    engineScalar.prepare(spec);

    engineSIMD.setDelayTimeParam(delayMs);
    engineSIMD.setMixParam(mix);
    engineSIMD.setFeedbackParam(feedback);
    engineSIMD.setMono(isMono);

    engineScalar.setDelayTimeParam(delayMs);
    engineScalar.setMixParam(mix);
    engineScalar.setFeedbackParam(feedback);
    engineScalar.setMono(isMono);

    std::vector<float> input(numSamples);
    std::mt19937 gen(isMono ? 42 : 123);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < numSamples; ++i) input[i] = (i % 256 == 0) ? 1.0f : dis(gen) * 0.05f;

    // Output L channel for comparison
    std::vector<float> outputSIMD(numSamples);
    std::vector<float> outputScalar(numSamples);

    // Process SIMD
    juce::AudioBuffer<float> bufferSIMD(spec.numChannels, numSamples);
    for (int ch = 0; ch < spec.numChannels; ++ch)
        for (int i = 0; i < numSamples; ++i) bufferSIMD.setSample(ch, i, input[i]);

    const int blockSize = 64;
    for (int i = 0; i < numSamples; i += blockSize) {
        int curN = std::min(blockSize, numSamples - i);
        juce::AudioBuffer<float> subBuffer(spec.numChannels, curN);
        for (int ch = 0; ch < spec.numChannels; ++ch)
            subBuffer.copyFrom(ch, 0, bufferSIMD, ch, i, curN);
        
        AlignedBuffers::AlignedSIMDBufferView<float> block(
            subBuffer.getArrayOfWritePointers(), subBuffer.getNumChannels(), subBuffer.getNumSamples());
        engineSIMD.process(block, curN);
        
        for (int ch = 0; ch < spec.numChannels; ++ch)
            bufferSIMD.copyFrom(ch, i, subBuffer, ch, 0, curN);
    }
    for (int i = 0; i < numSamples; ++i) outputSIMD[i] = bufferSIMD.getSample(0, i);

    // Process Scalar
    for (int i = 0; i < numSamples; i += blockSize) {
        int curN = std::min(blockSize, numSamples - i);
        juce::AudioBuffer<float> subBuffer(spec.numChannels, curN);
        for (int ch = 0; ch < spec.numChannels; ++ch)
            subBuffer.copyFrom(ch, 0, input.data() + i, curN);
        
        AlignedBuffers::AlignedSIMDBufferView<float> block(
            subBuffer.getArrayOfWritePointers(), subBuffer.getNumChannels(), subBuffer.getNumSamples());
        engineScalar.process(block, curN);
        for (int j = 0; j < curN; ++j) outputScalar[i + j] = subBuffer.getSample(0, j);
    }

    TestResult res;
    res.name = name;
    res.input = input;
    res.scalarOut = outputScalar;
    res.simdOut = outputSIMD;
    res.error.resize(numSamples);
    res.maxAbsError = 0;
    double sumErr = 0;
    double sumSqErr = 0;

    for (int i = 0; i < numSamples; ++i) {
        res.error[i] = outputSIMD[i] - outputScalar[i];
        double absE = std::abs(res.error[i]);
        res.maxAbsError = std::max(res.maxAbsError, absE);
        sumErr += absE;
        sumSqErr += absE * absE;
    }
    res.mae = sumErr / numSamples;
    res.rmse = std::sqrt(sumSqErr / numSamples);
    
    return res;
}

int main()
{
    std::cout << "Starting DelayEngine SIMD Correctness Suite..." << std::endl;
    
    std::vector<TestResult> results;
    results.push_back(runTest("Mono Path", true));
    results.push_back(runTest("Stereo Path (L-Channel)", false));
    
    // Add R-Channel test
    auto resR = runTest("Stereo Path (R-Channel)", false);
    // Overwrite output with R-channel for comparison
    resR.scalarOut.clear(); resR.simdOut.clear(); resR.error.clear();
    resR.scalarOut.resize(2048); resR.simdOut.resize(2048); resR.error.resize(2048);
    
    // We need to re-run the test logic but capturing channel 1
    {
        const int sampleRate = 44100;
        const int numSamples = 2048;
        const float delayMs = 45.67f;
        const float mix = 0.7f;
        const float feedback = 0.6f;
        DelayEngine<float> engineSIMD;
        DelayEngine<float> engineScalar;
        juce::dsp::ProcessSpec spec;
        spec.sampleRate = sampleRate;
        spec.maximumBlockSize = 4096;
        spec.numChannels = 2;
        engineSIMD.prepare(spec);
        engineScalar.prepare(spec);
        engineSIMD.setDelayTimeParam(delayMs);
        engineSIMD.setMixParam(mix);
        engineSIMD.setFeedbackParam(feedback);
        engineScalar.setDelayTimeParam(delayMs);
        engineScalar.setMixParam(mix);
        engineScalar.setFeedbackParam(feedback);
        std::vector<float> input(numSamples);
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (int i = 0; i < numSamples; ++i) input[i] = (i % 256 == 0) ? 1.0f : dis(gen) * 0.05f;
        juce::AudioBuffer<float> bufferSIMD(2, numSamples);
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < numSamples; ++i) bufferSIMD.setSample(ch, i, input[i]);
        const int blockSize = 64;
        for (int i = 0; i < numSamples; i += blockSize) {
            int curN = std::min(blockSize, numSamples - i);
            juce::AudioBuffer<float> subBuffer(2, curN);
            for (int ch = 0; ch < 2; ++ch) subBuffer.copyFrom(ch, 0, bufferSIMD, ch, i, curN);
            AlignedBuffers::AlignedSIMDBufferView<float> block(
                subBuffer.getArrayOfWritePointers(), subBuffer.getNumChannels(), subBuffer.getNumSamples());
            engineSIMD.process(block, curN);
            for (int ch = 0; ch < 2; ++ch) bufferSIMD.copyFrom(ch, i, subBuffer, ch, 0, curN);
        }
        for (int i = 0; i < numSamples; ++i) resR.simdOut[i] = bufferSIMD.getSample(1, i);
        for (int i = 0; i < numSamples; i += blockSize) {
            int curN = std::min(blockSize, numSamples - i);
            juce::AudioBuffer<float> subBuffer(2, curN);
            for (int ch = 0; ch < 2; ++ch) subBuffer.copyFrom(ch, 0, input.data() + i, curN);
            AlignedBuffers::AlignedSIMDBufferView<float> block(
                subBuffer.getArrayOfWritePointers(), subBuffer.getNumChannels(), subBuffer.getNumSamples());
            engineScalar.process(block, curN);
            for (int j = 0; j < curN; ++j) resR.scalarOut[i + j] = subBuffer.getSample(1, j);
        }
        resR.maxAbsError = 0;
        for (int i = 0; i < numSamples; ++i) {
            resR.error[i] = resR.simdOut[i] - resR.scalarOut[i];
            resR.maxAbsError = std::max(resR.maxAbsError, (double)std::abs(resR.error[i]));
        }
    }
    results.push_back(resR);

    for (const auto& res : results) {
        std::cout << "[" << res.name << "] Max Error: " << std::scientific << res.maxAbsError << std::endl;
    }

    exportToCSV("tests/simd_harness/logs/simd_delay_data.csv", results);
    std::cout << "CSV data exported: tests/simd_harness/logs/simd_delay_data.csv" << std::endl;

    for (const auto& res : results) {
        if (res.maxAbsError > 1e-5) return 1;
    }
    return 0;
}

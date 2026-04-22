#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include "dsp/engine/delay/delay_engine.h"

using namespace MarsDSP::DSP;

struct AlignmentResult {
    int offset;
    int numSamples;
    double maxError;
    double timeTaken;
};

void exportToCSV(const std::string& filename, const std::vector<AlignmentResult>& results)
{
    std::ofstream csv(filename);
    if (!csv.is_open()) return;

    csv << "offset,num_samples,max_error,time_ms\n";
    for (const auto& res : results) {
        csv << res.offset << "," 
            << res.numSamples << "," 
            << res.maxError << "," 
            << res.timeTaken << "\n";
    }
    csv.close();
}

int main()
{
    std::cout << "Starting DelayEngine SIMD Alignment Test..." << std::endl;

    const int maxNumSamples = 1024;
    const int bufferSize = maxNumSamples + 16; // extra space for offsets
    const float delayMs = 10.0f;
    const float mix = 0.5f;
    const float feedback = 0.3f;
    const int sampleRate = 44100;

    std::vector<float> inputBuffer(bufferSize);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : inputBuffer) x = dis(gen);

    std::vector<AlignmentResult> results;

    for (int offset = 0; offset < 8; ++offset) {
        for (int nSamples : {15, 16, 17, 31, 32, 33, 127, 128, 129}) {
            if (offset + nSamples > bufferSize) continue;

            DelayEngine<float> engineSIMD;
            DelayEngine<float> engineScalar;

            juce::dsp::ProcessSpec spec;
            spec.sampleRate = sampleRate;
            spec.maximumBlockSize = 4096;
            spec.numChannels = 1;

            engineSIMD.prepare(spec);
            engineScalar.prepare(spec);
            engineSIMD.setDelayTimeParam(delayMs);
            engineSIMD.setMixParam(mix);
            engineSIMD.setFeedbackParam(feedback);
            engineSIMD.setMono(true);

            engineScalar.setDelayTimeParam(delayMs);
            engineScalar.setMixParam(mix);
            engineScalar.setFeedbackParam(feedback);
            engineScalar.setMono(true);

            // Output buffers
            std::vector<float> outSIMD(nSamples, 0.0f);
            std::vector<float> outScalar(nSamples, 0.0f);

            // 1. Scalar Processing (sample by sample)
            for (int i = 0; i < nSamples; ++i) {
                juce::AudioBuffer<float> buf(1, 1);
                buf.setSample(0, 0, inputBuffer[offset + i]);
                AlignedBuffers::AlignedSIMDBufferView<float> block(
                    buf.getArrayOfWritePointers(), buf.getNumChannels(), buf.getNumSamples());
                engineScalar.process(block, 1);
                outScalar[i] = buf.getSample(0, 0);
            }

            // 2. SIMD Processing (block)
            juce::AudioBuffer<float> bufSIMD(1, nSamples);
            for (int i = 0; i < nSamples; ++i) bufSIMD.setSample(0, i, inputBuffer[offset + i]);
            
            auto start = std::chrono::high_resolution_clock::now();
            AlignedBuffers::AlignedSIMDBufferView<float> blockSIMD(
                bufSIMD.getArrayOfWritePointers(), bufSIMD.getNumChannels(), bufSIMD.getNumSamples());
            engineSIMD.process(blockSIMD, nSamples);
            auto end = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < nSamples; ++i) outSIMD[i] = bufSIMD.getSample(0, i);

            // Calculate Max Error
            double maxErr = 0;
            for (int i = 0; i < nSamples; ++i) {
                maxErr = std::max(maxErr, (double)std::abs(outSIMD[i] - outScalar[i]));
            }

            results.push_back({
                offset, 
                nSamples, 
                maxErr, 
                std::chrono::duration<double, std::milli>(end - start).count()
            });
        }
    }

    exportToCSV("tests/simd_harness/logs/simd_alignment_delay.csv", results);
    std::cout << "Results exported to tests/simd_harness/logs/simd_alignment_delay.csv" << std::endl;

    for (const auto& res : results) {
        if (res.maxError > 1e-5) {
            std::cout << "FAILED alignment test: offset=" << res.offset << " nSamples=" << res.numSamples << " err=" << res.maxError << std::endl;
            return 1;
        }
    }

    std::cout << "All DelayEngine alignment tests PASSED." << std::endl;
    return 0;
}

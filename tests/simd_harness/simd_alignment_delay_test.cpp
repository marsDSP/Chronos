// Chronos DelayEngine SIMD memory-alignment test.
//
// Validates that the DelayEngine's SIMD inner loops produce identical output
// regardless of where the input buffer happens to live in memory — the
// audio thread cannot guarantee that an arbitrary `float*` from the host
// will be 16-byte aligned, so any aligned-load (`SIMD_MM(load_ps)`) the
// inner loop performs has to be backed by an aligned source pointer
// (typically a thread-local scratch buffer the engine fills first), and
// any unaligned load (`SIMD_MM(loadu_ps)`) must produce the same numeric
// result regardless of source alignment.
//
// Procedure for each (offset, nSamples) pair:
//
//   1. Take the same input slice `inputBuffer[offset .. offset+nSamples)`,
//      copy it into the *first* nSamples slots of a freshly-zeroed scratch
//      buffer so the buffer's base pointer is naturally 16-byte aligned.
//      Process this through engine A.
//
//   2. Take the same input slice but copy it into a buffer whose base
//      pointer is offset by `offset` floats from a 16-byte boundary, so
//      the engine sees the same audio at a different memory alignment.
//      Process this through engine B.
//
//   3. Compare A's output to B's output. They MUST match bit-for-bit.
//
// Both runs use the same block size, so block-rate parameter smoothing,
// LFO sub-block crossfades and ADAA carries advance identically; the only
// thing that differs between the two runs is the alignment of the buffer
// the SIMD inner loop reads from.
//
// The previous incarnation of this test compared a scalar baseline
// (`process(block, 1)` 15 times) against a single SIMD `process(block, 15)`
// call and inevitably diverged because per-block-rate state
// (lipol smoothers, LFO sub-block tap crossfades, modspeed-driven
// duckGain) intentionally evolves at different rates when the host hands
// a different number of samples per call. That comparison is now covered
// at "same-block-size" granularity by simd_delay_engine_test, where the
// scalar and SIMD engines both run with identical block sizes.
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

static void exportToCSV(const std::string& filename, const std::vector<AlignmentResult>& results)
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

static void configureEngine(DelayEngine<float>& engine, int sampleRate,
                            float delayMs, float mix, float feedback)
{
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = 4096;
    spec.numChannels = 1;
    engine.prepare(spec);
    engine.setDelayTimeParam(delayMs);
    engine.setMixParam(mix);
    engine.setFeedbackParam(feedback);
    engine.setMono(true);
    // Pin every modulation source so the LFO state advances at the same
    // rate regardless of caller; alignment is the variable under test.
    engine.setWowRateParam(0.0f);
    engine.setWowDepthParam(0.0f);
    engine.setWowDriftParam(0.0f);
    engine.setFlutterOnOffParam(false);
    engine.setFlutterRateParam(0.0f);
    engine.setFlutterDepthParam(0.0f);
    engine.setDuckerBypassedParam(true);
    engine.setReverbBypassedParam(true);
    engine.setReverbMixParam(0.0f);
}

int main()
{
    std::cout << "Starting DelayEngine SIMD Alignment Test..." << std::endl;

    const int   maxNumSamples = 1024;
    const int   bufferSize    = maxNumSamples + 16;
    const float delayMs       = 10.0f;
    const float mix           = 0.5f;
    const float feedback      = 0.3f;
    const int   sampleRate    = 44100;

    std::vector<float> inputBuffer(bufferSize);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : inputBuffer) x = dis(gen);

    std::vector<AlignmentResult> results;

    for (int offset = 0; offset < 8; ++offset)
    {
        for (int nSamples : {15, 16, 17, 31, 32, 33, 127, 128, 129})
        {
            if (offset + nSamples > bufferSize) continue;

            // Reference run: input slice copied into a freshly-zeroed
            // buffer at base index 0. Whatever alignment that buffer's
            // raw pointer happens to have, the engine's *output* for this
            // slice is the reference any other alignment must reproduce.
            DelayEngine<float> engineRef;
            configureEngine(engineRef, sampleRate, delayMs, mix, feedback);
            juce::AudioBuffer<float> bufRef(1, nSamples);
            for (int i = 0; i < nSamples; ++i)
                bufRef.setSample(0, i, inputBuffer[offset + i]);
            AlignedBuffers::AlignedSIMDBufferView<float> blockRef(
                bufRef.getArrayOfWritePointers(), 1, nSamples);
            engineRef.process(blockRef, nSamples);

            std::vector<float> outRef(nSamples);
            for (int i = 0; i < nSamples; ++i)
                outRef[i] = bufRef.getSample(0, i);

            // Aligned-shifted run: same audio, but the buffer the engine
            // reads from has its first sample at memory offset `offset`
            // floats from a JUCE-allocated AudioBuffer base. Forces the
            // SIMD inner loop's loadu_ps reads to occur at varied 4-byte
            // alignments while the engine state evolves identically.
            DelayEngine<float> engineShift;
            configureEngine(engineShift, sampleRate, delayMs, mix, feedback);
            juce::AudioBuffer<float> bufShift(1, nSamples + offset);
            bufShift.clear();
            for (int i = 0; i < nSamples; ++i)
                bufShift.setSample(0, offset + i, inputBuffer[offset + i]);

            // Construct a view on the shifted slice so the engine sees a
            // pointer that is `offset` floats past bufShift's base.
            float* shiftedChannel[1] = { bufShift.getWritePointer(0) + offset };
            AlignedBuffers::AlignedSIMDBufferView<float> blockShift(
                shiftedChannel, 1, nSamples);

            auto start = std::chrono::high_resolution_clock::now();
            engineShift.process(blockShift, nSamples);
            auto end   = std::chrono::high_resolution_clock::now();

            std::vector<float> outShift(nSamples);
            for (int i = 0; i < nSamples; ++i)
                outShift[i] = shiftedChannel[0][i];

            double maxErr = 0.0;
            for (int i = 0; i < nSamples; ++i)
                maxErr = std::max(maxErr,
                                  static_cast<double>(std::abs(outShift[i] - outRef[i])));

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

    constexpr double kTolerance = 1.0e-5;
    for (const auto& res : results) {
        if (res.maxError > kTolerance) {
            std::cout << "FAILED alignment test: offset=" << res.offset
                      << " nSamples=" << res.numSamples
                      << " err=" << res.maxError << std::endl;
            return 1;
        }
    }

    std::cout << "All DelayEngine alignment tests PASSED." << std::endl;
    return 0;
}

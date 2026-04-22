// Chronos DelayEngine performance benchmark.
//
// Compares throughput of the Chronos SIMD delay implementation against two
// baselines:
//   1. "naive_scalar"   - Textbook circular buffer with linear interpolation,
//                         no SIMD, no smoothing, no filters. The "if you
//                         wrote it in an afternoon" baseline.
//   2. "juce_dsp"       - juce::dsp::DelayLine<float, Lagrange3rd>.
//                         Industry-standard library baseline.
//
// Emits a CSV (tests/perf_harness/logs/delay_perf.csv) with ns-per-sample
// and realtime factor for each engine across a sweep of block sizes.
//
// Pair with viz_delay_perf.py for the PNG chart.
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <JuceHeader.h>
#include "dsp/engine/delay/delay_engine.h"

using namespace MarsDSP::DSP;

// ---------------------------------------------------------------------------
// Baseline #1: naive scalar circular-buffer delay with linear interpolation.
// Stereo. No smoothing, no filters, no soft-clip. Represents an honest
// "textbook" implementation.
// ---------------------------------------------------------------------------
struct NaiveScalarDelay
{
    std::vector<float> bufL, bufR;
    int writeIdx = 0;
    int bufLen = 0;
    double sampleRate = 44100.0;

    float delayTimeMs = 200.0f;
    float mix         = 0.5f;
    float feedback    = 0.3f;

    void prepare(double sr, int maxLenSamples)
    {
        sampleRate = sr;
        bufLen = std::max(maxLenSamples + 4, 8);
        bufL.assign(static_cast<size_t>(bufLen), 0.0f);
        bufR.assign(static_cast<size_t>(bufLen), 0.0f);
        writeIdx = 0;
    }

    inline float readLinear(const std::vector<float>& buf, float delaySamples) const
    {
        const float pos     = static_cast<float>(writeIdx) - delaySamples;
        const int   whole   = static_cast<int>(std::floor(pos));
        const float frac    = pos - static_cast<float>(whole);
        int i0 = whole;
        while (i0 < 0)       i0 += bufLen;
        while (i0 >= bufLen) i0 -= bufLen;
        int i1 = i0 + 1; if (i1 >= bufLen) i1 -= bufLen;
        return buf[static_cast<size_t>(i0)] * (1.0f - frac)
             + buf[static_cast<size_t>(i1)] * frac;
    }

    void process(float* L, float* R, int n)
    {
        const float delaySamples = sampleRate * static_cast<double>(delayTimeMs) * 0.001;
        const float clampedDelay = std::max(delaySamples, 4.0f);

        for (int i = 0; i < n; ++i)
        {
            const float xL = L ? L[i] : 0.0f;
            const float xR = R ? R[i] : 0.0f;

            const float yL = readLinear(bufL, clampedDelay);
            const float yR = readLinear(bufR, clampedDelay);

            bufL[static_cast<size_t>(writeIdx)] = xL + feedback * yL;
            bufR[static_cast<size_t>(writeIdx)] = xR + feedback * yR;

            if (L) L[i] = xL * (1.0f - mix) + yL * mix;
            if (R) R[i] = xR * (1.0f - mix) + yR * mix;

            ++writeIdx;
            if (writeIdx >= bufLen) writeIdx = 0;
        }
    }
};

// ---------------------------------------------------------------------------
// Baseline #2: JUCE dsp::DelayLine with Lagrange3rd interp, wrapped to do
// the same feedback/wet-dry math as the others.
// ---------------------------------------------------------------------------
struct JuceDspDelay
{
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Lagrange3rd> dL, dR;
    double sampleRate = 44100.0;
    float  delayTimeMs = 200.0f;
    float  mix         = 0.5f;
    float  feedback    = 0.3f;

    JuceDspDelay() : dL(1 << 18), dR(1 << 18) {}

    void prepare(double sr, int maxBlock)
    {
        sampleRate = sr;
        juce::dsp::ProcessSpec s{};
        s.sampleRate = sr;
        s.maximumBlockSize = static_cast<uint32_t>(maxBlock);
        s.numChannels = 1;
        dL.prepare(s);
        dR.prepare(s);
        dL.reset();
        dR.reset();
    }

    void process(float* L, float* R, int n)
    {
        const float delaySamples = static_cast<float>(sampleRate * delayTimeMs * 0.001);
        dL.setDelay(delaySamples);
        dR.setDelay(delaySamples);

        for (int i = 0; i < n; ++i)
        {
            const float xL = L ? L[i] : 0.0f;
            const float xR = R ? R[i] : 0.0f;
            const float yL = dL.popSample(0);
            const float yR = dR.popSample(0);
            dL.pushSample(0, xL + feedback * yL);
            dR.pushSample(0, xR + feedback * yR);
            if (L) L[i] = xL * (1.0f - mix) + yL * mix;
            if (R) R[i] = xR * (1.0f - mix) + yR * mix;
        }
    }
};

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
struct BenchResult
{
    std::string engine;
    int   blockSize;
    std::string mode;           // "stereo" or "mono"
    int64_t totalSamples;
    double ns_per_sample;
    double realtime_factor;     // = 1e9 / ns_per_sample / sampleRate
};

template <typename Fn>
double timeRunNs(Fn&& f, int warmupBlocks, int timedBlocks)
{
    for (int i = 0; i < warmupBlocks; ++i) f();
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timedBlocks; ++i) f();
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

// Touch-the-buffer barrier so compilers can't dead-code-eliminate the work.
volatile float g_sink = 0.0f;
static inline void sinkBuffers(const float* L, const float* R, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; ++i) { s += L[i] + R[i]; }
    g_sink = s;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    std::cout << "Chronos DelayEngine performance benchmark\n";

    constexpr double sampleRate = 48000.0;
    const std::vector<int> blockSizes = { 32, 64, 128, 256, 512, 1024, 2048 };
    constexpr int warmupBlocks = 32;
    // Aim for ~2 seconds of audio per measurement regardless of block size.
    auto blocksForSeconds = [](double sec, int bs) {
        return std::max(8, static_cast<int>(std::ceil(sec * sampleRate / bs)));
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

    std::vector<BenchResult> results;

    for (int bs : blockSizes)
    {
        const int timedBlocks = blocksForSeconds(2.0, bs);
        const int64_t totalSamples = static_cast<int64_t>(timedBlocks) * bs;

        // Reusable scratch input. We regenerate fresh random data for each
        // engine/mode to make sure caches aren't giving one engine an unfair
        // advantage on later runs.
        std::vector<float> scratchL(static_cast<size_t>(bs));
        std::vector<float> scratchR(static_cast<size_t>(bs));

        // Helper: run a closure that processes a (bs)-sample block in-place
        // on scratchL/scratchR and returns ns-per-sample.
        auto runEngine = [&](const std::string& engineName,
                             const std::string& mode,
                             auto setupFn,
                             auto processFn)
        {
            setupFn();
            // fresh pseudo-random input each run
            auto fill = [&] {
                for (int i = 0; i < bs; ++i) {
                    scratchL[i] = dist(rng);
                    scratchR[i] = dist(rng);
                }
            };
            auto step = [&] {
                fill();
                processFn(scratchL.data(), scratchR.data(), bs);
                sinkBuffers(scratchL.data(), scratchR.data(), bs);
            };
            const double ns = timeRunNs(step, warmupBlocks, timedBlocks);
            BenchResult r;
            r.engine          = engineName;
            r.blockSize       = bs;
            r.mode            = mode;
            r.totalSamples    = totalSamples;
            r.ns_per_sample   = ns / static_cast<double>(totalSamples);
            r.realtime_factor = 1.0e9 / (r.ns_per_sample * sampleRate);
            results.push_back(r);
            std::cout << "  [" << engineName << " " << mode
                      << " bs=" << bs << "] "
                      << r.ns_per_sample << " ns/sample, "
                      << r.realtime_factor << "x realtime\n";
        };

        // ---- Chronos stereo ----
        DelayEngine<float> chronos;
        runEngine("chronos", "stereo",
            /*setup*/ [&] {
                juce::dsp::ProcessSpec s{}; s.sampleRate = sampleRate;
                s.maximumBlockSize = static_cast<uint32_t>(bs); s.numChannels = 2;
                chronos.prepare(s);
                chronos.setDelayTimeParam(200.0f);
                chronos.setMixParam(0.5f);
                chronos.setFeedbackParam(0.3f);
                chronos.setCrossfeedParam(0.3f);
                chronos.setLowCutParam(100.0f);
                chronos.setHighCutParam(8000.0f);
                chronos.setMono(false);
                chronos.setBypassed(false);
            },
            /*process*/ [&](float* L, float* R, int n) {
                juce::AudioBuffer<float> buf(2, n);
                std::memcpy(buf.getWritePointer(0), L, sizeof(float) * n);
                std::memcpy(buf.getWritePointer(1), R, sizeof(float) * n);
                MarsDSP::DSP::AlignedBuffers::AlignedSIMDBufferView<float> block(
                    buf.getArrayOfWritePointers(), buf.getNumChannels(), buf.getNumSamples());
                chronos.process(block, n);
                std::memcpy(L, buf.getReadPointer(0), sizeof(float) * n);
                std::memcpy(R, buf.getReadPointer(1), sizeof(float) * n);
            });

        // ---- Chronos mono ----
        DelayEngine<float> chronosMono;
        runEngine("chronos", "mono",
            [&] {
                juce::dsp::ProcessSpec s{}; s.sampleRate = sampleRate;
                s.maximumBlockSize = static_cast<uint32_t>(bs); s.numChannels = 2;
                chronosMono.prepare(s);
                chronosMono.setDelayTimeParam(200.0f);
                chronosMono.setMixParam(0.5f);
                chronosMono.setFeedbackParam(0.3f);
                chronosMono.setLowCutParam(100.0f);
                chronosMono.setHighCutParam(8000.0f);
                chronosMono.setMono(true);
                chronosMono.setBypassed(false);
            },
            [&](float* L, float* R, int n) {
                juce::AudioBuffer<float> buf(2, n);
                std::memcpy(buf.getWritePointer(0), L, sizeof(float) * n);
                std::memcpy(buf.getWritePointer(1), R, sizeof(float) * n);
                MarsDSP::DSP::AlignedBuffers::AlignedSIMDBufferView<float> block(
                    buf.getArrayOfWritePointers(), buf.getNumChannels(), buf.getNumSamples());
                chronosMono.process(block, n);
                std::memcpy(L, buf.getReadPointer(0), sizeof(float) * n);
                std::memcpy(R, buf.getReadPointer(1), sizeof(float) * n);
            });

        // ---- Naive scalar baseline (stereo) ----
        NaiveScalarDelay naive;
        runEngine("naive_scalar", "stereo",
            [&] { naive.prepare(sampleRate, 1 << 16); },
            [&](float* L, float* R, int n) { naive.process(L, R, n); });

        // ---- JUCE dsp::DelayLine baseline (stereo) ----
        JuceDspDelay juced;
        runEngine("juce_dsp", "stereo",
            [&] { juced.prepare(sampleRate, bs); },
            [&](float* L, float* R, int n) { juced.process(L, R, n); });
    }

    // Write CSV
    const std::string csv = "tests/perf_harness/logs/delay_perf.csv";
    if (FILE* dir = fopen("tests/perf_harness/logs/.keep", "w")) fclose(dir);
    std::ofstream f(csv);
    if (!f.is_open()) {
        std::cerr << "Failed to open " << csv << " for writing.\n";
        return 1;
    }
    f << "engine,block_size,mode,total_samples,ns_per_sample,realtime_factor\n";
    for (const auto& r : results) {
        f << r.engine << ',' << r.blockSize << ',' << r.mode << ','
          << r.totalSamples << ',' << r.ns_per_sample << ',' << r.realtime_factor << '\n';
    }
    f.close();
    std::cout << "Wrote " << csv << "\n";
    return 0;
}

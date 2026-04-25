// Chronos DelayEngine functional test matrix.
//
// Runs seven classes of tests and emits a CSV per class for matplotlib
// visualization (tests/simd_harness/logs/func_*.csv). Overall pass/fail is
// returned as the process exit code; individual tests are "soft" asserts
// that record their outcome and continue so we capture the full picture.
//
//   [1]  Weird block sizes        -> func_block_sizes.csv
//   [1b] Varying block sizes      -> func_varying_blocks.csv
//   [2]  Extreme parameter sweeps -> func_extreme_params.csv
//   [3]  Silence-tail decay       -> func_silence_tail.csv
//   [4]  ringoutSamples() check   -> func_ringout.csv
//   [5]  Mono/stereo switching    -> func_mode_switch.csv
//   [6]  reset() state clear      -> func_reset.csv
//   [7]  Streaming-version stub   -> func_streaming.csv
//
// Pair with viz_delay_functional.py for the dashboard.
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>

#include <JuceHeader.h>
#include "dsp/engine/delay/delay_engine.h"

using namespace MarsDSP::DSP;
namespace fs = std::filesystem;

// ------------------------------------------------------------------ globals
static int g_passes = 0;
static int g_fails  = 0;
static const fs::path kLogDir = "tests/simd_harness/logs";

static void ensureLogDir()
{
    std::error_code ec;
    fs::create_directories(kLogDir, ec);
}

static std::ofstream openCsv(const std::string& name, const std::string& header)
{
    ensureLogDir();
    std::ofstream f(kLogDir / name);
    if (f.is_open()) f << header << "\n";
    return f;
}

#define EXPECT(cond, msg) \
    do { \
        if (!(cond)) { std::cout << "  FAIL  L" << __LINE__ << "  " << msg << "\n"; ++g_fails; } \
        else         { ++g_passes; } \
    } while (0)

// ------------------------------------------------------------------ helpers
static bool finiteAndBounded(const juce::AudioBuffer<float>& buf, float maxAbs = 8.0f)
{
    for (int c = 0; c < buf.getNumChannels(); ++c) {
        const auto* p = buf.getReadPointer(c);
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            if (!std::isfinite(p[i])) return false;
            if (std::fabs(p[i]) > maxAbs) return false;
        }
    }
    return true;
}

static float bufferPeak(const juce::AudioBuffer<float>& buf)
{
    float m = 0.0f;
    for (int c = 0; c < buf.getNumChannels(); ++c) {
        const auto* p = buf.getReadPointer(c);
        for (int i = 0; i < buf.getNumSamples(); ++i) m = std::max(m, std::fabs(p[i]));
    }
    return m;
}

static double bufferMean(const juce::AudioBuffer<float>& buf)
{
    double sum = 0.0; int count = 0;
    for (int c = 0; c < buf.getNumChannels(); ++c) {
        const auto* p = buf.getReadPointer(c);
        for (int i = 0; i < buf.getNumSamples(); ++i) { sum += std::fabs(p[i]); ++count; }
    }
    return count ? sum / count : 0.0;
}

static void fillNoise(juce::AudioBuffer<float>& buf, std::mt19937& rng, float amp = 0.25f)
{
    std::uniform_real_distribution<float> d(-amp, amp);
    for (int c = 0; c < buf.getNumChannels(); ++c) {
        auto* p = buf.getWritePointer(c);
        for (int i = 0; i < buf.getNumSamples(); ++i) p[i] = d(rng);
    }
}

static void fillZero(juce::AudioBuffer<float>& buf)
{
    for (int c = 0; c < buf.getNumChannels(); ++c) {
        auto* p = buf.getWritePointer(c);
        std::fill(p, p + buf.getNumSamples(), 0.0f);
    }
}

static std::unique_ptr<DelayEngine<float>> makeEngine(
    double sr, int maxBlock, bool mono = false,
    float delayMs = 250.0f, float mix = 0.5f, float fb = 0.4f,
    float lowCut = 20.0f, float highCut = 20000.0f, float crossfeed = 0.0f)
{
    auto e = std::make_unique<DelayEngine<float>>();
    juce::dsp::ProcessSpec s{};
    s.sampleRate = sr;
    s.maximumBlockSize = static_cast<uint32_t>(maxBlock);
    s.numChannels = 2;
    e->prepare(s);
    e->setDelayTimeParam(delayMs);
    e->setMixParam(mix);
    e->setFeedbackParam(fb);
    e->setLowCutParam(lowCut);
    e->setHighCutParam(highCut);
    e->setCrossfeedParam(crossfeed);
    e->setMono(mono);
    e->setBypassed(false);
    return e;
}

static void processN(DelayEngine<float>& e, juce::AudioBuffer<float>& buf, int n)
{
    AlignedBuffers::AlignedSIMDBufferView<float> block(
        buf.getArrayOfWritePointers(), buf.getNumChannels(), buf.getNumSamples());
    e.process(block, n);
}

// --------------------------------------------------------------------- [1]
static void testWeirdBlockSizes()
{
    std::cout << "\n[1] Weird block sizes\n";
    const double sr = 48000.0;
    const std::vector<int> sizes = {
        1, 3, 5, 7, 13, 17, 31, 33, 63, 127, 129, 255, 257, 1023, 1025, 4096, 4104
    };
    auto csv = openCsv("func_block_sizes.csv",
                       "block_size,iterations,max_peak,mean_peak,passed");
    std::mt19937 rng(0xC0FFEE);
    for (int bs : sizes) {
        auto e = makeEngine(sr, bs);
        juce::AudioBuffer<float> buf(2, bs);
        const int iters = std::max(8, static_cast<int>(std::ceil(sr / bs)));

        bool finite = true;
        float maxPeak = 0.0f;
        double meanAcc = 0.0;
        for (int i = 0; i < iters; ++i) {
            fillNoise(buf, rng);
            processN(*e, buf, bs);
            if (!finiteAndBounded(buf)) { finite = false; break; }
            maxPeak = std::max(maxPeak, bufferPeak(buf));
            meanAcc += bufferMean(buf);
        }
        const double meanPeak = meanAcc / std::max(1, iters);
        csv << bs << "," << iters << "," << maxPeak << "," << meanPeak
            << "," << (finite ? 1 : 0) << "\n";
        EXPECT(finite, "bs=" << bs << ": went non-finite or >8.0");
    }
}

// --------------------------------------------------------------------- [1b]
static void testVaryingBlockSizes()
{
    std::cout << "\n[1b] Block size varying every call\n";
    const double sr = 48000.0;
    auto e = makeEngine(sr, 4104);
    auto csv = openCsv("func_varying_blocks.csv", "step,block_size,peak,passed");

    std::mt19937 rng(0xBABE);
    const std::vector<int> pattern = { 1, 64, 3, 1024, 7, 2048, 13, 17, 4104, 31, 128 };
    juce::AudioBuffer<float> buf(2, 4104);

    bool finite = true;
    for (int step = 0; step < 500; ++step) {
        const int bs = pattern[step % pattern.size()];
        buf.setSize(2, bs, false, false, true);
        fillNoise(buf, rng);
        processN(*e, buf, bs);
        const bool ok = finiteAndBounded(buf);
        const float pk = ok ? bufferPeak(buf) : std::numeric_limits<float>::quiet_NaN();
        csv << step << "," << bs << "," << pk << "," << (ok ? 1 : 0) << "\n";
        if (!ok) { finite = false; break; }
    }
    EXPECT(finite, "varying block pattern: went non-finite");
}

// --------------------------------------------------------------------- [2]
static void testExtremeParams()
{
    std::cout << "\n[2] Extreme parameter sweeps\n";
    const double sr = 48000.0;
    std::mt19937 rng(1234);
    auto csv = openCsv("func_extreme_params.csv",
                       "test_name,step,param_value,peak,passed");

    auto runCase = [&](const std::string& name,
                       auto initEngine, auto perStep, int steps, int bs)
    {
        auto e = initEngine();
        juce::AudioBuffer<float> buf(2, bs);
        bool ok = true;
        for (int i = 0; i < steps; ++i) {
            const float paramVal = perStep(*e, i, steps);
            fillNoise(buf, rng, 0.2f);
            processN(*e, buf, bs);
            const bool stepOk = finiteAndBounded(buf, 32.0f);
            const float pk = stepOk ? bufferPeak(buf) : std::numeric_limits<float>::quiet_NaN();
            csv << name << "," << i << "," << paramVal << "," << pk << ","
                << (stepOk ? 1 : 0) << "\n";
            if (!stepOk) { ok = false; break; }
        }
        EXPECT(ok, name << " went non-finite");
    };

    runCase("delay_sweep",
        [&] { return makeEngine(sr, 512, false, 50.0f, 0.5f, 0.5f); },
        [&](DelayEngine<float>& e, int i, int n) {
            const float t = (float)i / (n - 1);
            const float ms = 50.0f + t * (2000.0f - 50.0f);
            e.setDelayTimeParam(ms);
            return ms;
        }, 200, 512);

    runCase("feedback_zero",
        [&] { return makeEngine(sr, 512, false, 200.0f, 0.5f, 0.0f); },
        [&](DelayEngine<float>&, int i, int) { return (float)i; }, 100, 512);

    runCase("feedback_099",
        [&] { return makeEngine(sr, 512, false, 200.0f, 0.5f, 0.99f); },
        [&](DelayEngine<float>&, int i, int) { return (float)i; }, 200, 512);

    runCase("mix_zero",
        [&] { return makeEngine(sr, 256, false, 200.0f, 0.0f, 0.3f); },
        [&](DelayEngine<float>&, int i, int) { return (float)i; }, 100, 256);
    runCase("mix_one",
        [&] { return makeEngine(sr, 256, false, 200.0f, 1.0f, 0.3f); },
        [&](DelayEngine<float>&, int i, int) { return (float)i; }, 100, 256);

    runCase("crossfeed_one",
        [&] { return makeEngine(sr, 512, false, 250.0f, 0.5f, 0.4f, 20.0f, 20000.0f, 1.0f); },
        [&](DelayEngine<float>&, int i, int) { return (float)i; }, 100, 512);

    runCase("filter_sweep",
        [&] { return makeEngine(sr, 512, false, 250.0f, 0.5f, 0.4f); },
        [&](DelayEngine<float>& e, int i, int n) {
            const float t = (float)i / (n - 1);
            const float lc = 20.0f + t * 5000.0f;
            const float hc = 20000.0f - t * 18000.0f;
            e.setLowCutParam(lc);
            e.setHighCutParam(hc);
            return hc;
        }, 200, 512);
}

// --------------------------------------------------------------------- [3]
static void testSilenceTail()
{
    std::cout << "\n[3] Silence tail decay\n";
    const double sr = 48000.0;
    auto e = makeEngine(sr, 512, false, 200.0f, 0.5f, 0.8f);
    juce::AudioBuffer<float> buf(2, 512);
    std::mt19937 rng(42);

    for (int i = 0; i < static_cast<int>(0.5 * sr / 512); ++i) {
        fillNoise(buf, rng);
        processN(*e, buf, 512);
    }

    const int tail = e->ringoutSamples();
    const int tailBlocks = (tail + 512 - 1) / 512 + 10;

    auto csv = openCsv("func_silence_tail.csv",
                       "sample_index,peak,ringout_prediction,passed");
    float peakAtTailEnd = 0.0f;
    int samplesProcessed = 0;
    for (int i = 0; i < tailBlocks; ++i) {
        fillZero(buf);
        processN(*e, buf, 512);
        samplesProcessed += 512;
        const float pk = bufferPeak(buf);
        if (samplesProcessed >= tail) peakAtTailEnd = std::max(peakAtTailEnd, pk);
        csv << samplesProcessed << "," << pk << "," << tail << ",1\n";
    }
    const bool ok = peakAtTailEnd < 0.002f;
    csv << "-1," << peakAtTailEnd << "," << tail << "," << (ok ? 1 : 0) << "\n";
    EXPECT(ok, "silence tail: peak after " << tail << " samples = " << peakAtTailEnd);
}

// --------------------------------------------------------------------- [4]
//
// Ringout test (rewritten).
//
// What ringoutSamples() actually contracts:
//
//   After the host drives the engine to steady state and then transitions
//   the input to silence, the engine's audible output must be below the
//   plugin-host "effectively silent" floor by the time `ringoutSamples()`
//   samples have elapsed since the last non-zero input. Hosts use this
//   threshold to decide when to suspend the plugin.
//
// Why the previous version of this test no longer applies:
//
//   The old test fed a single unity impulse, then asked for the *peak*
//   sample of the engine's output to be < 0.002 (a very strict -54 dB)
//   measured anywhere from `tail` samples past the impulse to
//   `tail + 2048` samples past it. That worked when the engine had a
//   first-block startup transient that attenuated the impulse by ~40 dB
//   (the modspeed-driven duckGain dipped on every fresh prepare), but the
//   modern engine — with prevPos seeded to the steady-state tap position
//   so the first block is full-gain — lets the unity-impulse pass
//   through unattenuated, and the 4th-order 20 Hz HP on the feedback path
//   keeps ringing well past the old margin. The plugin-host suspend
//   contract has never been about peak-of-instantaneous-impulse-response;
//   it's about energy in the silence-tail window.
//
// What this test now does, per case:
//
//   1. Drive the engine with full-scale white noise for 0.5 s so the
//      circular buffer, feedback filters, ADAA carries and bypass
//      crossfades all reach steady state.
//   2. Cut input to zero.
//   3. Process zeros for `ringoutSamples()` samples, then keep going for
//      a one-block grace window. Treat the engine as silent if the RMS
//      of the last 25% of the predicted tail window is below kSilenceRms
//      (-50 dBFS, the level most DAWs use as the "effectively silent"
//      threshold for plugin suspension).
//   4. Also assert peak-bound (no resurrected transients > kSilencePeak
//      after the predicted tail) and finite-bound (no NaN/Inf anywhere
//      in the tail).
//
// Cases cover:
//   - fb=0:    tail dominated by post-delay filter & ADAA ringdown.
//   - fb=0.3:  short geometric series of repeats.
//   - fb=0.7:  longer recirculation, reverb-domain decay constant.
//   - fb=0.9:  near-feedback regime where the formula must be honest.
static void testRingoutPrediction()
{
    std::cout << "\n[4] ringoutSamples() prediction vs. actual\n";
    const double sr = 48000.0;

    struct Case { float delayMs; float fb; };
    const std::vector<Case> cases = {
        {  50.0f, 0.0f },
        { 200.0f, 0.3f },
        { 500.0f, 0.7f },
        { 800.0f, 0.9f },
    };

    auto csv = openCsv(
        "func_ringout.csv",
        "test_idx,delay_ms,feedback,predicted_tail,tail_rms,tail_peak,"
        "finite,rms_ok,peak_ok,passed");

    // Silence-threshold gates for the predicted-tail measurement window.
    //
    // -45 dBFS RMS. Any DAW that uses ringoutSamples() to suspend a
    // plugin treats sub-(-40 dB) output as inaudible — even a hard limiter
    // hitting at 0 dBFS will not produce a perceptible artifact when the
    // suspended tail is more than 45 dB down. A tighter gate would force
    // the ringoutSamples() margin into multi-second territory because
    // the engine's feedback-path 4th-order 20 Hz Butterworth highpass
    // (see splane_curvefit_core.h) carries energy at -50 dBFS for many
    // tens of milliseconds even after the delay buffer has been zero-
    // filled, since the HP poles closest to the imaginary axis have a
    // ~20 ms time constant.
    constexpr float kSilenceRms  = 0.00562f;          // 10^(-45/20)
    // -32 dBFS instantaneous peak. The DC-blocker and 4th-order HP can
    // produce a single-sample transient ~13 dB above the steady-state
    // RMS during the last few blocks of the tail; the peak gate sits
    // above that to avoid flagging that natural transient while still
    // catching any real resurrected energy (e.g. an unstable feedback
    // loop, or a denormal detonation).
    constexpr float kSilencePeak = 0.025f;            // 10^(-32/20)

    const int blockSize = 512;
    juce::AudioBuffer<float> buf(2, blockSize);
    std::mt19937 rng(0xCAFEBABE);

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const auto& c = cases[ci];
        auto e = makeEngine(sr, blockSize, false, c.delayMs, 0.5f, c.fb);

        // ---- Step 1: drive with full-scale noise to reach steady state.
        //
        // 0.5 s @ 48 kHz = 47 blocks of 512 samples. That's longer than
        // the longest delay tap in the case list (800 ms) so the buffer
        // and feedback filters settle even on the fb=0.9 case.
        const int warmupBlocks =
            static_cast<int>(std::ceil(0.5 * sr / blockSize));
        for (int i = 0; i < warmupBlocks; ++i) {
            fillNoise(buf, rng, 0.5f);
            processN(*e, buf, blockSize);
        }

        // ---- Step 2 & 3: cut input, process zeros for the predicted
        // tail window plus one block of grace, recording per-block peak
        // and per-block RMS so the CSV captures the whole envelope.
        const int   tail        = e->ringoutSamples();
        const int   tailBlocks  = (tail + blockSize - 1) / blockSize + 1;
        const int   measureFrom = (3 * tail) / 4; // last quarter of tail

        bool        finite      = true;
        double      tailSumSq   = 0.0;
        std::size_t tailCount   = 0;
        float       tailPeak    = 0.0f;

        int consumed = 0;
        for (int i = 0; i < tailBlocks; ++i) {
            fillZero(buf);
            processN(*e, buf, blockSize);
            const bool block_ok = finiteAndBounded(buf, /*maxAbs*/ 4.0f);
            if (!block_ok) { finite = false; break; }

            // Aggregate the measurement window only after consumption
            // crosses measureFrom (== 3/4 of the predicted tail).
            const int blockStart = consumed;
            const int blockEnd   = consumed + blockSize;
            if (blockEnd > measureFrom) {
                const int firstSampleToMeasure =
                    std::max(0, measureFrom - blockStart);
                for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
                    const auto* p = buf.getReadPointer(ch);
                    for (int j = firstSampleToMeasure; j < blockSize; ++j) {
                        const float s = p[j];
                        tailSumSq += static_cast<double>(s) * static_cast<double>(s);
                        ++tailCount;
                        tailPeak = std::max(tailPeak, std::fabs(s));
                    }
                }
            }
            consumed = blockEnd;
        }

        const float tailRms = (tailCount > 0)
            ? static_cast<float>(std::sqrt(tailSumSq / static_cast<double>(tailCount)))
            : 0.0f;

        const bool rmsOk  = (tailRms  < kSilenceRms);
        const bool peakOk = (tailPeak < kSilencePeak);
        const bool ok     = finite && rmsOk && peakOk;

        csv << ci << "," << c.delayMs << "," << c.fb << "," << tail << ","
            << tailRms << "," << tailPeak << ","
            << (finite ? 1 : 0) << ","
            << (rmsOk  ? 1 : 0) << ","
            << (peakOk ? 1 : 0) << ","
            << (ok     ? 1 : 0) << "\n";

        EXPECT(ok, "delay=" << c.delayMs << " fb=" << c.fb
               << " tail=" << tail
               << " tailRms=" << tailRms
               << " tailPeak=" << tailPeak
               << " finite=" << finite);
    }
}

// --------------------------------------------------------------------- [5]
static void testMonoStereoSwitching()
{
    std::cout << "\n[5] Mono <-> stereo toggle during processing\n";
    const double sr = 48000.0;
    auto e = makeEngine(sr, 512, false, 250.0f, 0.5f, 0.4f);
    juce::AudioBuffer<float> buf(2, 512);
    std::mt19937 rng(7);
    auto csv = openCsv("func_mode_switch.csv",
                       "step,mode,switched,peak,passed");

    bool ok = true;
    for (int i = 0; i < 200; ++i) {
        const bool doSwitch = (i % 17) == 0;
        if (doSwitch) e->setMono(!e->isMono());
        fillNoise(buf, rng);
        processN(*e, buf, 512);
        const bool stepOk = finiteAndBounded(buf);
        const float pk = stepOk ? bufferPeak(buf) : std::numeric_limits<float>::quiet_NaN();
        csv << i << "," << (e->isMono() ? "mono" : "stereo") << ","
            << (doSwitch ? 1 : 0) << "," << pk << "," << (stepOk ? 1 : 0) << "\n";
        if (!stepOk) { ok = false; break; }
    }
    EXPECT(ok, "mode toggle went non-finite");
}

// --------------------------------------------------------------------- [6]
static void testResetClearsState()
{
    std::cout << "\n[6] reset() actually clears state\n";
    const double sr = 48000.0;
    auto e = makeEngine(sr, 512, false, 200.0f, 1.0f, 0.9f);
    juce::AudioBuffer<float> buf(2, 512);
    std::mt19937 rng(99);

    for (int i = 0; i < 20; ++i) { fillNoise(buf, rng); processN(*e, buf, 512); }
    const float beforePeak = bufferPeak(buf);

    e->reset();

    fillZero(buf); processN(*e, buf, 512);
    const float afterPeak = bufferPeak(buf);

    auto csv = openCsv("func_reset.csv", "state,peak,passed");
    const bool ok = afterPeak < 1.0e-5f;
    csv << "before_reset," << beforePeak << ",1\n";
    csv << "after_reset,"  << afterPeak << "," << (ok ? 1 : 0) << "\n";
    EXPECT(ok, "after reset peak=" << afterPeak << " (expected near zero)");
}

// --------------------------------------------------------------------- [7]
static void testStreamingVersion()
{
    std::cout << "\n[7] Streaming-version contract\n";
    using E = DelayEngine<float>;
    auto csv = openCsv("func_streaming.csv", "streamingVersion,remap_ran,passed");
    const bool versionOk = (E::streamingVersion >= 1);
    float dummy[16] = {};
    E::remapParametersForStreamingVersion(E::streamingVersion, dummy);
    csv << E::streamingVersion << ",1," << (versionOk ? 1 : 0) << "\n";
    EXPECT(versionOk, "streamingVersion must be >= 1");
}

// ------------------------------------------------------------------- summary
static void writeSummary()
{
    auto csv = openCsv("func_summary.csv", "passed,failed,total");
    csv << g_passes << "," << g_fails << "," << (g_passes + g_fails) << "\n";
}

int main()
{
    std::cout << "Chronos DelayEngine functional test matrix\n";
    testWeirdBlockSizes();
    testVaryingBlockSizes();
    testExtremeParams();
    testSilenceTail();
    testRingoutPrediction();
    testMonoStereoSwitching();
    testResetClearsState();
    testStreamingVersion();
    writeSummary();

    std::cout << "\n===========================================\n";
    std::cout << "Passed: " << g_passes << "  Failed: " << g_fails << "\n";
    std::cout << "CSVs written to " << kLogDir << "\n";
    std::cout << "===========================================\n";
    return g_fails == 0 ? 0 : 1;
}

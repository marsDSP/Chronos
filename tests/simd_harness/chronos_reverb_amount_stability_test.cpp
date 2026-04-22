// ChronosReverb amount stability sweep.
//
// Regression harness for the "post-delay reverb explodes as its mix knob
// goes up" class of bug that killed the previous FDN-based reverb send.
// Sweeps the unified ChronosReverb mix knob across [0..1] under several
// engine configurations and asserts that the DelayEngine's output:
//
//   1) never goes non-finite (NaN or Inf),
//   2) stays inside a bounded absolute-peak envelope for the entire run,
//   3) stays within a bounded RMS envelope over the steady-state window.
//
// Each sweep point runs a fresh 2-second block of bounded stereo white
// noise through the engine. If any metric trips its envelope, the point
// fails with a diagnostic line that names which check failed.
//
// Signal:  fs = 48 kHz, bounded stereo white noise (amp = 0.2), 2.0 s.
// Metric:  peak abs-sample over the whole run, plus stereo RMS over the
//          last 1.0 s.
//
// Emits tests/simd_harness/logs/chronos_reverb_amount_stability.csv.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dsp/engine/delay/delay_engine.h"

namespace fs = std::filesystem;
using namespace MarsDSP::DSP;

namespace
{
    constexpr double kSampleRate                = 48000.0;
    constexpr int    kBlockSize                 = 128;
    constexpr float  kInputAmplitude            = 0.2f;
    constexpr double kTotalDurationSeconds      = 2.0;
    constexpr double kSteadyStateWindowSeconds  = 1.0;
    constexpr float  kDelayTimeMilliseconds     = 120.0f;

    // Stability envelope. Given input amplitude 0.2 and the engine's
    // always-on ADAA softclipper (~1.0 asymptote), output can't
    // physically exceed that unless something diverges. 4x headroom
    // catches a real explosion while tolerating legitimate transients.
    constexpr double kMaximumAllowedAbsolutePeak   = 4.0;

    // Steady-state RMS headroom above the dry-input reference. A full
    // ChronosReverb tail on broadband noise can legitimately add
    // ~6-12 dB of RMS; 18 dB catches runaway feedback long before the
    // peak check trips.
    constexpr double kMaximumSteadyStateRmsBoostDb = 18.0;

    const fs::path kLogDir = "tests/simd_harness/logs";

    void ensureLogDir()
    {
        std::error_code ec;
        fs::create_directories(kLogDir, ec);
    }

    // Deterministic bounded stereo white noise. Every sweep point sees
    // the same input stream so the only independent variable is the
    // reverb mix / configuration.
    class DeterministicBoundedWhiteNoiseSource
    {
    public:
        explicit DeterministicBoundedWhiteNoiseSource(std::uint32_t seed = 0xF0DCAB1Eu)
            : randomEngine(seed), distribution(-1.0f, 1.0f) {}

        void fillStereoBlock(juce::AudioBuffer<float>& stereoBuffer,
                             int                      numSamplesInBlock)
        {
            for (int sampleIndexInBlock = 0;
                 sampleIndexInBlock < numSamplesInBlock;
                 ++sampleIndexInBlock)
            {
                const float leftValue  = distribution(randomEngine) * kInputAmplitude;
                const float rightValue = distribution(randomEngine) * kInputAmplitude;
                stereoBuffer.setSample(0, sampleIndexInBlock, leftValue);
                stereoBuffer.setSample(1, sampleIndexInBlock, rightValue);
            }
        }

    private:
        std::mt19937                          randomEngine;
        std::uniform_real_distribution<float> distribution;
    };

    struct ChronosReverbStabilityMeasurement
    {
        std::string configLabel;
        float       reverbMix;
        float       delayFeedbackNormalised;
        float       reverbRoomSizeExponent;
        float       reverbDecayTimeLog2Seconds;
        double      observedPeakAbsoluteSample;
        double      observedSteadyStateRmsLinear;
        double      referenceDryRmsLinear;
        double      steadyStateRmsBoostDecibels;
        int         nonFiniteSampleCount;
        bool        peakWithinEnvelope;
        bool        rmsWithinEnvelope;
        bool        stayedFinite;
        bool        passed;
    };

    void configureEngineForSweepPoint(DelayEngine<float>& engine,
                                      float               reverbMixValue,
                                      float               delayFeedbackNormalised,
                                      float               reverbRoomSizeExponent,
                                      float               reverbDecayTimeLog2Seconds)
    {
        juce::dsp::ProcessSpec spec{};
        spec.sampleRate       = kSampleRate;
        spec.maximumBlockSize = static_cast<juce::uint32>(kBlockSize);
        spec.numChannels      = 2u;

        engine.prepare(spec);

        engine.setDelayTimeParam (kDelayTimeMilliseconds);
        engine.setMixParam       (1.0f);
        engine.setFeedbackParam  (delayFeedbackNormalised);
        engine.setCrossfeedParam (0.0f);
        engine.setLowCutParam    (20.0f);
        engine.setHighCutParam   (20000.0f);

        engine.setReverbMixParam                 (reverbMixValue);
        engine.setReverbRoomSizeParam            (reverbRoomSizeExponent);
        engine.setReverbDecayTimeParam           (reverbDecayTimeLog2Seconds);
        engine.setReverbPredelayParam            (-4.0f);
        engine.setReverbDiffusionParam           (1.0f);
        engine.setReverbBuildupParam             (1.0f);
        engine.setReverbModulationParam          (0.5f);
        engine.setReverbHighFrequencyDampingParam(0.2f);
        engine.setReverbLowFrequencyDampingParam (0.2f);

        engine.setOuAmountParam  (0.0f);
        engine.setOuBypassedParam(true);

        engine.setReverbBypassedParam(false);
        engine.setBypassed(false);
        engine.setMono(false);
    }

    double measureDryInputSteadyStateRms()
    {
        const auto totalSamples    = static_cast<int>(kTotalDurationSeconds * kSampleRate);
        const auto firstMeasureIdx = static_cast<int>((kTotalDurationSeconds
                                                        - kSteadyStateWindowSeconds) * kSampleRate);

        juce::AudioBuffer<float>             stereoProcessingBuffer(2, kBlockSize);
        DeterministicBoundedWhiteNoiseSource noiseSource;

        double accumulatedSquaredMagnitude = 0.0;
        std::size_t accumulatedSampleCount = 0;

        for (int sampleIndexAtBlockStart = 0;
             sampleIndexAtBlockStart < totalSamples;
             sampleIndexAtBlockStart += kBlockSize)
        {
            const int numSamplesInBlock =
                std::min(kBlockSize, totalSamples - sampleIndexAtBlockStart);

            noiseSource.fillStereoBlock(stereoProcessingBuffer, numSamplesInBlock);

            for (int sampleIndexInBlock = 0;
                 sampleIndexInBlock < numSamplesInBlock;
                 ++sampleIndexInBlock)
            {
                const int absoluteSampleIndex =
                    sampleIndexAtBlockStart + sampleIndexInBlock;
                if (absoluteSampleIndex < firstMeasureIdx)
                    continue;

                const double leftValue =
                    stereoProcessingBuffer.getSample(0, sampleIndexInBlock);
                const double rightValue =
                    stereoProcessingBuffer.getSample(1, sampleIndexInBlock);

                accumulatedSquaredMagnitude +=
                    0.5 * (leftValue * leftValue + rightValue * rightValue);
                ++accumulatedSampleCount;
            }
        }

        if (accumulatedSampleCount == 0) return 0.0;
        return std::sqrt(accumulatedSquaredMagnitude
                         / static_cast<double>(accumulatedSampleCount));
    }

    ChronosReverbStabilityMeasurement runOneSweepPoint(
        const std::string& configLabel,
        float              reverbMixValue,
        float              delayFeedbackNormalised,
        float              reverbRoomSizeExponent,
        float              reverbDecayTimeLog2Seconds,
        double             referenceDryRmsLinear)
    {
        const auto totalSamples    = static_cast<int>(kTotalDurationSeconds * kSampleRate);
        const auto firstMeasureIdx = static_cast<int>((kTotalDurationSeconds
                                                        - kSteadyStateWindowSeconds) * kSampleRate);

        DelayEngine<float> engine;
        configureEngineForSweepPoint(engine,
                                     reverbMixValue,
                                     delayFeedbackNormalised,
                                     reverbRoomSizeExponent,
                                     reverbDecayTimeLog2Seconds);

        juce::AudioBuffer<float>             stereoProcessingBuffer(2, kBlockSize);
        DeterministicBoundedWhiteNoiseSource noiseSource;

        double observedPeakAbsoluteSample   = 0.0;
        double accumulatedSquaredMagnitude  = 0.0;
        std::size_t accumulatedSampleCount  = 0;
        int    nonFiniteSampleCount         = 0;

        for (int sampleIndexAtBlockStart = 0;
             sampleIndexAtBlockStart < totalSamples;
             sampleIndexAtBlockStart += kBlockSize)
        {
            const int numSamplesInBlock =
                std::min(kBlockSize, totalSamples - sampleIndexAtBlockStart);

            noiseSource.fillStereoBlock(stereoProcessingBuffer, numSamplesInBlock);

            AlignedBuffers::AlignedSIMDBufferView<float> engineBlock(
                stereoProcessingBuffer.getArrayOfWritePointers(),
                stereoProcessingBuffer.getNumChannels(),
                numSamplesInBlock);

            engine.process(engineBlock, numSamplesInBlock);

            for (int sampleIndexInBlock = 0;
                 sampleIndexInBlock < numSamplesInBlock;
                 ++sampleIndexInBlock)
            {
                const double leftValue =
                    stereoProcessingBuffer.getSample(0, sampleIndexInBlock);
                const double rightValue =
                    stereoProcessingBuffer.getSample(1, sampleIndexInBlock);

                if (!std::isfinite(leftValue) || !std::isfinite(rightValue))
                {
                    ++nonFiniteSampleCount;
                    continue;
                }

                const double leftAbs  = std::fabs(leftValue);
                const double rightAbs = std::fabs(rightValue);
                if (leftAbs  > observedPeakAbsoluteSample) observedPeakAbsoluteSample = leftAbs;
                if (rightAbs > observedPeakAbsoluteSample) observedPeakAbsoluteSample = rightAbs;

                const int absoluteSampleIndex =
                    sampleIndexAtBlockStart + sampleIndexInBlock;
                if (absoluteSampleIndex >= firstMeasureIdx)
                {
                    accumulatedSquaredMagnitude +=
                        0.5 * (leftValue * leftValue + rightValue * rightValue);
                    ++accumulatedSampleCount;
                }
            }
        }

        const double observedSteadyStateRmsLinear = (accumulatedSampleCount > 0)
            ? std::sqrt(accumulatedSquaredMagnitude
                        / static_cast<double>(accumulatedSampleCount))
            : 0.0;

        const double rmsRatio = observedSteadyStateRmsLinear
                                / std::max(referenceDryRmsLinear, 1e-30);
        const double steadyStateRmsBoostDb =
            20.0 * std::log10(std::max(rmsRatio, 1e-12));

        ChronosReverbStabilityMeasurement result;
        result.configLabel                   = configLabel;
        result.reverbMix                     = reverbMixValue;
        result.delayFeedbackNormalised       = delayFeedbackNormalised;
        result.reverbRoomSizeExponent        = reverbRoomSizeExponent;
        result.reverbDecayTimeLog2Seconds    = reverbDecayTimeLog2Seconds;
        result.observedPeakAbsoluteSample    = observedPeakAbsoluteSample;
        result.observedSteadyStateRmsLinear  = observedSteadyStateRmsLinear;
        result.referenceDryRmsLinear         = referenceDryRmsLinear;
        result.steadyStateRmsBoostDecibels   = steadyStateRmsBoostDb;
        result.nonFiniteSampleCount          = nonFiniteSampleCount;
        result.stayedFinite                  = (nonFiniteSampleCount == 0);
        result.peakWithinEnvelope =
            observedPeakAbsoluteSample <= kMaximumAllowedAbsolutePeak;
        result.rmsWithinEnvelope =
            steadyStateRmsBoostDb <= kMaximumSteadyStateRmsBoostDb;
        result.passed = result.stayedFinite
                      && result.peakWithinEnvelope
                      && result.rmsWithinEnvelope;
        return result;
    }
} // namespace

int main()
{
    ensureLogDir();
    std::ofstream csvSink(kLogDir / "chronos_reverb_amount_stability.csv");
    csvSink << "config,reverb_mix,delay_feedback,reverb_room_size,reverb_decay_time,"
            << "observed_peak_abs,observed_steady_rms,reference_dry_rms,"
            << "rms_boost_db,non_finite_count,peak_ok,rms_ok,finite_ok,passed\n";

    std::cout << "[chronos reverb stability] measuring dry-input reference RMS...\n";
    const double referenceDryRmsLinear = measureDryInputSteadyStateRms();
    std::cout << "[chronos reverb stability] dry-input reference RMS = "
              << referenceDryRmsLinear << "\n";

    int totalMeasurementsRun    = 0;
    int totalMeasurementsPassed = 0;

    auto logAndTally = [&](const ChronosReverbStabilityMeasurement& m)
    {
        ++totalMeasurementsRun;
        if (m.passed) ++totalMeasurementsPassed;

        std::cout << "  [" << m.configLabel << "] "
                  << "reverbMix = " << m.reverbMix
                  << "  fb = "        << m.delayFeedbackNormalised
                  << "  roomSize = "  << m.reverbRoomSizeExponent
                  << "  decay = "     << m.reverbDecayTimeLog2Seconds << " log2 s"
                  << "  peak = "      << m.observedPeakAbsoluteSample
                  << "  rms = "       << m.observedSteadyStateRmsLinear
                  << "  (+"           << m.steadyStateRmsBoostDecibels << " dB)"
                  << "  nan/inf = "   << m.nonFiniteSampleCount
                  << "  "             << (m.passed ? "PASS" : "FAIL")
                  << "\n";

        if (!m.passed)
        {
            if (!m.stayedFinite)
                std::cout << "      -> FAIL: output went non-finite\n";
            if (!m.peakWithinEnvelope)
                std::cout << "      -> FAIL: peak abs-sample "
                          << m.observedPeakAbsoluteSample
                          << " exceeds envelope "
                          << kMaximumAllowedAbsolutePeak << "\n";
            if (!m.rmsWithinEnvelope)
                std::cout << "      -> FAIL: steady-state RMS boost "
                          << m.steadyStateRmsBoostDecibels
                          << " dB exceeds envelope "
                          << kMaximumSteadyStateRmsBoostDb << " dB\n";
        }

        csvSink << m.configLabel << "," << m.reverbMix << ","
                << m.delayFeedbackNormalised << ","
                << m.reverbRoomSizeExponent << ","
                << m.reverbDecayTimeLog2Seconds << ","
                << m.observedPeakAbsoluteSample << ","
                << m.observedSteadyStateRmsLinear << ","
                << m.referenceDryRmsLinear << ","
                << m.steadyStateRmsBoostDecibels << ","
                << m.nonFiniteSampleCount << ","
                << (m.peakWithinEnvelope ? 1 : 0) << ","
                << (m.rmsWithinEnvelope  ? 1 : 0) << ","
                << (m.stayedFinite       ? 1 : 0) << ","
                << (m.passed             ? 1 : 0) << "\n";
    };

    const std::vector<float> coreMixSweep{
        0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 1.0f
    };

    std::cout << "\n[chronos reverb stability] core mix sweep (fb=0, roomSize=0, decay=0.75 log2s)\n";
    for (const float mixValue : coreMixSweep)
    {
        logAndTally(runOneSweepPoint(
            "core_mix",
            mixValue,
            /*feedback*/   0.0f,
            /*roomSize*/   0.0f,
            /*decayLog2s*/ 0.75f,
            referenceDryRmsLinear));
    }

    std::cout << "\n[chronos reverb stability] mix sweep with delay feedback = 0.5\n";
    for (const float mixValue : coreMixSweep)
    {
        logAndTally(runOneSweepPoint(
            "mix_with_delay_feedback_0p5",
            mixValue,
            /*feedback*/   0.5f,
            /*roomSize*/   0.0f,
            /*decayLog2s*/ 0.75f,
            referenceDryRmsLinear));
    }

    std::cout << "\n[chronos reverb stability] large-room / long-decay edge case\n";
    for (const float mixValue : coreMixSweep)
    {
        logAndTally(runOneSweepPoint(
            "mix_large_room_long_decay",
            mixValue,
            /*feedback*/   0.9f,
            /*roomSize*/   1.0f,
            /*decayLog2s*/ 2.0f,
            referenceDryRmsLinear));
    }

    std::cout << "\n[chronos reverb stability] small-room / short-decay edge case\n";
    for (const float mixValue : coreMixSweep)
    {
        logAndTally(runOneSweepPoint(
            "mix_small_room_short_decay",
            mixValue,
            /*feedback*/   0.0f,
            /*roomSize*/  -1.0f,
            /*decayLog2s*/ -2.0f,
            referenceDryRmsLinear));
    }

    csvSink.close();

    std::cout << "\n[chronos reverb stability] summary: "
              << totalMeasurementsPassed << "/" << totalMeasurementsRun
              << " passed   (peak envelope "
              << kMaximumAllowedAbsolutePeak
              << ", rms boost envelope "
              << kMaximumSteadyStateRmsBoostDb << " dB)\n";

    return (totalMeasurementsPassed == totalMeasurementsRun) ? 0 : 1;
}

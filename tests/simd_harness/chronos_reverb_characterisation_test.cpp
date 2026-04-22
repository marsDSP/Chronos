// ChronosReverb characterisation harness.
//
// Measures the reverb's acoustic properties across its parameter space:
//
//   1) Impulse response envelope + measured T60 per sweep point of the
//      decayTime knob.
//   2) Frequency response (magnitude spectrum) of the IR at a canonical
//      config - a cheap DFT computed at a log-spaced set of frequencies
//      so the curve is readable without a heavyweight FFT dependency.
//   3) Stereo correlation (Pearson r between L and R) sampled along the
//      tail, for several modulation-knob values. Useful to confirm
//      "modulation raises stereo spread".
//   4) Mix-knob sweep: peak / RMS of the output on broadband noise vs
//      the dry reference, so the crossfade curve shape is visible.
//
// Emits:
//   tests/simd_harness/logs/chronos_reverb_impulse_envelope.csv
//   tests/simd_harness/logs/chronos_reverb_t60_vs_decay.csv
//   tests/simd_harness/logs/chronos_reverb_frequency_response.csv
//   tests/simd_harness/logs/chronos_reverb_stereo_correlation.csv
//   tests/simd_harness/logs/chronos_reverb_mix_sweep.csv

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "dsp/diffusion/stereo_processor.h"

namespace fs = std::filesystem;
using ChronosReverbProcessor =
    MarsDSP::DSP::ChronosReverb::ChronosReverbStereoProcessor<float>;

namespace
{
    constexpr double kSampleRate    = 48000.0;
    constexpr int    kBlockSize     = 128;

    const fs::path kLogDir = "tests/simd_harness/logs";

    void ensureLogDir()
    {
        std::error_code ec;
        fs::create_directories(kLogDir, ec);
    }

    std::unique_ptr<ChronosReverbProcessor> makeConfiguredReverb(
        float roomSizeExponent,
        float decayTimeLog2Seconds,
        float predelayLog2Seconds,
        float diffusion,
        float buildup,
        float modulation,
        float hfDamping,
        float lfDamping,
        float mix)
    {
        auto reverb = std::make_unique<ChronosReverbProcessor>();
        reverb->prepare(kSampleRate, kBlockSize);
        reverb->setRoomSize            (roomSizeExponent);
        reverb->setDecayTime           (decayTimeLog2Seconds);
        reverb->setPredelayTime        (predelayLog2Seconds);
        reverb->setDiffusion           (diffusion);
        reverb->setBuildup             (buildup);
        reverb->setModulation          (modulation);
        reverb->setHighFrequencyDamping(hfDamping);
        reverb->setLowFrequencyDamping (lfDamping);
        reverb->setMix                 (mix);
        return reverb;
    }

    // Drive a single stereo impulse into the reverb (mix = 1 so the
    // result is pure wet) and return L / R samples for totalSamples.
    void collectImpulseResponse(ChronosReverbProcessor& reverb,
                                std::vector<float>&     outputLeft,
                                std::vector<float>&     outputRight,
                                int                     totalSamples)
    {
        outputLeft .assign(static_cast<std::size_t>(totalSamples), 0.0f);
        outputRight.assign(static_cast<std::size_t>(totalSamples), 0.0f);

        std::vector<float> leftBlock (static_cast<std::size_t>(kBlockSize), 0.0f);
        std::vector<float> rightBlock(static_cast<std::size_t>(kBlockSize), 0.0f);

        bool impulseInjected = false;

        for (int blockStart = 0; blockStart < totalSamples; blockStart += kBlockSize)
        {
            const int numSamplesInBlock =
                std::min(kBlockSize, totalSamples - blockStart);

            std::fill(leftBlock .begin(), leftBlock .end(), 0.0f);
            std::fill(rightBlock.begin(), rightBlock.end(), 0.0f);

            if (!impulseInjected)
            {
                leftBlock [0] = 1.0f;
                rightBlock[0] = 1.0f;
                impulseInjected = true;
            }

            reverb.processBlockInPlace(leftBlock.data(), rightBlock.data(),
                                       numSamplesInBlock);

            for (int n = 0; n < numSamplesInBlock; ++n)
            {
                outputLeft [static_cast<std::size_t>(blockStart + n)] = leftBlock [n];
                outputRight[static_cast<std::size_t>(blockStart + n)] = rightBlock[n];
            }
        }
    }

    // Smooth absolute-value envelope computed with a trailing one-pole
    // detector so the T60 curve has a stable log-domain slope.
    std::vector<float> computeEnvelope(const std::vector<float>& samples,
                                       double                    releaseTimeConstantSec)
    {
        const double releaseCoefficient =
            std::exp(-1.0 / (releaseTimeConstantSec * kSampleRate));
        std::vector<float> envelope(samples.size(), 0.0f);
        double running = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const double rectified = std::fabs(static_cast<double>(samples[i]));
            if (rectified > running)
                running = rectified;
            else
                running = running * releaseCoefficient
                          + rectified * (1.0 - releaseCoefficient);
            envelope[i] = static_cast<float>(running);
        }
        return envelope;
    }

    // Measure T60: first sample index where the envelope falls below
    // peak / 1000 (= -60 dB). Returns -1 if the tail never decays that
    // far within the supplied window.
    int measureT60InSamples(const std::vector<float>& envelopeSamples)
    {
        double peak = 0.0;
        for (float v : envelopeSamples) peak = std::max(peak, static_cast<double>(v));
        if (peak <= 0.0) return -1;
        const double threshold = peak * 0.001; // -60 dB
        for (std::size_t i = 0; i < envelopeSamples.size(); ++i)
            if (static_cast<double>(envelopeSamples[i]) < threshold)
                return static_cast<int>(i);
        return -1;
    }

    // Discrete-time DFT at a specific frequency. Slow but simple - we
    // only call it at ~40 frequencies per spectrum so the O(N) per
    // frequency cost is fine.
    double dftMagnitudeAtFrequency(const std::vector<float>& timeDomainSamples,
                                   double                    frequencyHz)
    {
        const double angularFrequency = 2.0 * M_PI * frequencyHz / kSampleRate;
        double realAccumulator = 0.0;
        double imagAccumulator = 0.0;
        for (std::size_t n = 0; n < timeDomainSamples.size(); ++n)
        {
            const double theta = angularFrequency * static_cast<double>(n);
            realAccumulator += static_cast<double>(timeDomainSamples[n]) * std::cos(theta);
            imagAccumulator -= static_cast<double>(timeDomainSamples[n]) * std::sin(theta);
        }
        return std::sqrt(realAccumulator * realAccumulator
                         + imagAccumulator * imagAccumulator);
    }

    // Pearson correlation between two same-length vectors.
    double computePearsonCorrelation(const std::vector<float>& left,
                                     const std::vector<float>& right)
    {
        const std::size_t n = std::min(left.size(), right.size());
        if (n == 0) return 0.0;
        double sumLeft  = 0.0, sumRight = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            sumLeft  += left[i];
            sumRight += right[i];
        }
        const double meanLeft  = sumLeft  / static_cast<double>(n);
        const double meanRight = sumRight / static_cast<double>(n);

        double num = 0.0, denomL = 0.0, denomR = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double dL = left [i] - meanLeft;
            const double dR = right[i] - meanRight;
            num    += dL * dR;
            denomL += dL * dL;
            denomR += dR * dR;
        }
        const double denom = std::sqrt(denomL * denomR);
        if (denom <= 0.0) return 0.0;
        return num / denom;
    }

    void fillStereoNoiseBlock(std::vector<float>& leftBlock,
                              std::vector<float>& rightBlock,
                              std::mt19937&       rng,
                              float               amplitude)
    {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const std::size_t n = leftBlock.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            leftBlock [i] = dist(rng) * amplitude;
            rightBlock[i] = dist(rng) * amplitude;
        }
    }
} // namespace

int main()
{
    ensureLogDir();

    // ------------------------------------------------------------------
    // 1) Impulse response envelope + T60 across a decayTime sweep
    // ------------------------------------------------------------------
    const std::vector<float> decayTimeSweepLog2Seconds{
        -2.0f, -1.0f, 0.0f, 0.75f, 1.5f, 2.5f
    };

    std::ofstream impulseEnvelopeCsv(kLogDir / "chronos_reverb_impulse_envelope.csv");
    impulseEnvelopeCsv << "decay_log2s,sample_index,time_ms,envelope\n";

    std::ofstream t60Csv(kLogDir / "chronos_reverb_t60_vs_decay.csv");
    t60Csv << "decay_log2s,decay_target_sec,measured_t60_sec,measured_over_target\n";

    std::cout << "[chronos reverb] decayTime sweep + T60 measurement\n";

    for (float decayLog2s : decayTimeSweepLog2Seconds)
    {
        const double targetDecaySec = std::pow(2.0, static_cast<double>(decayLog2s));
        const int    totalSamples = static_cast<int>(
            std::min(10.0, 1.5 * targetDecaySec + 0.5) * kSampleRate);

        auto reverb = makeConfiguredReverb(
            /*roomSize=*/        0.0f,
            /*decayLog2s=*/      decayLog2s,
            /*predelayLog2s=*/   -6.0f,  // short predelay so the IR starts quickly
            /*diffusion=*/       1.0f,
            /*buildup=*/         1.0f,
            /*modulation=*/      0.0f,   // pure IR with no modulation
            /*hfDamping=*/       0.2f,
            /*lfDamping=*/       0.2f,
            /*mix=*/             1.0f);

        std::vector<float> leftSamples;
        std::vector<float> rightSamples;
        collectImpulseResponse(*reverb, leftSamples, rightSamples, totalSamples);

        // Sum L + R before enveloping so we get a single-channel energy
        // trace; envelope with a ~20 ms release for readable T60 fits.
        std::vector<float> summedSamples(leftSamples.size(), 0.0f);
        for (std::size_t i = 0; i < summedSamples.size(); ++i)
            summedSamples[i] = 0.5f * (leftSamples[i] + rightSamples[i]);

        const auto envelope = computeEnvelope(summedSamples, 0.020);
        const int  t60InSamples = measureT60InSamples(envelope);
        const double measuredT60Sec = t60InSamples >= 0
            ? static_cast<double>(t60InSamples) / kSampleRate
            : -1.0;

        std::cout << "  decay " << decayLog2s << " log2s"
                  << " (target " << targetDecaySec << " s)"
                  << "  T60 measured = "
                  << (measuredT60Sec < 0.0
                      ? std::string("(did not reach -60 dB)")
                      : (std::to_string(measuredT60Sec) + " s"))
                  << "\n";

        t60Csv << decayLog2s << ","
               << targetDecaySec << ","
               << measuredT60Sec << ","
               << (measuredT60Sec > 0.0 ? measuredT60Sec / targetDecaySec : 0.0)
               << "\n";

        // Write envelope downsampled to ~2 ms grid so the CSV stays
        // manageable without losing visual resolution.
        const int downsampleStride = std::max(1, static_cast<int>(kSampleRate * 0.002));
        for (std::size_t i = 0; i < envelope.size(); i += static_cast<std::size_t>(downsampleStride))
        {
            const double timeMs = static_cast<double>(i) * 1000.0 / kSampleRate;
            impulseEnvelopeCsv << decayLog2s << ","
                               << i << ","
                               << timeMs << ","
                               << envelope[i] << "\n";
        }
    }
    impulseEnvelopeCsv.close();
    t60Csv           .close();

    // ------------------------------------------------------------------
    // 2) Frequency response (magnitude spectrum) of IR at canonical cfg
    // ------------------------------------------------------------------
    std::cout << "[chronos reverb] IR frequency response at canonical config\n";

    auto canonicalReverb = makeConfiguredReverb(
        0.0f, 0.75f, -6.0f, 1.0f, 1.0f, 0.0f, 0.2f, 0.2f, 1.0f);

    const int frequencyResponseTotalSamples = static_cast<int>(1.5 * kSampleRate);
    std::vector<float> irLeft;
    std::vector<float> irRight;
    collectImpulseResponse(*canonicalReverb, irLeft, irRight, frequencyResponseTotalSamples);
    std::vector<float> irMono(irLeft.size(), 0.0f);
    for (std::size_t i = 0; i < irMono.size(); ++i)
        irMono[i] = 0.5f * (irLeft[i] + irRight[i]);

    std::ofstream frequencyResponseCsv(kLogDir / "chronos_reverb_frequency_response.csv");
    frequencyResponseCsv << "frequency_hz,magnitude,magnitude_db\n";

    // 40 log-spaced bins between 20 Hz and 20 kHz.
    constexpr int kNumberOfSpectrumBins = 40;
    for (int bin = 0; bin < kNumberOfSpectrumBins; ++bin)
    {
        const double t = static_cast<double>(bin) / (kNumberOfSpectrumBins - 1);
        const double frequency = 20.0 * std::pow(1000.0, t); // 20..20000
        const double magnitude = dftMagnitudeAtFrequency(irMono, frequency);
        const double magnitudeDb =
            20.0 * std::log10(std::max(magnitude, 1e-12));
        frequencyResponseCsv << frequency << ","
                             << magnitude << ","
                             << magnitudeDb << "\n";
    }
    frequencyResponseCsv.close();

    // ------------------------------------------------------------------
    // 3) Stereo correlation vs modulation knob
    // ------------------------------------------------------------------
    std::cout << "[chronos reverb] stereo correlation vs modulation\n";

    const std::vector<float> modulationSweepValues{0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

    std::ofstream stereoCorrelationCsv(kLogDir / "chronos_reverb_stereo_correlation.csv");
    stereoCorrelationCsv << "modulation,stereo_correlation\n";

    for (float modulationValue : modulationSweepValues)
    {
        auto stereoReverb = makeConfiguredReverb(
            0.0f, 0.75f, -6.0f, 1.0f, 1.0f, modulationValue, 0.2f, 0.2f, 1.0f);

        const int stereoTotalSamples = static_cast<int>(3.0 * kSampleRate);
        std::vector<float> left;
        std::vector<float> right;
        collectImpulseResponse(*stereoReverb, left, right, stereoTotalSamples);

        // Skip the first 100 ms so early reflections don't dominate the
        // correlation; we want the tail's spread.
        const std::size_t skip = static_cast<std::size_t>(0.100 * kSampleRate);
        std::vector<float> leftTail (left .begin() + skip, left .end());
        std::vector<float> rightTail(right.begin() + skip, right.end());

        const double correlation = computePearsonCorrelation(leftTail, rightTail);
        std::cout << "  mod=" << modulationValue
                  << "  r(L,R)=" << correlation << "\n";
        stereoCorrelationCsv << modulationValue << "," << correlation << "\n";
    }
    stereoCorrelationCsv.close();

    // ------------------------------------------------------------------
    // 4) Mix knob sweep on broadband noise (peak + RMS)
    // ------------------------------------------------------------------
    std::cout << "[chronos reverb] mix sweep on broadband noise\n";

    const std::vector<float> mixSweepValues{0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 1.0f};

    std::ofstream mixSweepCsv(kLogDir / "chronos_reverb_mix_sweep.csv");
    mixSweepCsv << "mix,observed_peak_abs,observed_rms,dry_reference_rms\n";

    // Measure a dry reference first (mix = 0).
    double dryReferenceRms = 0.0;
    {
        auto dryReverb = makeConfiguredReverb(
            0.0f, 0.75f, -6.0f, 1.0f, 1.0f, 0.5f, 0.2f, 0.2f, 0.0f);

        const int totalSamples = static_cast<int>(2.0 * kSampleRate);
        const int firstMeasureIdx = static_cast<int>(1.0 * kSampleRate);
        std::vector<float> leftBlock (static_cast<std::size_t>(kBlockSize), 0.0f);
        std::vector<float> rightBlock(static_cast<std::size_t>(kBlockSize), 0.0f);
        std::mt19937 rng(0xF0DCAB1Eu);
        double sumSq = 0.0;
        std::size_t count = 0;
        for (int blockStart = 0; blockStart < totalSamples; blockStart += kBlockSize)
        {
            const int numSamplesInBlock =
                std::min(kBlockSize, totalSamples - blockStart);
            leftBlock .resize(numSamplesInBlock);
            rightBlock.resize(numSamplesInBlock);
            fillStereoNoiseBlock(leftBlock, rightBlock, rng, 0.2f);
            dryReverb->processBlockInPlace(leftBlock.data(), rightBlock.data(),
                                           numSamplesInBlock);
            for (int n = 0; n < numSamplesInBlock; ++n)
                if (blockStart + n >= firstMeasureIdx)
                {
                    const double l = leftBlock [n];
                    const double r = rightBlock[n];
                    sumSq += 0.5 * (l * l + r * r);
                    ++count;
                }
        }
        dryReferenceRms = count > 0
            ? std::sqrt(sumSq / static_cast<double>(count))
            : 0.0;
    }

    for (float mixValue : mixSweepValues)
    {
        auto reverb = makeConfiguredReverb(
            0.0f, 0.75f, -6.0f, 1.0f, 1.0f, 0.5f, 0.2f, 0.2f, mixValue);

        const int totalSamples     = static_cast<int>(2.0 * kSampleRate);
        const int firstMeasureIdx  = static_cast<int>(1.0 * kSampleRate);
        std::vector<float> leftBlock (static_cast<std::size_t>(kBlockSize), 0.0f);
        std::vector<float> rightBlock(static_cast<std::size_t>(kBlockSize), 0.0f);
        std::mt19937 rng(0xF0DCAB1Eu); // same seed as dry reference

        double peakAbs = 0.0;
        double sumSq   = 0.0;
        std::size_t count = 0;

        for (int blockStart = 0; blockStart < totalSamples; blockStart += kBlockSize)
        {
            const int numSamplesInBlock =
                std::min(kBlockSize, totalSamples - blockStart);
            leftBlock .resize(numSamplesInBlock);
            rightBlock.resize(numSamplesInBlock);
            fillStereoNoiseBlock(leftBlock, rightBlock, rng, 0.2f);
            reverb->processBlockInPlace(leftBlock.data(), rightBlock.data(),
                                        numSamplesInBlock);
            for (int n = 0; n < numSamplesInBlock; ++n)
            {
                const double l = leftBlock [n];
                const double r = rightBlock[n];
                peakAbs = std::max({peakAbs, std::fabs(l), std::fabs(r)});
                if (blockStart + n >= firstMeasureIdx)
                {
                    sumSq += 0.5 * (l * l + r * r);
                    ++count;
                }
            }
        }
        const double rms = count > 0
            ? std::sqrt(sumSq / static_cast<double>(count))
            : 0.0;
        std::cout << "  mix=" << mixValue
                  << "  peak=" << peakAbs
                  << "  rms="  << rms
                  << "  (dry ref=" << dryReferenceRms << ")\n";
        mixSweepCsv << mixValue << ","
                    << peakAbs  << ","
                    << rms      << ","
                    << dryReferenceRms << "\n";
    }
    mixSweepCsv.close();

    std::cout << "[chronos reverb] wrote all characterisation CSVs to " << kLogDir << "\n";
    return 0;
}

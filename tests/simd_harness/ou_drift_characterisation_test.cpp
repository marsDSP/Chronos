// OU drift characterisation harness.
//
// Sweeps the OU drift amount knob across [0..1] and, for each sweep
// point, collects:
//
//   * summary stats: mean, stdev, min, max of the per-sample drift
//     output over a long run,
//   * autocorrelation at a handful of lags, so the effective
//     decorrelation time can be read off (the process has a 10 Hz
//     mean-reversion lowpass so the autocorrelation should drop to
//     ~0.37 by the time we're a few tens of ms in),
//   * a histogram of output values so the distribution shape can be
//     checked (expected: Gaussian-ish, centred on 0, variance scaling
//     with amount),
//   * an impulse-response-style "mean reversion" trace: initialise the
//     internal state to a known large value and run with
//     driftAmount = 0 so the only active force is mean reversion; log
//     the first ~200 ms of the decay.
//
// Emits three CSVs:
//   tests/simd_harness/logs/ou_drift_stats.csv         - per-amount stats
//   tests/simd_harness/logs/ou_drift_autocorr.csv      - lag vs autocorr per amount
//   tests/simd_harness/logs/ou_drift_distribution.csv  - histogram per amount
//   tests/simd_harness/logs/ou_drift_mean_reversion.csv - decay trace

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "dsp/modulation/ou_drift_process.h"

namespace fs = std::filesystem;

namespace
{
    constexpr double kSampleRate       = 48000.0;
    constexpr int    kBlockSize        = 128;
    constexpr double kRunDurationSec   = 5.0;
    constexpr int    kHistogramBins    = 81;
    constexpr double kHistogramRange   = 0.25; // symmetric around 0
    // Autocorrelation lags expressed in milliseconds, converted to
    // samples at kSampleRate below. Covers the 1 kHz .. 0.5 Hz band.
    constexpr double kAutocorrLagsInMilliseconds[] = {
        0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0
    };
    constexpr std::size_t kNumberOfAutocorrLags =
        sizeof(kAutocorrLagsInMilliseconds) / sizeof(double);

    const fs::path kLogDir = "tests/simd_harness/logs";

    void ensureLogDir()
    {
        std::error_code ec;
        fs::create_directories(kLogDir, ec);
    }

    // Fill one block of the OU process into the output vector starting
    // at absolute sample index `writeOffset` (sample-by-sample, using
    // channel 0). Returns the number of samples actually produced.
    int produceBlockOfOuDriftSamples(MarsDSP::DSP::Modulation::OuDriftProcess& ouProcess,
                                     float                                     driftAmount,
                                     std::vector<float>&                       outputSamples,
                                     int                                       writeOffset,
                                     int                                       numSamplesInBlock)
    {
        ouProcess.prepareBlock(driftAmount, numSamplesInBlock);
        for (int sampleIndex = 0; sampleIndex < numSamplesInBlock; ++sampleIndex)
            outputSamples[static_cast<std::size_t>(writeOffset + sampleIndex)] =
                ouProcess.process(sampleIndex, 0);
        return numSamplesInBlock;
    }

    // Run the OU process for kRunDurationSec and return the full sample
    // vector, so the caller can compute stats / autocorrelation / etc.
    std::vector<float> collectOuDriftRun(float driftAmountNormalised)
    {
        const int totalSamples =
            static_cast<int>(kRunDurationSec * kSampleRate);

        MarsDSP::DSP::Modulation::OuDriftProcess ouProcess;
        ouProcess.prepare(kSampleRate, kBlockSize, 1);

        std::vector<float> samples(static_cast<std::size_t>(totalSamples), 0.0f);

        for (int writeOffset = 0; writeOffset < totalSamples; writeOffset += kBlockSize)
        {
            const int numSamplesInBlock =
                std::min(kBlockSize, totalSamples - writeOffset);
            produceBlockOfOuDriftSamples(ouProcess, driftAmountNormalised,
                                         samples, writeOffset, numSamplesInBlock);
        }
        return samples;
    }

    // Run an amount=0 decay from a known "kicked" initial state. We
    // mimic the non-zero state by first running with a large drift
    // amount for a short pre-roll so the state accumulates, then flip
    // the amount to zero and capture the tail.
    std::vector<float> collectMeanReversionDecayTrace()
    {
        const int prerollSamples = static_cast<int>(0.100 * kSampleRate);   // 100 ms kick
        const int decaySamples   = static_cast<int>(0.500 * kSampleRate);   // 500 ms decay

        MarsDSP::DSP::Modulation::OuDriftProcess ouProcess;
        ouProcess.prepare(kSampleRate, kBlockSize, 1);

        // Pre-roll at amount = 1 so the state walks away from zero.
        std::vector<float> prerollScratch(static_cast<std::size_t>(prerollSamples), 0.0f);
        for (int writeOffset = 0; writeOffset < prerollSamples; writeOffset += kBlockSize)
        {
            const int numSamplesInBlock =
                std::min(kBlockSize, prerollSamples - writeOffset);
            produceBlockOfOuDriftSamples(ouProcess, 1.0f,
                                         prerollScratch, writeOffset, numSamplesInBlock);
        }

        // Tail at amount = 0 captures pure mean-reversion behaviour.
        std::vector<float> decaySamples_(static_cast<std::size_t>(decaySamples), 0.0f);
        for (int writeOffset = 0; writeOffset < decaySamples; writeOffset += kBlockSize)
        {
            const int numSamplesInBlock =
                std::min(kBlockSize, decaySamples - writeOffset);
            produceBlockOfOuDriftSamples(ouProcess, 0.0f,
                                         decaySamples_, writeOffset, numSamplesInBlock);
        }
        return decaySamples_;
    }

    struct OuDriftSummaryStatistics
    {
        float  driftAmountNormalised;
        double meanValue;
        double standardDeviation;
        double minimumValue;
        double maximumValue;
        double peakAbsoluteValue;
    };

    OuDriftSummaryStatistics computeSummary(float driftAmountNormalised,
                                            const std::vector<float>& samples)
    {
        double sum          = 0.0;
        double sumOfSquares = 0.0;
        double minValue     =  std::numeric_limits<double>::infinity();
        double maxValue     = -std::numeric_limits<double>::infinity();

        for (float sampleValue : samples)
        {
            const double value = static_cast<double>(sampleValue);
            sum          += value;
            sumOfSquares += value * value;
            if (value < minValue) minValue = value;
            if (value > maxValue) maxValue = value;
        }
        const double count  = static_cast<double>(samples.size());
        const double mean   = sum / count;
        const double var    = std::max(0.0, sumOfSquares / count - mean * mean);
        const double stdev  = std::sqrt(var);
        const double peakAbs = std::max(std::fabs(minValue), std::fabs(maxValue));

        return OuDriftSummaryStatistics{driftAmountNormalised,
                                        mean, stdev, minValue, maxValue, peakAbs};
    }

    // Autocorrelation at lag 'lagInSamples' computed over the supplied
    // sample vector, minus the mean, normalised by the total variance.
    double computeNormalisedAutocorrelationAtLag(const std::vector<float>& samples,
                                                 int                       lagInSamples)
    {
        if (lagInSamples <= 0 || static_cast<int>(samples.size()) <= lagInSamples)
            return 0.0;

        double sum = 0.0;
        for (float s : samples) sum += static_cast<double>(s);
        const double mean = sum / static_cast<double>(samples.size());

        double numerator = 0.0;
        double totalVar  = 0.0;
        for (std::size_t i = 0; i + static_cast<std::size_t>(lagInSamples) < samples.size(); ++i)
        {
            const double a = static_cast<double>(samples[i])                                - mean;
            const double b = static_cast<double>(samples[i + static_cast<std::size_t>(lagInSamples)]) - mean;
            numerator += a * b;
        }
        for (float s : samples)
        {
            const double centered = static_cast<double>(s) - mean;
            totalVar += centered * centered;
        }
        if (totalVar <= 0.0) return 0.0;
        return numerator / totalVar;
    }

    // Bin samples into a histogram centred on 0 with half-range
    // kHistogramRange so sparse / dense parts are both visible.
    std::array<int, kHistogramBins> computeHistogram(const std::vector<float>& samples)
    {
        std::array<int, kHistogramBins> bins{};
        const double lowerEdge = -kHistogramRange;
        const double upperEdge = +kHistogramRange;
        const double binWidth  = (upperEdge - lowerEdge) / static_cast<double>(kHistogramBins);
        for (float sampleValue : samples)
        {
            const double value = static_cast<double>(sampleValue);
            int binIndex = static_cast<int>(std::floor((value - lowerEdge) / binWidth));
            binIndex = std::clamp(binIndex, 0, kHistogramBins - 1);
            ++bins[static_cast<std::size_t>(binIndex)];
        }
        return bins;
    }
} // namespace

int main()
{
    ensureLogDir();

    const std::vector<float> amountSweepValues{
        0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f
    };

    // ------------------------------------------------------------------
    // 1) Per-amount summary + histogram + autocorrelation
    // ------------------------------------------------------------------
    std::ofstream statsCsv         (kLogDir / "ou_drift_stats.csv");
    statsCsv << "amount,mean,stdev,min_value,max_value,peak_abs\n";

    std::ofstream autocorrCsv      (kLogDir / "ou_drift_autocorr.csv");
    autocorrCsv << "amount,lag_ms,autocorr\n";

    std::ofstream histogramCsv     (kLogDir / "ou_drift_distribution.csv");
    histogramCsv << "amount,bin_centre,count\n";

    std::cout << "[ou drift] amount sweep (" << kRunDurationSec << " s at "
              << kSampleRate << " Hz per point)\n";

    for (float amountValue : amountSweepValues)
    {
        const std::vector<float> samples = collectOuDriftRun(amountValue);

        const OuDriftSummaryStatistics summary = computeSummary(amountValue, samples);
        std::cout << "  amount = " << summary.driftAmountNormalised
                  << "  mean="  << summary.meanValue
                  << "  stdev=" << summary.standardDeviation
                  << "  peak=" << summary.peakAbsoluteValue
                  << "\n";
        statsCsv << summary.driftAmountNormalised << ","
                 << summary.meanValue            << ","
                 << summary.standardDeviation    << ","
                 << summary.minimumValue         << ","
                 << summary.maximumValue         << ","
                 << summary.peakAbsoluteValue    << "\n";

        // Autocorrelation scan.
        for (std::size_t lagIndex = 0; lagIndex < kNumberOfAutocorrLags; ++lagIndex)
        {
            const double lagInMilliseconds = kAutocorrLagsInMilliseconds[lagIndex];
            const int    lagInSamples      =
                static_cast<int>(lagInMilliseconds * 0.001 * kSampleRate);
            const double autocorrelation =
                computeNormalisedAutocorrelationAtLag(samples, lagInSamples);
            autocorrCsv << amountValue << ","
                        << lagInMilliseconds << ","
                        << autocorrelation << "\n";
        }

        // Histogram.
        const auto histogram = computeHistogram(samples);
        const double binWidth = (2.0 * kHistogramRange) / static_cast<double>(kHistogramBins);
        for (std::size_t binIndex = 0; binIndex < kHistogramBins; ++binIndex)
        {
            const double binCentre = -kHistogramRange
                                     + (static_cast<double>(binIndex) + 0.5) * binWidth;
            histogramCsv << amountValue << ","
                         << binCentre << ","
                         << histogram[binIndex] << "\n";
        }
    }

    statsCsv    .close();
    autocorrCsv .close();
    histogramCsv.close();

    // ------------------------------------------------------------------
    // 2) Mean-reversion decay trace
    // ------------------------------------------------------------------
    std::cout << "[ou drift] mean-reversion decay trace (100 ms kick, 500 ms decay)\n";
    const std::vector<float> decayTrace = collectMeanReversionDecayTrace();
    std::ofstream decayCsv(kLogDir / "ou_drift_mean_reversion.csv");
    decayCsv << "sample_index,time_ms,value\n";
    for (std::size_t sampleIndex = 0; sampleIndex < decayTrace.size(); ++sampleIndex)
    {
        const double timeInMilliseconds =
            static_cast<double>(sampleIndex) * 1000.0 / kSampleRate;
        decayCsv << sampleIndex << ","
                 << timeInMilliseconds << ","
                 << decayTrace[sampleIndex] << "\n";
    }
    decayCsv.close();

    std::cout << "[ou drift] wrote "
              << (kLogDir / "ou_drift_stats.csv")         << ", "
              << (kLogDir / "ou_drift_autocorr.csv")      << ", "
              << (kLogDir / "ou_drift_distribution.csv")  << ", "
              << (kLogDir / "ou_drift_mean_reversion.csv")
              << "\n";
    return 0;
}

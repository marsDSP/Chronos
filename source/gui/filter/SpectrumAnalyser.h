#pragma once

#ifndef CHRONOS_SPECTRUM_ANALYSER_H
#define CHRONOS_SPECTRUM_ANALYSER_H

#include "math/FFT.h"
#include "math/SeqMakima.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <numbers>
#include <vector>

namespace MarsDSP::GUI {

// The FILTER page spectrum pipeline. Header-only and JUCE-free, so the
// harness links it without an editor. Message thread only: the FFT
// allocates in prepare.
//
//   sliding history (+ DC-blocking high-pass on the input)
//     -> periodic Hann window scaled by 2/N
//     -> real FFT -> power |X|^2
//     -> 1/n-octave boxcar smoothing, applied twice, via cumulative sums
//     -> power to dB (10 log10)
//     -> tilt (+ slope * log2(f / 1 kHz)), so pink material reads flat
//     -> decay (instant attack via max, exponential release toward a floor)
//     -> modified-Akima interpolation from the bin knots to the pixel columns
//
// The dB, Hz, and time constants below are model values, not dimensions,
// so they live here rather than in Metrics.
class SpectrumAnalyser {
public:
    static constexpr int kOrder = 12;
    static constexpr int kSize = 1 << kOrder;          // 4096 samples, 85 ms at 48 kHz
    static constexpr int kBins = kSize / 2 + 1;        // DC .. Nyquist
    static constexpr float kFloorDb = -72.0f;          // the display bottom; 0 dB at the top
    static constexpr float kTiltDbPerOct = 4.5f;       // pink noise reads flat
    static constexpr double kSmoothOctaves = 1.0 / 6.0;
    static constexpr float kRefreshHz = 30.0f;
    static constexpr float kDecaySeconds = 0.5f;       // release from 0 dB to the floor

    void prepare(const double sampleRate)
    {
        sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
        fft_.setSize(static_cast<std::size_t>(kSize));
        buildWindow_();
        buildSmoother_();
        buildTilt_();
        buildBinFreqs_();
        // The spline references the knot arrays in place; they are members
        // with fixed addresses, so it stays valid for the analyser's life.
        makima_ = std::make_unique<Interpolation::SeqMakima<float>>(binFreq_.data(), specDb_.data(),
                                                                    static_cast<std::size_t>(kBins), 0.0f, 0.0f);
        decayP_ = decayCoefficient_(kRefreshHz, kFloorDb, kDecaySeconds);
        clear();
        prepared_ = true;
    }

    [[nodiscard]] bool isPrepared() const noexcept { return prepared_; }
    [[nodiscard]] double sampleRate() const noexcept { return sampleRate_; }

    // Forget the history and drop the trace to silence.
    void clear() noexcept
    {
        history_.fill(0.0f);
        writePos_ = 0;
        dcX_ = dcY_ = 0.0f;
        specDb_.fill(kSilentDb);
        decayState_.fill(kSilentDb);
        hasData_ = false;
    }

    // Append samples to the sliding history through the DC blocker.
    void push(const float* in, const int n) noexcept
    {
        for (int i = 0; i < n; ++i)
        {
            const float x = in[i];
            const float y = x - dcX_ + kDcPole * dcY_;
            dcX_ = x;
            dcY_ = y;
            history_[static_cast<std::size_t>(writePos_)] = y;
            writePos_ = (writePos_ + 1) & (kSize - 1);
        }
        if (n > 0)
            hasData_ = true;
    }

    // Recompute the trace from the most recent kSize samples.
    void update()
    {
        if (! prepared_)
            return;

        // Oldest sample first, so the window's centre sits mid-history.
        for (int i = 0; i < kSize; ++i)
        {
            const int idx = (writePos_ + i) & (kSize - 1);
            fftIn_[static_cast<std::size_t>(i)] = history_[static_cast<std::size_t>(idx)]
                                                 * window_[static_cast<std::size_t>(i)];
        }
        fft_.forward(fftIn_.data(), fftOut_.data());

        // The real transform packs DC in out[0].real() and Nyquist in
        // out[0].imag(); bins 1 .. N/2-1 are the complex values.
        constexpr int half = kSize / 2;
        power_[0] = fftOut_[0].real() * fftOut_[0].real();
        power_[static_cast<std::size_t>(half)] = fftOut_[0].imag() * fftOut_[0].imag();
        for (int k = 1; k < half; ++k)
        {
            const auto& c = fftOut_[static_cast<std::size_t>(k)];
            power_[static_cast<std::size_t>(k)] = c.real() * c.real() + c.imag() * c.imag();
        }

        boxcarAverage_();
        boxcarAverage_();

        for (int k = 0; k < kBins; ++k)
        {
            const auto u = static_cast<std::size_t>(k);
            const float db = 10.0f * std::log10(std::max(power_[u], kPowerFloor)) + tilt_[u];
            const float prev = decayState_[u];
            const float released = prev + decayP_ * (db - prev); // exponential release
            const float v = std::max(db, released);              // instant attack
            decayState_[u] = v;
            specDb_[u] = v;
        }
    }

    [[nodiscard]] bool hasData() const noexcept { return hasData_; }
    [[nodiscard]] static constexpr int numBins() noexcept { return kBins; }
    [[nodiscard]] float binDb(const int k) const noexcept { return specDb_[static_cast<std::size_t>(k)]; }
    [[nodiscard]] double binFrequency(const int k) const noexcept
    {
        return static_cast<double>(k) * sampleRate_ / static_cast<double>(kSize);
    }

    // The trace at n ascending frequencies through the makima spline over
    // the bin knots: the smooth curve the display draws, one value per
    // pixel column. Outside the knot range the ends hold.
    void evaluateDb(const float* freqsAscending, float* outDb, const int n)
    {
        if (! prepared_ || makima_ == nullptr || n <= 0)
            return;
        makima_->prepare();
        makima_->eval(freqsAscending, outDb, static_cast<std::size_t>(n));
    }

    // The trace at a frequency, interpolated linearly between the two
    // neighbouring bins. Below bin 1 the trace holds bin 1.
    [[nodiscard]] float magnitudeDbAt(const float freqHz) const noexcept
    {
        const double pos = static_cast<double>(freqHz) * static_cast<double>(kSize) / sampleRate_;
        if (pos <= 1.0)
            return specDb_[1];
        const double maxPos = static_cast<double>(kBins - 1);
        if (pos >= maxPos)
            return specDb_[static_cast<std::size_t>(kBins - 1)];
        const int k0 = static_cast<int>(pos);
        const float t = static_cast<float>(pos - static_cast<double>(k0));
        const float a = specDb_[static_cast<std::size_t>(k0)];
        const float b = specDb_[static_cast<std::size_t>(k0 + 1)];
        return a + t * (b - a);
    }

private:
    static constexpr float kDcPole = 0.9999f;
    static constexpr float kPowerFloor = 1.0e-12f;
    static constexpr float kSilentDb = -240.0f;

    // Periodic Hann scaled by 2/N: a full-scale sine reads its amplitude.
    void buildWindow_() noexcept
    {
        const double twoOverN = 2.0 / static_cast<double>(kSize);
        for (int i = 0; i < kSize; ++i)
            window_[static_cast<std::size_t>(i)] = static_cast<float>(
                twoOverN * 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi_v<double> * static_cast<double>(i)
                                                 / static_cast<double>(kSize))));
    }

    // Per-bin boxcar bounds spanning +/- kSmoothOctaves/2 around the bin.
    void buildSmoother_() noexcept
    {
        const double factor = std::pow(2.0, kSmoothOctaves / 2.0);
        const double inv = 1.0 / factor;
        for (int i = 0; i < kBins; ++i)
        {
            const auto u = static_cast<std::size_t>(i);
            loIdx_[u] = static_cast<int>(std::lround(static_cast<double>(i) * inv));
            hiIdx_[u] = std::min(kBins - 1, static_cast<int>(std::lround(static_cast<double>(i) * factor)) + 1);
            const int cnt = std::max(1, hiIdx_[u] - loIdx_[u]);
            invCount_[u] = 1.0f / static_cast<float>(cnt);
        }
    }

    // The knot x positions for the spline: the centre frequency of each bin.
    void buildBinFreqs_() noexcept
    {
        for (int i = 0; i < kBins; ++i)
            binFreq_[static_cast<std::size_t>(i)] = static_cast<float>(binFrequency(i));
    }

    // Tilt in dB per bin: + slope * log2(f / 1 kHz). Bin 0 borrows bin 1.
    void buildTilt_() noexcept
    {
        const double delta = sampleRate_ * 0.5 / static_cast<double>(kBins - 1);
        for (int i = 1; i < kBins; ++i)
            tilt_[static_cast<std::size_t>(i)] = static_cast<float>(
                std::log2(static_cast<double>(i) * delta / 1000.0) * static_cast<double>(kTiltDbPerOct));
        tilt_[0] = tilt_[1];
    }

    // The release coefficient that carries 0 dB down to minDb in
    // decaySeconds at refreshRate frames per second, against a -120 dB
    // floor of the exponential.
    static float decayCoefficient_(const float refreshRate, const float minDb, const float decaySeconds) noexcept
    {
        constexpr float floorDb = -120.0f;
        constexpr float startDb = 0.0f;
        if (decaySeconds <= 0.0f || minDb <= floorDb)
            return 1.0f;
        const float nFrames = refreshRate * decaySeconds;
        const float startDist = startDb - floorDb;
        const float targetDist = minDb - floorDb;
        return std::clamp(1.0f - std::pow(targetDist / startDist, 1.0f / nFrames), 0.0f, 1.0f);
    }

    // One pass of the 1/n-octave boxcar through a cumulative sum.
    void boxcarAverage_() noexcept
    {
        cumSum_[0] = 0.0;
        for (int i = 0; i < kBins; ++i)
            cumSum_[static_cast<std::size_t>(i + 1)] = cumSum_[static_cast<std::size_t>(i)]
                                                      + static_cast<double>(power_[static_cast<std::size_t>(i)]);
        for (int i = 0; i < kBins; ++i)
        {
            const auto u = static_cast<std::size_t>(i);
            power_[u] = static_cast<float>((cumSum_[static_cast<std::size_t>(hiIdx_[u])]
                                            - cumSum_[static_cast<std::size_t>(loIdx_[u])])
                                           * static_cast<double>(invCount_[u]));
        }
    }

    double sampleRate_ = 48000.0;
    bool prepared_ = false;
    bool hasData_ = false;
    float decayP_ = 0.05f;
    float dcX_ = 0.0f;
    float dcY_ = 0.0f;
    int writePos_ = 0;

    MarsDSP::MathOps::RealFFT<float> fft_;
    std::unique_ptr<Interpolation::SeqMakima<float>> makima_;
    std::array<float, kBins> binFreq_ {};
    std::array<float, kSize> history_ {};
    std::array<float, kSize> window_ {};
    std::array<float, kSize> fftIn_ {};
    std::array<std::complex<float>, kSize / 2> fftOut_ {};
    std::array<float, kBins> power_ {};
    std::array<float, kBins> tilt_ {};
    std::array<float, kBins> specDb_ {};
    std::array<float, kBins> decayState_ {};
    std::array<int, kBins> loIdx_ {};
    std::array<int, kBins> hiIdx_ {};
    std::array<float, kBins> invCount_ {};
    std::array<double, kBins + 1> cumSum_ {};
};

} // namespace MarsDSP::GUI

#endif

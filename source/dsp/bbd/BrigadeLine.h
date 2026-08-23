#pragma once

#ifndef CHRONOS_BBD_BRIGADE_LINE_H
#define CHRONOS_BBD_BRIGADE_LINE_H

#include "PoleBank.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>

namespace MarsDSP::BBD
{
    /** Clocked bucket-brigade delay line.
     *  The analog pole banks run at the audio rate. A shift register of
     *  kStages buckets moves the charge in a two-phase event loop.
     */
    class BrigadeLine
    {
    public:
        static constexpr int kStages = 4096;

        // The split-step transport offset for the BBD timing compensation.
        // The read-before-write hand-off shifts the line delay by one sample.
        // See docs/dsp-notes.md, "BrigadeLine split-step transport".
        static constexpr float kSplitStepOffset = -1.0f;

        static constexpr std::size_t bbdStorageFloats (int numChannels) noexcept
        {
            constexpr std::size_t perChan = (static_cast<std::size_t> (kStages + 1) + 15u) & ~static_cast<std::size_t> (15u);
            return static_cast<std::size_t> (numChannels) * perChan;
        }

        BrigadeLine() = default;

        void prepare (double sampleRate, float* storage = nullptr)
        {
            fs_ = static_cast<float> (sampleRate > 0.0 ? sampleRate : 48000.0);
            Ts_ = 1.0f / fs_;

            kMinClockHz_ = fs_ / 30.0f;
            kMaxClockHz_ = 100.0f * fs_;

            inputBank_ = InputPoleBank (Ts_);
            outputBank_ = OutputPoleBank (Ts_);
            H0_ = outputBank_.calcH0();

            storage_ = storage;
            setDelaySeconds (0.375f);
            reset();
        }

        void setStorage (float* storage) noexcept
        {
            storage_ = storage;
            reset();
        }

        void reset() noexcept
        {
            bufferPtr_ = 0;
            if (storage_ != nullptr)
                std::fill (storage_, storage_ + (kStages + 1), 0.0f);
            yBBD_old_ = 0.0f;
            tn_ = 0.0f;
            evenOn_ = true;

            inputBank_.reset();
            outputBank_.reset();
            H0_ = outputBank_.calcH0();
        }

        void setInputFilterFreq (float freqHz = kInputCutoffHz) noexcept
        {
            inputBank_.set_freq (freqHz);
            inputBank_.set_time (tn_);
        }

        void setOutputFilterFreq (float freqHz = kOutputCutoffHz) noexcept
        {
            outputBank_.set_freq (freqHz);
            outputBank_.set_time (tn_);
            H0_ = outputBank_.calcH0();
        }

        void setClockHz (float clkHz) noexcept
        {
            const float clkClamped = std::clamp (clkHz, kMinClockHz_, kMaxClockHz_);
            Ts_bbd_ = 1.0f / clkClamped;
            const float deltaNorm = 2.0f * Ts_bbd_ * fs_;
            inputBank_.set_delta (deltaNorm);
            outputBank_.set_delta (deltaNorm);
        }

        // One clock authority for the line and the loop. The formula
        // (2 * kStages + 0.5) * fs / delay matches ClockModel. See
        // docs/dsp-notes.md, "BrigadeLine split-step transport".
        [[nodiscard]] static float clockForDelay (float dEffSamples, double sampleRate) noexcept
        {
            const float minClk = static_cast<float> (sampleRate / 30.0);
            const float maxClk = static_cast<float> (100.0 * sampleRate);
            const auto kStagesD = static_cast<double> (kStages);
            const double minDelay = (2.0 * kStagesD + 0.5) * sampleRate / static_cast<double> (maxClk);
            const double safeDelay = std::max (minDelay, static_cast<double> (dEffSamples));
            const double fClk = (2.0 * kStagesD + 0.5) * sampleRate / safeDelay;
            return std::clamp (static_cast<float> (fClk), minClk, maxClk);
        }

        void setDelaySeconds (float delaySec) noexcept
        {
            const float dSamples = std::max (Ts_, delaySec) * fs_;
            setClockHz (clockForDelay (dSamples, static_cast<double> (fs_)));
        }

        [[nodiscard]] float getClockHz() const noexcept
        {
            return Ts_bbd_ > 0.0f ? (1.0f / Ts_bbd_) : kMinClockHz_;
        }

        [[nodiscard]] static double getBankGroupDelayAtDC (double sampleRate) noexcept
        {
            const double w0In = 2.0 * std::numbers::pi * static_cast<double> (kInputCutoffHz);
            const double w0Out = 2.0 * std::numbers::pi * static_cast<double> (kOutputCutoffHz);
            const double qFactor = (1.0 / kButterworthQ1 + 1.0 / kButterworthQ2);
            return sampleRate * (qFactor / w0In + qFactor / w0Out);
        }

        // Read the delayed tap from the bucket line and output bank.
        inline float readTap() noexcept
        {
            if (storage_ == nullptr)
                return 0.0f;

            inputBank_.set_time (tn_);
            outputBank_.set_time (tn_);

            std::array<std::complex<float>, 4> xOutAccum {};
            float yBBD = 0.0f;
            float delta = 0.0f;
            int iterations = 0;

            const float stepNorm = Ts_bbd_ * fs_;
            while (tn_ < 1.0f)
            {
                if (++iterations > 10000)
                    break;

                if (evenOn_)
                {
                    inputBank_.calcG (tn_);
                    float val = 0.0f;
                    for (int m = 0; m < 4; ++m)
                    {
                        const auto term = inputBank_.Gcalc[static_cast<std::size_t>(m)] * inputBank_.x[static_cast<std::size_t>(m)]
                                        + ((inputBank_.Gcalc[static_cast<std::size_t>(m)] - inputBank_.getG0(m)) / inputBank_.getPole(m)) * lastIn_;
                        val += term.real();
                    }

                    if (!std::isfinite (val))
                        val = 0.0f;

                    storage_[bufferPtr_++] = val;
                    if (bufferPtr_ >= kStages)
                        bufferPtr_ = 0;
                }
                else
                {
                    yBBD = storage_[bufferPtr_];
                    delta = yBBD - yBBD_old_;
                    yBBD_old_ = yBBD;
                    outputBank_.calcG (tn_);
                    for (int m = 0; m < 4; ++m)
                        xOutAccum[m] += outputBank_.Gcalc[m] * delta;
                }

                evenOn_ = !evenOn_;
                tn_ += stepNorm;
            }
            tn_ -= 1.0f;

            outputBank_.process (xOutAccum);

            float sumOut = 0.0f;
            for (int m = 0; m < 4; ++m)
                sumOut += outputBank_.x[m].real();

            const float out = H0_ * yBBD_old_ + sumOut;
            if (!std::isfinite (out))
            {
                reset();
                return 0.0f;
            }
            return out;
        }

        /// Push one sample into the input bank.
        inline void writeSample (float u) noexcept
        {
            if (!std::isfinite (u))
                u = 0.0f;
            lastIn_ = u;
            inputBank_.process (u);
        }

        /// Read the tap, then push the sample. One full audio-rate step.
        inline float process (float u) noexcept
        {
            const float out = readTap();
            writeSample (u);
            return out;
        }

    private:
        float fs_ { 48000.0f };
        float Ts_ { 1.0f / 48000.0f };
        float Ts_bbd_ { 1.0f / 48000.0f };
        float kMinClockHz_ { 1600.0f };
        float kMaxClockHz_ { 4800000.0f };
        float lastIn_ { 0.0f };

        InputPoleBank inputBank_ {};
        OutputPoleBank outputBank_ {};
        float H0_ { 1.0f };

        float* storage_ { nullptr };
        int bufferPtr_ { 0 };

        float yBBD_old_ { 0.0f };
        float tn_ { 0.0f };
        bool evenOn_ { true };
    };
}
#endif

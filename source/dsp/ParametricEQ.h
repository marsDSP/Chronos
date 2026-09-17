#pragma once

#ifndef CHRONOS_PARAMETRIC_EQ_H
#define CHRONOS_PARAMETRIC_EQ_H

#include "StateVariable.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace MarsDSP::Filters
{
    /** The four free bands of the FILTER page EQ.
     *
     *  Each band is one SimdSVF (L/R in lanes 0/1) of type Bell, Low
     *  Shelf, High Shelf, or Notch. The low and high cuts are not here:
     *  they are the OutputFilterStage pair, which runs before this stage.
     *
     *  A band that is off is skipped entirely, so the stage is
     *  bit-transparent with every band off. A band toggled on resets its
     *  state first, so it never replays a stale tail. Coefficients ramp
     *  across 32-sample sub-blocks through setCoeffForBlock, the same
     *  zipper-free idiom as the Digital cut path. Zero latency.
     *
     *  magnitudeDb() is the exact digital response of the same
     *  (g, k, m0, m1, m2) evaluated at Omega = tan(pi f / fs) / g, so the
     *  curve the display draws is the response the audio path has.
     */
    class ParametricEQ
    {
    public:
        static constexpr int kNumBands = 4;
        static constexpr int kSubBlock = 32;

        // The free band types, in the order of the eqNType choice parameter.
        enum class Type
        {
            Bell = 0,
            LowShelf = 1,
            HighShelf = 2,
            Notch = 3
        };

        struct Band
        {
            bool on = false;
            int type = 0; // Type as an int, the choice index
            float freqHz = 1000.0f;
            float gainDb = 0.0f;
            float q = 0.707f;
        };

        void prepare (double sampleRate, int numChannels) noexcept
        {
            sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
            numChannels_ = numChannels > 1 ? 2 : 1;
            reset();
        }

        void reset() noexcept
        {
            for (auto& s : svf_)
                s.reset();
        }

        // Set one band. Turning a band on resets its filter state.
        void setBand (const int i, const Band& b) noexcept
        {
            auto& cur = bands_[static_cast<std::size_t> (i)];
            if (b.on && ! cur.on)
                svf_[static_cast<std::size_t> (i)].reset();
            cur = b;
        }

        [[nodiscard]] const Band& band (const int i) const noexcept
        {
            return bands_[static_cast<std::size_t> (i)];
        }

        // Process in place. inR may be null for mono.
        void process (float* inL, float* inR, const int n) noexcept
        {
            if (n <= 0 || inL == nullptr)
                return;
            const bool hasR = (numChannels_ > 1 && inR != nullptr);

            for (int i = 0; i < kNumBands; ++i)
            {
                const auto& b = bands_[static_cast<std::size_t> (i)];
                if (! b.on)
                    continue;

                auto& svf = svf_[static_cast<std::size_t> (i)];
                const auto type = svfTypeFor (b.type);

                for (int offset = 0; offset < n;)
                {
                    const int sub = std::min (kSubBlock, n - offset);
                    svf.setCoeffForBlock (type, sampleRate_,
                                          static_cast<double> (b.freqHz),
                                          static_cast<double> (b.q),
                                          static_cast<double> (b.gainDb), sub);
                    for (int s = 0; s < sub; ++s)
                    {
                        float l = inL[offset + s];
                        float r = hasR ? inR[offset + s] : 0.0f;
                        svf.processBlockStep (l, r);
                        inL[offset + s] = l;
                        if (hasR)
                            inR[offset + s] = r;
                    }
                    offset += sub;
                }
            }
        }

        // ------------------------------------------------------------------
        // The analytic response, shared by the display and the harness.
        // ------------------------------------------------------------------

        static SimdSVF::SVFType svfTypeFor (const int type) noexcept
        {
            switch (static_cast<Type> (type))
            {
                case Type::LowShelf:  return SimdSVF::SVFType::LowShelf;
                case Type::HighShelf: return SimdSVF::SVFType::HighShelf;
                case Type::Notch:     return SimdSVF::SVFType::Notch;
                case Type::Bell:
                default:              return SimdSVF::SVFType::Bell;
            }
        }

        // |H(f)| in dB of one SimdSVF section. The coefficients follow
        // SimdSVF::setCoeffPostGK exactly, in double, and H(s) with
        // s = j * Omega, Omega = tan(pi f / fs) / g, is the exact digital
        // magnitude of the trapezoidal SVF.
        static double magnitudeDb (const SimdSVF::SVFType type, const double sampleRate,
                                   const double f, double freqHz, double q, const double gainDb) noexcept
        {
            constexpr double pi = std::numbers::pi_v<double>;
            const double fs = sampleRate > 0.0 ? sampleRate : 48000.0;
            const double nyq = 0.49 * fs;
            freqHz = std::clamp (freqHz, 10.0, nyq);
            q = std::max (q, 0.025);

            const double A = std::pow (10.0, gainDb / 40.0);
            const double sqrtA = std::sqrt (A);
            const double gt = std::tan (pi * freqHz / fs);
            const double kk = 1.0 / q;

            double g = gt, k = kk, m0 = 1.0, m1 = 0.0, m2 = 0.0;
            switch (type)
            {
                case SimdSVF::SVFType::LowPass:   m0 = 0.0; m1 = 0.0;  m2 = 1.0;  break;
                case SimdSVF::SVFType::HighPass:  m0 = 1.0; m1 = -kk;  m2 = -1.0; break;
                case SimdSVF::SVFType::BandPass:  m0 = 0.0; m1 = kk;   m2 = 0.0;  break;
                case SimdSVF::SVFType::Notch:     m0 = 1.0; m1 = -kk;  m2 = 0.0;  break;
                case SimdSVF::SVFType::AllPass:   m0 = 1.0; m1 = -2.0 * kk; m2 = 0.0; break;
                case SimdSVF::SVFType::Bell:      k = kk / A; m0 = 1.0; m1 = k * (A * A - 1.0); m2 = 0.0; break;
                case SimdSVF::SVFType::LowShelf:  g = gt / sqrtA; m0 = 1.0; m1 = k * (A - 1.0); m2 = A * A - 1.0; break;
                case SimdSVF::SVFType::HighShelf: g = gt * sqrtA; m0 = A * A; m1 = k * (1.0 - A) * A; m2 = 1.0 - A * A; break;
                case SimdSVF::SVFType::TiltShelf: g = gt * sqrtA; m0 = A; m1 = kk * (1.0 - A); m2 = 1.0 / A - A; break;
            }

            if (g <= 0.0)
                return 0.0;
            const double fClamped = std::clamp (f, 1.0, nyq);
            const double omega = std::tan (pi * fClamped / fs) / g;
            const double w2 = omega * omega;
            const double numRe = -m0 * w2 + (m0 + m2);
            const double numIm = (m0 * k + m1) * omega;
            const double denRe = 1.0 - w2;
            const double denIm = k * omega;
            const double num2 = numRe * numRe + numIm * numIm;
            const double den2 = denRe * denRe + denIm * denIm;
            if (den2 <= 0.0)
                return 0.0;
            return 10.0 * std::log10 (std::max (num2 / den2, 1.0e-30));
        }

        // One free band by its choice type.
        static double bandMagnitudeDb (const int type, const double sampleRate, const double f,
                                       const double freqHz, const double gainDb, const double q) noexcept
        {
            return magnitudeDb (svfTypeFor (type), sampleRate, f, freqHz, q, gainDb);
        }

        // One cut of the OutputFilterStage pair: second order at Q 0.7071.
        // Exact for the Digital SVF; the Analog Sallen-Key shares the
        // Butterworth prototype (sallen_key_response_check bounds it).
        static constexpr double kCutQ = 0.7071;
        static double cutMagnitudeDb (const bool highPass, const double sampleRate,
                                      const double f, const double cutoffHz) noexcept
        {
            return magnitudeDb (highPass ? SimdSVF::SVFType::HighPass : SimdSVF::SVFType::LowPass,
                                sampleRate, f, cutoffHz, kCutQ, 0.0);
        }

        // The whole free-band cascade at its current settings.
        [[nodiscard]] double responseDb (const double f) const noexcept
        {
            double sum = 0.0;
            for (const auto& b : bands_)
                if (b.on)
                    sum += bandMagnitudeDb (b.type, sampleRate_, f, b.freqHz, b.gainDb, b.q);
            return sum;
        }

    private:
        double sampleRate_ { 48000.0 };
        int numChannels_ { 2 };
        std::array<Band, kNumBands> bands_ {};
        std::array<SimdSVF, kNumBands> svf_ {};
    };
}
#endif

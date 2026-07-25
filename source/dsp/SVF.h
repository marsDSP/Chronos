#pragma once

#ifndef CHRONOS_SVF_H
#define CHRONOS_SVF_H

#include <cmath>
#include <numbers>
#include <algorithm>
#include "simd/Config.h"
#include "math/Trigonometry.h"

// Scalar bridge to the FMA-accurate mmTan(M128) kernel for bilinear pre-warping.
// Precondition |π·f/fs| < 1.55 is enforced by the 0.49·fs clamp in setParams;
// do not loosen to Cytomic's 0.499 (π·0.499 ≈ 1.568 would exceed the range).
namespace {
    inline double mmTanScalar(const double x) noexcept {
        const M128 v = MM(set1_ps)(static_cast<float>(x));
        return MM(cvtss_f32)(mmTan(v));
    }
}

namespace MarsDSP::Filters {
    class OnePoleTPT {
    public:
        enum class Type
        {
            LowPass, HighPass
        };

        void reset() noexcept
        {
            z = 0.0;
        }

        void setParams(const Type t, const double sampleRate, double freqHz) noexcept
        {
            constexpr double pi = std::numbers::pi_v<double>;
            const double fs = (sampleRate > 0.0) ? sampleRate : 48000.0;
            const double nyq = 0.49 * fs;
            freqHz = std::clamp(freqHz, 10.0, nyq);
            type = t;
            gNorm = mmTanScalar(pi * freqHz / fs);
            G = gNorm / (1.0 + gNorm);
        }

        template <std::floating_point SampleType>
        SampleType processSample(const SampleType in) noexcept
        {
            const double x = in;
            const double v = (x - z) * G;
            const double lp = v + z;
            z = lp + v;
            return static_cast<SampleType>(type == Type::LowPass ? lp : x - lp);
        }

        [[nodiscard("result is the only effect; discarding wastes the computation")]]
        double magnitude(double freqHz, const double sampleRate) const noexcept
        {
            constexpr double pi = std::numbers::pi_v<double>;
            if (gNorm <= 0.0) return 1.0;
            const double fs = (sampleRate > 0.0) ? sampleRate : 48000.0;
            freqHz = std::clamp(freqHz, 10.0, 0.49 * fs);
            const double omega = mmTanScalar(pi * (freqHz / fs)) / gNorm;
            const double denom = std::sqrt(1.0 + (omega * omega));
            return (type == Type::LowPass) ? (1.0 / denom) : (omega / denom);
        }

    private:
        Type type = Type::LowPass;
        double gNorm = 0.0;
        double G = 0.0;
        double z = 0.0;
    };

    class TwoPoleSVF {
    public:
        enum class SVFType
        {
            LowPass, HighPass, BandPass, Notch, Bell, LowShelf, HighShelf, AllPass, TiltShelf
        };

        void reset() noexcept
        {
            ic1eq = 0.0;
            ic2eq = 0.0;
        }

        void setParams(const SVFType type, const double sampleRate, double freqHz, double Q, const double gainDB) noexcept
        {
            constexpr double pi = std::numbers::pi_v<double>;
            const double fs = (sampleRate > 0.0) ? sampleRate : 48000.0;
            const double nyq = 0.49 * fs;
            freqHz = std::clamp(freqHz, 10.0, nyq);
            setParamsFromG(type, Q, gainDB, mmTanScalar(pi * freqHz / fs));
        }

        void setParamsFromG(const SVFType type, double Q, const double gainDB, const double gt) noexcept
        {
            Q = std::max(Q, 0.025);

            constexpr double ln10 = std::numbers::ln10_v<double>;
            const double A = std::exp(gainDB * (ln10 / 40.0));
            const double sqrtA = std::sqrt(A);
            const double kk = 1.0 / Q;

            switch (type)
            {
                case SVFType::LowPass:
                    g = gt;
                    k = kk;
                    m0 = 0.0;
                    m1 = 0.0;
                    m2 = 1.0;
                    break;
                case SVFType::HighPass:
                    g = gt;
                    k = kk;
                    m0 = 1.0;
                    m1 = -kk;
                    m2 = -1.0;
                    break;
                case SVFType::BandPass:
                    g = gt;
                    k = kk;
                    m0 = 0.0;
                    m1 = kk;
                    m2 = 0.0;
                    break;
                case SVFType::Notch:
                    g = gt;
                    k = kk;
                    m0 = 1.0;
                    m1 = -kk;
                    m2 = 0.0;
                    break;
                case SVFType::AllPass:
                    g = gt;
                    k = kk;
                    m0 = 1.0;
                    m1 = -2.0 * kk;
                    m2 = 0.0;
                    break;
                case SVFType::Bell:
                    g = gt;
                    k = 1.0 / (Q * A);
                    m0 = 1.0;
                    m1 = k * (A*A - 1.0);
                    m2 = 0.0;
                    break;
                case SVFType::LowShelf:
                    g = gt / sqrtA;
                    k = kk;
                    m0 = 1.0;
                    m1 = k * (A - 1.0);
                    m2 = A*A - 1.0;
                    break;
                case SVFType::HighShelf:
                    g = gt * sqrtA;
                    k = kk;
                    m0 = A*A;
                    m1 = k * (1.0 - A)*A;
                    m2 = 1.0 - A*A;
                    break;
                case SVFType::TiltShelf:
                    g = gt * sqrtA;
                    k = kk;
                    m0 = A;
                    m1 = kk * (1.0 - A);
                    m2 = 1.0/A - A;
                    break;
            }
            a1 = 1.0 / (1.0 + g * (g + k));
            a2 = g * a1;
            a3 = g * a2;
        }

        template <std::floating_point SampleType>
        SampleType processSample(const SampleType in) noexcept
        {
            const double v0 = in;
            const double v3 = v0 - ic2eq;
            const double v1 = a1 * ic1eq + a2 * v3;
            const double v2 = ic2eq + a2 * ic1eq + a3 * v3;
            ic1eq = 2.0 * v1 - ic1eq;
            ic2eq = 2.0 * v2 - ic2eq;
            return static_cast<SampleType>(m0 * v0 + m1 * v1 + m2 * v2);
        }

        [[nodiscard("result is the only effect; discarding wastes the computation")]]
        double magnitude(double freqHz, const double sampleRate) const noexcept
        {
            constexpr double pi = std::numbers::pi_v<double>;
            if (g <= 0.0) return 1.0;
            const double fs = (sampleRate > 0.0) ? sampleRate : 48000.0;
            freqHz = std::clamp(freqHz, 10.0, 0.49 * fs);
            const double omega = mmTanScalar(pi * freqHz / fs) / g;
            const double w2 = omega * omega;
            const double numRe = -m0 * w2 + (m0 + m2);
            const double numIm = (m0 * k + m1) * omega;
            const double denRe = 1.0 - w2;
            const double denIm = k * omega;
            const double denMag = std::sqrt(denRe * denRe + denIm * denIm);
            if (denMag <= 0.0) return 1.0;
            return std::sqrt (numRe * numRe + numIm * numIm) / denMag;
        }

    private:
        double g = 0.0;
        double k = 1.0;
        double m0 = 1.0;
        double m1 = 0.0;
        double m2 = 0.0;
        double a1 = 1.0;
        double a2 = 0.0;
        double a3 = 0.0;
        double ic1eq = 0.0;
        double ic2eq = 0.0;
    };
}
#endif
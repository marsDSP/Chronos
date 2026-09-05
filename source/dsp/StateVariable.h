#pragma once

#ifndef CHRONOS_SVF_H
#define CHRONOS_SVF_H

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <algorithm>
#include "simd/Config.h"
#include "math/Trigonometry.h"

namespace MarsDSP::Filters::detail {
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
            gNorm = detail::mmTanScalar(pi * freqHz / fs);
            G = gNorm / (1.0 + gNorm);
        }

        template<std::floating_point SampleType>
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
            const double omega = detail::mmTanScalar(pi * (freqHz / fs)) / gNorm;
            const double denom = std::sqrt(1.0 + (omega * omega));
            return (type == Type::LowPass) ? (1.0 / denom) : (omega / denom);
        }

    private:
        Type type = Type::LowPass;
        double gNorm = 0.0;
        double G = 0.0;
        double z = 0.0;
    };

    class SimdSVF {
    public:
        enum class SVFType
        {
            LowPass, HighPass, BandPass, Notch, Bell, LowShelf, HighShelf, AllPass, TiltShelf
        };

        void reset() noexcept
        {
            ic1eq = MM(setzero_ps)();
            ic2eq = MM(setzero_ps)();
            firstBlock = true;
            da1 = MM(setzero_ps)();
            da2 = MM(setzero_ps)();
            da3 = MM(setzero_ps)();
            dm0 = MM(setzero_ps)();
            dm1 = MM(setzero_ps)();
            dm2 = MM(setzero_ps)();
        }

        void setCoeff(const SVFType type, const double sampleRate, double freqHz, double Q, const double gainDB) noexcept
        {
            constexpr double pi = std::numbers::pi_v<double>;
            const double fs = (sampleRate > 0.0) ? sampleRate : 48000.0;
            const double nyq = 0.49 * fs;
            freqHz = std::clamp(freqHz, 10.0, nyq);
            const auto gt = static_cast<float>(detail::mmTanScalar(pi * freqHz / fs));
            setCoeffPostGK(type, MM(set1_ps)(gt), Q, gainDB);
        }

        void setCoeff(const SVFType type, const M128 angles, double Q, const double gainDB) noexcept
        {
            setCoeffPostGK(type, mmTan(angles), Q, gainDB);
        }

        static M128 step(SimdSVF &svf, M128 input) noexcept
        {
            const M128 v3 = MM(sub_ps)(input, svf.ic2eq);
            const M128 v1 = MM(add_ps)(MM(mul_ps)(svf.a1, svf.ic1eq), MM(mul_ps)(svf.a2, v3));
            const M128 v2 = MM(add_ps)(svf.ic2eq, MM(add_ps)(MM(mul_ps)(svf.a2, svf.ic1eq), MM(mul_ps)(svf.a3, v3)));
            svf.ic1eq = MM(sub_ps)(MM(mul_ps)(svf.two, v1), svf.ic1eq);
            svf.ic2eq = MM(sub_ps)(MM(mul_ps)(svf.two, v2), svf.ic2eq);
            return MM(add_ps)(MM(mul_ps)(svf.m0, input), MM(add_ps)(MM(mul_ps)(svf.m1, v1), MM(mul_ps)(svf.m2, v2)));
        }

        static void step(SimdSVF &svf, float &L, float &R) noexcept
        {
            const M128 input = MM(set_ps)(0, 0, R, L);
            const M128 out = step(svf, input);
            alignas(16) std::array<float, 4> lanes {};
            MM(store_ps)(lanes.data(), out);
            L = lanes[0];
            R = lanes[1];
        }

        void setCoeffForBlock(const SVFType type, const double sampleRate, double freqHz,
                              double Q, const double gainDB, const int numSamples) noexcept
        {
            const M128 a1_prior = a1;
            const M128 a2_prior = a2;
            const M128 a3_prior = a3;
            const M128 m0_prior = m0;
            const M128 m1_prior = m1;
            const M128 m2_prior = m2;

            setCoeff(type, sampleRate, freqHz, Q, gainDB);

            if (firstBlock)
            {
                da1 = MM(setzero_ps)();
                da2 = MM(setzero_ps)();
                da3 = MM(setzero_ps)();
                dm0 = MM(setzero_ps)();
                dm1 = MM(setzero_ps)();
                dm2 = MM(setzero_ps)();
                firstBlock = false;
                return;
            }

            const M128 obs = MM(set1_ps)(1.0f / static_cast<float>(numSamples));
            da1 = MM(mul_ps)(MM(sub_ps)(a1, a1_prior), obs);
            da2 = MM(mul_ps)(MM(sub_ps)(a2, a2_prior), obs);
            da3 = MM(mul_ps)(MM(sub_ps)(a3, a3_prior), obs);
            dm0 = MM(mul_ps)(MM(sub_ps)(m0, m0_prior), obs);
            dm1 = MM(mul_ps)(MM(sub_ps)(m1, m1_prior), obs);
            dm2 = MM(mul_ps)(MM(sub_ps)(m2, m2_prior), obs);

            a1 = a1_prior;
            a2 = a2_prior;
            a3 = a3_prior;
            m0 = m0_prior;
            m1 = m1_prior;
            m2 = m2_prior;
        }

        M128 processBlockStep(const M128 input) noexcept
        {
            const M128 badIn = nonFiniteMask(input);
            const M128 badSt = MM(or_ps)(nonFiniteMask(ic1eq), nonFiniteMask(ic2eq));
            M128 in = input;
            if (MM(movemask_ps)(MM(or_ps)(badIn, badSt)) != 0)
            {
                in = MM(andnot_ps)(badIn, input); // clear bad lanes to +0.0f
                ic1eq = MM(setzero_ps)();
                ic2eq = MM(setzero_ps)();
            }
            const M128 out = step(*this, in);
            a1 = MM(add_ps)(a1, da1);
            a2 = MM(add_ps)(a2, da2);
            a3 = MM(add_ps)(a3, da3);
            m0 = MM(add_ps)(m0, dm0);
            m1 = MM(add_ps)(m1, dm1);
            m2 = MM(add_ps)(m2, dm2);
            return out;
        }

        void processBlockStep(float &L, float &R) noexcept
        {
            const M128 input = MM(set_ps)(0, 0, R, L);
            const M128 out = processBlockStep(input);
            alignas(16) std::array<float, 4> lanes {};
            MM(store_ps)(lanes.data(), out);
            L = lanes[0];
            R = lanes[1];
        }

    private:
        // Lanes holding NaN (unordered) or ±inf, as an all-ones mask.
        static M128 nonFiniteMask(const M128 x) noexcept
        {
            const M128 nanMask = MM(cmpunord_ps)(x, x);
            const M128 infMask = MM(cmpeq_ps)(MM(andnot_ps)(MM(set1_ps)(-0.0f), x),
                                              MM(set1_ps)(std::numeric_limits<float>::infinity()));
            return MM(or_ps)(nanMask, infMask);
        }

        void setCoeffPostGK(const SVFType type, const M128 gt, double Q, const double gainDB) noexcept
        {
            Q = std::max(Q, 0.025);
            constexpr double ln10 = std::numbers::ln10_v<double>;
            const auto A = static_cast<float>(std::exp(gainDB * (ln10 / 40.0)));
            const float sqrtA = std::sqrt(A);
            const auto kk = static_cast<float>(1.0 / Q);

            const M128 vA = MM(set1_ps)(A);
            const M128 vSqrtA = MM(set1_ps)(sqrtA);

            g = gt;
            k = MM(set1_ps)(kk);

            switch (type)
            {
                case SVFType::LowPass:
                    m0 = MM(setzero_ps)();
                    m1 = MM(setzero_ps)();
                    m2 = one;
                    break;
                case SVFType::HighPass:
                    m0 = one;
                    m1 = MM(sub_ps)(MM(setzero_ps)(), k);
                    m2 = negOne;
                    break;
                case SVFType::BandPass:
                    m0 = MM(setzero_ps)();
                    m1 = k;
                    m2 = MM(setzero_ps)();
                    break;
                case SVFType::Notch:
                    m0 = one;
                    m1 = MM(sub_ps)(MM(setzero_ps)(), k);
                    m2 = MM(setzero_ps)();
                    break;
                case SVFType::AllPass:
                    m0 = one;
                    m1 = MM(mul_ps)(negTwo, k);
                    m2 = MM(setzero_ps)();
                    break;
                case SVFType::Bell:
                    k = MM(div_ps)(k, vA);
                    m0 = one;
                    m1 = MM(mul_ps)(k, MM(sub_ps)(MM(mul_ps)(vA, vA), one));
                    m2 = MM(setzero_ps)();
                    break;
                case SVFType::LowShelf:
                    g = MM(div_ps)(gt, vSqrtA);
                    m0 = one;
                    m1 = MM(mul_ps)(k, MM(sub_ps)(vA, one));
                    m2 = MM(sub_ps)(MM(mul_ps)(vA, vA), one);
                    break;
                case SVFType::HighShelf:
                    g = MM(mul_ps)(gt, vSqrtA);
                    m0 = MM(mul_ps)(vA, vA);
                    m1 = MM(mul_ps)(MM(mul_ps)(k, MM(sub_ps)(one, vA)), vA);
                    m2 = MM(sub_ps)(one, MM(mul_ps)(vA, vA));
                    break;
                case SVFType::TiltShelf:
                    g = MM(mul_ps)(gt, vSqrtA);
                    m0 = vA;
                    m1 = MM(mul_ps)(k, MM(sub_ps)(one, vA));
                    m2 = MM(sub_ps)(MM(div_ps)(one, vA), vA);
                    break;
            }
            gk = MM(add_ps)(g, k);
            a1 = MM(div_ps)(one, MM(add_ps)(one, MM(mul_ps)(g, gk)));
            a2 = MM(mul_ps)(g, a1);
            a3 = MM(mul_ps)(g, a2);
        }

        M128 ic1eq{MM(setzero_ps)()};
        M128 ic2eq{MM(setzero_ps)()};
        M128 g;
        M128 k;
        M128 gk;
        M128 a1;
        M128 a2;
        M128 a3;
        M128 m0;
        M128 m1;
        M128 m2;
        M128 one{MM(set1_ps)(1.0f)};
        M128 negOne{MM(set1_ps)(-1.0f)};
        M128 two{MM(set1_ps)(2.0f)};
        M128 negTwo{MM(set1_ps)(-2.0f)};

        bool firstBlock{true};
        M128 da1{MM(setzero_ps)()};
        M128 da2{MM(setzero_ps)()};
        M128 da3{MM(setzero_ps)()};
        M128 dm0{MM(setzero_ps)()};
        M128 dm1{MM(setzero_ps)()};
        M128 dm2{MM(setzero_ps)()};
    };
}
#endif

#pragma once

#ifndef CHRONOS_BBD_POLE_BANK_H
#define CHRONOS_BBD_POLE_BANK_H

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>

namespace MarsDSP::BBD
{
    /**
     *  Fourth-order Butterworth low-pass banks for BBD antialiasing and reconstruction.
     *  Poles and residues come in closed form from two cascaded Sallen-Key
     *  stages with Q1 and Q2, discretized by the matched-Z transform.
     */
    constexpr float kInputCutoffHz = 9900.0f;
    constexpr float kOutputCutoffHz = 9500.0f;

    constexpr double kButterworthQ1 = 0.5411961001;
    constexpr double kButterworthQ2 = 1.3065629649;

    struct PoleResidueCoeffs
    {
        std::array<std::complex<double>, 4> poles;
        std::array<std::complex<double>, 4> residues;
    };

    inline PoleResidueCoeffs computeButterworthPolesAndResidues (double fc) noexcept
    {
        const double w0 = 2.0 * std::numbers::pi * fc;
        const double q1 = kButterworthQ1;
        const double q2 = kButterworthQ2;

        const double alpha1 = -1.0 / (2.0 * q1);
        const double beta1  = std::sqrt (std::max (0.0, 1.0 - 1.0 / (4.0 * q1 * q1)));
        const double alpha2 = -1.0 / (2.0 * q2);
        const double beta2  = std::sqrt (std::max (0.0, 1.0 - 1.0 / (4.0 * q2 * q2)));

        PoleResidueCoeffs res {};
        res.poles[0] = w0 * std::complex<double> (alpha1, -beta1);
        res.poles[1] = w0 * std::complex<double> (alpha1,  beta1);
        res.poles[2] = w0 * std::complex<double> (alpha2, -beta2);
        res.poles[3] = w0 * std::complex<double> (alpha2,  beta2);

        const double num = w0 * w0 * w0 * w0;
        for (int m = 0; m < 4; ++m)
        {
            std::complex<double> denom (1.0, 0.0);
            for (int n = 0; n < 4; ++n)
            {
                if (n != m)
                    denom *= (res.poles[m] - res.poles[n]);
            }
            res.residues[m] = num / denom;
        }

        return res;
    }

    class InputPoleBank
    {
    public:
        explicit InputPoleBank (float sampleTime = 1.0f / 48000.0f) : Ts_ (sampleTime)
        {
            set_freq (kInputCutoffHz);
            reset();
        }

        void reset() noexcept
        {
            for (int i = 0; i < 4; ++i)
            {
                x[i] = std::complex<float> (0.0f, 0.0f);
                Gcalc[i] = gCoef[i];
            }
        }

        void set_freq (float freqHz) noexcept
        {
            const auto pr = computeButterworthPolesAndResidues (static_cast<double> (freqHz));
            for (int m = 0; m < 4; ++m)
            {
                const std::complex<double> pCorr = std::exp (pr.poles[m] * static_cast<double> (Ts_));
                pole_corr[m] = static_cast<std::complex<float>> (pCorr);
                pole_corr_angle[m] = static_cast<float> (std::arg (pCorr));
                pole[m] = static_cast<std::complex<float>> (pr.poles[m]);
                gCoef[m] = static_cast<std::complex<float>> (pr.residues[m]);
                bCoef[m] = static_cast<std::complex<float>> ((pCorr - 1.0) / pr.poles[m]);
            }
        }

        void set_time (float tn) noexcept
        {
            for (int m = 0; m < 4; ++m)
                Gcalc[m] = gCoef[m] * std::pow (pole_corr[m], tn);
        }

        void set_delta (float delta) noexcept
        {
            for (int m = 0; m < 4; ++m)
            {
                Aplus[m] = std::pow (pole_corr[m], delta);
            }
        }

        void calcG (float tn) noexcept
        {
            for (int m = 0; m < 4; ++m)
                Gcalc[m] = gCoef[m] * std::pow (pole_corr[m], tn);
        }

        void process (float u) noexcept
        {
            for (int m = 0; m < 4; ++m)
                x[m] = pole_corr[m] * x[m] + bCoef[m] * u;
        }

        [[nodiscard]] std::complex<float> getG0 (int m) const noexcept { return gCoef[static_cast<std::size_t>(m)]; }
        [[nodiscard]] std::complex<float> getPole (int m) const noexcept { return pole[static_cast<std::size_t>(m)]; }

        std::array<std::complex<float>, 4> x {};
        std::array<std::complex<float>, 4> Gcalc {};

    private:
        float Ts_ { 1.0f / 48000.0f };
        std::array<std::complex<float>, 4> pole_corr {};
        std::array<float, 4> pole_corr_angle {};
        std::array<std::complex<float>, 4> pole {};
        std::array<std::complex<float>, 4> gCoef {};
        std::array<std::complex<float>, 4> bCoef {};
        std::array<std::complex<float>, 4> Aplus {};
    };

    class OutputPoleBank
    {
    public:
        explicit OutputPoleBank (float sampleTime = 1.0f / 48000.0f) : Ts_ (sampleTime)
        {
            set_freq (kOutputCutoffHz);
            reset();
        }

        void reset() noexcept
        {
            for (int i = 0; i < 4; ++i)
            {
                x[i] = std::complex<float> (0.0f, 0.0f);
                Gcalc[i] = Amult[i];
            }
        }

        [[nodiscard]] float calcH0() const noexcept
        {
            float sum = 0.0f;
            for (int m = 0; m < 4; ++m)
                sum += gCoef[m].real();
            return -sum;
        }

        void set_freq (float freqHz) noexcept
        {
            const auto pr = computeButterworthPolesAndResidues (static_cast<double> (freqHz));
            for (int m = 0; m < 4; ++m)
            {
                const std::complex<double> pCorr = std::exp (pr.poles[m] * static_cast<double> (Ts_));
                pole_corr[m] = static_cast<std::complex<float>> (pCorr);
                pole_corr_angle[m] = static_cast<float> (std::arg (pCorr));
                gCoef[m] = static_cast<std::complex<float>> (pr.residues[m] / pr.poles[m]);
                Amult[m] = gCoef[m] * pole_corr[m];
            }
        }

        void set_time (float tn) noexcept
        {
            for (int m = 0; m < 4; ++m)
                Gcalc[m] = Amult[m] * std::pow (pole_corr[m], 1.0f - tn);
        }

        void set_delta (float delta) noexcept
        {
            for (int m = 0; m < 4; ++m)
            {
                Aplus[m] = std::pow (pole_corr[m], -delta);
            }
        }

        void calcG (float tn) noexcept
        {
            for (int m = 0; m < 4; ++m)
                Gcalc[m] = Amult[m] * std::pow (pole_corr[m], 1.0f - tn);
        }

        void process (const std::array<std::complex<float>, 4>& u) noexcept
        {
            for (int m = 0; m < 4; ++m)
                x[m] = pole_corr[m] * x[m] + u[m];
        }

        std::array<std::complex<float>, 4> x {};
        std::array<std::complex<float>, 4> Gcalc {};

    private:
        float Ts_ { 1.0f / 48000.0f };
        std::array<std::complex<float>, 4> pole_corr {};
        std::array<float, 4> pole_corr_angle {};
        std::array<std::complex<float>, 4> gCoef {};
        std::array<std::complex<float>, 4> Amult {};
        std::array<std::complex<float>, 4> Aplus {};
    };
}
#endif

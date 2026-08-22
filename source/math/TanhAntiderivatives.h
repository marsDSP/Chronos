#pragma once

#ifndef CHRONOS_TANH_ANTIDERIVATIVES_H
#define CHRONOS_TANH_ANTIDERIVATIVES_H

#include <array>
#include <cmath>

#include "math/Dilogarithm.h"

/**
 * Regional minimax kernels for the tanh antiderivatives F1 and F2.
 * F1(x) = ln cosh(x) is even; F2(x) = integral of ln cosh is odd.
 * Three regions with crossovers a0 = 1 and a1 = 19. See dsp-notes.md
 * for the derivation and the seam and evaluation reasoning.
 * Coefficients are derived and checked by scripts/python/remez_*.py.
 */

namespace MarsDSP::Math
{
    inline constexpr double kTanA0 = 1.0; // region I/II crossover
    inline constexpr double kTanA1 = 19.0; // region II/III crossover

    // C2 = pi^2/24 - ln^2(2)/2. The nearest double suffices.
    inline constexpr double kTanC2 = 0.1710070097529559;

    // ln(2) as a hi/lo pair: |ln2 - (hi + lo)| = 5.7e-34.
    inline constexpr double kTanLn2Hi = 0.6931471805599453;
    inline constexpr double kTanLn2Lo = 2.3190468138462996e-17;

    // P(u), degree 14. F2(x) = x*u*P(u), u = x^2, |x| <= 1. P(0) = 1/6.
    inline constexpr std::array<double, 15> kF2RegionI{
        {
            0.16666666666666666, // 0x1.5555555555555p-3
            -0.016666666666666594, // -0x1.11111111110fcp-6
            0.003174603174597562, // 0x1.a01a01a016d8cp-9
            -0.0007495590827244051, // -0x1.88fc1df948e6ap-11
            0.00019881352950739333, // 0x1.a0f1336e7b6b6p-13
            -5.6815587325387805e-05, // -0x1.dc9a8b739b9e8p-15
            1.710521585280172e-05, // 0x1.1efa57b9cc50dp-16
            -5.351664370785777e-06, // -0x1.6724e4d0854c6p-18
            1.723185777048654e-06, // 0x1.ce906cc7680d9p-20
            -5.648235330986546e-07, // -0x1.2f3cc15a62e98p-21
            1.8413852923818012e-07, // 0x1.8b6f3a1d4aff0p-23
            -5.656464155399535e-08, // -0x1.e5e2f653ff567p-25
            1.4778914550004e-08, // 0x1.fbccb507b92b5p-27
            -2.772833176805685e-09, // -0x1.7d1864ec74041p-29
            2.671491545614661e-10, // 0x1.25bbcd53b407ap-32
        }
    };

    // S(u), degree 14. F1(x) = u*S(u), u = x^2, |x| <= 1. S(0) = 1/2.
    inline constexpr std::array<double, 15> kF1RegionI{
        {
            0.5, // 0x1.0000000000000p-1
            -0.08333333333333116, // -0x1.55555555554b9p-4
            0.022222222222058514, // 0x1.6c16c16c0b3c5p-6
            -0.0067460317411381305, // -0x1.ba1ba1b4ba339p-8
            0.0021869487768058212, // 0x1.1ea5d2f2e5a1ep-9
            -0.0007386022322834027, // -0x1.833d837593464p-11
            0.00025657604293932094, // 0x1.0d0a1b69edb8ep-12
            -9.097020257139513e-05, // -0x1.7d8e82e0896a2p-14
            3.271979510502662e-05, // 0x1.1279399695185p-15
            -1.1823955824000437e-05, // -0x1.8cbf01d4606f3p-17
            4.188108537857103e-06, // 0x1.190f28249ebf9p-18
            -1.3734342529622273e-06, // -0x1.70adaee5dbcacp-20
            3.760822745593593e-07, // 0x1.93d0b55e43c47p-22
            -7.280182752600742e-08, // -0x1.38ae74b5169a3p-24
            7.156534675180096e-09, // 0x1.ebcb16e57ace4p-28
        }
    };

    // psi(t) = -Li2(-t)/t, degree 10 on [0, e^-2]. psi(0) = 1.
    inline constexpr std::array<double, 11> kF2RegionIIPsi{
        {
            1.0, // 0x1.0000000000000p+0
            -0.24999999999999908, // -0x1.fffffffffffdfp-3
            0.11111111111083918, // 0x1.c71c71c717a92p-4
            -0.062499999968526385, // -0x1.fffffffbac9eep-5
            0.039999998128492524, // 0x1.47ae1379a976fp-5
            -0.027777712682628235, // -0x1.c71c2be1e484ep-6
            0.02040674580837912, // 0x1.4e58187758729p-6
            -0.015605040437770511, // -0x1.ff58912a9f266p-7
            0.012162896722470562, // 0x1.8e8dc5d2e5c20p-7
            -0.008934639114660923, // -0x1.24c52f6646a65p-7
            0.0045376647188292134, // 0x1.296161914d37ep-8
        }
    };

    // L(t) = log1p(t)/t, degree 10 on [0, e^-2]. L(0) = 1.
    inline constexpr std::array<double, 11> kF1RegionIIL{
        {
            1.0, // 0x1.0000000000000p+0
            -0.49999999999998945, // -0x1.fffffffffff42p-2
            0.3333333333302044, // 0x1.5555555547927p-2
            -0.24999999963821531, // -0x1.fffffff391b55p-3
            0.19999997849986126, // 0x1.999996b6dc613p-3
            -0.16666591894168303, // -0x1.5554f0f9b2d28p-3
            0.14284085417648296, // 0x1.2489bee9c3fe0p-3
            -0.1247703631890595, // -0x1.ff0f35568349dp-4
            0.10900311629907711, // 0x1.be7a0d3aaa7a8p-4
            -0.08765558117293032, // -0x1.670989e731994p-4
            0.04729716544174639, // 0x1.8375585881738p-5
        }
    };

    namespace detail
    {
        // Estrin evaluation, degree 14, depth 4.
        inline double estrin15(const std::array<double, 15> &c, double u) noexcept
        {
            const double u2 = u * u;
            const double u4 = u2 * u2;
            const double u8 = u4 * u4;
            const double p0 = std::fma(c[1], u, c[0]);
            const double p1 = std::fma(c[3], u, c[2]);
            const double p2 = std::fma(c[5], u, c[4]);
            const double p3 = std::fma(c[7], u, c[6]);
            const double p4 = std::fma(c[9], u, c[8]);
            const double p5 = std::fma(c[11], u, c[10]);
            const double p6 = std::fma(c[13], u, c[12]);
            const double p7 = c[14];
            const double q0 = std::fma(p1, u2, p0);
            const double q1 = std::fma(p3, u2, p2);
            const double q2 = std::fma(p5, u2, p4);
            const double q3 = std::fma(p7, u2, p6);
            const double r0 = std::fma(q1, u4, q0);
            const double r1 = std::fma(q3, u4, q2);
            return std::fma(r1, u8, r0);
        }

        // Estrin evaluation, degree 10, depth 4.
        inline double estrin11(const std::array<double, 11> &c, double u) noexcept
        {
            const double u2 = u * u;
            const double u4 = u2 * u2;
            const double u8 = u4 * u4;
            const double p0 = std::fma(c[1], u, c[0]);
            const double p1 = std::fma(c[3], u, c[2]);
            const double p2 = std::fma(c[5], u, c[4]);
            const double p3 = std::fma(c[7], u, c[6]);
            const double p4 = std::fma(c[9], u, c[8]);
            const double p5 = c[10];
            const double q0 = std::fma(p1, u2, p0);
            const double q1 = std::fma(p3, u2, p2);
            const double q2 = std::fma(p5, u2, p4);
            const double r0 = std::fma(q1, u4, q0);
            return std::fma(q2, u8, r0);
        }
    } // namespace detail

    /// F2(x) = integral of ln cosh. Odd. F2(0) is exactly zero.
    inline double f2Tanh(const double x) noexcept
    {
        const double a = std::fabs(x);
        if (a <= kTanA0)
        {
            // Region I: no transcendental. The sign rides the leading x.
            const double u = x * x;
            return x * u * detail::estrin15(kF2RegionI, u);
        }
        const double h = (a - kTanLn2Hi) - kTanLn2Lo;
        if (a < kTanA1)
        {
            // Region II: completed square minus the dilogarithm remainder.
            const double t = std::exp(-2.0 * a);
            const double e = 0.5 * t * detail::estrin11(kF2RegionIIPsi, t);
            const double m = std::fma(0.5 * h, h, kTanC2) - e;
            return x < 0.0 ? -m : m;
        }
        // Region III: the remainder is below 1e-17 relative, so it is dropped.
        const double m = std::fma(0.5 * h, h, kTanC2);
        return x < 0.0 ? -m : m;
    }

    /// F1(x) = ln cosh(x). Even. F1(0) is exactly zero. F1 is non-negative.
    inline double f1Tanh(const double x) noexcept
    {
        const double a = std::fabs(x);
        if (a <= kTanA0)
        {
            const double u = x * x;
            return u * detail::estrin15(kF1RegionI, u);
        }
        const double h = (a - kTanLn2Hi) - kTanLn2Lo;
        if (a < kTanA1)
        {
            const double t = std::exp(-2.0 * a);
            return h + t * detail::estrin11(kF1RegionIIL, t);
        }
        return h;
    }

    // Reference twins for the test oracle. Keep them; do not optimize.
    namespace Ref
    {
        inline double signRef(const double x) noexcept
        {
            return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
        }

        inline double f1TanhRef(const double x) noexcept
        {
            const double a = std::fabs(x);
            return a - 0.6931471805599453 + std::log1p(std::exp(-2.0 * a));
        }

        inline double f2TanhRef(const double x) noexcept
        {
            const double a = std::fabs(x);
            const double t = std::exp(-2.0 * a);
            const double g = 0.5 * dilogNeg(t) + 0.4112335167120566;
            return signRef(x) * (0.5 * a * a - a * 0.6931471805599453 + g);
        }
    }
}
#endif

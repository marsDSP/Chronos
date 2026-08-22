// tests/harnesses/cd/rtype_solver_check.cpp
//
// Verification harness for WDF R-type scattering solvers.
// Compares RJunctionFast against RJunctionMNA and WdfOracle (double-double MNA).

#include "dsp/wdf/wdft/RTypeJunctionFast.h"
#include "dsp/wdf/wdft/RTypeJunctionMNA.h"
#include "dsp/SallenKeyJunction.h"
#include "wdf_dd_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <print>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace
{
    const char* g_section = "(startup)";

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

    using MarsDSP::WDF::RJunctionFast;
    using MarsDSP::WDF::RJunctionMNA;

    template <typename TOut>
    using SMatrix4 = std::array<std::array<TOut, 4>, 4>;

    template <typename Solver, typename TOut>
    double solveJunction (double Rb, double Rc, double Rd, SMatrix4<TOut> &S)
    {
        enum { nA = 0, nB = 1, nO = 2, numNodes = 3, numPorts = 4 };
        Solver m;
        m.stampConductance (nB, nO, 1.0 / MarsDSP::Filters::opAmpInputRes);
        m.stampOpAmp (nB, nO, nO, MarsDSP::Filters::opAmpGain, MarsDSP::Filters::opAmpOutputRes);
        m.setPort (0, nA, Solver::ground);
        m.setPort (1, nO, nA);
        m.setPort (2, nB, nA);
        m.setPort (3, nB, Solver::ground);
        const std::array<double, 4> pr { 0.0, Rb, Rc, Rd };
        return m.solveScattering (pr, 0, S);
    }

    double estimateSpectralRadius (const SMatrix4<double> &S)
    {
        // Estimate the spectral radius via power iteration on the normalized matrix
        // or computing the eigenvalue magnitude.
        std::array<double, 4> v { 0.5, 0.5, 0.5, 0.5 };
        double lambda = 0.0;
        for (int it = 0; it < 200; ++it)
        {
            std::array<double, 4> w { 0.0, 0.0, 0.0, 0.0 };
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    w[i] += S[i][j] * v[j];

            double vDotW = 0.0;
            double vDotV = 0.0;
            for (int i = 0; i < 4; ++i)
            {
                vDotW += v[i] * w[i];
                vDotV += v[i] * v[i];
            }
            if (vDotV > 0.0)
                lambda = std::fabs (vDotW / vDotV);

            double norm = std::sqrt (w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
            if (norm == 0.0)
                return 0.0;
            for (int i = 0; i < 4; ++i)
                v[i] = w[i] / norm;
        }
        return lambda;
    }
} // namespace

int main()
{
    const std::array sampleRates { 44100.0, 48000.0, 88200.0, 96000.0, 176400.0, 192000.0 };
    const std::array qValues { 0.05, 0.2, 0.7071, 2.0, 5.0, 10.9999 };
    constexpr int numCutoffs = 81;

    double maxRelErrFastVsLuS = 0.0;
    double maxRelErrFastVsLuRa = 0.0;
    double maxRelErrFastVsDDS = 0.0;
    double maxRelErrFastVsDDRa = 0.0;
    double maxRelErrFloatRoundTrip = 0.0;
    double maxSpectralRadiusPassive = 0.0;
    double maxSpectralRadius = 0.0;

    double worstPointFs = 0.0;
    double worstPointF = 0.0;
    double worstPointQ = 0.0;
    const char* worstPointType = "";

    double minPortRes = 1.0e30;
    double maxPortRes = 0.0;

    for (double fs : sampleRates)
    {
        for (int fi = 0; fi < numCutoffs; ++fi)
        {
            const double f = 20.0 * std::pow (1000.0, static_cast<double> (fi) / (numCutoffs - 1));
            for (double q : qValues)
            {
                // 1. Low-pass operating point
                {
                    constexpr double capVal = 1.0e-8;
                    constexpr double capRatio = 22.0;
                    constexpr double C1 = capVal * capRatio;
                    constexpr double C2 = capVal / capRatio;

                    const double Rb = 1.0 / (2.0 * fs * C1);
                    const double Rd = 1.0 / (2.0 * fs * C2);

                    const double fClamped = std::clamp (f, 10.0, 0.49 * fs);
                    const double wa = 2.0 * fs * std::tan (std::numbers::pi * fClamped / fs);
                    const double Rv = 1.0 / (wa * capVal);
                    const double sp = 2.0 * q / (capRatio + std::sqrt (capRatio * capRatio - 4.0 * q * q));
                    const double Rc = Rv / sp;

                    minPortRes = std::min ({ minPortRes, Rb, Rc, Rd });
                    maxPortRes = std::max ({ maxPortRes, Rb, Rc, Rd });

                    SMatrix4<double> SFast {};
                    SMatrix4<double> SLu {};
                    SMatrix4<float> SFloat {};
                    const double RaFast = solveJunction<RJunctionFast<3, 4>> (Rb, Rc, Rd, SFast);
                    const double RaLu   = solveJunction<RJunctionMNA<3, 4>>  (Rb, Rc, Rd, SLu);
                    solveJunction<RJunctionFast<3, 4>> (Rb, Rc, Rd, SFloat);

                    const std::array<double, 4> pr { 0.0, Rb, Rc, Rd };
                    const auto oracleRes = WdfOracle::solveScatteringDD (pr, 0);
                    const double RaDD = F2Oracle::toDouble (oracleRes.Ra);

                    // Section 3: Structural identities
                    g_section = "identities_lpf";
                    CHECK (SFast[0][0] == 0.0);
                    CHECK (std::isfinite (RaFast) && RaFast > 0.0);

                    // Section 1: Fast vs LU
                    const double errRaLu = std::fabs (RaFast - RaLu) / RaLu;
                    maxRelErrFastVsLuRa = std::max (maxRelErrFastVsLuRa, errRaLu);

                    for (int i = 0; i < 4; ++i)
                    {
                        for (int j = 0; j < 4; ++j)
                        {
                            CHECK (std::isfinite (SFast[i][j]));
                            const double scale = std::max (std::fabs (SLu[i][j]), 1.0e-3);
                            const double errS = std::fabs (SFast[i][j] - SLu[i][j]) / scale;
                            maxRelErrFastVsLuS = std::max (maxRelErrFastVsLuS, errS);

                            const double sDD = F2Oracle::toDouble (oracleRes.S[i][j]);
                            const double scaleDD = std::max (std::fabs (sDD), 1.0e-3);
                            const double errSDD = std::fabs (SFast[i][j] - sDD) / scaleDD;
                            if (errSDD > maxRelErrFastVsDDS)
                            {
                                maxRelErrFastVsDDS = errSDD;
                                worstPointFs = fs;
                                worstPointF = f;
                                worstPointQ = q;
                                worstPointType = "LPF";
                            }

                            const double errFloat = std::fabs (static_cast<double> (SFloat[i][j]) - SFast[i][j]);
                            const double relFloat = errFloat / std::max (std::fabs (SFast[i][j]), 1.0e-3);
                            maxRelErrFloatRoundTrip = std::max (maxRelErrFloatRoundTrip, relFloat);
                        }
                    }

                    const double errRaDD = std::fabs (RaFast - RaDD) / RaDD;
                    maxRelErrFastVsDDRa = std::max (maxRelErrFastVsDDRa, errRaDD);

                    // Section 4: Passivity proxy
                    const double rho = estimateSpectralRadius (SFast);
                    if (q <= 0.7071)
                        maxSpectralRadiusPassive = std::max (maxSpectralRadiusPassive, rho);
                    maxSpectralRadius = std::max (maxSpectralRadius, rho);
                }

                // 2. High-pass operating point
                {
                    constexpr double capVal = 1.0e-8;
                    const double fClamped = std::clamp (f, 10.0, 0.49 * fs);
                    const double wa = 2.0 * fs * std::tan (std::numbers::pi * fClamped / fs);
                    const double Rv = 1.0 / (wa * capVal);
                    const double sp = 2.0 * q;

                    const double R1 = Rv / sp;
                    const double Rc = 1.0 / (2.0 * fs * capVal);
                    const double Rd = Rv * sp;
                    const double Rb = R1;

                    minPortRes = std::min ({ minPortRes, Rb, Rc, Rd });
                    maxPortRes = std::max ({ maxPortRes, Rb, Rc, Rd });

                    SMatrix4<double> SFast {};
                    SMatrix4<double> SLu {};
                    SMatrix4<float> SFloat {};
                    const double RaFast = solveJunction<RJunctionFast<3, 4>> (Rb, Rc, Rd, SFast);
                    const double RaLu   = solveJunction<RJunctionMNA<3, 4>>  (Rb, Rc, Rd, SLu);
                    solveJunction<RJunctionFast<3, 4>> (Rb, Rc, Rd, SFloat);

                    const std::array<double, 4> pr { 0.0, Rb, Rc, Rd };
                    const auto oracleRes = WdfOracle::solveScatteringDD (pr, 0);
                    const double RaDD = F2Oracle::toDouble (oracleRes.Ra);

                    // Section 3: Structural identities
                    g_section = "identities_hpf";
                    CHECK (SFast[0][0] == 0.0);
                    CHECK (std::isfinite (RaFast) && RaFast > 0.0);

                    // Section 1: Fast vs LU
                    const double errRaLu = std::fabs (RaFast - RaLu) / RaLu;
                    maxRelErrFastVsLuRa = std::max (maxRelErrFastVsLuRa, errRaLu);

                    for (int i = 0; i < 4; ++i)
                    {
                        for (int j = 0; j < 4; ++j)
                        {
                            CHECK (std::isfinite (SFast[i][j]));
                            const double scale = std::max (std::fabs (SLu[i][j]), 1.0e-3);
                            const double errS = std::fabs (SFast[i][j] - SLu[i][j]) / scale;
                            maxRelErrFastVsLuS = std::max (maxRelErrFastVsLuS, errS);

                            const double sDD = F2Oracle::toDouble (oracleRes.S[i][j]);
                            const double scaleDD = std::max (std::fabs (sDD), 1.0e-3);
                            const double errSDD = std::fabs (SFast[i][j] - sDD) / scaleDD;
                            if (errSDD > maxRelErrFastVsDDS)
                            {
                                maxRelErrFastVsDDS = errSDD;
                                worstPointFs = fs;
                                worstPointF = f;
                                worstPointQ = q;
                                worstPointType = "HPF";
                            }

                            const double errFloat = std::fabs (static_cast<double> (SFloat[i][j]) - SFast[i][j]);
                            const double relFloat = errFloat / std::max (std::fabs (SFast[i][j]), 1.0e-3);
                            maxRelErrFloatRoundTrip = std::max (maxRelErrFloatRoundTrip, relFloat);
                        }
                    }

                    const double errRaDD = std::fabs (RaFast - RaDD) / RaDD;
                    maxRelErrFastVsDDRa = std::max (maxRelErrFastVsDDRa, errRaDD);

                    // Section 4: Passivity proxy
                    const double rho = estimateSpectralRadius (SFast);
                    if (q <= 0.7071)
                        maxSpectralRadiusPassive = std::max (maxSpectralRadiusPassive, rho);
                    maxSpectralRadius = std::max (maxSpectralRadius, rho);
                }
            }
        }
    }

    g_section = "gates";
    CHECK (maxRelErrFastVsLuS < 1.0e-6);
    CHECK (maxRelErrFastVsLuRa < 1.0e-6);
    CHECK (maxRelErrFastVsDDS < 1.0e-6);
    CHECK (maxSpectralRadius <= 1.10);
    CHECK (maxRelErrFloatRoundTrip < 6.0e-8);

    std::println("=== rtype_solver_check OK ===");
    std::println("  Fast vs LU (rel): S {:.3}, Ra {:.3}", maxRelErrFastVsLuS, maxRelErrFastVsLuRa);
    std::println("  Fast vs DD (rel): S {:.3}, Ra {:.3} (worst: {} fs={:.0} f={:.1} Q={:.3})",
                 maxRelErrFastVsDDS, maxRelErrFastVsDDRa, worstPointType, worstPointFs, worstPointF, worstPointQ);
    std::println("  Spectral radius max: {:.6}", maxSpectralRadius);
    std::println("  Float round trip max: {:.3}", maxRelErrFloatRoundTrip);
    std::println("  Port resistance span: [{:.2}, {:.2}] Ohms", minPortRes, maxPortRes);

    return 0;
}

/**
 * Correctness harness for HalfSampleFir, the symmetric half-sample-delay
 * FIR that compensates ADAA1's 0.5-sample box-kernel latency on the wet
 * path. Plain main(), exit code, always-live CHECK/FAIL.
 */

#include "dsp/align/HalfSampleFir.h"
#include "dsp/DelayInterpolator.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <print>
#include <vector>

// MSVC's <cmath> does not expose the POSIX M_PI macro unless
// _USE_MATH_DEFINES is defined before include.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr int    kN  = MarsDSP::Align::kHalfSampleTaps;
constexpr double kD  = (kN - 1) / 2.0;
constexpr double kFs = 48000.0;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

/// H(w) = sum_m h[m] e^{-j w m} (double, from the float32 coefficients).
std::complex<double> evaluateH(double w)
{
    std::complex<double> h(0.0, 0.0);
    for (int m = 0; m < kN; ++m)
        h += std::complex<double>(MarsDSP::Align::kHalfSampleCoeffs[m], 0.0)
             * std::polar(1.0, -w * m);
    return h;
}

/// Group delay tau(w) = (A*C' - C*A') / (A^2 + C^2), with H = A - j C.
/// Exact (and equal to D) for symmetric coefficients; computed in double.
double groupDelay(double w)
{
    double A = 0.0;
    double C = 0.0;
    double dA = 0.0;
    double dC = 0.0;
    for (int m = 0; m < kN; ++m)
    {
        const double cs = std::cos(w * m);
        const double sn = std::sin(w * m);
        const double hm = MarsDSP::Align::kHalfSampleCoeffs[m];
        A += hm * cs;
        C += hm * sn;
        dA += -static_cast<double>(m) * hm * sn;
        dC +=  static_cast<double>(m) * hm * cs;
    }
    const double denom = A * A + C * C;
    return (A * dC - C * dA) / denom;
}

double magnitudeDb(double fNorm)
{
    const double w = 2.0 * M_PI * fNorm;
    return 20.0 * std::log10(std::max(std::abs(evaluateH(w)), 1e-300));
}

int runAll()
{
    using MarsDSP::Align::HalfSampleFir;
    using MarsDSP::Align::kHalfSampleCoeffs;

    // 1. Coefficient symmetry (bit-exact).
    g_section = "coefficient symmetry";
    for (int j = 0; j < kN; ++j)
        if (kHalfSampleCoeffs[j] != kHalfSampleCoeffs[kN - 1 - j])
            FAIL("h[{{}}] = {{}} != h[{{}}] = {{}}", j, static_cast<double>(kHalfSampleCoeffs[j]),
                 kN - 1 - j, static_cast<double>(kHalfSampleCoeffs[kN - 1 - j]));
    std::println("coefficient symmetry (bit-exact): PASS");

    // 2. DC gain.
    g_section = "DC gain";
    {
        double s = 0.0;
        for (int j = 0; j < kN; ++j) s += kHalfSampleCoeffs[j];
        CHECK(std::abs(s - 1.0) < 1e-7);
        std::println("DC gain: sum(h) = {:.17g}  (|sum-1| = {:.3e}): PASS", s, std::abs(s - 1.0));
    }

    // 3. Impulse-response centroid.
    g_section = "impulse centroid";
    {
        HalfSampleFir fir;
        fir.reset();
        std::array<float, kN> y{};
        for (int n = 0; n < kN; ++n)
            y[static_cast<std::size_t>(n)] = fir.process(n == 0 ? 1.0f : 0.0f);

        double num = 0.0;
        double den = 0.0;
        for (int n = 0; n < kN; ++n) { num += static_cast<double>(n) * y[static_cast<std::size_t>(n)]; den += y[static_cast<std::size_t>(n)]; }
        const double centroid = num / den;
        if (std::abs(centroid - kD) > 1e-6)
            FAIL("centroid = {{:.9f}} != D = {{:.1f}}", centroid, kD);
        std::println("impulse centroid: {:.9f} (expected {:.1f}): PASS", centroid, kD);
    }

    // 4. Group delay is flat (linear phase).
    g_section = "group delay flat";
    {
        double maxGdErr = 0.0;
        double maxImagRot = 0.0;
        constexpr int kPts = 1000;
        for (int i = 0; i < kPts; ++i)
        {
            const double w = (0.9 * M_PI) * static_cast<double>(i) / static_cast<double>(kPts - 1);
            const double tau = groupDelay(w);
            maxGdErr = std::max(maxGdErr, std::abs(tau - kD));
            const std::complex<double> R = evaluateH(w) * std::polar(1.0, w * kD);
            maxImagRot = std::max(maxImagRot, std::abs(R.imag()));
        }
        CHECK(maxGdErr < 1e-9);
        CHECK(maxImagRot < 1e-9);
        std::println("group delay flat: max|tau-D| = {:.3e}, max|Im(R)| = {:.3e} (< 1e-9): PASS",
                    maxGdErr, maxImagRot);
    }

    // 5. Passband magnitude.
    g_section = "passband magnitude";
    {
        const std::array<double, 5> khz = {1.0, 5.0, 10.0, 15.0, 20.0};
        double mag15 = 0.0;
        for (double kh : khz)
        {
            const double db = magnitudeDb((kh * 1000.0) / kFs);
            std::println("    {:5.1f} kHz : {:+.6f} dB", kh, db);
            if (kh == 15.0) mag15 = db;
        }
        CHECK(mag15 > -0.1);
        std::println("passband magnitude (gate: > -0.1 dB at 15 kHz, got {:+.6f}): PASS", mag15);
    }

    // 6. Nyquist null.
    g_section = "Nyquist null";
    {
        double s = 0.0;
        for (int m = 0; m < kN; ++m) s += kHalfSampleCoeffs[m] * ((m & 1) ? -1.0 : 1.0);
        CHECK(std::abs(s) < 1e-6);
        std::println("Nyquist null: |H(pi)| = {:.3e} (< 1e-6): PASS", std::abs(s));
    }

    // 7. Lagrange cross-check (active only at kHalfSampleTaps == 6).
    g_section = "Lagrange cross-check";
    if constexpr (MarsDSP::Align::kHalfSampleTaps == 6)
    {
        const auto c = MarsDSP::Delays::makeCoeffs(MarsDSP::Delays::Interpolation::Lagrange5th, 0.5f);
        for (int j = 0; j < 6; ++j)
            CHECK(c.c[j] == c.c[5 - j]);
        double num = 0.0;
        double den = 0.0;
        double nyq = 0.0;
        for (int j = 0; j < 6; ++j)
        {
            num += static_cast<double>(j) * c.c[j];
            den += c.c[j];
            nyq += c.c[j] * ((j & 1) ? -1.0 : 1.0);
        }
        CHECK(std::abs(num / den - 2.5) < 1e-6);
        CHECK(std::abs(den - 1.0) < 1e-6);
        CHECK(std::abs(nyq) < 1e-6);
        std::println("Lagrange cross-check (N=6, centroid/DC/Nyquist): PASS");
    }
    else
    {
        std::println("Lagrange cross-check: dormant (kHalfSampleTaps = {}, active at 6)", kN);
    }

    // 8. Reset reproducibility.
    g_section = "reset";
    {
        HalfSampleFir fir;
        std::array<float, 100> in{};
        std::array<float, 100> y1{};
        std::array<float, 100> y2{};
        for (int n = 0; n < 100; ++n)
            in[n] = 0.5f * std::sin(0.3f * static_cast<float>(n)) + 0.3f * std::sin(1.1f * static_cast<float>(n));
        fir.reset();
        for (int n = 0; n < 100; ++n) y1[n] = fir.process(in[n]);
        fir.reset();
        for (int n = 0; n < 100; ++n) y2[n] = fir.process(in[n]);
        for (int n = 0; n < 100; ++n)
            if (y1[n] != y2[n])
                FAIL("reset mismatch at n={{}}: {{}} != {{}}", n, static_cast<double>(y1[n]), static_cast<double>(y2[n]));
        std::println("reset reproducibility (bit-exact): PASS");
    }

    // 9. Impulse-response bit-equality.
    g_section = "impulse-response bit-equality";
    {
        HalfSampleFir fir;
        fir.reset();
        for (int n = 0; n < kN; ++n)
        {
            const float y = fir.process(n == 0 ? 1.0f : 0.0f);
            const float exp = kHalfSampleCoeffs[static_cast<std::size_t>(n)];
            if (y != exp)
                FAIL("impulse response n={{}}: got {{}}, expected coeff {{}}",
                     n, static_cast<double>(y), static_cast<double>(exp));
        }
        std::println("impulse-response bit-equality (coeffs): PASS");
    }

    // 10. Memmove-vs-circular parity.
    g_section = "memmove parity";
    {
        // Local twin that keeps the old memmove-shift logic.
        // It also returns the sum of term magnitudes. Use mag to set the gate.
        struct HalfSampleFirOld {
            std::array<float, kN> z_{};
            void reset() noexcept { z_.fill(0.0f); }
            float process(float x, float& mag) noexcept {
                std::memmove(z_.data() + 1, z_.data(),
                             static_cast<std::size_t>(kN - 1) * sizeof(float));
                z_.front() = x;
                float acc = 0.0f;
                mag = 0.0f;
                for (int j = 0; j < kN / 2; ++j)
                {
                    const float term =
                        kHalfSampleCoeffs[static_cast<std::size_t>(j)]
                        * (z_[static_cast<std::size_t>(j)]
                           + z_[static_cast<std::size_t>(kN - 1 - j)]);
                    acc += term;
                    mag += std::fabs(term);
                }
                return acc;
            }
        };

        HalfSampleFir neu;
        HalfSampleFirOld old;
        neu.reset();
        old.reset();

        constexpr int kM = 2000;
        double maxErr = 0.0;
        int worstN = 0;
        // 2^-24 is one ULP of a float32 value with magnitude 1.
        constexpr float k2neg24 = 5.960464e-8f;
        for (int n = 0; n < kM; ++n)
        {
            const float x = 0.5f * std::sin(0.3f * static_cast<float>(n))
                          + 0.3f * std::sin(1.1f * static_cast<float>(n))
                          + 0.01f * static_cast<float>(n);
            float mag = 0.0f;
            const float yn = neu.process(x);
            const float yo = old.process(x, mag);
            const float e = std::fabs(yn - yo);
            if (e > maxErr) { maxErr = e; worstN = n; }
            // The SIMD kernel uses FMADD, which reassociates the MAC order.
            // Scale the gate by mag, not by the output (the output can be small).
            if (e > 16.0f * k2neg24 * mag)
                FAIL("memmove parity n={{}}: new={{}} old={{}} err={{:.3e}} > 16ULP={{:.3e}}",
                     n, static_cast<double>(yn), static_cast<double>(yo), static_cast<double>(e),
                     static_cast<double>(16.0f * k2neg24 * mag));
        }
        std::println("memmove-vs-circular parity ({} samples, V3 FMADD, max err {:.3e} at n={}): PASS",
                    kM, maxErr, worstN);
    }

    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos HalfSampleFir correctness harness ===");
    std::println("kHalfSampleTaps = {}  D = {:.1f}  fs = {:.0f} Hz", kN, kD, kFs);
    std::println();

    const int r = runAll();

    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

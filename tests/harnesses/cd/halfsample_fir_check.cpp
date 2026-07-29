// tests/harnesses/cd/halfsample_fir_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Correctness harness for MarsDSP::Align::HalfSampleFir, the symmetric
// half-sample-delay FIR that compensates ADAA1's 0.5-sample box-kernel
// latency on the wet path. The coefficients are re-derived by
// scripts/python/halfsample_fir.py (Kaiser-windowed sinc), which doubles
// as a regression check on the constants in source/dsp/align/HalfSampleFir.h.
//
//   1. Coefficient symmetry   – h[j] == h[N-1-j] bit-exact (linear phase
//                               depends on this exactly, not approximately).
//   2. DC gain                 – sum(h) within 1e-7 of 1.0.
//   3. Impulse-response centroid – feed an impulse through process() and
//                               assert sum(m*y[m])/sum(y) == (N-1)/2 to
//                               1e-6. Catches a tap-order / off-by-one in
//                               process(), which is otherwise silent.
//   4. Group delay is flat     – evaluate H(w) on 1000 points over
//                               [0, 0.9*pi]; assert the phase is exactly
//                               linear (the rotated response H*e^{jwD} is
//                               real) and group-delay ripple < 1e-9 samples.
//   5. Passband magnitude      – |H| in dB at 1/5/10/15/20 kHz (48 kHz).
//                               Gate: > -0.1 dB at 15 kHz.
//   6. Nyquist null            – |H(pi)| < 1e-6 (structural for even-length
//                               symmetric FIRs; fails if not symmetric).
//   7. Lagrange cross-check    – at kHalfSampleTaps == 6, verify the repo's
//                               existing makeCoeffs(Lagrange5th, 0.5f) is
//                               itself a valid symmetric half-sample delay,
//                               tying this harness to already-tested code.
//   8. Reset                   – process 100 samples, reset(), reprocess,
//                               assert bit-exact match.
//   9. Impulse-response bit-equality – feed an impulse, assert the
//                               16 output samples are bit-exactly the
//                               coefficient array (the impulse response of
//                               a folded-symmetric FIR is its coefficients,
//                               so this catches a tap-order or indexing
//                               error in the circular buffer).
//  10. Memmove-vs-circular parity – a local twin that keeps the old
//                               memmove-shift logic, compared bit-for-bit
//                               against the new circular-buffer HalfSampleFir
//                               over 2000 samples of a sine+ramp signal.
//
// Conventions (matching ring_buffer_check): plain main(), exit code, printf,
// always-live CHECK/FAIL (NOT assert — NDEBUG in Release would void every
// test). Links SharedCode only; no JUCE. No forced -O2 so any assert
// preconditions in the headers stay armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/align/HalfSampleFir.h"
#include "dsp/DelayInterpolator.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// MSVC's <cmath> does not expose the POSIX M_PI macro unless
// _USE_MATH_DEFINES is defined before include. Same fallback as
// source/math/Trigonometry.h.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr int    kN  = MarsDSP::Align::kHalfSampleTaps;
constexpr double kD  = (kN - 1) / 2.0;
constexpr double kFs = 48000.0;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

// H(w) = sum_m h[m] e^{-j w m}  (double, from the float32 coefficients).
std::complex<double> evaluateH(double w)
{
    std::complex<double> h(0.0, 0.0);
    for (int m = 0; m < kN; ++m)
        h += std::complex<double>(MarsDSP::Align::kHalfSampleCoeffs[m], 0.0)
             * std::polar(1.0, -w * m);
    return h;
}

// Group delay tau(w) = (A*C' - C*A') / (A^2 + C^2), with H = A - j C.
// Exact (and equal to D) for symmetric coefficients; computed in double.
double groupDelay(double w)
{
    double A = 0.0, C = 0.0, dA = 0.0, dC = 0.0;
    for (int m = 0; m < kN; ++m)
    {
        const double cs = std::cos(w * m);
        const double sn = std::sin(w * m);
        const double hm = MarsDSP::Align::kHalfSampleCoeffs[m];
        A  += hm * cs;
        C  += hm * sn;
        dA += -double(m) * hm * sn;   // d/dw A = -sum m h sin
        dC +=  double(m) * hm * cs;   // d/dw C =  sum m h cos
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

    // ── 1. Coefficient symmetry (bit-exact) ───────────────────────────────
    g_section = "coefficient symmetry";
    for (int j = 0; j < kN; ++j)
        if (kHalfSampleCoeffs[j] != kHalfSampleCoeffs[kN - 1 - j])
            FAIL("h[%d] = %g != h[%d] = %g", j, (double)kHalfSampleCoeffs[j],
                 kN - 1 - j, (double)kHalfSampleCoeffs[kN - 1 - j]);
    std::printf("coefficient symmetry (bit-exact): PASS\n");

    // ── 2. DC gain ────────────────────────────────────────────────────────
    g_section = "DC gain";
    {
        double s = 0.0;
        for (int j = 0; j < kN; ++j) s += kHalfSampleCoeffs[j];
        CHECK(std::abs(s - 1.0) < 1e-7);
        std::printf("DC gain: sum(h) = %.17g  (|sum-1| = %.3e): PASS\n", s, std::abs(s - 1.0));
    }

    // ── 3. Impulse-response centroid ──────────────────────────────────────
    g_section = "impulse centroid";
    {
        HalfSampleFir fir;
        fir.reset();
        // Impulse at sample 0; collect kN output samples (the FIR rings out
        // completely within N samples for an impulse at sample 0).
        std::array<float, kN> y{};
        for (int n = 0; n < kN; ++n)
            y[n] = fir.process(n == 0 ? 1.0f : 0.0f);

        double num = 0.0, den = 0.0;
        for (int n = 0; n < kN; ++n) { num += double(n) * y[n]; den += y[n]; }
        const double centroid = num / den;
        if (std::abs(centroid - kD) > 1e-6)
            FAIL("centroid = %.9f != D = %.1f", centroid, kD);
        std::printf("impulse centroid: %.9f (expected %.1f): PASS\n", centroid, kD);
    }

    // ── 4. Group delay is flat (linear phase) ─────────────────────────────
    g_section = "group delay flat";
    {
        double maxGdErr = 0.0;
        double maxImagRot = 0.0; // |Im(H e^{j w D})| — must be ~0 (R real)
        constexpr int kPts = 1000;
        for (int i = 0; i < kPts; ++i)
        {
            const double w = (0.9 * M_PI) * double(i) / double(kPts - 1);
            const double tau = groupDelay(w);
            maxGdErr = std::max(maxGdErr, std::abs(tau - kD));
            // Rotated response R(w) = H(w) e^{j w D} must be real.
            const std::complex<double> R = evaluateH(w) * std::polar(1.0, w * kD);
            maxImagRot = std::max(maxImagRot, std::abs(R.imag()));
        }
        CHECK(maxGdErr < 1e-9);
        CHECK(maxImagRot < 1e-9);
        std::printf("group delay flat: max|tau-D| = %.3e, max|Im(R)| = %.3e (< 1e-9): PASS\n",
                    maxGdErr, maxImagRot);
    }

    // ── 5. Passband magnitude ─────────────────────────────────────────────
    g_section = "passband magnitude";
    {
        const struct { double khz; } pts[] = {{1.0}, {5.0}, {10.0}, {15.0}, {20.0}};
        double mag15 = 0.0;
        for (const auto& p : pts)
        {
            const double db = magnitudeDb((p.khz * 1000.0) / kFs);
            std::printf("    %5.1f kHz : %+.6f dB\n", p.khz, db);
            if (p.khz == 15.0) mag15 = db;
        }
        CHECK(mag15 > -0.1);
        std::printf("passband magnitude (gate: > -0.1 dB at 15 kHz, got %+.6f): PASS\n", mag15);
    }

    // ── 6. Nyquist null ───────────────────────────────────────────────────
    g_section = "Nyquist null";
    {
        double s = 0.0;
        for (int m = 0; m < kN; ++m) s += kHalfSampleCoeffs[m] * ((m & 1) ? -1.0 : 1.0);
        CHECK(std::abs(s) < 1e-6);
        std::printf("Nyquist null: |H(pi)| = %.3e (< 1e-6): PASS\n", std::abs(s));
    }

    // ── 7. Lagrange cross-check (active only at kHalfSampleTaps == 6) ─────
    // The header's kHalfSampleTaps is a namespace-scoped constexpr, not a
    // macro, so the preprocessor #if the spec wrote cannot see it; this is
    // realised as if constexpr, which is dormant (compiled, not executed)
    // at 16 and active at 6. makeCoeffs is always available, so the block
    // compiles at both. NOTE: a Kaiser-windowed sinc and a Lagrange
    // interpolator are different designs, so kHalfSampleCoeffs does NOT
    // equal the Lagrange coefficients even at N=6; instead we verify the
    // Lagrange coefficients are *themselves* a valid symmetric half-sample
    // delay, tying this harness to already-tested code.
    g_section = "Lagrange cross-check";
    if constexpr (MarsDSP::Align::kHalfSampleTaps == 6)
    {
        const auto c = MarsDSP::Delays::makeCoeffs(MarsDSP::Delays::Interpolation::Lagrange5th, 0.5f);
        for (int j = 0; j < 6; ++j)
            CHECK(c.c[j] == c.c[5 - j]);
        double num = 0.0, den = 0.0, nyq = 0.0;
        for (int j = 0; j < 6; ++j)
        {
            num += double(j) * c.c[j];
            den += c.c[j];
            nyq += c.c[j] * ((j & 1) ? -1.0 : 1.0);
        }
        CHECK(std::abs(num / den - 2.5) < 1e-6);
        CHECK(std::abs(den - 1.0) < 1e-6);
        CHECK(std::abs(nyq) < 1e-6);
        std::printf("Lagrange cross-check (N=6, centroid/DC/Nyquist): PASS\n");
    }
    else
    {
        std::printf("Lagrange cross-check: dormant (kHalfSampleTaps = %d, active at 6)\n", kN);
    }

    // ── 8. Reset reproducibility ──────────────────────────────────────────
    g_section = "reset";
    {
        HalfSampleFir fir;
        std::array<float, 100> in{}, y1{}, y2{};
        for (int n = 0; n < 100; ++n)
            in[n] = 0.5f * std::sin(0.3f * float(n)) + 0.3f * std::sin(1.1f * float(n));

        fir.reset();
        for (int n = 0; n < 100; ++n) y1[n] = fir.process(in[n]);

        fir.reset();
        for (int n = 0; n < 100; ++n) y2[n] = fir.process(in[n]);

        for (int n = 0; n < 100; ++n)
            if (y1[n] != y2[n])
                FAIL("reset mismatch at n=%d: %g != %g", n, (double)y1[n], (double)y2[n]);
        std::printf("reset reproducibility (bit-exact): PASS\n");
    }

    // ── 9. Impulse-response bit-equality (C5) ──────────────────────────────
    g_section = "impulse-response bit-equality";
    {
        HalfSampleFir fir;
        fir.reset();
        for (int n = 0; n < kN; ++n)
        {
            const float y = fir.process(n == 0 ? 1.0f : 0.0f);
            const float exp = kHalfSampleCoeffs[static_cast<std::size_t>(n)];
            if (y != exp)
                FAIL("impulse response n=%d: got %g, expected coeff %g",
                     n, (double)y, (double)exp);
        }
        std::printf("impulse-response bit-equality (coeffs): PASS\n");
    }

    // ── 10. Memmove-vs-circular parity (C5) ────────────────────────────────
    g_section = "memmove parity";
    {
        // Local twin that keeps the OLD memmove-shift logic.
        struct HalfSampleFirOld {
            std::array<float, kN> z_{};
            void reset() noexcept { z_.fill(0.0f); }
            float process(float x) noexcept {
                std::memmove(z_.data() + 1, z_.data(),
                             static_cast<std::size_t>(kN - 1) * sizeof(float));
                z_.front() = x;
                float acc = 0.0f;
                for (int j = 0; j < kN / 2; ++j)
                    acc += kHalfSampleCoeffs[static_cast<std::size_t>(j)]
                         * (z_[static_cast<std::size_t>(j)]
                            + z_[static_cast<std::size_t>(kN - 1 - j)]);
                return acc;
            }
        };

        HalfSampleFir neu;
        HalfSampleFirOld old;
        neu.reset();
        old.reset();

        constexpr int kM = 2000;
        for (int n = 0; n < kM; ++n)
        {
            const float x = 0.5f * std::sin(0.3f * float(n))
                          + 0.3f * std::sin(1.1f * float(n))
                          + 0.01f * float(n);   // ramp breaks any DC symmetry
            const float yn = neu.process(x);
            const float yo = old.process(x);
            if (yn != yo)
                FAIL("memmove parity n=%d: new=%g old=%g", n, (double)yn, (double)yo);
        }
        std::printf("memmove-vs-circular parity (%d samples): PASS\n", kM);
    }

    return 0;
}

} // namespace

int main()
{
    std::printf("=== Chronos HalfSampleFir correctness harness ===\n");
    std::printf("kHalfSampleTaps = %d  D = %.1f  fs = %.0f Hz\n\n", kN, kD, kFs);

    int r = runAll();

    std::printf("\n=== %s ===\n", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

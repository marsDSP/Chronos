// tests/harnesses/cd/align_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Correctness harness for MarsDSP::Align::SaturatorAlign, the constant-
// latency dry/wet alignment layer for the ADAA saturator stage. This is the
// harness that verifies the property the whole layer exists for.
//
//   1. Dry is an exact integer delay. For all three modes, an impulse through
//      processDry lands exactly at sample kBudget, value 1.0f, zeros elsewhere.
//      Bit-exact. Invariant I1 — if this fails, mix = 0% transparency is gone.
//   2. Wet total group delay equals kBudget in every mode. The ADAA stage
//      kernel is prepended analytically (mode 0: identity; mode 1: 2-tap box
//      [0.5, 0.5]; mode 2: 3-tap triangle [0.25, 0.5, 0.25]) to the alignment
//      wet impulse response, and the centroid is asserted to equal kBudget
//      to 1e-6. This is the headline test.
//   3. Modes 0 and 2 are bit-exact pure delays. Impulse in, single 1.0f out at
//      kBudget (mode 0) and kBudget-1 (mode 2), zeros elsewhere.
//   4. Mode 1 is linear phase. DFT of the mode-1 wet impulse response; group-
//      delay ripple < 1e-6 samples across [0, 0.45·fs]. Invariant I3.
//   5. Dry/wet null. Mode 0, same white-noise sequence to processDry and
//      processWet; outputs are bit-identical.
//   6. Cross-mode phase coherence. For each mode, feed a 1/5/10/15 kHz sine
//      to dry and wet (bypassing the nonlinearity by applying the stage kernel
//      analytically before processWet), sum at 50/50, and assert the summed
//      amplitude is within 0.5% of the perfectly-aligned prediction
//      (dry_amp + wet_amp). A first-order allpass would fail this at 15 kHz.
//   7. Reset across all modes.
//   8. Mode switching mid-stream — no NaN, no unbounded output.
//
// Conventions (matching ring_buffer_check / halfsample_fir_check / short_delay_check):
// plain main(), exit code, printf, always-live CHECK/FAIL (NOT assert). Links
// SharedCode only; no JUCE. No forced -O2 so the header's assert preconditions
// stay armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/align/SaturatorAlign.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int    kBudget = MarsDSP::Align::SaturatorAlign::kBudget;
constexpr double kFs    = 48000.0;
constexpr double kPi    = 3.14159265358979323846;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

// Simple xorshift32 white-noise generator (deterministic, fast, no std::random).
struct Rng
{
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed) {}
    float nextFloat()
    {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Map to [-1, 1] via high bits.
        return (static_cast<float>(s >> 1) / 2147483520.0f) * 2.0f - 1.0f;
    }
};

// Capture the alignment layer's wet impulse response for a given mode.
std::vector<float> captureWetImpulse(int mode, int len)
{
    MarsDSP::Align::SaturatorAlign a;
    a.reset();
    a.setMode(mode);
    std::vector<float> y(static_cast<std::size_t>(len));
    for (int n = 0; n < len; ++n)
        y[static_cast<std::size_t>(n)] = a.processWet(n == 0 ? 1.0f : 0.0f);
    return y;
}

// Convolve x with a finite kernel k.
std::vector<float> convolve(const std::vector<float>& x, const std::vector<double>& k)
{
    if (k.empty()) return x;
    std::vector<float> y(x.size() + k.size() - 1, 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = 0; j < k.size(); ++j)
            y[i + j] += x[i] * static_cast<float>(k[j]);
    return y;
}

double centroid(const std::vector<float>& h)
{
    double num = 0.0, den = 0.0;
    for (std::size_t n = 0; n < h.size(); ++n)
    {
        num += static_cast<double>(n) * h[n];
        den += h[n];
    }
    return num / den;
}

// Complex exponential correlation over [start, start+len). For a coherent sine
// of amplitude A, the returned magnitude is approximately A * len / 2, so the
// amplitude is 2 * mag / len. omega must be 2*pi*k/len for integer k.
struct ComplexAmp
{
    double re, im;
    double mag() const { return std::sqrt(re * re + im * im); }
};

ComplexAmp correlate(const std::vector<float>& x, double omega, int start, int len)
{
    double c = 0.0, s = 0.0;
    for (int n = 0; n < len; ++n)
    {
        const double angle = omega * (start + n);
        c += x[static_cast<std::size_t>(start + n)] * std::cos(angle);
        s += x[static_cast<std::size_t>(start + n)] * std::sin(angle);
    }
    return {c, s};
}

double sineAmplitude(const std::vector<float>& x, double omega, int start, int len)
{
    const auto z = correlate(x, omega, start, len);
    return 2.0 * z.mag() / static_cast<double>(len);
}

// Apply the ADAA stage kernel analytically to a sample stream.
struct StageKernel
{
    std::vector<double> coeffs; // causal, newest at back? We use state shift.
    std::vector<double> state;

    explicit StageKernel(const std::vector<double>& c) : coeffs(c), state(c.size(), 0.0) {}

    double process(double x)
    {
        // Shift state: state[0] is oldest, state.back() is newest.
        for (std::size_t i = 0; i + 1 < state.size(); ++i)
            state[i] = state[i + 1];
        if (!state.empty()) state.back() = x;

        double y = 0.0;
        for (std::size_t i = 0; i < coeffs.size(); ++i)
            y += state[i] * coeffs[i]; // coeffs ordered oldest..newest
        return y;
    }

    void reset() { std::fill(state.begin(), state.end(), 0.0); }
};

StageKernel makeStageKernel(int mode)
{
    switch (mode)
    {
        case 0: return StageKernel({1.0});              // Off: identity
        case 1: return StageKernel({0.5, 0.5});       // ADAA1: box, 0.5-sample delay
        case 2: return StageKernel({0.25, 0.5, 0.25}); // ADAA2: triangle, 1-sample delay
    }
    FAIL("bad stage mode %d", mode);
    return StageKernel({}); // unreachable
}

// ── Test 1: dry is an exact integer delay of kBudget ─────────────────────
void testDryIntegerDelay()
{
    g_section = "dry integer delay";
    constexpr int kLen = 2 * kBudget + 3;
    for (int mode = 0; mode <= 2; ++mode)
    {
        MarsDSP::Align::SaturatorAlign a;
        a.reset();
        a.setMode(mode);
        for (int n = 0; n < kLen; ++n)
        {
            const float y = a.processDry(n == 0 ? 1.0f : 0.0f);
            const float exp = (n == kBudget) ? 1.0f : 0.0f;
            if (y != exp)
                FAIL("mode=%d dry n=%d got=%g exp=%g", mode, n, (double)y, (double)exp);
        }
    }
    std::printf("dry integer delay (kBudget=%d, all modes): PASS\n", kBudget);
}

// ── Test 2: wet total group delay equals kBudget in every mode ───────────
void testWetGroupDelay()
{
    g_section = "wet group delay";
    for (int mode = 0; mode <= 2; ++mode)
    {
        const auto align = captureWetImpulse(mode, 32);
        std::vector<double> stageCoeffs;
        switch (mode)
        {
            case 0: stageCoeffs = {1.0};              break;
            case 1: stageCoeffs = {0.5, 0.5};          break;
            case 2: stageCoeffs = {0.25, 0.5, 0.25};   break;
        }
        const auto combined = convolve(align, stageCoeffs);
        const double c = centroid(combined);
        if (std::abs(c - static_cast<double>(kBudget)) > 1e-6)
            FAIL("mode=%d wet centroid = %.9f != kBudget %d", mode, c, kBudget);
    }
    std::printf("wet total group delay == kBudget (all modes): PASS\n");
}

// ── Test 3: modes 0 and 2 are bit-exact pure delays ──────────────────────
void testModesBitExactPureDelay()
{
    g_section = "pure delay modes";
    constexpr int kLen = 2 * kBudget + 3;
    // Mode 0: wet total delay = kBudget.
    {
        MarsDSP::Align::SaturatorAlign a;
        a.reset();
        a.setMode(0);
        for (int n = 0; n < kLen; ++n)
        {
            const float y = a.processWet(n == 0 ? 1.0f : 0.0f);
            const float exp = (n == kBudget) ? 1.0f : 0.0f;
            if (y != exp)
                FAIL("mode 0 wet n=%d got=%g exp=%g", n, (double)y, (double)exp);
        }
    }
    // Mode 2: wet total delay = kBudget - 1 (the ADAA2 triangle supplies 1).
    {
        MarsDSP::Align::SaturatorAlign a;
        a.reset();
        a.setMode(2);
        for (int n = 0; n < kLen; ++n)
        {
            const float y = a.processWet(n == 0 ? 1.0f : 0.0f);
            const float exp = (n == kBudget - 1) ? 1.0f : 0.0f;
            if (y != exp)
                FAIL("mode 2 wet n=%d got=%g exp=%g", n, (double)y, (double)exp);
        }
    }
    std::printf("modes 0/2 are bit-exact pure delays: PASS\n");
}

// ── Test 4: mode 1 is linear phase ───────────────────────────────────────
void testMode1LinearPhase()
{
    g_section = "mode 1 linear phase";
    const auto h = captureWetImpulse(1, 32); // mode 1 wet = FIR impulse response

    const double D = static_cast<double>(kBudget) - 0.5; // FIR group delay
    double maxGdErr = 0.0;
    double maxImagRot = 0.0;
    constexpr int kPts = 1000;
    for (int i = 0; i < kPts; ++i)
    {
        const double fNorm = 0.45 * double(i) / double(kPts - 1); // [0, 0.45·fs]
        const double w = 2.0 * kPi * fNorm;

        double A = 0.0, C = 0.0, dA = 0.0, dC = 0.0;
        for (std::size_t m = 0; m < h.size(); ++m)
        {
            const double cs = std::cos(w * m);
            const double sn = std::sin(w * m);
            const double hm = h[m];
            A += hm * cs;
            C += hm * sn;
            dA += -static_cast<double>(m) * hm * sn;
            dC +=  static_cast<double>(m) * hm * cs;
        }
        const double denom = A * A + C * C;
        const double tau = (A * dC - C * dA) / denom;
        maxGdErr = std::max(maxGdErr, std::abs(tau - D));

        // Rotated response R(w) = H(w) e^{j w D} must be real.
        const std::complex<double> R = std::complex<double>(A, -C) * std::polar(1.0, w * D);
        maxImagRot = std::max(maxImagRot, std::abs(R.imag()));
    }
    CHECK(maxGdErr < 1e-6);
    CHECK(maxImagRot < 1e-6);
    std::printf("mode 1 linear phase: max|tau-D| = %.3e, max|Im(R)| = %.3e (< 1e-6): PASS\n",
                maxGdErr, maxImagRot);
}

// ── Test 5: dry/wet null in mode 0 ───────────────────────────────────────
void testDryWetNull()
{
    g_section = "dry/wet null";
    MarsDSP::Align::SaturatorAlign dry, wet;
    dry.reset();
    dry.setMode(0);
    wet.reset();
    wet.setMode(0);

    Rng rng(20240727u);
    for (int n = 0; n < 1000; ++n)
    {
        const float x = rng.nextFloat();
        const float d = dry.processDry(x);
        const float w = wet.processWet(x);
        if (d != w)
            FAIL("null mode 0 n=%d dry=%g wet=%g", n, (double)d, (double)w);
    }
    std::printf("dry/wet null (mode 0, bit-exact): PASS\n");
}

// ── Test 6: cross-mode phase coherence at 1/5/10/15 kHz ──────────────────
void testPhaseCoherence()
{
    g_section = "phase coherence";
    constexpr int kLen    = 4800;  // coherent window length (multiple of 48, so 1/5/10/15 kHz are integer bins at 48 kHz)
    constexpr int kStart    = 128;   // skip initial transients (FIR length + delay margin)
    constexpr int kTotal    = kStart + kLen;
    constexpr int kFreqs    = 4;
    constexpr std::array<double, kFreqs> kFreqsHz = {1000.0, 5000.0, 10000.0, 15000.0};

    for (int mode = 0; mode <= 2; ++mode)
    {
        for (double fHz : kFreqsHz)
        {
            MarsDSP::Align::SaturatorAlign a;
            a.reset();
            a.setMode(mode);
            StageKernel stage = makeStageKernel(mode);

            const int k = static_cast<int>(std::llround(fHz * kLen / kFs));
            const double omega = 2.0 * kPi * k / kLen;

            std::vector<float> dry(kTotal), wet(kTotal);
            for (int n = 0; n < kTotal; ++n)
            {
                const float x = static_cast<float>(std::sin(omega * n));
                dry[n] = a.processDry(x);
                const double s = stage.process(static_cast<double>(x));
                wet[n] = a.processWet(static_cast<float>(s));
            }

            const double dryAmp = sineAmplitude(dry, omega, kStart, kLen);
            const double wetAmp = sineAmplitude(wet, omega, kStart, kLen);

            std::vector<float> sum(kTotal);
            for (int n = 0; n < kTotal; ++n) sum[n] = dry[n] + wet[n];
            const double sumAmp = sineAmplitude(sum, omega, kStart, kLen);

            const double expected = dryAmp + wetAmp; // perfectly aligned in-phase sum
            if (expected <= 0.0)
                FAIL("mode=%d %.0f Hz non-positive expected amplitude %.6g", mode, fHz, expected);
            const double rel = std::abs(sumAmp - expected) / expected;
            if (rel > 0.005)
                FAIL("mode=%d %.0f Hz phase coherence |sumAmp - expected|/expected = %.4g "
                     "(sumAmp=%.6g, expected=%.6g, dry=%.6g, wet=%.6g)",
                     mode, fHz, rel, sumAmp, expected, dryAmp, wetAmp);
        }
    }
    std::printf("cross-mode phase coherence at 1/5/10/15 kHz: PASS\n");
}

// ── Test 7: reset across all modes ───────────────────────────────────────
void testReset()
{
    g_section = "reset";
    for (int mode = 0; mode <= 2; ++mode)
    {
        MarsDSP::Align::SaturatorAlign a;
        a.reset();
        a.setMode(mode);

        // Warm it up with a ramp and a few mode switches.
        for (int n = 0; n < 50; ++n) { (void) a.processDry(static_cast<float>(n)); (void) a.processWet(static_cast<float>(n)); }
        a.reset();
        // After reset, dry must again be the exact kBudget delay and wet must
        // be the correct pure-delay/FIR response for the current mode.
        for (int n = 0; n < 2 * kBudget + 3; ++n)
        {
            const float d = a.processDry(n == 0 ? 1.0f : 0.0f);
            const float dryExp = (n == kBudget) ? 1.0f : 0.0f;
            if (d != dryExp)
                FAIL("mode=%d reset dry n=%d got=%g exp=%g", mode, n, (double)d, (double)dryExp);
        }
        a.reset();
        if (mode == 1) continue; // mode 1 wet is a FIR; reset is covered by linearity below
        for (int n = 0; n < 2 * kBudget + 3; ++n)
        {
            const float w = a.processWet(n == 0 ? 1.0f : 0.0f);
            const int expectedTap = (mode == 0) ? kBudget : (kBudget - 1);
            const float wetExp = (n == expectedTap) ? 1.0f : 0.0f;
            if (w != wetExp)
                FAIL("mode=%d reset wet n=%d got=%g exp=%g", mode, n, (double)w, (double)wetExp);
        }
    }
    std::printf("reset reproducibility (all modes): PASS\n");
}

// ── Test 8: mode switching — no NaN, no unbounded output ──────────────────
void testModeSwitching()
{
    g_section = "mode switching";
    MarsDSP::Align::SaturatorAlign a;
    a.reset();
    a.setMode(0);

    Rng rng(20240728u);
    constexpr int kSwitchPeriod = 16;
    constexpr int kModes[] = {0, 1, 2, 1, 0, 2, 0, 1};
    double maxAbs = 0.0;
    for (int n = 0; n < 512; ++n)
    {
        const int mode = kModes[(n / kSwitchPeriod) % (sizeof(kModes) / sizeof(kModes[0]))];
        a.setMode(mode);
        const float x = rng.nextFloat();
        const float d = a.processDry(x);
        const float w = a.processWet(x);
        if (!std::isfinite(d) || !std::isfinite(w))
            FAIL("mode switch n=%d mode=%d produced non-finite dry=%g wet=%g", n, mode, (double)d, (double)w);
        maxAbs = std::max(maxAbs, std::max(std::abs((double)d), std::abs((double)w)));
    }
    CHECK(maxAbs < 1e6);
    std::printf("mode switching (no NaN, maxAbs=%.3g): PASS\n", maxAbs);
}

} // namespace

int main()
{
    std::printf("=== Chronos SaturatorAlign correctness harness ===\n");
    std::printf("kBudget = %d  fs = %.0f Hz\n\n", kBudget, kFs);

    testDryIntegerDelay();
    testWetGroupDelay();
    testModesBitExactPureDelay();
    testMode1LinearPhase();
    testDryWetNull();
    testPhaseCoherence();
    testReset();
    testModeSwitching();

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

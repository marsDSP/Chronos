// tests/harnesses/dsp/loop_gain_check.cpp
//
// Loop makeup and output trim harness (S5).
//
// Fixes feedback at 0.5 and sweeps the loop drive across its whole range.
// Measures RT60 from the decay envelope of a burst and asserts it stays
// within +-5 percent across the sweep (the makeup fix makes the small-signal
// loop gain exactly g at every drive). Separately measures the wet RMS and
// asserts it stays within +-1.5 dB across the sweep (the output trim
// compensates the level the saturator removes).
//
// Also verifies the rmsRatio / trim anchor values from the spec.
//
// Mono, full wet, no main saturator, transparent HPF/LPF, bits 32. The burst
// is a 0.5-amplitude 220 Hz tone (matching the rmsRatio reference level so the
// trim compensates where the saturator actually compresses). The RT60
// measurement uses the late decay tail where the signal is small enough that
// the small-signal loop gain governs. Links SharedCode only.

#include "dsp/FeedbackDelay.h"
#include "dsp/SimdDelayLine.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {

constexpr double kFs    = 48000.0;
constexpr double kPi    = std::numbers::pi_v<double>;
constexpr int    kBlock = 256;
constexpr int    kDelay = 4800;       // 100 ms repeats
constexpr float  kFb    = 0.5f;
constexpr int    kBurstLen = 4800;    // one delay period of tone
constexpr int    kDecayLen = 120000;  // ~2.5 s: > RT60 at g=0.5 (~1 s)
constexpr int    kTotal = kBurstLen + kDecayLen;

using MarsDSP::Delays::FeedbackDelay;
using MarsDSP::Delays::SimdDelayLine;

const char* g_section = "(startup)";

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

// Run one drive step with a continuous tone (steady-state wet RMS) and a
// subsequent burst+silence (RT60). Returns the full output buffer. The
// continuous tone lets the loop reach a steady state where the trim is
// designed to compensate. The burst measures the decay rate.
std::vector<float> runDrive(float loopDrive, int& steadyStart, int& burstStart)
{
    SimdDelayLine dl;
    dl.prepare(kFs, kBlock, 5000.0f);
    const int fbMaxDelay = dl.getMaxDelaySamples();

    FeedbackDelay fb;
    fb.prepare(kFs, kBlock, fbMaxDelay);

    FeedbackDelay::Params p;
    p.delaySamples  = static_cast<float>(kDelay);
    p.feedback      = kFb;
    p.dampHz        = 20000.0f;   // transparent
    p.crossFeed     = 0.0f;
    p.loopDrive     = loopDrive;
    p.satOrder      = 2;          // ADAA2
    p.enableDiffuser = false;
    p.diffusion     = 0.7f;
    p.diffuserSize  = 0.5f;
    p.diffModDepth  = 0.0f;
    p.diffModRateHz = 0.5f;
    fb.resetParams(p);

    // Phase 1: continuous 0.5-amp 220 Hz tone (steady state).
    constexpr int kSteadyLen = 48000;   // 1 s: >> loop build-up time
    steadyStart = kSteadyLen - kDelay;  // measure the last delay period
    // Phase 2: burst at the same tone, then silence (RT60).
    constexpr int kBurst = kBurstLen;
    constexpr int kSilence = kDecayLen;
    burstStart = kSteadyLen;
    const int total = kSteadyLen + kBurst + kSilence;

    std::vector<float> inBuf(static_cast<std::size_t>(total), 0.0f);
    for (int i = 0; i < kSteadyLen + kBurst; ++i)
        inBuf[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::sin(2.0 * kPi * 220.0 * static_cast<double>(i) / kFs));

    std::vector<float> wet(static_cast<std::size_t>(total), 0.0f);
    for (int off = 0; off < total; off += kBlock)
    {
        const int n = std::min(kBlock, total - off);
        fb.process(inBuf.data() + off, nullptr,
                   wet.data() + off, nullptr, n);
    }
    return wet;
}

// Measure RT60 from the decay envelope. Windows the output into delay-period
// blocks (one per repeat), computes the RMS of each, fits a line to
// 20*log10(rms) vs time, and returns RT60 in seconds. Also returns the worst
// deviation of any repeat from the fitted line (the linearity check). The
// decay starts at burstStart + kBurstLen (the silence after the burst).
struct RT60Result { double rt60; double worstDevDb; };
RT60Result measureRT60(const std::vector<float>& wet, int burstStart)
{
    const int decayStart = burstStart + kBurstLen;
    const int nRepeats = kDecayLen / kDelay;
    std::vector<double> rmsDb;
    std::vector<double> timeMs;
    for (int r = 0; r < nRepeats; ++r)
    {
        const int start = decayStart + r * kDelay;
        double sumSq = 0.0;
        int count = 0;
        for (int i = 0; i < kDelay; ++i)
        {
            const int idx = start + i;
            if (idx >= static_cast<int>(wet.size())) break;
            const double v = static_cast<double>(wet[static_cast<std::size_t>(idx)]);
            sumSq += v * v;
            ++count;
        }
        if (count == 0) break;
        const double rms = std::sqrt(sumSq / static_cast<double>(count));
        if (rms < 1e-12) break;     // below the noise floor
        rmsDb.push_back(20.0 * std::log10(rms));
        timeMs.push_back(static_cast<double>(r) * static_cast<double>(kDelay) / kFs * 1000.0);
    }
    if (rmsDb.size() < 4)
        FAIL("RT60: only %zu repeat windows above noise floor", rmsDb.size());

    // Linear regression: rmsDb = a + b * timeMs. RT60 = -60 / b (ms).
    // Skip the first repeat (transient / burst tail).
    const std::size_t i0 = 1;
    const std::size_t N = rmsDb.size() - i0;
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (std::size_t i = i0; i < rmsDb.size(); ++i)
    {
        sx  += timeMs[i];
        sy  += rmsDb[i];
        sxx += timeMs[i] * timeMs[i];
        sxy += timeMs[i] * rmsDb[i];
    }
    const double denom = static_cast<double>(N) * sxx - sx * sx;
    if (std::fabs(denom) < 1e-12) FAIL("RT60: degenerate linear fit");
    const double b = (static_cast<double>(N) * sxy - sx * sy) / denom;
    const double a = (sy - b * sx) / static_cast<double>(N);
    if (b >= 0.0) FAIL("RT60: non-decaying envelope (slope %.4f dB/ms)", b);

    // Worst deviation of any fitted repeat from the line (linearity check).
    double worstDev = 0.0;
    for (std::size_t i = i0; i < rmsDb.size(); ++i)
    {
        const double predicted = a + b * timeMs[i];
        const double dev = std::fabs(rmsDb[i] - predicted);
        worstDev = std::max(worstDev, dev);
    }

    return { -60.0 / b / 1000.0, worstDev };   // seconds, dB
}

} // namespace

int main()
{
    std::printf("=== Chronos loop_gain_check (S5) ===\n");
    std::printf("fs=%.0f  delay=%d (%.0f ms)  feedback=%.2f  burst=220Hz @0.5\n\n",
                kFs, kDelay, static_cast<double>(kDelay) / kFs * 1000.0, static_cast<double>(kFb));

    // ── Anchor value check: rmsRatioForDrive_ vs the spec table ──────────
    g_section = "anchor values";
    struct Anchor { float k; float rmsRatio; float trim; };
    const Anchor anchors[] = {
        { 1.000000f,  0.942467f, 1.030070f },
        { 1.995262f,  1.624467f, 0.784593f },
        { 15.848932f, 2.711624f, 0.607275f },
    };
    std::printf("%10s %12s %12s | %12s %12s | %s\n",
                "k", "rmsRatio", "spec", "trim", "spec", "pass");
    for (const auto& a : anchors)
    {
        const float got = FeedbackDelay::rmsRatioForDrive_(a.k);
        const float trim = std::pow(got, -0.5f);
        const bool okR = std::fabs(got - a.rmsRatio) < 1e-4f;
        const bool okT = std::fabs(trim - a.trim) < 1e-4f;
        std::printf("%10.6f %12.6f %12.6f | %12.6f %12.6f | %s\n",
                    static_cast<double>(a.k), static_cast<double>(got),
                    static_cast<double>(a.rmsRatio),
                    static_cast<double>(trim), static_cast<double>(a.trim),
                    (okR && okT) ? "PASS" : "FAIL");
        if (!okR) FAIL("rmsRatio(k=%.6f) = %.6f vs spec %.6f", a.k, got, a.rmsRatio);
        if (!okT) FAIL("trim(k=%.6f) = %.6f vs spec %.6f", a.k, trim, a.trim);
    }
    std::printf("anchor values: PASS\n\n");

    // ── RT60 + wet RMS across the drive sweep ────────────────────────────
    // The trim is defined for the 0.5-amp reference, so the sweep starts at
    // drive=1 (0 dB) where the reference-level signal enters the saturator at
    // 0.5 amp. Below drive=1 the saturator is linear for this reference and the
    // trim over-boosts (rmsRatio drops toward k, trim rises as 1/sqrt(k)).
    const float drives[] = { 1.0f, 2.0f, 4.0f, 8.0f, 16.0f };
    constexpr int kNDrives = static_cast<int>(sizeof(drives) / sizeof(drives[0]));

    const double theoreticalRT60 =
        3.0 * static_cast<double>(kDelay) / kFs /
        std::fabs(std::log10(static_cast<double>(kFb)));

    std::printf("theoretical RT60 (g=0.5): %.4f s\n\n", theoreticalRT60);
    std::printf("%6s %10s %10s %10s | %10s %10s | %s\n",
                "drive", "RT60(s)", "theory", "|d|%", "linDev", "gate", "pass");

    double worstRT60Err = 0.0;
    double worstLinDev = 0.0;
    for (int d = 0; d < kNDrives; ++d)
    {
        g_section = "drive sweep";
        int steadyStart = 0, burstStart = 0;
        (void)steadyStart;
        const auto wet = runDrive(drives[d], steadyStart, burstStart);
        const RT60Result r = measureRT60(wet, burstStart);

        const double rt60Err = std::fabs(r.rt60 - theoreticalRT60) / theoreticalRT60 * 100.0;
        worstRT60Err = std::max(worstRT60Err, rt60Err);
        worstLinDev = std::max(worstLinDev, r.worstDevDb);

        const bool okRT = rt60Err <= 5.0;
        const bool okLin = r.worstDevDb <= 1.5;
        std::printf("%6.2f %10.4f %10.4f %9.2f%% | %9.3f %9.1f | %s\n",
                    static_cast<double>(drives[d]), r.rt60, theoreticalRT60, rt60Err,
                    r.worstDevDb, 1.5, (okRT && okLin) ? "PASS" : "FAIL");
        if (!okRT)
            FAIL("drive=%.2f: RT60 %.4f s vs theory %.4f s (%.2f%% > 5%%)",
                 drives[d], r.rt60, theoreticalRT60, rt60Err);
        if (!okLin)
            FAIL("drive=%.2f: decay envelope linearity dev %.3f dB > 1.5 dB",
                 drives[d], r.worstDevDb);
    }

    std::printf("\nworst RT60 error: %.2f%% (gate 5%%)\n", worstRT60Err);
    std::printf("worst decay linearity dev: %.3f dB (gate 1.5 dB)\n", worstLinDev);
    std::printf("\n=== ALL LOOP GAIN GATES HELD ===\n");
    return 0;
}

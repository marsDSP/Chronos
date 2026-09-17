/**
 * SpectrumAnalyser harness. Host-free: links SharedCode only (the
 * analyser is header-only and JUCE-free).
 *
 *   1. A 1 kHz full-scale tone peaks within two bins of 1 kHz, at a
 *      level between the raw amplitude reading (-6 dB) and the smoothing
 *      loss (bounded at -24 dB), and sits 50 dB above the trace at 10 kHz.
 *   2. Silence after the tone releases below the floor within the
 *      specified decay time plus margin, and never rises above it again.
 *   3. Tilt: white noise has a flat power density, and the boxcar takes
 *      the mean power per bin, so its trace slopes by exactly the tilt:
 *      the 4 kHz octave reads 2 * kTiltDbPerOct above the 1 kHz octave,
 *      within 2 dB. A pure tone, whose fixed line energy is spread over a
 *      constant-Q window, reads 3 dB/oct less: 2 * (tilt - 3.01) dB.
 *   4. magnitudeDbAt interpolates between bins and clamps at both ends.
 */

#include "gui/filter/SpectrumAnalyser.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <print>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

using MarsDSP::GUI::SpectrumAnalyser;
constexpr double kFs = 48000.0;

// Push `seconds` of a sine at freqHz, calling update() every 1/refresh s
// the way the display timer does. Return the analyser filled.
void runTone(SpectrumAnalyser& a, const double freqHz, const float amp, const double seconds)
{
    const int perTick = static_cast<int>(kFs / static_cast<double>(SpectrumAnalyser::kRefreshHz));
    const int ticks = static_cast<int>(seconds * static_cast<double>(SpectrumAnalyser::kRefreshHz));
    std::vector<float> buf(static_cast<std::size_t>(perTick));
    double phase = 0.0;
    const double inc = 2.0 * std::numbers::pi_v<double> * freqHz / kFs;
    for (int t = 0; t < ticks; ++t)
    {
        for (auto& v : buf)
        {
            v = amp * static_cast<float>(std::sin(phase));
            phase += inc;
        }
        a.push(buf.data(), perTick);
        a.update();
    }
}

struct Peak { int bin; float db; };
Peak findPeak(const SpectrumAnalyser& a, const double fMin, const double fMax)
{
    Peak p { -1, -1.0e9f };
    for (int k = 1; k < SpectrumAnalyser::numBins(); ++k)
    {
        const double f = a.binFrequency(k);
        if (f < fMin || f > fMax) continue;
        if (a.binDb(k) > p.db) { p.db = a.binDb(k); p.bin = k; }
    }
    return p;
}

int runAll()
{
    // ------------------------------------------------------------------
    // 1. Tone in bin.
    // ------------------------------------------------------------------
    g_section = "tone";
    float peak1k = 0.0f;
    {
        SpectrumAnalyser a;
        a.prepare(kFs);
        CHECK(a.isPrepared());
        runTone(a, 1000.0, 1.0f, 1.0);

        const auto p = findPeak(a, 20.0, 20000.0);
        const double binHz = kFs / static_cast<double>(SpectrumAnalyser::kSize);
        const double fPeak = a.binFrequency(p.bin);
        if (std::abs(fPeak - 1000.0) > 2.0 * binHz)
            FAIL("1 kHz tone peaked at {:.1f} Hz (bin {}), more than two bins off", fPeak, p.bin);

        // Raw amplitude reading is -6 dB (power of A/2); the two boxcar
        // passes spread the main lobe over about 1/6 octave, so the
        // smoothed peak lands lower but stays above -24 dB.
        if (p.db > -6.0f + 0.5f || p.db < -24.0f)
            FAIL("1 kHz peak level {:.2f} dB outside [-24, -5.5]", p.db);
        peak1k = p.db;

        // Far from the tone the trace sits well below the peak (the tilt
        // adds +15 dB at 10 kHz, still leaving a wide margin).
        const float far = a.magnitudeDbAt(10000.0f);
        if (peak1k - far < 50.0f)
            FAIL("10 kHz trace {:.1f} dB is only {:.1f} dB under the 1 kHz peak", far, peak1k - far);

        std::println("1 kHz tone: peak at {:.1f} Hz, {:.2f} dB; 10 kHz trace {:.1f} dB: PASS", fPeak, peak1k, far);
    }

    // ------------------------------------------------------------------
    // 2. Release to the floor on silence.
    // ------------------------------------------------------------------
    g_section = "release";
    {
        SpectrumAnalyser a;
        a.prepare(kFs);
        runTone(a, 1000.0, 1.0f, 1.0);
        const int bin = findPeak(a, 20.0, 20000.0).bin;

        const int perTick = static_cast<int>(kFs / static_cast<double>(SpectrumAnalyser::kRefreshHz));
        std::vector<float> zeros(static_cast<std::size_t>(perTick), 0.0f);
        // Twice the specified decay time, plus the 85 ms the window takes
        // to slide the tone out of the history.
        const int ticks = static_cast<int>(2.0 * SpectrumAnalyser::kDecaySeconds * SpectrumAnalyser::kRefreshHz) + 4;
        float last = 0.0f;
        for (int t = 0; t < ticks; ++t)
        {
            a.push(zeros.data(), perTick);
            a.update();
            last = a.binDb(bin);
        }
        if (last > SpectrumAnalyser::kFloorDb)
            FAIL("after {} silent frames the peak bin reads {:.1f} dB, above the floor {}", ticks, last, SpectrumAnalyser::kFloorDb);

        // It stays down.
        for (int t = 0; t < 30; ++t)
        {
            a.push(zeros.data(), perTick);
            a.update();
            CHECK(a.binDb(bin) <= last + 1.0e-3f);
        }
        std::println("release: {:.1f} dB after {} silent frames, monotone after: PASS", last, ticks);
    }

    // ------------------------------------------------------------------
    // 3. Tilt slope.
    // ------------------------------------------------------------------
    g_section = "tilt";
    {
        // White noise: flat density, so the trace slopes by the tilt alone.
        SpectrumAnalyser a;
        a.prepare(kFs);
        const int perTick = static_cast<int>(kFs / static_cast<double>(SpectrumAnalyser::kRefreshHz));
        std::vector<float> buf(static_cast<std::size_t>(perTick));
        std::uint32_t rng = 0x7F4A7C15u;
        for (int t = 0; t < 60; ++t)
        {
            for (auto& v : buf)
            {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                v = static_cast<float>(rng >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
            }
            a.push(buf.data(), perTick);
            a.update();
        }
        auto octaveMeanDb = [&](const double centre)
        {
            double sum = 0.0;
            int n = 0;
            for (int k = 1; k < SpectrumAnalyser::numBins(); ++k)
            {
                const double f = a.binFrequency(k);
                if (f < centre / std::sqrt(2.0) || f > centre * std::sqrt(2.0)) continue;
                sum += a.binDb(k);
                ++n;
            }
            return static_cast<float>(sum / std::max(1, n));
        };
        const float want = 2.0f * SpectrumAnalyser::kTiltDbPerOct;
        const float got = octaveMeanDb(4000.0) - octaveMeanDb(1000.0);
        if (std::abs(got - want) > 2.0f)
            FAIL("white noise: 4 kHz octave - 1 kHz octave = {:.2f} dB, expected the tilt {:.2f}", got, want);

        // A tone reads 3 dB/oct less: its line energy is a mean over a
        // window whose width grows with frequency.
        SpectrumAnalyser b;
        b.prepare(kFs);
        runTone(b, 4000.0, 1.0f, 1.0);
        const float toneWant = 2.0f * (SpectrumAnalyser::kTiltDbPerOct - 10.0f * std::log10(2.0f));
        const float toneGot = findPeak(b, 20.0, 20000.0).db - peak1k;
        if (std::abs(toneGot - toneWant) > 1.5f)
            FAIL("tone: 4 kHz - 1 kHz peak = {:.2f} dB, expected {:.2f}", toneGot, toneWant);

        std::println("tilt: white noise {:.2f} dB / 2 oct (expected {:.1f}); tone {:.2f} dB (expected {:.2f}): PASS",
                     got, want, toneGot, toneWant);
    }

    // ------------------------------------------------------------------
    // 4. magnitudeDbAt interpolation and clamps.
    // ------------------------------------------------------------------
    g_section = "interp";
    {
        SpectrumAnalyser a;
        a.prepare(kFs);
        runTone(a, 1000.0, 1.0f, 1.0);
        const int k = 100;
        const float f0 = static_cast<float>(a.binFrequency(k));
        const float f1 = static_cast<float>(a.binFrequency(k + 1));
        const float mid = a.magnitudeDbAt(0.5f * (f0 + f1));
        const float want = 0.5f * (a.binDb(k) + a.binDb(k + 1));
        CHECK(std::abs(mid - want) < 1.0e-3f);
        CHECK(a.magnitudeDbAt(0.0f) == a.binDb(1));
        CHECK(a.magnitudeDbAt(1.0e9f) == a.binDb(SpectrumAnalyser::numBins() - 1));
        std::println("magnitudeDbAt: midpoint interpolates, ends clamp: PASS");
    }

    std::println("spectrum_check: ALL PASS");
    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos SpectrumAnalyser harness ===");
    std::println();
    const int r = runAll();
    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

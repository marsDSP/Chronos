/**
 * ParametricEQ response harness. Host-free: links SharedCode only.
 *
 * The property the FILTER page rests on: the curve the display draws
 * (ParametricEQ::magnitudeDb, analytic) is the response the audio path
 * has (the SimdSVF bands). Each configuration is measured as a 65536-
 * sample impulse response through the vendored FFT and compared bin by
 * bin from 30 Hz to 20 kHz.
 *
 *   1. Single band: Bell / Low Shelf / High Shelf / Notch over a grid of
 *      frequency x gain x Q, measured vs analytic within kTolDb where the
 *      analytic response is above -40 dB (deep notch nulls are skipped).
 *   2. Cascade additivity: four bands on, measured vs the sum of the four
 *      analytic responses.
 *   3. Bit transparency: every band off passes the input unchanged;
 *      a Bell at 0 dB passes it unchanged too.
 *   4. No stale replay: a band driven hard, toggled off, then on again
 *      into silence emits silence.
 *   5. The cut pair model: OutputFilterStage (Digital) HPF + LPF measured
 *      vs cutMagnitudeDb within kTolDb.
 */

#include "dsp/ParametricEQ.h"
#include "dsp/OutputFilterStage.h"
#include "math/FFT.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <print>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kFs = 48000.0;
constexpr int kN = 65536;           // IR length: 0.73 Hz bins, a Q 12 bell at 60 Hz decays well inside
constexpr int kBlock = 256;         // process in blocks, so the sub-block ramps run
constexpr double kTolDb = 0.1;
constexpr double kFloorDb = -40.0;  // skip bins the analytic curve puts below this
constexpr double kFMin = 30.0;
constexpr double kFMax = 20000.0;

using MarsDSP::Filters::ParametricEQ;

// Deterministic noise.
std::uint32_t g_rng = 0x9E3779B9u;
float noise()
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return static_cast<float>(g_rng >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

// Magnitude in dB per bin (index k -> k * fs / N) of an impulse pushed through
// `run(l, r, n)` in kBlock-sample blocks. Stereo: L carries the impulse, R is
// silent, and the R output must stay silent (lane independence).
template <typename Run>
std::vector<double> measureIrDb(Run&& run)
{
    std::vector<float> l(kN, 0.0f), r(kN, 0.0f);
    l[0] = 1.0f;
    for (int off = 0; off < kN; off += kBlock)
        run(l.data() + off, r.data() + off, std::min(kBlock, kN - off));

    for (int i = 0; i < kN; ++i)
    {
        CHECK(std::isfinite(l[static_cast<std::size_t>(i)]));
        CHECK(r[static_cast<std::size_t>(i)] == 0.0f);
    }

    std::vector<double> x(kN);
    for (int i = 0; i < kN; ++i)
        x[static_cast<std::size_t>(i)] = l[static_cast<std::size_t>(i)];

    MarsDSP::MathOps::RealFFT<double> fft(static_cast<std::size_t>(kN));
    std::vector<std::complex<double>> X(kN / 2);
    fft.forward(x.data(), X.data());

    std::vector<double> db(kN / 2, -200.0);
    for (int k = 1; k < kN / 2; ++k)
        db[static_cast<std::size_t>(k)] = 20.0 * std::log10(std::max(std::abs(X[static_cast<std::size_t>(k)]), 1.0e-12));
    return db;
}

// Compare a measured IR spectrum against an analytic dB function over
// [kFMin, kFMax], skipping bins the analytic curve puts below kFloorDb.
template <typename Analytic>
double compareDb(const std::vector<double>& measured, Analytic&& analytic, const char* what)
{
    double worst = 0.0;
    int worstBin = 0;
    for (int k = 1; k < kN / 2; ++k)
    {
        const double f = static_cast<double>(k) * kFs / static_cast<double>(kN);
        if (f < kFMin || f > kFMax) continue;
        const double want = analytic(f);
        if (want < kFloorDb) continue;
        const double err = std::abs(measured[static_cast<std::size_t>(k)] - want);
        if (err > worst) { worst = err; worstBin = k; }
    }
    if (worst > kTolDb)
        FAIL("{}: worst {:.4f} dB at {:.1f} Hz > {:.3f} dB", what, worst,
             static_cast<double>(worstBin) * kFs / static_cast<double>(kN), kTolDb);
    return worst;
}

ParametricEQ::Band makeBand(const int type, const float freq, const float gain, const float q)
{
    ParametricEQ::Band b;
    b.on = true;
    b.type = type;
    b.freqHz = freq;
    b.gainDb = gain;
    b.q = q;
    return b;
}

int runAll()
{
    // ------------------------------------------------------------------
    // 1. Single bands over the grid.
    // ------------------------------------------------------------------
    g_section = "single_band";
    {
        const int types[] = { 0, 1, 2, 3 };
        const char* typeNames[] = { "Bell", "LowShelf", "HighShelf", "Notch" };
        const float freqs[] = { 60.0f, 400.0f, 2500.0f, 9000.0f };
        const float gains[] = { -15.0f, -6.0f, 6.0f, 15.0f };
        const float qs[] = { 0.4f, 0.707f, 3.0f, 12.0f };

        double worstAll = 0.0;
        int configs = 0;
        for (const int type : types)
        {
            const bool hasGain = (type != 3);
            for (const float freq : freqs)
                for (const float q : qs)
                    for (const float gain : gains)
                    {
                        if (! hasGain && gain != gains[0]) continue; // one pass for the notch

                        ParametricEQ eq;
                        eq.prepare(kFs, 2);
                        eq.setBand(0, makeBand(type, freq, hasGain ? gain : 0.0f, q));

                        const auto db = measureIrDb([&](float* l, float* r, int n) { eq.process(l, r, n); });
                        const double worst = compareDb(db, [&](double f)
                        {
                            return ParametricEQ::bandMagnitudeDb(type, kFs, f, freq, hasGain ? gain : 0.0, q);
                        }, typeNames[type]);
                        worstAll = std::max(worstAll, worst);
                        ++configs;
                    }
        }
        std::println("single band: {} configurations, worst deviation {:.4f} dB (gate {:.2f}): PASS",
                     configs, worstAll, kTolDb);
    }

    // ------------------------------------------------------------------
    // 2. Cascade additivity: one band of each type.
    // ------------------------------------------------------------------
    g_section = "cascade";
    {
        ParametricEQ eq;
        eq.prepare(kFs, 2);
        const ParametricEQ::Band bands[4] = {
            makeBand(1, 150.0f, 6.0f, 0.707f),     // low shelf
            makeBand(0, 1000.0f, -8.0f, 2.0f),     // bell
            makeBand(3, 3000.0f, 0.0f, 4.0f),      // notch
            makeBand(2, 8000.0f, 4.0f, 0.707f),    // high shelf
        };
        for (int i = 0; i < 4; ++i)
            eq.setBand(i, bands[i]);

        const auto db = measureIrDb([&](float* l, float* r, int n) { eq.process(l, r, n); });
        const double worst = compareDb(db, [&](double f) { return eq.responseDb(f); }, "cascade");
        std::println("cascade of four bands vs the analytic sum: worst {:.4f} dB: PASS", worst);
    }

    // ------------------------------------------------------------------
    // 3. Bit transparency.
    // ------------------------------------------------------------------
    g_section = "transparency";
    {
        std::vector<float> l(4096), r(4096), l0, r0;
        for (auto& v : l) v = noise();
        for (auto& v : r) v = noise();
        l0 = l; r0 = r;

        ParametricEQ eq;
        eq.prepare(kFs, 2);
        eq.process(l.data(), r.data(), 4096);
        for (int i = 0; i < 4096; ++i)
        {
            CHECK(l[static_cast<std::size_t>(i)] == l0[static_cast<std::size_t>(i)]);
            CHECK(r[static_cast<std::size_t>(i)] == r0[static_cast<std::size_t>(i)]);
        }

        // A bell at 0 dB: m1 = k (A^2 - 1) = 0 exactly, so out = in.
        eq.setBand(1, makeBand(0, 1000.0f, 0.0f, 0.707f));
        eq.process(l.data(), r.data(), 4096);
        for (int i = 0; i < 4096; ++i)
        {
            CHECK(l[static_cast<std::size_t>(i)] == l0[static_cast<std::size_t>(i)]);
            CHECK(r[static_cast<std::size_t>(i)] == r0[static_cast<std::size_t>(i)]);
        }
        std::println("all bands off, and a flat bell: input passes unchanged: PASS");
    }

    // ------------------------------------------------------------------
    // 4. No stale replay on toggle.
    // ------------------------------------------------------------------
    g_section = "toggle";
    {
        ParametricEQ eq;
        eq.prepare(kFs, 2);
        auto band = makeBand(0, 200.0f, 15.0f, 12.0f); // a ringing bell
        eq.setBand(2, band);

        std::vector<float> l(4096), r(4096);
        for (auto& v : l) v = noise();
        for (auto& v : r) v = noise();
        eq.process(l.data(), r.data(), 4096);

        band.on = false;
        eq.setBand(2, band);
        std::fill(l.begin(), l.end(), 0.0f);
        std::fill(r.begin(), r.end(), 0.0f);
        eq.process(l.data(), r.data(), 4096);
        for (int i = 0; i < 4096; ++i)
            CHECK(l[static_cast<std::size_t>(i)] == 0.0f && r[static_cast<std::size_t>(i)] == 0.0f);

        band.on = true;
        eq.setBand(2, band);
        eq.process(l.data(), r.data(), 4096);
        for (int i = 0; i < 4096; ++i)
            CHECK(std::abs(l[static_cast<std::size_t>(i)]) < 1.0e-12f && std::abs(r[static_cast<std::size_t>(i)]) < 1.0e-12f);
        std::println("off skips the band, on again starts from silence: PASS");
    }

    // ------------------------------------------------------------------
    // 5. The cut pair model vs OutputFilterStage (Digital).
    // ------------------------------------------------------------------
    g_section = "cuts";
    {
        MarsDSP::Filters::OutputFilterStage stage;
        stage.prepare(kFs, 2);
        stage.setModeImmediate(MarsDSP::Filters::OutputFilterStage::Mode::Digital);
        constexpr float hpf = 150.0f, lpf = 6000.0f;
        stage.setCutoffs(hpf, lpf);

        const auto db = measureIrDb([&](float* l, float* r, int n)
        {
            // OutputFilterStage is out of place; run it into scratch and copy back.
            std::vector<float> ol(static_cast<std::size_t>(n)), orr(static_cast<std::size_t>(n));
            stage.process(l, r, ol.data(), orr.data(), n);
            std::memcpy(l, ol.data(), sizeof(float) * static_cast<std::size_t>(n));
            std::memcpy(r, orr.data(), sizeof(float) * static_cast<std::size_t>(n));
        });
        const double worst = compareDb(db, [&](double f)
        {
            return ParametricEQ::cutMagnitudeDb(true, kFs, f, hpf) + ParametricEQ::cutMagnitudeDb(false, kFs, f, lpf);
        }, "cut pair");
        std::println("OutputFilterStage Digital HPF {} / LPF {} vs cutMagnitudeDb: worst {:.4f} dB: PASS", hpf, lpf, worst);
    }

    std::println("eq_response_check: ALL PASS");
    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos ParametricEQ response harness ===");
    std::println();
    const int r = runAll();
    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

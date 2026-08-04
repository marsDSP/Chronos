// tests/harnesses/dsp/diffuser_ring_check.cpp
//
// Diffuser ring-decay harness. Measures the cascade impulse response
// envelope and asserts the tapered coefficient at 0.78 rings at least 6 dB
// less than the untapered 0.92 baseline in the late tail.
//
// The baseline is a hand-rolled eight-section Schroeder cascade. It uses the
// same prime-snapped path lengths, the same size cut, and the same FracDelayTap
// as the Diffuser, but holds every section at g = 0.92 with no taper. This is
// the pre-S20 design. The tapered path uses the real Diffuser at master g 0.78.
//
// An allpass cascade preserves total energy. A higher coefficient spreads more
// energy into the late tail. The gate measures the RMS energy in a late window
// and compares the two in dB. Links SharedCode only; no JUCE.

#include "dsp/Diffuser.h"
#include "dsp/FracDelayTap.h"
#include "dsp/Pow2RingBuffer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr double kFs      = 48000.0;
constexpr int    kBlock   = 256;
constexpr int    kSettle  = 256;       // one block; prime() snaps the smoothers
constexpr int    kCapture = 131072;    // 2.7 s: captures the late tail
constexpr int    kTotal   = kSettle + kCapture;
constexpr float  kBaselineCoef = 0.92f; // pre master coefficient
constexpr float  kTaperedCoef  = 0.78f; // master coefficient
constexpr int    kNumSec = MarsDSP::Diffusion::Diffuser::kNumSections;
constexpr double kGateDb = 6.0;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

using D = MarsDSP::Diffusion::Diffuser;

// Hand-rolled baseline cascade: eight Schroeder allpass sections at a fixed
// coefficient, no per-section taper. Mirrors the pre-Diffuser chain_.
struct BaselineSection
{
    MarsDSP::Delays::Pow2RingBuffer ring;
    int len = 0;
    int w   = 0;
};

void baselinePrepare(BaselineSection* bank, double sr)
{
    D d;
    d.prepare(sr);
    const int headroom = D::modHeadroomFor(sr);
    for (int i = 0; i < kNumSec; ++i)
    {
        bank[i].len = d.sectionLenL(i);
        const int minCap = bank[i].len + headroom
                         + MarsDSP::Delays::Pow2RingBuffer::kTail + 8;
        bank[i].ring.prepare(minCap);
        bank[i].w = 0;
    }
}

void baselineReset(BaselineSection* bank)
{
    for (int i = 0; i < kNumSec; ++i) { bank[i].ring.clear(); bank[i].w = 0; }
}

float baselineChain(BaselineSection* bank, float x, float size, float coef)
{
    for (int i = 0; i < kNumSec; ++i)
    {
        auto& sec = bank[i];
        const float lenF = static_cast<float>(sec.len);
        float eff = D::effLen(lenF, size);
        eff = std::nearbyintf(eff);
        eff = std::clamp(eff, D::kMinDelay, lenF);
        const float g = coef * D::sectionSign(i);
        const float dd = MarsDSP::Delays::FracDelayTap::read(sec.ring, sec.w, eff);
        float v = x - g * dd;
        if (!std::isfinite(v)) v = 0.0f;
        const float y = dd + g * v;
        sec.ring.writeBlock(&v, sec.w, 1);
        sec.ring.refreshMirror(sec.w, 1);
        sec.w = (sec.w + 1) & sec.ring.mask();
        x = y;
    }
    return x;
}

std::vector<float> renderBaseline(float size, float coef)
{
    BaselineSection bank[kNumSec];
    baselinePrepare(bank, kFs);
    baselineReset(bank);
    std::vector<float> buf(static_cast<std::size_t>(kTotal), 0.0f);
    buf[static_cast<std::size_t>(kSettle)] = 1.0f;
    for (int s = 0; s < kTotal; ++s)
        buf[static_cast<std::size_t>(s)] =
            baselineChain(bank, buf[static_cast<std::size_t>(s)], size, coef);
    return buf;
}

std::vector<float> renderTapered(float size, float coef)
{
    D d;
    d.prepare(kFs);
    const float amount = coef / D::kMaxCoefficient; // master g == coef
    d.setDiffusion(amount);
    d.setSize(size);
    d.setModDepthSamples(0.0f);   // no LFO: deterministic
    d.setModRateHz(0.5f);
    d.prime();                     // snap the size/coef smoothers

    std::vector<float> buf(static_cast<std::size_t>(kTotal), 0.0f);
    buf[static_cast<std::size_t>(kSettle)] = 1.0f;
    for (int off = 0; off < kTotal; off += kBlock)
    {
        const int n = std::min(kBlock, kTotal - off);
        d.processBlock(buf.data() + off, nullptr, n); // left bank only
    }
    return buf;
}

// RMS energy in dBFS over a window [w0, w1).
double windowRmsDb(const std::vector<float>& ir, int w0, int w1)
{
    double sumSq = 0.0;
    int count = 0;
    for (int n = w0; n < w1; ++n)
    {
        const double v = static_cast<double>(ir[static_cast<std::size_t>(n)]);
        sumSq += v * v;
        ++count;
    }
    if (count <= 0 || sumSq <= 0.0) return -999.0;
    const double rms = std::sqrt(sumSq / static_cast<double>(count));
    return 20.0 * std::log10(rms);
}

// Last sample whose magnitude exceeds -60 dBFS, past the impulse.
int lastAboveMinus60(const std::vector<float>& ir)
{
    constexpr double kThr = 0.001; // -60 dBFS
    int last = -1;
    for (int n = kSettle; n < kTotal; ++n)
        if (std::fabs(static_cast<double>(ir[static_cast<std::size_t>(n)])) >= kThr)
            last = n;
    return last;
}

} // namespace

int main()
{
    std::printf("=== Chronos diffuser_ring_check (S20: coefficient taper) ===\n");
    std::printf("fs=%.0f  size=1.0  baseline g=%.2f (untapered)  tapered g=%.2f\n\n",
                kFs, static_cast<double>(kBaselineCoef), static_cast<double>(kTaperedCoef));

    const float size = 1.0f;
    const auto baseIr = renderBaseline(size, kBaselineCoef);
    const auto tapIr  = renderTapered(size, kTaperedCoef);

    g_section = "finite";
    for (int n = kSettle; n < kTotal; ++n)
    {
        CHECK(std::isfinite(baseIr[static_cast<std::size_t>(n)]));
        CHECK(std::isfinite(tapIr[static_cast<std::size_t>(n)]));
    }

    // Transport at size 1.0 (full path). The late window starts past the
    // direct arrivals so the gate measures the recirculating tail.
    D d;
    d.prepare(kFs);
    const double transport = static_cast<double>(d.baseTransportSamples(size));
    const int w0 = static_cast<int>(4.0 * transport);
    const int w1 = kTotal;
    std::printf("transport=%.0f samples (%.1f ms); late window [%d, %d)\n\n",
                transport, transport / kFs * 1000.0, w0, w1);

    const double dbBase = windowRmsDb(baseIr, w0, w1);
    const double dbTap  = windowRmsDb(tapIr,  w0, w1);
    const double delta  = dbBase - dbTap;

    const int lastBase = lastAboveMinus60(baseIr);
    const int lastTap  = lastAboveMinus60(tapIr);
    std::printf("late-tail RMS:  baseline(0.92 untapered) = %8.2f dBFS\n", dbBase);
    std::printf("                tapered(0.78)            = %8.2f dBFS\n", dbTap);
    std::printf("                delta                    = %8.2f dB (gate >= %.1f)\n\n",
                delta, kGateDb);
    std::printf("-60 dBFS decay:  baseline last sample = %d (%.1f ms)\n",
                lastBase, lastBase < 0 ? 0.0 : static_cast<double>(lastBase) / kFs * 1000.0);
    std::printf("                 tapered  last sample = %d (%.1f ms)\n",
                lastTap, lastTap < 0 ? 0.0 : static_cast<double>(lastTap) / kFs * 1000.0);

    if (delta < kGateDb)
        FAIL("tail decay delta %.2f dB < %.1f dB (baseline %.2f, tapered %.2f)",
             delta, kGateDb, dbBase, dbTap);

    std::printf("\ntapered 0.78 rings at least 6 dB shorter than untapered 0.92: PASS\n");
    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

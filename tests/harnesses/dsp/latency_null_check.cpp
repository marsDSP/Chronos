// tests/harnesses/dsp/latency_null_check.cpp
#include "dsp/SimdDelayLine.h"
#include "dsp/StateVariable.h"
#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"
#include "dsp/align/SaturatorAlign.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {

constexpr int    kBudget = MarsDSP::Align::SaturatorAlign::kBudget;
constexpr double kFs     = 48000.0;
constexpr double kPi     = 3.14159265358979323846;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

struct Chain
{
    MarsDSP::Delays::SimdDelayLine delayLine;
    MarsDSP::Align::SaturatorAlign align;
    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2;
    using SVF = MarsDSP::Filters::SimdSVF;
    SVF hpf;
    SVF lpf;

    double fs         = kFs;
    int    mode       = 0;       // 0=Off, 1=ADAA1, 2=ADAA2
    float  mix        = 0.0f;    // 0..1
    float  driveLin   = 1.0f;    // linear gain into the saturator
    float  delaySamples = 0.0f;
    float  hpfFreq    = 10.0f;   // clamp-min (transparent)
    float  lpfFreq    = 23520.0f;// ~0.49*fs (transparent)
    bool   applySVF   = false;
    float  gainLin    = 1.0f;

    std::vector<float> wetL;

    void prepare(int maxBlock)
    {
        delayLine.prepare(fs, maxBlock, 5000.0f);
        delayLine.setInterpolation(MarsDSP::Delays::Interpolation::Lagrange5th);
        reset();
        wetL.resize(static_cast<std::size_t>(maxBlock));
    }

    void reset()
    {
        delayLine.reset();
        align.reset();
        adaa1.reset();
        adaa2.reset();
        hpf.reset();
        lpf.reset();
    }

    // Process one mono block: in[0..n) -> out[0..n).
    void process(const float* in, float* out, int n)
    {
        delayLine.process(in, nullptr, wetL.data(), nullptr, n, delaySamples, delaySamples);
        if (applySVF)
        {
            hpf.setCoeffForBlock(SVF::SVFType::HighPass, fs, hpfFreq, 0.7071, 0.0, n);
            lpf.setCoeffForBlock(SVF::SVFType::LowPass,  fs, lpfFreq, 0.7071, 0.0, n);
        }
        align.setMode(mode);

        const float theta    = mix * (0.5f * std::numbers::pi_v<float>);
        const float dryGain  = mmCos(theta);
        const float wetGain  = mmSin(theta);

        for (int s = 0; s < n; ++s)
        {
            const float dry0a = align.processDry(in[s]);   // dry delayed kBudget

            float sat0;
            switch (mode)
            {
                case 0:  sat0 = wetL[s]; break;
                case 1:  sat0 = static_cast<float>(adaa1.process(static_cast<double>(driveLin * wetL[s]))); break;
                default: sat0 = static_cast<float>(adaa2.process(static_cast<double>(driveLin * wetL[s]))); break;
            }

            sat0 = align.processWet(sat0);   // saturator -> alignment -> SVF

            float wetOut;
            if (applySVF)
            {
                const M128 wetV = MM(set_ps)(0.0f, 0.0f, 0.0f, sat0);
                const M128 hpV  = hpf.processBlockStep(wetV);
                const M128 lpV  = lpf.processBlockStep(hpV);
                alignas(16) std::array<float, 4> lanes;
                MM(storeu_ps)(lanes.data(), lpV);
                wetOut = lanes[0];
            }
            else
            {
                wetOut = sat0;
            }

            out[s] = (dry0a * dryGain + wetOut * wetGain) * gainLin;
        }
    }
};

// Coherent amplitude of the component at freqHz over x[start, start+len).
// freqHz must be an integer bin of len (freqHz*len/fs integer) for zero
// spectral leakage. Returns the sine amplitude (2*|X|/len).
double measureAmp(const std::vector<float>& x, double freqHz, int start, int len)
{
    const double omega = 2.0 * kPi * freqHz / kFs;
    double c = 0.0;
    double s = 0.0;
    for (int n = 0; n < len; ++n)
    {
        const double ang = omega * (start + n);
        const double v   = static_cast<double>(x[static_cast<std::size_t>(start + n)]);
        c += v * std::cos(ang);
        s += v * std::sin(ang);
    }
    return 2.0 * std::sqrt(c * c + s * s) / static_cast<double>(len);
}

// Test 1: full-dry null (mix = 0%), bit-exact, all three modes
void testFullDryNull()
{
    g_section = "full-dry null";
    constexpr int kN = 256;
    std::vector<float> in(static_cast<std::size_t>(kN));
    for (int n = 0; n < kN; ++n) in[n] = static_cast<float>(n + 1); // unique ramp, no -0.0

    std::vector<float> out0(static_cast<std::size_t>(kN));
    for (int mode = 0; mode <= 2; ++mode)
    {
        Chain c;
        c.mode = mode;
        c.mix  = 0.0f;          // 0% wet
        c.delaySamples = 0.0f;
        c.gainLin = 1.0f;
        c.applySVF = false;
        c.prepare(kN);
        c.process(in.data(), out0.data(), kN);

        for (int n = 0; n < kN; ++n)
        {
            const float exp = (n >= kBudget) ? in[static_cast<std::size_t>(n - kBudget)] : 0.0f;
            if (out0[n] != exp)
                FAIL("mode={} n={} got={} exp={} (full-dry null, bit-exact)",
                     mode, n, static_cast<double>(out0[n]), static_cast<double>(exp));
        }
    }

    // Mode-independence: the three mode outputs must be bit-identical (I2).
    std::vector<float> ref(static_cast<std::size_t>(kN));
    {
        Chain c; c.mode = 0; c.mix = 0.0f; c.delaySamples = 0.0f; c.gainLin = 1.0f;
        c.prepare(kN); c.process(in.data(), ref.data(), kN);
    }
    for (int mode = 1; mode <= 2; ++mode)
    {
        std::vector<float> o(static_cast<std::size_t>(kN));
        Chain c; c.mode = mode; c.mix = 0.0f; c.delaySamples = 0.0f; c.gainLin = 1.0f;
        c.prepare(kN); c.process(in.data(), o.data(), kN);
        for (int n = 0; n < kN; ++n)
            if (o[n] != ref[n])
                FAIL("mode-independence: mode={} n={} differs from mode 0 ({} != {})",
                     mode, n, static_cast<double>(o[n]), static_cast<double>(ref[n]));
    }
    std::println("full-dry null (mix=0%, bit-exact, all 3 modes identical): PASS");
}

void testFullWetOffDelay()
{
    g_section = "full-wet Off delay";
    constexpr int kN = 4096;
    constexpr float kDelay = 100.0f;   // integer -> exact Lagrange5th tap
    const double fHz = 1000.0;

    std::vector<float> in(static_cast<std::size_t>(kN));
    for (int n = 0; n < kN; ++n)
        in[n] = static_cast<float>(std::sin(2.0 * kPi * fHz * n / kFs));

    Chain c;
    c.mode = 0;                  // Off (saturator = identity)
    c.mix  = 1.0f;               // 100% wet
    c.delaySamples = kDelay;
    c.gainLin = 1.0f;
    c.applySVF = false;          // SVF group delay would shift the null
    c.prepare(kN);
    std::vector<float> out(static_cast<std::size_t>(kN));
    c.process(in.data(), out.data(), kN);

    const int totalDelay = static_cast<int>(kDelay) + kBudget;
    double maxErr = 0.0;
    for (int n = totalDelay; n < kN; ++n)
    {
        const double err = std::abs(static_cast<double>(out[n]) - static_cast<double>(in[n - totalDelay]));
        maxErr = std::max(maxErr, err);
    }
    if (maxErr > 1e-5)
        FAIL("full-wet Off max|out - in[delay+kBudget]| = {:.3} (> 1e-5)", maxErr);
    std::println("full-wet Off delay (delaySamples+kBudget = {}, max err {:.3} < 1e-5): PASS",
                totalDelay, maxErr);
}

double sweepAlignmentDeviation(int mode, float amplitude, double fLo, double fHi)
{
    constexpr int kN       = 16384;
    constexpr int kSkip    = 1024;   // FIR (16) + kBudget (8) + ADAA ring-out + margin
    constexpr int kMeasLen = kN - kSkip;
    const float theta4    = 0.5f * (0.5f * std::numbers::pi_v<float>);  // pi/4
    const float cosGain   = mmCos(theta4);
    const float sinGain   = mmSin(theta4);

    auto runChain = [&](float mix, double fHz) -> double
    {
        Chain c;
        c.mode = mode; c.mix = mix; c.delaySamples = 0.0f;
        c.driveLin = 1.0f; c.gainLin = 1.0f; c.applySVF = false;
        c.prepare(kN);
        std::vector<float> in(static_cast<std::size_t>(kN));
        for (int n = 0; n < kN; ++n)
            in[n] = static_cast<float>(amplitude * std::sin(2.0 * kPi * fHz * n / kFs));
        std::vector<float> out(static_cast<std::size_t>(kN));
        c.process(in.data(), out.data(), kN);
        return measureAmp(out, fHz, kSkip, kMeasLen);
    };

    const auto binToHz = [](int k) { return k * kFs / kMeasLen; };
    const int kLo = std::max(1, static_cast<int>(std::round(fLo * kMeasLen / kFs)));
    const int kHi = static_cast<int>(std::round(fHi * kMeasLen / kFs));

    double maxDevDb = 0.0;
    int worstK = kLo;
    for (int i = 0; i <= 80; ++i)
    {
        const double frac = static_cast<double>(i) / 80.0;
        const double kd = kLo * std::pow(static_cast<double>(kHi) / kLo, frac);
        const int k = std::clamp(static_cast<int>(std::round(kd)), kLo, kHi);
        const double fHz = binToHz(k);

        const double dry = runChain(0.0f, fHz);   // mix=0% -> dry path only
        const double wet = runChain(1.0f, fHz);   // mix=100% -> wet path only
        const double sum = runChain(0.5f, fHz);   // 50/50
        const double predicted = dry * cosGain + wet * sinGain;
        if (predicted <= 0.0) FAIL("mode={} {:.1} Hz predicted <= 0", mode, fHz);
        const double devDb = 20.0 * std::log10(sum / predicted);
        if (std::abs(devDb) > std::abs(maxDevDb)) { maxDevDb = devDb; worstK = k; }
    }
    std::println("    mode={} sweep [{:.1}, {:.1}] Hz: max dev {:.4} dB at {:.1} Hz (vs in-phase prediction)",
                mode, binToHz(kLo), binToHz(kHi), maxDevDb, binToHz(worstK));
    return std::abs(maxDevDb);
}

void test3_50pctOffSweep()
{
    g_section = "50% mix Off sweep";
    const double dev = sweepAlignmentDeviation(0, 1.0f, 20.0, 20000.0);
    CHECK(dev <= 0.05);
    std::println("50% mix Off sweep 20 Hz-20 kHz (gate +-0.05 dB vs aligned prediction, got {:.4}): PASS", dev);
}

void test4_50pctADAA()
{
    g_section = "50% mix ADAA1";
    {
        const double dev = sweepAlignmentDeviation(1, 0.01f, 20.0, 15000.0);
        CHECK(dev <= 0.2);
        std::println("50% mix ADAA1 sweep to 15 kHz (gate +-0.2 dB vs aligned prediction, got {:.4}): PASS", dev);
    }
    g_section = "50% mix ADAA2";
    {
        const double dev = sweepAlignmentDeviation(2, 0.01f, 20.0, 15000.0);
        CHECK(dev <= 0.1);
        std::println("50% mix ADAA2 sweep to 15 kHz (gate +-0.1 dB vs aligned prediction, got {:.4}): PASS", dev);
    }
}

} // namespace

int main()
{
    std::println("=== Chronos end-to-end latency/alignment harness ===");
    std::println("kBudget = {}  fs = {:.0} Hz\n", kBudget, kFs);

    testFullDryNull();
    testFullWetOffDelay();
    test3_50pctOffSweep();
    test4_50pctADAA();

    std::println("\n=== ALL PROPERTIES HELD ===");
    return 0;
}

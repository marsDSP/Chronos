/**
 * Adversarial-input safety net for the signal chain, hand-assembled in
 * ChronosEngine::process order. Gates: no NaN or Inf for any finite input;
 * output within the analytic bound; zeros in give +0.0f out bit-exact with
 * the dither off; injected non-finite samples are scrubbed and the output
 * rejoins the reference trajectory. See docs/dsp-notes.md for the bound
 * derivation. NDEBUG is forced on for this target: the battery intentionally
 * violates DSP header preconditions and needs IEEE propagation.
 */

#include "dsp/SimdDelayLine.h"
#include "dsp/StateVariable.h"
#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"
#include "dsp/align/SaturatorAlign.h"
#include "math/Trigonometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <vector>

namespace {

constexpr double kFs     = 48000.0;
constexpr double kPi     = std::numbers::pi_v<double>;
constexpr double kHpfHz  = 200.0;
constexpr double kLpfHz  = 8000.0;
constexpr double kSvfQ   = 0.7071;          // Butterworth, matches processor
constexpr int    kN      = 8192;            // samples per battery run
constexpr int    kBits   = 24;
constexpr int    kBudget = MarsDSP::Align::SaturatorAlign::kBudget;
constexpr auto   kInterp = MarsDSP::Delays::Interpolation::Lagrange5th;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// xorshift32, identical to ChronosProcessor::nextUniform.
inline float nextUniform(std::uint32_t& s) noexcept
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
}

// The chain under test
// Mirrors ChronosProcessor::processBlock's per-sample loop with flat
// parameters and a harness-only dither kill switch. Stereo always.
struct FuzzChain
{
    MarsDSP::Delays::SimdDelayLine delayLine;
    MarsDSP::Align::SaturatorAlign alignL;
    MarsDSP::Align::SaturatorAlign alignR;
    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1L;
    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1R;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2L;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2R;
    MarsDSP::Filters::SimdSVF hpf;
    MarsDSP::Filters::SimdSVF lpf;
    std::uint32_t xsL = 0x12345678u;
    std::uint32_t xsR = 0x9abcdef0u;
    std::vector<float> wetL;
    std::vector<float> wetR;
    std::vector<float> workL;
    std::vector<float> workR;

    int   mode    = 2;      // 0=Off, 1=ADAA1, 2=ADAA2
    float mixPct  = 100.0f; // 0..100
    float driveDb = 12.0f;
    float delayMs = 5.0f;
    bool  dither  = true;

    void prepare(int maxBlock)
    {
        delayLine.prepare(kFs, maxBlock, 5000.0f);   // matches the processor
        delayLine.setInterpolation(kInterp);
        wetL.resize(static_cast<std::size_t>(maxBlock));
        wetR.resize(static_cast<std::size_t>(maxBlock));
        reset();
    }

    void reset()
    {
        delayLine.reset();
        alignL.reset(); alignR.reset();
        adaa1L.reset(); adaa1R.reset();
        adaa2L.reset(); adaa2R.reset();
        hpf.reset(); lpf.reset();
    }

    [[nodiscard]] float driveLin() const { return std::pow(10.0f, driveDb / 20.0f); }
    [[nodiscard]] float delaySamples() const { return delayMs * 0.001f * static_cast<float>(kFs); }
    [[nodiscard]] static float lsb() { return std::ldexp(1.0f, 1 - kBits); }

    // Process n samples (stereo, identical L/R input) in block-sized chunks,
    // overwriting a work copy of the input exactly like the processor
    // overwrites the host buffer.
    void process(const float* in, float* out, int n, int block)
    {
        workL.assign(in, in + n);
        workR.assign(in, in + n);
        for (int off = 0; off < n; off += block)
        {
            const int len = std::min(block, n - off);
            processBlock(workL.data() + off, workR.data() + off, len);
        }
        std::memcpy(out, workL.data(), sizeof(float) * static_cast<std::size_t>(n));
        std::memcpy(out + n, workR.data(), sizeof(float) * static_cast<std::size_t>(n));
    }

    void processBlock(float* d0, float* d1, int n)
    {
        const float drv = driveLin();
        const float dly = delaySamples();
        delayLine.process(d0, d1, wetL.data(), wetR.data(), n, dly, dly);
        hpf.setCoeffForBlock(MarsDSP::Filters::SimdSVF::SVFType::HighPass, kFs, kHpfHz, kSvfQ, 0.0, n);
        lpf.setCoeffForBlock(MarsDSP::Filters::SimdSVF::SVFType::LowPass,  kFs, kLpfHz, kSvfQ, 0.0, n);
        alignL.setMode(mode);
        alignR.setMode(mode);

        const float theta = (mixPct * 0.01f) * (std::numbers::pi_v<float> * 0.5f);
        const float dryGain = mmCos(theta);
        const float wetGain = mmSin(theta);

        for (int s = 0; s < n; ++s)
        {
            const float dry0a = alignL.processDry(d0[s]);
            const float dry1a = alignR.processDry(d1[s]);
            const float wet0 = wetL[static_cast<std::size_t>(s)];
            const float wet1 = wetR[static_cast<std::size_t>(s)];

            float sat0;
            float sat1;
            switch (mode)
            {
                case 0:
                    sat0 = wet0; sat1 = wet1;
                    break;
                case 1:
                    sat0 = static_cast<float>(adaa1L.process(static_cast<double>(drv * wet0)));
                    sat1 = static_cast<float>(adaa1R.process(static_cast<double>(drv * wet1)));
                    break;
                default:
                    sat0 = static_cast<float>(adaa2L.process(static_cast<double>(drv * wet0)));
                    sat1 = static_cast<float>(adaa2R.process(static_cast<double>(drv * wet1)));
                    break;
            }

            sat0 = alignL.processWet(sat0);
            sat1 = alignR.processWet(sat1);

            const M128 wetV = MM(set_ps)(0.0f, 0.0f, sat1, sat0);
            const M128 hpV  = hpf.processBlockStep(wetV);
            const M128 lpV  = lpf.processBlockStep(hpV);
            alignas(16) float lanes[4];
            MM(storeu_ps)(lanes, lpV);

            d0[s] = dry0a * dryGain + lanes[0] * wetGain;
            d1[s] = dry1a * dryGain + lanes[1] * wetGain;

            const float lsbV = lsb();
            if (dither)
            {
                const float di0 = (nextUniform(xsL) - nextUniform(xsL)) * lsbV;
                const float di1 = (nextUniform(xsR) - nextUniform(xsR)) * lsbV;
                d0[s] = std::round((d0[s] + di0) / lsbV) * lsbV;
                d1[s] = std::round((d1[s] + di1) / lsbV) * lsbV;
            }
            else
            {
                d0[s] = std::round(d0[s] / lsbV) * lsbV;
                d1[s] = std::round(d1[s] / lsbV) * lsbV;
            }
        }
    }
};

// Assertion helpers
struct Cfg
{
    int mode; float mixPct; float driveDb; float delayMs;
};

// Analytic output bound - derivation in the header comment, assertion (b).
float outBound(const Cfg& c, float maxIn)
{
    const float theta  = (c.mixPct * 0.01f) * (std::numbers::pi_v<float> * 0.5f);
    const float dryB   = maxIn;
    const float satB   = (c.mode == 0) ? maxIn : 1.00001f;
    const float wetB   = 3.0f * satB;   // measured SVF cascade worst ~2.44x, see header
    return dryB * std::fabs(mmCos(theta)) + wetB * std::fabs(mmSin(theta))
           + 2.0f * FuzzChain::lsb() + 1e-6f;
}

// Run one finite-input battery case; assert (a) finite everywhere and
// (b) bounded, both channels, every sample.
void runFinite(const Cfg& c, const char* tag, const std::vector<float>& in,
               float maxIn, int block)
{
    g_section = tag;
    FuzzChain ch;
    ch.mode = c.mode; ch.mixPct = c.mixPct; ch.driveDb = c.driveDb; ch.delayMs = c.delayMs;
    ch.prepare(512);

    std::vector<float> out(2 * in.size());
    ch.process(in.data(), out.data(), static_cast<int>(in.size()), block);

    const float bound = outBound(c, maxIn);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const float v = out[i];
        if (!std::isfinite(v))
            FAIL("{}: non-finite output at i={} (mode={} mix={:.0} drive={:.0} delay={:.0})",
                 tag, i, c.mode, static_cast<double>(c.mixPct), static_cast<double>(c.driveDb), static_cast<double>(c.delayMs));
        if (std::fabs(v) > bound)
            FAIL("{}: |out|={} exceeds analytic bound {} at i={} (mode={} mix={:.0} drive={:.0} delay={:.0})",
                 tag, static_cast<double>(std::fabs(v)), static_cast<double>(bound), i,
                 c.mode, static_cast<double>(c.mixPct), static_cast<double>(c.driveDb), static_cast<double>(c.delayMs));
    }
}

// Signal generators (all length kN unless noted)
std::vector<float> genZeros()        { return std::vector<float>(kN, 0.0f); }

std::vector<float> genAlternating(double mag)
{
    std::vector<float> x(kN);
    for (int i = 0; i < kN; ++i)
        x[static_cast<std::size_t>(i)] = static_cast<float>((i & 1) ? -mag : mag);
    return x;
}

std::vector<float> genDc(double dc)
{
    std::vector<float> x(kN);
    for (int i = 0; i < kN; ++i)
        x[static_cast<std::size_t>(i)] = static_cast<float>(dc);
    return x;
}

std::vector<float> genSine(double amp, double fHz)
{
    std::vector<float> x(kN);
    for (int i = 0; i < kN; ++i)
        x[static_cast<std::size_t>(i)] = static_cast<float>(amp * std::sin(2.0 * kPi * fHz * static_cast<double>(i) / kFs));
    return x;
}

std::vector<float> genImpulse(double amp)
{
    std::vector<float> x(kN, 0.0f);
    x[0] = static_cast<float>(amp);
    return x;
}

// Alternating ±step at every block boundary over 16 blocks of `block` samples.
std::vector<float> genBlockSteps(double amp, int block)
{
    std::vector<float> x(static_cast<std::size_t>(block * 16));
    for (int b = 0; b < 16; ++b)
        for (int i = 0; i < block; ++i)
            x[static_cast<std::size_t>(b * block + i)] = static_cast<float>((b & 1) ? -amp : amp);
    return x;
}

// Test 1: zeros - homogeneity, bit-exact with dither off; lsb with dither
void test1_zeros(const Cfg& c)
{
    g_section = "zeros";
    const std::vector<float> in = genZeros();

    {
        FuzzChain ch;
        ch.mode = c.mode; ch.mixPct = c.mixPct; ch.driveDb = c.driveDb; ch.delayMs = c.delayMs;
        ch.dither = false;
        ch.prepare(512);
        std::vector<float> out(2 * kN);
        ch.process(in.data(), out.data(), kN, 256);
        for (std::size_t i = 0; i < out.size(); ++i)
            if (out[i] != 0.0f || std::signbit(out[i]))
                FAIL("zeros (dither off): out[{}] = {}, expected +0.0f bit-exact (mode={} mix={:.0})",
                     i, static_cast<double>(out[i]), c.mode, static_cast<double>(c.mixPct));
    }
    {
        FuzzChain ch;
        ch.mode = c.mode; ch.mixPct = c.mixPct; ch.driveDb = c.driveDb; ch.delayMs = c.delayMs;
        ch.prepare(512);   // dither on
        std::vector<float> out(2 * kN);
        ch.process(in.data(), out.data(), kN, 256);
        for (std::size_t i = 0; i < out.size(); ++i)
            if (std::fabs(out[i]) > FuzzChain::lsb())
                FAIL("zeros (dither on): |out[{}]| = {} > lsb = {} (mode={} mix={:.0})",
                     i, static_cast<double>(std::fabs(out[i])), static_cast<double>(FuzzChain::lsb()), c.mode, static_cast<double>(c.mixPct));
    }
}

// Test 4: NaN/inf injection - output stays finite, chain recovers
// Assertions (c1)/(c2) from the header. For each injection the reference is
// the identical run with the bad sample zeroed; dither streams match (same
// seeds, same draw count), so the comparison isolates the scrub/transit.
void test4_injection(const Cfg& c, long& runs)
{
    g_section = "NaN/inf injection";
    const float bads[3] = { std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN() };
    const std::array<int, 4> poss = {{ 0, 1, 2, kN / 2 }};
    const int delaySmp = static_cast<int>(std::ceil(c.delayMs * 0.001f * static_cast<float>(kFs)));

    for (float bad : bads)
        for (int pos : poss)
        {
            std::vector<float> in = genSine(0.5, 440.0);
            in[static_cast<std::size_t>(pos)] = bad;

            std::vector<float> refIn = in;
            refIn[static_cast<std::size_t>(pos)] = 0.0f;

            std::vector<float> out(2 * kN), ref(2 * kN);
            {
                FuzzChain ch;
                ch.mode = c.mode; ch.mixPct = c.mixPct; ch.driveDb = c.driveDb; ch.delayMs = c.delayMs;
                ch.prepare(512);
                ch.process(in.data(), out.data(), kN, 256);
            }
            {
                FuzzChain ch;
                ch.mode = c.mode; ch.mixPct = c.mixPct; ch.driveDb = c.driveDb; ch.delayMs = c.delayMs;
                ch.prepare(512);
                ch.process(refIn.data(), ref.data(), kN, 256);
            }
            ++runs;

            // (c1) every output sample stays finite, even during transit.
            for (std::size_t i = 0; i < out.size(); ++i)
                if (!std::isfinite(out[i]))
                    FAIL("injection: non-finite output at i={} (bad={} pos={} mode={} mix={:.0} drive={:.0} delay={:.0})",
                         i, static_cast<double>(bad), pos, c.mode, static_cast<double>(c.mixPct), static_cast<double>(c.driveDb), static_cast<double>(c.delayMs));

            // (c2) rejoin the reference trajectory after the flush+settle
            // window: pos + delaySamples + 300 samples, tol 1e-2 (header).
            const std::size_t thr = static_cast<std::size_t>(pos + delaySmp + 300);
            for (std::size_t i = thr; i < static_cast<std::size_t>(kN); ++i)
            {
                if (std::fabs(out[i] - ref[i]) > 1e-2f)
                    FAIL("injection: L not rejoined at i={} (thr={}): out={} ref={} (bad={} pos={} mode={} mix={:.0})",
                         i, thr, static_cast<double>(out[i]), static_cast<double>(ref[i]), static_cast<double>(bad), pos, c.mode, static_cast<double>(c.mixPct));
                if (std::fabs(out[static_cast<std::size_t>(kN) + i] - ref[static_cast<std::size_t>(kN) + i]) > 1e-2f)
                    FAIL("injection: R not rejoined at i={} (thr={}): out={} ref={} (bad={} pos={} mode={} mix={:.0})",
                         i, thr, static_cast<double>(out[static_cast<std::size_t>(kN) + i]), static_cast<double>(ref[static_cast<std::size_t>(kN) + i]),
                         static_cast<double>(bad), pos, c.mode, static_cast<double>(c.mixPct));
            }
        }
}

// Test 9: parameter step changes mid-stream
void test9_paramSteps()
{
    g_section = "parameter steps";
    FuzzChain ch;
    ch.mixPct = 50.0f;
    ch.driveDb = 0.0f;
    ch.delayMs = 5.0f;
    ch.mode = 0;
    ch.prepare(512);

    const int block = 256;
    const int nBlocks = 12;
    const std::vector<float> in = genSine(0.5, 440.0);

    for (int b = 0; b < nBlocks; ++b)
    {
        switch (b)   // harsher than the plugin: unsmoothed, instantaneous
        {
            case 2:  ch.delayMs = 50.0f; break;
            case 4:  ch.delayMs = 5.0f;  break;
            case 6:  ch.driveDb = 40.0f; break;
            case 8:  ch.mode = 1; break;
            case 10: ch.mode = 2; break;
            default: break;
        }
        const Cfg now { ch.mode, ch.mixPct, ch.driveDb, ch.delayMs };
        const float bound = outBound(now, 0.5f);

        std::vector<float> blk(static_cast<std::size_t>(block));
        std::copy_n(in.begin() + b * block, block, blk.begin());
        std::vector<float> wL(blk), wR(blk);
        ch.processBlock(wL.data(), wR.data(), block);

        for (int s = 0; s < block; ++s)
        {
            if (!std::isfinite(wL[static_cast<std::size_t>(s)]) || !std::isfinite(wR[static_cast<std::size_t>(s)]))
                FAIL("param steps: non-finite output at block {} sample {}", b, s);
            if (std::fabs(wL[static_cast<std::size_t>(s)]) > bound || std::fabs(wR[static_cast<std::size_t>(s)]) > bound)
                FAIL("param steps: |out| exceeds bound {} at block {} sample {} (mode={} drive={:.0})",
                     static_cast<double>(bound), b, s, ch.mode, static_cast<double>(ch.driveDb));
        }
    }
    std::println("parameter steps (delay 5→50→5 ms, drive 0→40 dB, mode 0→1→2): PASS");
}

} // namespace

int main()
{
    std::println("=== Chronos chain_fuzz_check: adversarial-input safety net ===");
    std::println("kBudget={}  fs={:.0}  run={} samples  bits={} (lsb={:.3})\n",
                kBudget, kFs, kN, kBits, static_cast<double>(FuzzChain::lsb()));

    const std::array<int, 3> modes = {{ 0, 1, 2 }};
    const std::array<float, 3> mixes = {{ 0.0f, 50.0f, 100.0f }};
    const std::array<float, 3> drives = {{ 0.0f, 12.0f, 40.0f }};
    const std::array<float, 2> delays = {{ 5.0f, 50.0f }};

    long injRuns = 0;
    long finiteRuns = 0;

    for (int mode : modes)
    for (float mix : mixes)
    for (float drive : drives)
    for (float delay : delays)
    {
        const Cfg c { mode, mix, drive, delay };

        // 1. Zeros: homogeneity (bit-exact, dither off) + dither bound.
        test1_zeros(c); ++finiteRuns;

        // 2. Denormals: ±1e-320 (flushes to ±0.0f in the float buffer -
        //    documents the flush) and ±1e-45f (true float denormal).
        runFinite(c, "denormals 1e-320", genAlternating(1e-320), 0.0f, 256); ++finiteRuns;
        runFinite(c, "denormals 1e-45f", genAlternating(1e-45), 1e-45f, 256); ++finiteRuns;

        // 3. Full scale ±1.0 and ±10.0 (hosts do send > 0 dBFS).
        for (double dc : { 1.0, -1.0, 10.0, -10.0 })
        {
            runFinite(c, "full-scale DC", genDc(dc), static_cast<float>(std::fabs(dc)), 256);
            ++finiteRuns;
        }

        // 5. DC at every power of two from 2^-30 to 2^0.
        for (int k = -30; k <= 0; ++k)
        {
            runFinite(c, "DC power-of-two", genDc(std::ldexp(1.0, k)),
                      static_cast<float>(std::ldexp(1.0, k)), 256);
            ++finiteRuns;
        }

        // 6. Nyquist alternation - fires ADAA2 branch (b). Block 61 forces
        //    the SIMD delay's sub-block scalar tail as well.
        runFinite(c, "nyquist alternation", genAlternating(1.0), 1.0f, 256); ++finiteRuns;
        runFinite(c, "nyquist alternation (blk=61)", genAlternating(1.0), 1.0f, 61); ++finiteRuns;

        // 7. Single-sample impulse of amplitude 1e6.
        runFinite(c, "impulse 1e6", genImpulse(1e6), 1e6f, 256); ++finiteRuns;

        // 8. Step discontinuities at block boundaries, two block sizes,
        //    plus a silence→step case.
        {
            const std::vector<float> s64  = genBlockSteps(0.8, 64);
            const std::vector<float> s256 = genBlockSteps(0.8, 256);
            runFinite(c, "block steps (blk=64)",  s64,  0.8f, 64);  ++finiteRuns;
            runFinite(c, "block steps (blk=256)", s256, 0.8f, 256); ++finiteRuns;
            std::vector<float> rise(kN, 0.0f);
            for (int i = 512; i < kN; ++i) rise[static_cast<std::size_t>(i)] = 0.8f;
            runFinite(c, "silence-to-step", rise, 0.8f, 256); ++finiteRuns;
        }

        // 4. ±inf / NaN at samples 0, 1, 2, mid-block - GATED (c1/c2).
        test4_injection(c, injRuns);
    }

    // 9. Parameter step changes mid-stream.
    test9_paramSteps();

    std::println("finite-input battery: {} runs × {}×2 samples — no NaN/inf, all within analytic bounds: PASS",
                finiteRuns, kN);
    std::println("zeros-in/zeros-out bit-exact (dither off) and |out| <= lsb (dither on): PASS");
    std::println("NaN/inf injection: {} runs — output always finite, rejoins reference within", injRuns);
    std::println("  pos + delaySamples + 300 samples (tol 1e-2): PASS");

    std::println("\n=== ALL FUZZ PROPERTIES HELD ===");
    return 0;
}

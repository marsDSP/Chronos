// tests/harnesses/dsp/oversized_block_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// invariant test: an oversized host block (larger than samplesPerBlock)
// chunked at wetBufCapacity (2x samplesPerBlock) must produce output
// BIT-IDENTICAL to the same signal pushed in samplesPerBlock-sized blocks.
//
// The chain is hand-assembled (SharedCode only, no JUCE AudioProcessor) in
// processBlock order: delay → drive → ADAA → align → HPF → LPF → crossfade
// → gain + TPDF dither + quantization. Parameters are FLAT (no smoothers).
// the test isolates the chunking invariant, not the smoothing behaviour.
// Dither seeds are fixed and reset for both paths, so the RNG state at each
// sample position is identical regardless of chunk count (the RNG advances
// 2x per sample per channel, and the total sample count is the same).
//
// Why bit-exactness holds for static parameters:
//  • Delay line: the posSmoother is at its target after the first block, so
//    processN produces no change — the ring read is identical regardless of
//    how many process() calls chunk the block.
//  • SVF: setCoeffForBlock computes da1 = (a1_new - a1_prior) / numSamples.
//    For static cutoff, a1_new == a1_prior, so da1 = 0 for every call after
//    the first (which has firstBlock=true, also da1=0). The coefficient ramp
//    is zero regardless of numSamples.
//  • ADAA, align, crossfade, dither: all purely per-sample — block-size-
//    independent.
//
// Matrix: blockSize ∈ {64, 65, 128, 129, 512, 1024} × adaaOrder ∈ {0,1,2}.
// Prepared at samplesPerBlock = 64, wetBufCapacity = 128.
// Reference: chunk at 64 (samplesPerBlock). Test: chunk at 128 (wetBufCapacity).
//
// Conventions (matching latency_null_check): plain main(), exit code, printf,
// always-live CHECK/FAIL. Links SharedCode only; no JUCE. No forced -O2 so
// header assert preconditions stay armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

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
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <vector>

namespace {

constexpr double kFs     = 48000.0;
constexpr double kPi     = std::numbers::pi_v<double>;
constexpr double kHpfHz  = 200.0;
constexpr double kLpfHz  = 8000.0;
constexpr double kSvfQ   = 0.7071;
constexpr int    kBits   = 24;
constexpr auto   kInterp = MarsDSP::Delays::Interpolation::Lagrange5th;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

inline float nextUniform(std::uint32_t& s) noexcept
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
}

// Hand-assembled chain with flat parameters, mirroring
// ChronosProcessor::processBlock's inner loop. Stereo always.
struct Chain
{
    MarsDSP::Delays::SimdDelayLine delayLine;
    MarsDSP::Align::SaturatorAlign alignL, alignR;
    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1L, adaa1R;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2L, adaa2R;
    MarsDSP::Filters::SimdSVF hpf, lpf;
    std::uint32_t xsL = 0x12345678u, xsR = 0x9abcdef0u;
    std::vector<float> wetL, wetR, workL, workR;
    int wetBufCapacity = 0;

    // Flat parameters.
    int   mode    = 2;
    float mixPct  = 100.0f;
    float driveDb = 12.0f;
    float delayMs = 5.0f;

    void prepare(int samplesPerBlock)
    {
        wetBufCapacity = std::max(1, 2 * samplesPerBlock);
        delayLine.prepare(kFs, wetBufCapacity, 5000.0f);
        delayLine.setInterpolation(kInterp);
        wetL.resize(static_cast<std::size_t>(wetBufCapacity));
        wetR.resize(static_cast<std::size_t>(wetBufCapacity));
        reset();
    }

    void reset()
    {
        delayLine.reset();
        alignL.reset(); alignR.reset();
        adaa1L.reset(); adaa1R.reset();
        adaa2L.reset(); adaa2R.reset();
        hpf.reset(); lpf.reset();
        xsL = 0x12345678u;
        xsR = 0x9abcdef0u;
    }

    [[nodiscard]] float driveLin() const { return std::pow(10.0f, driveDb / 20.0f); }
    [[nodiscard]] float delaySamples() const { return delayMs * 0.001f * static_cast<float>(kFs); }
    [[nodiscard]] static float lsb() { return std::ldexp(1.0f, 1 - kBits); }

    // Process one block of n samples (n <= wetBufCapacity), in-place on d0/d1.
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
        const float lsbV = lsb();

        for (int s = 0; s < n; ++s)
        {
            const float dry0a = alignL.processDry(d0[s]);
            const float dry1a = alignR.processDry(d1[s]);
            const float wet0 = wetL[static_cast<std::size_t>(s)];
            const float wet1 = wetR[static_cast<std::size_t>(s)];

            float sat0, sat1;
            switch (mode)
            {
                case 0:  sat0 = wet0; sat1 = wet1; break;
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

            const float sc0 = d0[s];
            const float di0 = (nextUniform(xsL) - nextUniform(xsL)) * lsbV;
            d0[s] = std::round((sc0 + di0) / lsbV) * lsbV;
            const float sc1 = d1[s];
            const float di1 = (nextUniform(xsR) - nextUniform(xsR)) * lsbV;
            d1[s] = std::round((sc1 + di1) / lsbV) * lsbV;
        }
    }

    // Process n samples (stereo, identical L/R input), chunking at chunkSize.
    // Mirrors ChronosProcessor::processBlock's chunking loop.
    void process(const float* in, float* out, int n, int chunkSize)
    {
        workL.assign(in, in + n);
        workR.assign(in, in + n);
        for (int off = 0; off < n; off += chunkSize)
        {
            const int chunk = std::min(chunkSize, n - off);
            processBlock(workL.data() + off, workR.data() + off, chunk);
        }
        std::memcpy(out, workL.data(), sizeof(float) * static_cast<std::size_t>(n));
        std::memcpy(out + n, workR.data(), sizeof(float) * static_cast<std::size_t>(n));
    }
};

// Test one (blockSize, mode) pair: oversized chunked at wetBufCapacity must
// equal the reference chunked at samplesPerBlock (64).
void testOne(int blockSize, int mode)
{
    g_section = "oversized block";
    constexpr int kSamplesPerBlock = 64;

    // Generate a test signal: sine + ramp (breaks symmetry so multiple taps
    // are nonzero — the test that catches indexing/flush bugs, not a DC null).
    std::vector<float> in(static_cast<std::size_t>(blockSize));
    for (int i = 0; i < blockSize; ++i)
        in[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::sin(0.3 * static_cast<double>(i)))
          + 0.3f * static_cast<float>(std::sin(1.1 * static_cast<double>(i)))
          + 0.01f * static_cast<float>(i);

    // Reference: chunk at samplesPerBlock (64).
    std::vector<float> ref(2 * blockSize);
    {
        Chain c;
        c.mode = mode;
        c.prepare(kSamplesPerBlock);
        c.process(in.data(), ref.data(), blockSize, kSamplesPerBlock);
    }

    // Test: chunk at wetBufCapacity (2 * 64 = 128) — one oversized block.
    std::vector<float> tst(2 * blockSize);
    {
        Chain c;
        c.mode = mode;
        c.prepare(kSamplesPerBlock);
        c.process(in.data(), tst.data(), blockSize, c.wetBufCapacity);
    }

    for (int i = 0; i < 2 * blockSize; ++i)
        if (ref[static_cast<std::size_t>(i)] != tst[static_cast<std::size_t>(i)])
            FAIL("blockSize=%d mode=%d i=%d: ref=%g tst=%g",
                 blockSize, mode, i,
                 (double)ref[static_cast<std::size_t>(i)],
                 (double)tst[static_cast<std::size_t>(i)]);

    std::printf("  blockSize=%-5d mode=%d: BIT-IDENTICAL: PASS\n", blockSize, mode);
}

} // namespace

int main()
{
    std::printf("=== Chronos oversized_block_check (S1) ===\n");
    std::printf("samplesPerBlock=64  wetBufCapacity=128  fs=%.0f  bits=%d\n\n", kFs, kBits);

    const int blockSizes[] = { 64, 65, 128, 129, 512, 1024 };
    const int modes[]      = { 0, 1, 2 };

    for (int bs : blockSizes)
        for (int m : modes)
            testOne(bs, m);

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

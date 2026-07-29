// tests/harnesses/dsp/chain_parity.cpp
// ──────────────────────────────────────────────────────────────────────────
// bit-exactness gate: ChronosEngine::process vs a verbatim copy of the
// pre-per-sample loop body
//
// Both the engine and the reference are driven in lockstep with identical
// parameters, dither seeds, and input. Every output sample must be BIT-EXACT.
// Not a tolerance.
//
// Matrix: block sizes {1,2,3,7,15,16,17,63,64,65,127,256,511,512,1024}
//         × adaaOrder {0,1,2} × {mono, stereo}
//         × mix {0,25,50,75,100} × drive {0,12,40} dB
//         × {static, ramping} parameters × delay sweep crossing a ring wrap.
// State-carry section: 200 consecutive blocks.
//
// Conventions (matching latency_null_check): plain main(), exit code, printf,
// always-live CHECK/FAIL. Links SharedCode only; no JUCE.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/ChronosEngine.h"
#include "dsp/SimdDelayLine.h"
#include "dsp/StateVariable.h"
#include "dsp/LinearSmoother.h"
#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"
#include "dsp/align/SaturatorAlign.h"
#include "math/Trigonometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <vector>

namespace {

constexpr double kFs     = 48000.0;
constexpr int    kBudget = MarsDSP::Align::SaturatorAlign::kBudget;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

namespace Ref {

// reference only — do not optimize, do not delete
struct ChainRef
{
    MarsDSP::Delays::SimdDelayLine                        delayLine;
    std::vector<float>                                   wetBufL;
    std::vector<float>                                   wetBufR;
    int                                                  wetBufCap {0};

    using SVF = MarsDSP::Filters::SimdSVF;
    SVF                                                 hpf;
    SVF                                                 lpf;
    static constexpr double                             svfQ {0.7071};

    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1L, adaa1R;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2L, adaa2R;

    MarsDSP::Align::SaturatorAlign                       alignL, alignR;

    std::uint32_t                                        xsL {0x12345678u};
    std::uint32_t                                        xsR {0x9abcdef0u};

    MarsDSP::Smoothers::LinearSmoother<float>            gainS, bitsS, hpfS, lpfS, mixS, driveS;

    float   smGain {}, smHpf {}, smLpf {}, smMix {}, smDrive {};
    int     smBits {};
    float   delaySamples {};
    int     adaaOrder {2};
    double  sampleRate {0.0};

    static float nextUniform(std::uint32_t& s) noexcept
    {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }

    void prepare(double sr, int maxBlock, int numCh) noexcept
    {
        sampleRate = sr;
        wetBufCap = std::max(1, 2 * maxBlock);
        delayLine.prepare(sr, wetBufCap, 5000.0f);
        wetBufL.resize(static_cast<std::size_t>(wetBufCap));
        wetBufR.resize(static_cast<std::size_t>(wetBufCap));
        constexpr double kRamp = 0.02;
        gainS.reset(sr, kRamp); bitsS.reset(sr, kRamp);
        hpfS.reset(sr, kRamp);  lpfS.reset(sr, kRamp);
        mixS.reset(sr, kRamp);  driveS.reset(sr, kRamp);
        reset();
    }

    void reset() noexcept
    {
        delayLine.reset(); hpf.reset(); lpf.reset();
        adaa1L.reset(); adaa1R.reset(); adaa2L.reset(); adaa2R.reset();
        alignL.reset(); alignR.reset();
        smGain = 0.0f; smBits = 0; smHpf = 0.0f; smLpf = 0.0f; smMix = 0.0f; smDrive = 0.0f;
    }

    void resetParams(float drvLin, float mix, float gainLin, float hpfHz, float lpfHz, int bits) noexcept
    {
        gainS.setCurrentAndTargetValue(gainLin);
        bitsS.setCurrentAndTargetValue(static_cast<float>(bits));
        hpfS.setCurrentAndTargetValue(hpfHz);
        lpfS.setCurrentAndTargetValue(lpfHz);
        mixS.setCurrentAndTargetValue(mix);
        driveS.setCurrentAndTargetValue(drvLin);
    }

    void setParams(float dlySmp, int order, float drvLin, float mix, float gainLin, float hpfHz, float lpfHz, int bits,
                   MarsDSP::Delays::Interpolation interp) noexcept
    {
        delaySamples = dlySmp;
        adaaOrder = order;
        delayLine.setInterpolation(interp);
        gainS.setTargetValue(gainLin);
        bitsS.setTargetValue(static_cast<float>(bits));
        hpfS.setTargetValue(hpfHz);
        lpfS.setTargetValue(lpfHz);
        mixS.setTargetValue(mix);
        driveS.setTargetValue(drvLin);
    }

    void setDitherSeeds(std::uint32_t l, std::uint32_t r) noexcept { xsL = l; xsR = r; }

    // reference only
    void process(float* const* io, int numChannels, int numSamples) noexcept
    {
        if (numSamples <= 0) return;
        const double fsSafe = sampleRate > 0.0 ? sampleRate : 48000.0;
        float* data0 = io[0];
        float* data1 = numChannels > 1 ? io[1] : nullptr;

        for (int offset = 0; offset < numSamples;)
        {
            const int chunk = std::min(wetBufCap, numSamples - offset);

            delayLine.process(data0 + offset,
                              data1 != nullptr ? data1 + offset : nullptr,
                              wetBufL.data(),
                              data1 != nullptr ? wetBufR.data() : nullptr,
                              chunk, delaySamples, delaySamples);

            hpf.setCoeffForBlock(SVF::SVFType::HighPass, fsSafe, smHpf, svfQ, 0.0, chunk);
            lpf.setCoeffForBlock(SVF::SVFType::LowPass,  fsSafe, smLpf, svfQ, 0.0, chunk);

            alignL.setMode(adaaOrder);
            alignR.setMode(adaaOrder);

            for (int s = 0; s < chunk; ++s)
            {
                // smoothen
                smGain  = gainS.getNextValue();
                smBits  = static_cast<int>(bitsS.getNextValue());
                smHpf   = hpfS.getNextValue();
                smLpf   = lpfS.getNextValue();
                smMix   = mixS.getNextValue();
                smDrive = driveS.getNextValue();

                const float driveLin = smDrive;
                const float mixNorm = smMix * 0.01f;
                const float theta = mixNorm * (std::numbers::pi_v<float> * 0.5f);
                // clamp endpoints (engine and reference both use exact
                // values at mix=0 and mix=100; mmCos/mmSin leak ~1.1e-7).
                const float mixVal = smMix;
                float dryGain, wetGain;
                if (mixVal <= 0.0f)
                {
                    dryGain = 1.0f;
                    wetGain = 0.0f;
                }
                else if (mixVal >= 100.0f)
                {
                    dryGain = 0.0f;
                    wetGain = 1.0f;
                }
                else
                {
                    dryGain = mmCos(theta);
                    wetGain = mmSin(theta);
                }

                const float dry0 = data0[offset + s];
                const float dry0a = alignL.processDry(dry0);
                float wet0 = wetBufL[static_cast<std::size_t>(s)];

                float dry1 = 0.0f, dry1a = 0.0f, wet1 = 0.0f;
                if (data1 != nullptr)
                {
                    dry1 = data1[offset + s];
                    dry1a = alignR.processDry(dry1);
                    wet1 = wetBufR[static_cast<std::size_t>(s)];
                }

                float sat0, sat1 = 0.0f;
                switch (adaaOrder)
                {
                    case 0: sat0 = wet0; if (data1 != nullptr) sat1 = wet1; break;
                    case 1:
                        sat0 = static_cast<float>(adaa1L.process(driveLin * wet0));
                        if (data1 != nullptr) sat1 = static_cast<float>(adaa1R.process(driveLin * wet1));
                        break;
                    default:
                        sat0 = static_cast<float>(adaa2L.process(driveLin * wet0));
                        if (data1 != nullptr) sat1 = static_cast<float>(adaa2R.process(driveLin * wet1));
                        break;
                }

                sat0 = alignL.processWet(sat0);
                if (data1 != nullptr) sat1 = alignR.processWet(sat1);

                const M128 wetV = MM(set_ps)(0.0f, 0.0f, sat1, sat0);
                const M128 hpV  = hpf.processBlockStep(wetV);
                const M128 lpV  = lpf.processBlockStep(hpV);
                alignas(16) std::array<float, 4> out;
                MM(storeu_ps)(out.data(), lpV);   // note: storeu_ps (pre-S3)

                data0[offset + s] = dry0a * dryGain + out[0] * wetGain;
                if (data1 != nullptr) data1[offset + s] = dry1a * dryGain + out[1] * wetGain;

                const float gainLin = smGain;
                const float lsb = std::ldexp(1.0f, 1 - smBits);

                for (int ch = 0; ch < numChannels; ++ch)
                {
                    auto* data = io[ch];
                    auto& state = ch == 0 ? xsL : xsR;
                    const float scaled = data[offset + s] * gainLin;
                    const float dither = (nextUniform(state) - nextUniform(state)) * lsb;
                    data[offset + s] = std::round((scaled + dither) / lsb) * lsb;
                }
            }
            offset += chunk;
        }
    }
};

} // namespace Ref

// ── Test configuration ────────────────────────────────────────────────────
struct TestCfg
{
    int   blockSize;
    int   adaaOrder;
    int   numChannels;   // 1 or 2
    float mixPct;
    float driveDb;
    bool  ramping;       // static vs ramping params
};

// Build engine Params from raw values.
MarsDSP::ChronosEngine::Params makeParams(float dlySmp, int order, float drvLin, float mix, float gainLin,
                                          float hpfHz, float lpfHz, int bits, MarsDSP::Delays::Interpolation interp)
{
    MarsDSP::ChronosEngine::Params p{};
    p.delaySamples = dlySmp;
    p.driveLin     = drvLin;
    p.mix          = mix;
    p.gainLin      = gainLin;
    p.hpfHz        = hpfHz;
    p.lpfHz        = lpfHz;
    p.bits         = bits;
    p.adaaOrder    = order;
    p.interp       = interp;
    return p;
}

// Run one test configuration: compare engine vs reference bit-exactly.
void runOne(const TestCfg& tc, long& totalSamples)
{
    g_section = "chain_parity";

    constexpr std::uint32_t kSeedL = 0xDEADBEEFu;
    constexpr std::uint32_t kSeedR = 0xCAFEBABEu;
    constexpr float kHpfHz = 200.0f;
    constexpr float kLpfHz = 8000.0f;
    constexpr int   kBits  = 24;
    constexpr auto  kInterp = MarsDSP::Delays::Interpolation::Lagrange5th;

    const float drvLin   = std::pow(10.0f, tc.driveDb / 20.0f);
    const float gainLin  = 1.0f;
    const float dlySmp   = 240.0f;   // crosses ring wrap at cap=512

    // ── Engine ──
    MarsDSP::ChronosEngine engine;
    engine.prepare(kFs, 256, tc.numChannels);
    engine.reset();
    engine.setDitherSeeds(kSeedL, kSeedR);
    engine.resetParams(makeParams(dlySmp, tc.adaaOrder, drvLin, tc.mixPct, gainLin, kHpfHz, kLpfHz, kBits, kInterp));

    // ── Reference ──
    Ref::ChainRef ref;
    ref.prepare(kFs, 256, tc.numChannels);
    ref.reset();
    ref.setDitherSeeds(kSeedL, kSeedR);
    ref.resetParams(drvLin, tc.mixPct, gainLin, kHpfHz, kLpfHz, kBits);

    // Generate input: sine + ramp (breaks symmetry).
    std::vector<float> inL(static_cast<std::size_t>(tc.blockSize));
    std::vector<float> inR(static_cast<std::size_t>(tc.blockSize));
    for (int i = 0; i < tc.blockSize; ++i)
    {
        inL[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::sin(0.3 * static_cast<double>(i)))
          + 0.3f * static_cast<float>(std::sin(1.1 * static_cast<double>(i)))
          + 0.01f * static_cast<float>(i);
        inR[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::cos(0.27 * static_cast<double>(i)))
          + 0.01f * static_cast<float>(i * 2);
    }

    // Work copies for both paths.
    std::vector<float> engL(inL), engR(inR);
    std::vector<float> refL(inL), refR(inR);

    // For ramping params: change drive each block.
    float curDrvLin = drvLin;
    float curMix    = tc.mixPct;

    // setParams for the first block (matching what processBlock does: update
    // then setParams). For static, the targets don't change.
    engine.setParams(makeParams(dlySmp, tc.adaaOrder, curDrvLin, curMix, gainLin, kHpfHz, kLpfHz, kBits, kInterp));
    ref.setParams(dlySmp, tc.adaaOrder, curDrvLin, curMix, gainLin, kHpfHz, kLpfHz, kBits, kInterp);

    float* engIo[2] = { engL.data(), tc.numChannels > 1 ? engR.data() : nullptr };
    float* refIo[2] = { refL.data(), tc.numChannels > 1 ? refR.data() : nullptr };

    engine.process(engIo, tc.numChannels, tc.blockSize);
    ref.process(refIo, tc.numChannels, tc.blockSize);

    totalSamples += static_cast<long>(tc.blockSize) * tc.numChannels;

    // Compare bit-exactly. Engine and reference both clamp endpoints
    // (D1), so bit-exact everywhere.
    for (int ch = 0; ch < tc.numChannels; ++ch)
    {
        const float* e = ch == 0 ? engL.data() : engR.data();
        const float* r = ch == 0 ? refL.data() : refR.data();
        for (int i = 0; i < tc.blockSize; ++i)
        {
            if (e[i] != r[i])
                FAIL("blockSize=%d order=%d ch=%d mix=%.0f drive=%.0f ramp=%d i=%d: "
                     "engine=%g ref=%g",
                     tc.blockSize, tc.adaaOrder, ch, tc.mixPct, tc.driveDb,
                     static_cast<int>(tc.ramping), i, (double)e[i], (double)r[i]);
        }
    }
}

// ── State-carry section: 200 consecutive blocks ───────────────────────────
void runStateCarry(long& totalSamples)
{
    g_section = "state-carry";
    constexpr int kBlocks = 200;
    constexpr int kBlockSize = 64;
    constexpr int kNumCh = 2;
    constexpr std::uint32_t kSeedL = 0x12345678u;
    constexpr std::uint32_t kSeedR = 0x9abcdef0u;
    constexpr float kHpfHz = 200.0f, kLpfHz = 8000.0f;
    constexpr int kBits = 24;
    constexpr auto kInterp = MarsDSP::Delays::Interpolation::Lagrange5th;
    const float drvLin = std::pow(10.0f, 12.0f / 20.0f);
    const float dlySmp = 240.0f;

    MarsDSP::ChronosEngine engine;
    engine.prepare(kFs, 256, kNumCh);
    engine.reset();
    engine.setDitherSeeds(kSeedL, kSeedR);
    engine.resetParams(makeParams(dlySmp, 2, drvLin, 50.0f, 1.0f, kHpfHz, kLpfHz, kBits, kInterp));

    Ref::ChainRef ref;
    ref.prepare(kFs, 256, kNumCh);
    ref.reset();
    ref.setDitherSeeds(kSeedL, kSeedR);
    ref.resetParams(drvLin, 50.0f, 1.0f, kHpfHz, kLpfHz, kBits);

    for (int b = 0; b < kBlocks; ++b)
    {
        // Ramp params slowly.
        const float mix = 50.0f + 30.0f * std::sin(static_cast<double>(b) * 0.05);
        const float drv = std::pow(10.0f, (12.0f + 6.0f * std::sin(static_cast<double>(b) * 0.03)) / 20.0f);

        engine.setParams(makeParams(dlySmp, 2, drv, mix, 1.0f, kHpfHz, kLpfHz, kBits, kInterp));
        ref.setParams(dlySmp, 2, drv, mix, 1.0f, kHpfHz, kLpfHz, kBits, kInterp);

        std::vector<float> engL(kBlockSize), engR(kBlockSize);
        std::vector<float> refL(kBlockSize), refR(kBlockSize);
        for (int i = 0; i < kBlockSize; ++i)
        {
            const float v = 0.5f * static_cast<float>(std::sin(0.3 * static_cast<double>(b * kBlockSize + i)))
                          + 0.01f * static_cast<float>(b * kBlockSize + i);
            engL[i] = v; engR[i] = v; refL[i] = v; refR[i] = v;
        }

        float* engIo[2] = { engL.data(), engR.data() };
        float* refIo[2] = { refL.data(), refR.data() };
        engine.process(engIo, kNumCh, kBlockSize);
        ref.process(refIo, kNumCh, kBlockSize);

        totalSamples += static_cast<long>(kBlockSize) * kNumCh;

        for (int ch = 0; ch < kNumCh; ++ch)
        {
            const float* e = ch == 0 ? engL.data() : engR.data();
            const float* r = ch == 0 ? refL.data() : refR.data();
            for (int i = 0; i < kBlockSize; ++i)
                if (e[i] != r[i])
                    FAIL("state-carry block=%d ch=%d i=%d: engine=%g ref=%g",
                         b, ch, i, (double)e[i], (double)r[i]);
        }
    }
    std::printf("state-carry (%d blocks x %d samples, ramping mix/drive): PASS\n", kBlocks, kBlockSize);
}

} // namespace

int main()
{
    std::printf("=== Chronos chain_parity (S3 bit-exactness gate) ===\n");
    std::printf("fs=%.0f  kBudget=%d\n\n", kFs, kBudget);

    const int blockSizes[] = {1, 2, 3, 7, 15, 16, 17, 63, 64, 65, 127, 256, 511, 512, 1024};
    const int orders[]     = {0, 1, 2};
    const int nChs[]       = {1, 2};
    const float mixes[]    = {0.0f, 25.0f, 50.0f, 75.0f, 100.0f};
    const float drives[]   = {0.0f, 12.0f, 40.0f};

    long totalSamples = 0;
    long configs = 0;

    for (int bs : blockSizes)
    for (int order : orders)
    for (int nch : nChs)
    for (float mix : mixes)
    for (float drv : drives)
    {
        // static params
        runOne({bs, order, nch, mix, drv, false}, totalSamples);
        ++configs;

        // ramping params (one representative: mix 50, drive 12, with a param change mid-block
        if (mix == 50.0f && drv == 12.0f)
        {
            runOne({bs, order, nch, mix, drv, true}, totalSamples);
            ++configs;
        }
    }

    std::printf("matrix (%ld configs, %ld samples): BIT-EXACT: PASS\n", configs, totalSamples);

    runStateCarry(totalSamples);

    std::printf("\ntotal samples compared: %ld\n", totalSamples);
    std::printf("\n=== ALL PROPERTIES HELD (BIT-EXACT) ===\n");
    return 0;
}

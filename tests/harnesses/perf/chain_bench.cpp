// tests/harnesses/perf/chain_bench.cpp
// ──────────────────────────────────────────────────────────────────────────
// Stage-attributed end-to-end throughput benchmark for the Chronos signal
// chain, hand-assembled in ChronosProcessor::processBlock order
// (delay → drive gain → ADAA → alignment → HPF → LPF → equal-power crossfade
// → output gain → TPDF dither → quantization), the way
// tests/harnesses/dsp/latency_null_check.cpp assembles it. Links SharedCode
// only — no JUCE AudioProcessor.
//
// For each of the 8 stages it reports ns/sample in isolation (streaming over
// the stage's own recorded real input), plus the full fused chain, over the
// matrix
//   adaaOrder {0,1,2} × mix {0,50,100} × blockSize {64,128,256,512} × {mono,stereo}
//
// Stand-ins and fidelity notes (read before comparing to the plugin):
//  * JUCE LinearSmoothedValue is replaced by LinRamp, a scalar linear ramp
//    between two endpoints. Every configuration uses equal endpoints (flat
//    ramp): the plugin's smoothers sit settled at their targets in the steady
//    state this benchmark measures. A mid-move parameter would add one FMA
//    per sample per parameter to the loops below.
//  * bits is read from a flat float ramp and truncated per sample, so the
//    per-sample ldexp (ChronosProcessor.cpp:247) is not loop-invariant and
//    codegen matches the plugin's.
//  * Isolated stages stream their ~2 MB recorded inputs from L3; the fused
//    chain is cache-resident. The stage sum therefore overestimates the full
//    chain. Attribution is still meaningful for the expensive stages.
//  * Isolated per-sample loops are compiled with auto-vectorization disabled
//    (CHRONOS_NO_VECTORIZE, same pragma as tan_bench) so they cost like their
//    scalar in-chain counterparts, and use one doNotOptimize barrier per
//    block rather than per sample — a per-sample barrier would exceed the
//    work of the cheapest stage (drive-gain is sub-ns/sample).
//  * The delay line is prepared outside the timed rep (its 4 MB ring
//    alloc+zero would pollute the measurement); its state carries across
//    reps, which is immaterial because the kernel cost is data-independent.
//    All other stages are reconstructed fresh inside each rep.
//  * No ScopedNoDenormals (JUCE-free harness); the input is a clean
//    0.5-amplitude sine pair, denormal-free.
//  * Representative fixed parameters (not matrix axes): fs 48 kHz, delay
//    347.5 samples (fractional → exercises Lagrange5th), drive 12 dB, output
//    gain 0 dB, bits 24, HPF 200 Hz, LPF 8 kHz, Q 0.7071.
//
// Informational only: no pass/fail gate (a timing gate fires on machine
// noise). Exit non-zero only if the untimed recording pass produces a NaN/inf,
// which would mean the chain was assembled wrong.
//
// Build: cmake -S . -B build -DBUILD_TEST_HARNESSES=ON
//        cmake --build build --target chain_bench
// Run:   ./build/tests/chain_bench [--csv tests/logs/<arch>/chain_bench.csv]
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/ChronosEngine.h"
#include "dsp/SimdDelayLine.h"
#include "dsp/StateVariable.h"
#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"
#include "dsp/align/SaturatorAlign.h"
#include "math/Trigonometry.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace {

constexpr double kFs           = 48000.0;
constexpr double kPi           = std::numbers::pi_v<double>;
constexpr float  kDelaySamples = 347.5f;   // fractional → Lagrange5th path
constexpr float  kDriveDb      = 12.0f;
constexpr float  kGainDb       = 0.0f;
constexpr int    kBits         = 24;
constexpr double kHpfHz        = 200.0;
constexpr double kLpfHz        = 8000.0;
constexpr double kSvfQ         = 0.7071;   // Butterworth, matches processor
constexpr int    kSamples      = 1 << 19;  // 524288 per rep (÷ every block size)
constexpr int    kReps         = 5;
constexpr int    kMaxBlock     = 512;

using Clock = std::chrono::steady_clock;

// ── tan_bench timing idiom ────────────────────────────────────────────────
#ifdef __clang__
#define CHRONOS_NO_VECTORIZE _Pragma("clang loop vectorize(disable)")
#else
#define CHRONOS_NO_VECTORIZE
#endif

#if defined(__clang__) || defined(__GNUC__)
template <class T>
inline void doNotOptimize(T const& v) noexcept
{
    asm volatile("" : : "r,m"(v) : "memory");
}
#else
template <class T>
inline void doNotOptimize(T const& v) noexcept
{
    volatile T sink = v; (void)sink;   // MSVC fallback (weaker; no x64 inline asm)
}
#endif

// Run fn (which performs `ops` samples of work) `reps` times; return the best
// (min) ns/sample and sink the accumulators into sinkOut so loops stay live.
template <class Fn>
double benchNsPerOp(Fn fn, std::size_t ops, std::size_t reps, double& sinkOut)
{
    double best  = std::numeric_limits<double>::infinity();
    double total = 0.0;
    for (std::size_t r = 0; r < reps; ++r)
    {
        const auto t0  = Clock::now();
        const double a = fn();
        const auto t1  = Clock::now();
        total += a;
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    sinkOut = total;
    return best / static_cast<double>(ops);
}

// Scalar linear ramp between two endpoints — stand-in for JUCE
// LinearSmoothedValue (see header comment; endpoints are always equal here).
struct LinRamp
{
    float cur = 0.0f, step = 0.0f;
    void set(float from, float to, int n) noexcept
    {
        cur  = from;
        step = n > 1 ? (to - from) / static_cast<float>(n - 1) : 0.0f;
    }
    float next() noexcept { const float v = cur; cur += step; return v; }
};

// xorshift32, identical to ChronosProcessor::nextUniform.
inline float nextUniform(std::uint32_t& s) noexcept
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
}

// ── Configuration and buffers ─────────────────────────────────────────────
struct Cfg
{
    int   mode;   // 0=Off, 1=ADAA1, 2=ADAA2
    float mix;    // 0..100
    int   block;  // samples per block
    int   ch;     // 1=mono, 2=stereo
};

// Every per-stage intermediate, recorded untimed by the same loops that are
// later timed (template<bool Store> below), so each isolated stage streams
// exactly the input the fused chain would hand it.
struct Bufs
{
    std::vector<float> inL, inR;   // test signal
    std::vector<float> dL, dR;     // delay out      = drive-gain in
    std::vector<float> vL, vR;     // drive out      = adaa in
    std::vector<float> aL, aR;     // adaa out       = align wet in
    std::vector<float> wL, wR;     // align wet out  = hpf in
    std::vector<float> hL, hR;     // hpf out        = lpf in
    std::vector<float> pL, pR;     // lpf out        = crossfade wet in
    std::vector<float> yL, yR;     // align dry out  = crossfade dry in
    std::vector<float> mL, mR;     // crossfade out  = tail in

    void alloc(int n)
    {
        inL.resize(static_cast<std::size_t>(n)); inR.resize(static_cast<std::size_t>(n));
        dL.resize(static_cast<std::size_t>(n));  dR.resize(static_cast<std::size_t>(n));
        vL.resize(static_cast<std::size_t>(n));  vR.resize(static_cast<std::size_t>(n));
        aL.resize(static_cast<std::size_t>(n));  aR.resize(static_cast<std::size_t>(n));
        wL.resize(static_cast<std::size_t>(n));  wR.resize(static_cast<std::size_t>(n));
        hL.resize(static_cast<std::size_t>(n));  hR.resize(static_cast<std::size_t>(n));
        pL.resize(static_cast<std::size_t>(n));  pR.resize(static_cast<std::size_t>(n));
        yL.resize(static_cast<std::size_t>(n));  yR.resize(static_cast<std::size_t>(n));
        mL.resize(static_cast<std::size_t>(n));  mR.resize(static_cast<std::size_t>(n));
    }
};

constexpr auto kInterp = MarsDSP::Delays::Interpolation::Lagrange5th;

// ── Stage: delay (block-rate SimdDelayLine) ───────────────────────────────
// dl is prepared by the caller, outside the timed rep (see header comment).
template <bool Store>
double stageDelayBody(const Cfg& c, const Bufs& b, Bufs* rec,
                      MarsDSP::Delays::SimdDelayLine& dl,
                      float* oL, float* oR)
{
    double acc = 0.0;
    for (int off = 0; off < kSamples; off += c.block)
    {
        const int n = c.block;
        dl.process(b.inL.data() + off,
                   c.ch > 1 ? b.inR.data() + off : nullptr,
                   oL, c.ch > 1 ? oR : nullptr,
                   n, kDelaySamples, kDelaySamples);
        if constexpr (Store)
        {
            std::memcpy(rec->dL.data() + off, oL, sizeof(float) * static_cast<std::size_t>(n));
            if (c.ch > 1)
                std::memcpy(rec->dR.data() + off, oR, sizeof(float) * static_cast<std::size_t>(n));
        }
        else
        {
            acc += oL[n - 1];
            if (c.ch > 1) acc += oR[n - 1];
            doNotOptimize(acc);
        }
    }
    return acc;
}

// ── Stage: drive-gain (per-sample scalar multiply, flat ramp) ─────────────
template <bool Store>
double stageDrive(const Cfg& c, const Bufs& b, Bufs* rec, float driveLin)
{
    double acc = 0.0;
    LinRamp ramp; ramp.set(driveLin, driveLin, kSamples);
    for (int off = 0; off < kSamples; off += c.block)
    {
        CHRONOS_NO_VECTORIZE
        for (int s = 0; s < c.block; ++s)
        {
            const int i = off + s;
            const float g = ramp.next();
            const float v0 = g * b.dL[static_cast<std::size_t>(i)];
            if constexpr (Store)
            {
                rec->vL[static_cast<std::size_t>(i)] = v0;
                if (c.ch > 1) rec->vR[static_cast<std::size_t>(i)] = g * b.dR[static_cast<std::size_t>(i)];
            }
            else
            {
                acc += v0;
                if (c.ch > 1) acc += g * b.dR[static_cast<std::size_t>(i)];
            }
        }
        if constexpr (!Store) doNotOptimize(acc);
    }
    return acc;
}

// ── Stage: adaa (mode-dispatched saturator) ───────────────────────────────
template <bool Store>
double stageAdaa(const Cfg& c, const Bufs& b, Bufs* rec)
{
    using namespace MarsDSP::Nonlinear;
    double acc = 0.0;
    ADAA1<TanhNL> a1L, a1R; a1L.reset(); a1R.reset();
    ADAA2<TanhNL> a2L, a2R; a2L.reset(); a2R.reset();

    for (int off = 0; off < kSamples; off += c.block)
    {
        CHRONOS_NO_VECTORIZE
        for (int s = 0; s < c.block; ++s)
        {
            const int i = off + s;
            const std::size_t u = static_cast<std::size_t>(i);
            float v0 = 0.0f, v1 = 0.0f;
            switch (c.mode)   // read once per block, like the processor
            {
                case 0:
                    v0 = b.vL[u];
                    if (c.ch > 1) v1 = b.vR[u];
                    break;
                case 1:
                    v0 = static_cast<float>(a1L.process(static_cast<double>(b.vL[u])));
                    if (c.ch > 1) v1 = static_cast<float>(a1R.process(static_cast<double>(b.vR[u])));
                    break;
                default:
                    v0 = static_cast<float>(a2L.process(static_cast<double>(b.vL[u])));
                    if (c.ch > 1) v1 = static_cast<float>(a2R.process(static_cast<double>(b.vR[u])));
                    break;
            }
            if constexpr (Store)
            {
                rec->aL[u] = v0;
                if (c.ch > 1) rec->aR[u] = v1;
            }
            else
            {
                acc += v0 + v1;
            }
        }
        if constexpr (!Store) doNotOptimize(acc);
    }
    return acc;
}

// ── Stage: align (dry + wet SaturatorAlign paths) ─────────────────────────
template <bool Store>
double stageAlign(const Cfg& c, const Bufs& b, Bufs* rec)
{
    double acc = 0.0;
    MarsDSP::Align::SaturatorAlign alL, alR;
    alL.reset(); alR.reset();

    for (int off = 0; off < kSamples; off += c.block)
    {
        alL.setMode(c.mode);   // once per block, like the processor
        alR.setMode(c.mode);
        CHRONOS_NO_VECTORIZE
        for (int s = 0; s < c.block; ++s)
        {
            const int i = off + s;
            const std::size_t u = static_cast<std::size_t>(i);
            const float d0 = alL.processDry(b.inL[u]);
            const float w0 = alL.processWet(b.aL[u]);
            float d1 = 0.0f, w1 = 0.0f;
            if (c.ch > 1)
            {
                d1 = alR.processDry(b.inR[u]);
                w1 = alR.processWet(b.aR[u]);
            }
            if constexpr (Store)
            {
                rec->yL[u] = d0; rec->wL[u] = w0;
                if (c.ch > 1) { rec->yR[u] = d1; rec->wR[u] = w1; }
            }
            else
            {
                acc += d0 + w0 + d1 + w1;
            }
        }
        if constexpr (!Store) doNotOptimize(acc);
    }
    return acc;
}

// ── Stage: one SVF (hp or lp), stereo packed into lanes 0,1 ───────────────
template <bool Store>
double stageSvf(const Cfg& c, const std::vector<float>& xL, const std::vector<float>& xR,
                std::vector<float>* oL, std::vector<float>* oR,
                MarsDSP::Filters::SimdSVF::SVFType type, double freqHz)
{
    double acc = 0.0;
    MarsDSP::Filters::SimdSVF svf;
    svf.reset();

    for (int off = 0; off < kSamples; off += c.block)
    {
        svf.setCoeffForBlock(type, kFs, freqHz, kSvfQ, 0.0, c.block);
        for (int s = 0; s < c.block; ++s)
        {
            const std::size_t u = static_cast<std::size_t>(off + s);
            const float r = c.ch > 1 ? xR[u] : 0.0f;
            const M128 in  = MM(set_ps)(0.0f, 0.0f, r, xL[u]);
            const M128 out = svf.processBlockStep(in);
            alignas(16) float lanes[4];
            MM(storeu_ps)(lanes, out);
            if constexpr (Store)
            {
                (*oL)[u] = lanes[0];
                if (c.ch > 1) (*oR)[u] = lanes[1];
            }
            else
            {
                acc += lanes[0] + lanes[1];
            }
        }
        if constexpr (!Store) doNotOptimize(acc);
    }
    return acc;
}

// ── Stage: equal-power crossfade (minimax mmCos/mmSin, flat mix ramp) ─────
template <bool Store>
double stageXfade(const Cfg& c, const Bufs& b, Bufs* rec)
{
    double acc = 0.0;
    LinRamp ramp; ramp.set(c.mix, c.mix, kSamples);

    for (int off = 0; off < kSamples; off += c.block)
    {
        CHRONOS_NO_VECTORIZE
        for (int s = 0; s < c.block; ++s)
        {
            const std::size_t u = static_cast<std::size_t>(off + s);
            const float theta = (ramp.next() * 0.01f) * (std::numbers::pi_v<float> * 0.5f);
            const float dryGain = mmCos(theta);
            const float wetGain = mmSin(theta);
            const float v0 = b.yL[u] * dryGain + b.pL[u] * wetGain;
            if constexpr (Store)
            {
                rec->mL[u] = v0;
                if (c.ch > 1) rec->mR[u] = b.yR[u] * dryGain + b.pR[u] * wetGain;
            }
            else
            {
                acc += v0;
                if (c.ch > 1) acc += b.yR[u] * dryGain + b.pR[u] * wetGain;
            }
        }
        if constexpr (!Store) doNotOptimize(acc);
    }
    return acc;
}

// ── Stage: output gain + TPDF dither + quantization ───────────────────────
template <bool Store>
double stageTail(const Cfg& c, const Bufs& b, Bufs* rec, float gainLin, std::uint32_t& xsL, std::uint32_t& xsR)
{
    double acc = 0.0;
    LinRamp gainR; gainR.set(gainLin, gainLin, kSamples);
    LinRamp bitsR; bitsR.set(static_cast<float>(kBits), static_cast<float>(kBits), kSamples);

    for (int off = 0; off < kSamples; off += c.block)
    {
        CHRONOS_NO_VECTORIZE
        for (int s = 0; s < c.block; ++s)
        {
            const std::size_t u = static_cast<std::size_t>(off + s);
            const float g   = gainR.next();
            const float lsb = std::ldexp(1.0f, 1 - static_cast<int>(bitsR.next()));
            const float sc0 = b.mL[u] * g;
            const float di0 = (nextUniform(xsL) - nextUniform(xsL)) * lsb;
            const float v0  = std::round((sc0 + di0) / lsb) * lsb;
            if constexpr (Store)
            {
                (void)v0;   // tail output is the final buffer; nothing records it
            }
            else
            {
                acc += v0;
            }
            if (c.ch > 1)
            {
                const float sc1 = b.mR[u] * g;
                const float di1 = (nextUniform(xsR) - nextUniform(xsR)) * lsb;
                const float v1  = std::round((sc1 + di1) / lsb) * lsb;
                if constexpr (!Store) acc += v1;
            }
        }
        if constexpr (!Store) doNotOptimize(acc);
    }
    return acc;
}

// ── Recording pass (untimed): run the same Store=true loops in chain order ─
// Returns false if any intermediate went NaN/inf (chain assembled wrong).
bool recordAll(const Cfg& c, Bufs& b, float driveLin, float gainLin)
{
    MarsDSP::Delays::SimdDelayLine dl;
    dl.prepare(kFs, kMaxBlock, 5000.0f);
    dl.setInterpolation(kInterp);
    dl.reset();
    std::vector<float> oL(static_cast<std::size_t>(kMaxBlock)), oR(static_cast<std::size_t>(kMaxBlock));
    stageDelayBody<true>(c, b, &b, dl, oL.data(), oR.data());
    stageDrive<true>(c, b, &b, driveLin);
    stageAdaa<true>(c, b, &b);
    stageAlign<true>(c, b, &b);
    stageSvf<true>(c, b.wL, b.wR, &b.hL, &b.hR, MarsDSP::Filters::SimdSVF::SVFType::HighPass, kHpfHz);
    stageSvf<true>(c, b.hL, b.hR, &b.pL, &b.pR, MarsDSP::Filters::SimdSVF::SVFType::LowPass, kLpfHz);
    stageXfade<true>(c, b, &b);

    for (int i = 0; i < kSamples; ++i)
    {
        const std::size_t u = static_cast<std::size_t>(i);
        if (!std::isfinite(b.mL[u])) return false;
        if (c.ch > 1 && !std::isfinite(b.mR[u])) return false;
    }
    (void)gainLin;
    return true;
}

// ── Full fused chain (processBlock minus JUCE), state prepared outside ─────
struct FullChain
{
    MarsDSP::Delays::SimdDelayLine delayLine;
    MarsDSP::Align::SaturatorAlign alignL, alignR;
    MarsDSP::Nonlinear::ADAA1<MarsDSP::Nonlinear::TanhNL> adaa1L, adaa1R;
    MarsDSP::Nonlinear::ADAA2<MarsDSP::Nonlinear::TanhNL> adaa2L, adaa2R;
    MarsDSP::Filters::SimdSVF hpf, lpf;
    std::uint32_t xsL = 0x12345678u, xsR = 0x9abcdef0u;   // fixed seeds: deterministic bench
    std::vector<float> wetL, wetR;
    std::vector<float> workL, workR;   // in-place work copy of the input block span

    void prepare()
    {
        delayLine.prepare(kFs, kMaxBlock, 5000.0f);
        delayLine.setInterpolation(kInterp);
        delayLine.reset();
        alignL.reset(); alignR.reset();
        adaa1L.reset(); adaa1R.reset();
        adaa2L.reset(); adaa2R.reset();
        hpf.reset(); lpf.reset();
        wetL.resize(static_cast<std::size_t>(kMaxBlock));
        wetR.resize(static_cast<std::size_t>(kMaxBlock));
        workL.resize(static_cast<std::size_t>(kSamples));
        workR.resize(static_cast<std::size_t>(kSamples));
    }

    // One rep: process kSamples in c.block-sized blocks. Mirrors
    // ChronosProcessor::processBlock's per-sample loop (flat LinRamp stand-ins).
    double run(const Cfg& c, const Bufs& b, float driveLin, float gainLin)
    {
        std::memcpy(workL.data(), b.inL.data(), sizeof(float) * static_cast<std::size_t>(kSamples));
        if (c.ch > 1)
            std::memcpy(workR.data(), b.inR.data(), sizeof(float) * static_cast<std::size_t>(kSamples));

        double acc = 0.0;
        LinRamp driveR; driveR.set(driveLin, driveLin, kSamples);
        LinRamp mixR;   mixR.set(c.mix, c.mix, kSamples);
        LinRamp gainR;  gainR.set(gainLin, gainLin, kSamples);
        LinRamp bitsR;  bitsR.set(static_cast<float>(kBits), static_cast<float>(kBits), kSamples);

        for (int off = 0; off < kSamples; off += c.block)
        {
            const int n = c.block;
            float* d0 = workL.data() + off;
            float* d1 = c.ch > 1 ? workR.data() + off : nullptr;

            delayLine.process(d0, d1, wetL.data(), c.ch > 1 ? wetR.data() : nullptr,
                              n, kDelaySamples, kDelaySamples);
            hpf.setCoeffForBlock(MarsDSP::Filters::SimdSVF::SVFType::HighPass, kFs, kHpfHz, kSvfQ, 0.0, n);
            lpf.setCoeffForBlock(MarsDSP::Filters::SimdSVF::SVFType::LowPass,  kFs, kLpfHz, kSvfQ, 0.0, n);
            alignL.setMode(c.mode);
            alignR.setMode(c.mode);

            for (int s = 0; s < n; ++s)
            {
                const float drv = driveR.next();
                const float theta = (mixR.next() * 0.01f) * (std::numbers::pi_v<float> * 0.5f);
                const float dryGain = mmCos(theta);
                const float wetGain = mmSin(theta);

                const float dry0a = alignL.processDry(d0[s]);
                const float wet0  = wetL[static_cast<std::size_t>(s)];
                float dry1a = 0.0f, wet1 = 0.0f;
                if (d1 != nullptr)
                {
                    dry1a = alignR.processDry(d1[s]);
                    wet1  = wetR[static_cast<std::size_t>(s)];
                }

                float sat0, sat1 = 0.0f;
                switch (c.mode)
                {
                    case 0:
                        sat0 = wet0;
                        if (d1 != nullptr) sat1 = wet1;
                        break;
                    case 1:
                        sat0 = static_cast<float>(adaa1L.process(static_cast<double>(drv * wet0)));
                        if (d1 != nullptr) sat1 = static_cast<float>(adaa1R.process(static_cast<double>(drv * wet1)));
                        break;
                    default:
                        sat0 = static_cast<float>(adaa2L.process(static_cast<double>(drv * wet0)));
                        if (d1 != nullptr) sat1 = static_cast<float>(adaa2R.process(static_cast<double>(drv * wet1)));
                        break;
                }

                sat0 = alignL.processWet(sat0);
                if (d1 != nullptr) sat1 = alignR.processWet(sat1);

                const M128 wetV = MM(set_ps)(0.0f, 0.0f, sat1, sat0);
                const M128 hpV  = hpf.processBlockStep(wetV);
                const M128 lpV  = lpf.processBlockStep(hpV);
                alignas(16) float out[4];
                MM(storeu_ps)(out, lpV);

                d0[s] = dry0a * dryGain + out[0] * wetGain;
                if (d1 != nullptr) d1[s] = dry1a * dryGain + out[1] * wetGain;

                const float g   = gainR.next();
                const float lsb = std::ldexp(1.0f, 1 - static_cast<int>(bitsR.next()));

                const float sc0 = d0[s] * g;
                const float di0 = (nextUniform(xsL) - nextUniform(xsL)) * lsb;
                d0[s] = std::round((sc0 + di0) / lsb) * lsb;
                acc += d0[s];
                if (d1 != nullptr)
                {
                    const float sc1 = d1[s] * g;
                    const float di1 = (nextUniform(xsR) - nextUniform(xsR)) * lsb;
                    d1[s] = std::round((sc1 + di1) / lsb) * lsb;
                    acc += d1[s];
                }
            }
            doNotOptimize(acc);
        }
        return acc;
    }
};

// ── Real engine row: ChronosEngine::process end-to-end ───────────────────
// chain_bench's FullChain hand-assembly is a single fused per-sample loop —
// it does NOT model the engine's stage-split block structure (scratch spans
// between block loops). C10 (loop fusion inside the engine) is therefore
// invisible to the FullChain number; this row times the real engine so
// engine-structure changes have a measurement vehicle. Parameters mirror
// the FullChain config (feedback 0, diffuser off, fixed dither seeds); the
// input memcpy sits inside the timed rep, matching FullChain::run.
double benchEngine(const Cfg& c, const Bufs& b, float driveLin, float gainLin,
                   double& sink)
{
    MarsDSP::ChronosEngine eng;
    eng.prepare(kFs, c.block, c.ch);
    eng.setDitherSeeds(0x12345678u, 0x9abcdef0u);

    MarsDSP::ChronosEngine::Params ep{};
    ep.delaySamples = kDelaySamples;
    ep.driveLin     = driveLin;
    ep.mix          = c.mix;
    ep.gainLin      = gainLin;
    ep.hpfHz        = static_cast<float>(kHpfHz);
    ep.lpfHz        = static_cast<float>(kLpfHz);
    ep.bits         = kBits;
    ep.adaaOrder    = c.mode;
    ep.interp       = kInterp;
    eng.resetParams(ep);

    std::vector<float> eL(static_cast<std::size_t>(kSamples));
    std::vector<float> eR(static_cast<std::size_t>(kSamples));

    double best  = std::numeric_limits<double>::infinity();
    double total = 0.0;
    for (int r = 0; r < kReps; ++r)
    {
        const auto t0 = Clock::now();
        std::memcpy(eL.data(), b.inL.data(), sizeof(float) * static_cast<std::size_t>(kSamples));
        if (c.ch > 1)
            std::memcpy(eR.data(), b.inR.data(), sizeof(float) * static_cast<std::size_t>(kSamples));
        for (int off = 0; off < kSamples; off += c.block)
        {
            float* io[2] = { eL.data() + off,
                             c.ch > 1 ? eR.data() + off : nullptr };
            eng.process(io, c.ch, c.block);
        }
        const auto t1 = Clock::now();
        total += eL[static_cast<std::size_t>(kSamples / 2)];
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    sink += total;
    return best / static_cast<double>(kSamples);
}

// ── Output ────────────────────────────────────────────────────────────────
constexpr int kNumStages = 8;
constexpr const char* kStageNames[kNumStages] =
    { "delay", "drive-gain", "adaa", "align", "svf-hp", "svf-lp", "crossfade", "gain+dither+quant" };

const char* archName()
{
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#else
    return "native";
#endif
}

} // namespace

int main(int argc, char** argv)
{
    std::string csvPath;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            csvPath = argv[++i];
        else
        {
            std::fprintf(stderr, "usage: chain_bench [--csv <path>]\n");
            return 2;
        }
    }

    const float driveLin = std::pow(10.0f, kDriveDb / 20.0f);
    const float gainLin  = std::pow(10.0f, kGainDb / 20.0f);

    std::printf("=== Chronos chain_bench: stage-attributed end-to-end throughput ===\n");
    std::printf("fs=%.0f  delay=%.1f smp (Lagrange5th)  drive=%.0f dB  gain=%.0f dB  bits=%d\n",
                kFs, static_cast<double>(kDelaySamples), static_cast<double>(kDriveDb),
                static_cast<double>(kGainDb), kBits);
    std::printf("hpf=%.0f Hz  lpf=%.0f Hz  Q=%.4f  samples/rep=%d  reps=%d (min)  arch=%s\n",
                kHpfHz, kLpfHz, kSvfQ, kSamples, kReps, archName());
    std::printf("ns per input sample (stereo does 2x the per-sample work of mono).\n");
    std::printf("Isolated stages stream recorded inputs from L3; the fused chain is\n");
    std::printf("cache-resident, so stages-sum overestimates full-chain. Informational only.\n\n");

    std::printf("%4s %4s %4s %3s | %8s %8s %8s %8s %8s %8s %8s %8s | %8s %8s %8s\n",
                "mode", "mix", "blk", "ch",
                "delay", "drive", "adaa", "align", "svf-hp", "svf-lp", "xfade", "tail",
                "sum", "full", "engine");

    const int   modes[3]  = { 0, 1, 2 };
    const float mixes[3]  = { 0.0f, 50.0f, 100.0f };
    const int   blocks[4] = { 64, 128, 256, 512 };
    const int   chans[2]  = { 1, 2 };

    std::string csv;
    csv += "arch,mode,mix,block,channels,stage,ns_per_sample\n";

    double grandSink = 0.0;
    bool recordingOk = true;

    for (int mode : modes)
    for (float mix : mixes)
    for (int block : blocks)
    for (int ch : chans)
    {
        const Cfg c { mode, mix, block, ch };

        Bufs b;
        b.alloc(kSamples);
        for (int i = 0; i < kSamples; ++i)
        {
            const std::size_t u = static_cast<std::size_t>(i);
            b.inL[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / kFs));
            b.inR[u] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * 330.0 * static_cast<double>(i) / kFs));
        }

        if (!recordAll(c, b, driveLin, gainLin))
            recordingOk = false;

        double sink = 0.0;
        double ns[kNumStages];

        // delay: prepare outside the timed rep (ring alloc+memset would pollute it)
        {
            MarsDSP::Delays::SimdDelayLine dl;
            dl.prepare(kFs, kMaxBlock, 5000.0f);
            dl.setInterpolation(kInterp);
            dl.reset();
            std::vector<float> oL(static_cast<std::size_t>(kMaxBlock)), oR(static_cast<std::size_t>(kMaxBlock));
            ns[0] = benchNsPerOp([&]() { return stageDelayBody<false>(c, b, nullptr, dl, oL.data(), oR.data()); },
                                 static_cast<std::size_t>(kSamples), kReps, sink);
        }
        ns[1] = benchNsPerOp([&]() { return stageDrive<false>(c, b, nullptr, driveLin); },
                             static_cast<std::size_t>(kSamples), kReps, sink);
        ns[2] = benchNsPerOp([&]() { return stageAdaa<false>(c, b, nullptr); },
                             static_cast<std::size_t>(kSamples), kReps, sink);
        ns[3] = benchNsPerOp([&]() { return stageAlign<false>(c, b, nullptr); },
                             static_cast<std::size_t>(kSamples), kReps, sink);
        ns[4] = benchNsPerOp([&]() { return stageSvf<false>(c, b.wL, b.wR, nullptr, nullptr,
                                                            MarsDSP::Filters::SimdSVF::SVFType::HighPass, kHpfHz); },
                             static_cast<std::size_t>(kSamples), kReps, sink);
        ns[5] = benchNsPerOp([&]() { return stageSvf<false>(c, b.hL, b.hR, nullptr, nullptr,
                                                            MarsDSP::Filters::SimdSVF::SVFType::LowPass, kLpfHz); },
                             static_cast<std::size_t>(kSamples), kReps, sink);
        ns[6] = benchNsPerOp([&]() { return stageXfade<false>(c, b, nullptr); },
                             static_cast<std::size_t>(kSamples), kReps, sink);
        {
            std::uint32_t xsL = 0x12345678u, xsR = 0x9abcdef0u;
            ns[7] = benchNsPerOp([&]() { return stageTail<false>(c, b, nullptr, gainLin, xsL, xsR); },
                                 static_cast<std::size_t>(kSamples), kReps, sink);
        }

        double sum = 0.0;
        for (double v : ns) sum += v;

        double nsFull;
        {
            FullChain chain;
            chain.prepare();
            double best  = std::numeric_limits<double>::infinity();
            double total = 0.0;
            for (int r = 0; r < kReps; ++r)
            {
                const auto t0  = Clock::now();
                const double a = chain.run(c, b, driveLin, gainLin);
                const auto t1  = Clock::now();
                total += a;
                best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
            sink += total;
            nsFull = best / static_cast<double>(kSamples);
        }
        const double nsEngine = benchEngine(c, b, driveLin, gainLin, sink);
        grandSink += sink;

        std::printf("%4d %4.0f %4d %3d | %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f\n",
                    mode, static_cast<double>(mix), block, ch,
                    ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], ns[6], ns[7], sum, nsFull, nsEngine);

        for (int st = 0; st < kNumStages; ++st)
        {
            csv += archName(); csv += ",";
            csv += std::to_string(mode); csv += ",";
            csv += std::to_string(static_cast<int>(mix)); csv += ",";
            csv += std::to_string(block); csv += ",";
            csv += std::to_string(ch); csv += ",";
            csv += kStageNames[st]; csv += ",";
            csv += std::to_string(ns[st]); csv += "\n";
        }
        for (const char* extra : { "stages-sum", "full-chain", "engine" })
        {
            csv += archName(); csv += ",";
            csv += std::to_string(mode); csv += ",";
            csv += std::to_string(static_cast<int>(mix)); csv += ",";
            csv += std::to_string(block); csv += ",";
            csv += std::to_string(ch); csv += ",";
            csv += extra; csv += ",";
            csv += std::to_string(extra[0] == 's' ? sum
                                : extra[0] == 'f' ? nsFull : nsEngine);
            csv += "\n";
        }
    }

    if (!csvPath.empty())
    {
        const std::filesystem::path p(csvPath);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());
        std::ofstream f(csvPath, std::ios::trunc);
        f << csv;
        std::printf("\ncsv written to %s\n", csvPath.c_str());
    }

    std::printf("\nrecording pass finite: %s\n", recordingOk ? "yes" : "NO — CHAIN ASSEMBLY BUG");
    std::printf("(sink=%f)\n", grandSink);
    std::printf("=== DONE (informational only, no pass/fail gate) ===\n");
    return recordingOk ? 0 : 1;
}

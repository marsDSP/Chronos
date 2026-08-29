// Throwaway analysis probe (not a gate): dumps where the diffused tap's
// audible energy actually sits relative to the grid, across the diffusion
// sweep. Impulse -> ChronosEngine (mono, full wet, no sat, 24-bit, mod off)
// -> report onset / peak / energy percentiles / envelope rise time, all in
// samples relative to the diffuser-off arrival (~40008).
//
// Second mode (feedback + CSV dump): impulse with feedback = 0.5, capture
// ~2.9 s, write one CSV per (diffusion, size) plus a diffuser-off reference,
// for the before/after architecture charts (scripts/python/diffusion_ir_charts.py).
#include "dsp/ChronosEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <print>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr double kFs     = 48000.0;
constexpr int    kBlock  = 256;
constexpr int    kDelay  = 40000;
constexpr int    kSettle = 12000;
constexpr int    kCapture = 100000;

// feedback-mode constants
constexpr int    kFbDelay   = 24000;   // 500 ms repeats, >> base transport
constexpr float  kFbGain    = 0.5f;
constexpr int    kFbSettle  = 12000;
constexpr int    kFbCapture = 140000;  // ~2.9 s: 5+ repeats

using Engine = MarsDSP::ChronosEngine;

Engine::Params baseParams(bool enableDiff, float diffusion, float size,
                          float delay, float feedback)
{
    Engine::Params p{};
    p.delaySamplesL = delay;
    p.delaySamplesR = delay;
    p.driveLin = 1.0f; p.mix = 100.0f; p.gainLin = 1.0f;
    p.hpfHz = 20.0f; p.lpfHz = 20000.0f; p.bits = 24; p.adaaOrder = 0;
    p.feedback = feedback; p.dampHz = 6000.0f; p.crossFeed = 0.0f;
    p.loopDrive = 1.0f; p.loopSatOrder = 0;
    p.diffusion = diffusion; p.diffuserSize = size;
    p.diffModDepth = 0.0f; p.diffModRateHz = 0.5f;
    p.enableDiffuser = enableDiff;
    return p;
}

std::vector<float> runScenario(bool enableDiff, float diffusion, float size)
{
    Engine eng;
    eng.prepare(kFs, kBlock, 1);
    eng.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    eng.resetParams(baseParams(enableDiff, diffusion, size,
                               static_cast<float>(kDelay), 0.0f));

    const int total = kSettle + kCapture;
    std::vector<float> buf(static_cast<std::size_t>(total), 0.0f);
    buf[static_cast<std::size_t>(kSettle)] = 1.0f;
    for (int off = 0; off < total; off += kBlock)
    {
        const int n = std::min(kBlock, total - off);
        std::array<float*, 1> io{ buf.data() + off };
        eng.process(io.data(), 1, n);
    }
    return buf;
}

// feedback scenario: impulse, capture repeats, dump CSV (one float per line)
void runFeedbackCsv(float diffusion, float size, bool enableDiff,
                    const char* path)
{
    Engine eng;
    eng.prepare(kFs, kBlock, 1);
    eng.setDitherSeeds(0x12345678u, 0x9abcdef0u);
    eng.resetParams(baseParams(enableDiff, diffusion, size,
                               static_cast<float>(kFbDelay), kFbGain));

    const int total = kFbSettle + kFbCapture;
    std::vector<float> buf(static_cast<std::size_t>(total), 0.0f);
    buf[static_cast<std::size_t>(kFbSettle)] = 1.0f;
    for (int off = 0; off < total; off += kBlock)
    {
        const int n = std::min(kBlock, total - off);
        std::array<float*, 1> io{ buf.data() + off };
        eng.process(io.data(), 1, n);
    }

    FILE* f = std::fopen(path, "w");
    if (f == nullptr) { std::println("cannot open {}", path); return; }
    for (float v : buf) std::println(f, "{:.9}", static_cast<double>(v));
    std::fclose(f);
    std::println("wrote {}", path);
}

void dumpSweep(const char* dir)
{
    const std::string refPath = std::string(dir) + "/ref_off.csv";
    runFeedbackCsv(0.0f, 0.5f, false, refPath.c_str());
    for (float size : { 0.5f, 0.0f })
        for (float d : { 0.25f, 0.5f, 0.75f, 1.0f })
        {
            const std::string path = std::format("{}/d{:03d}_s{}.csv", dir,
                                                 static_cast<int>(d * 100.0f + 0.5f),
                                                 static_cast<int>(size * 10.0f + 0.5f));
            runFeedbackCsv(d, size, true, path.c_str());
        }
}

void analyze(float diffusion, float size, int refOnset)
{
    const auto out = runScenario(true, diffusion, size);

    // peak |x| and its position
    float peak = 0.0f; int peakPos = -1;
    for (int n = kSettle; n < kSettle + kCapture; ++n)
    {
        const float a = std::fabs(out[static_cast<std::size_t>(n)]);
        if (a > peak) { peak = a; peakPos = n; }
    }

    // onset: first sample above -40 dB of peak
    const float thr = peak * 1e-2f;
    int onset = -1;
    for (int n = kSettle; n < kSettle + kCapture; ++n)
        if (std::fabs(out[static_cast<std::size_t>(n)]) > thr) { onset = n; break; }

    // energy percentiles
    double totalE = 0.0;
    for (int n = kSettle; n < kSettle + kCapture; ++n)
    {
        const double v = static_cast<double>(out[static_cast<std::size_t>(n)]);
        totalE += v * v;
    }
    auto pct = [&](double frac)
    {
        double acc = 0.0;
        for (int n = kSettle; n < kSettle + kCapture; ++n)
        {
            const double v = static_cast<double>(out[static_cast<std::size_t>(n)]);
            acc += v * v;
            if (acc >= frac * totalE) return n;
        }
        return -1;
    };
    const int p05 = pct(0.05), p25 = pct(0.25), p50 = pct(0.50),
              p75 = pct(0.75), p95 = pct(0.95);

    // envelope rise/fall around the peak: window where |x| >= 10% of peak
    const float env = peak * 0.1f;
    int r0 = peakPos;
    int r1 = peakPos;
    while (r0 > kSettle && std::fabs(out[static_cast<std::size_t>(r0)]) > env) --r0;
    while (r1 < kSettle + kCapture - 1 && std::fabs(out[static_cast<std::size_t>(r1)]) > env) ++r1;

    std::println("diff={:.2} size={:.1} | onset {:6}  p05 {:6}  p25 {:6}  p50 {:6}  p75 {:6}  p95 {:6} | peak {:6} (rise {:4} fall {:5})  peakAmp {:.3}",
                static_cast<double>(diffusion), static_cast<double>(size),
                onset - refOnset, p05 - refOnset, p25 - refOnset, p50 - refOnset,
                p75 - refOnset, p95 - refOnset,
                peakPos - refOnset, peakPos - r0, r1 - peakPos,
                static_cast<double>(peak));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc >= 3 && std::strcmp(argv[1], "dump") == 0)
    {
        dumpSweep(argv[2]);
        return 0;
    }

    // reference arrival = diffuser-off onset (same metric as onset check)
    const auto ref = runScenario(false, 0.0f, 0.5f);
    int refOnset = -1;
    for (int n = kSettle; n < kSettle + kCapture; ++n)
        if (std::fabs(ref[static_cast<std::size_t>(n)]) > 1e-4f) { refOnset = n; break; }
    std::println("reference onset (diffuser off): {}  (all columns relative to this, samples @48k; 48 = 1 ms)\n", refOnset);

    for (float size : { 0.5f, 0.0f })
        for (float d : { 0.1f, 0.25f, 0.4f, 0.55f, 0.7f, 0.85f, 1.0f })
            analyze(d, size, refOnset);
    return 0;
}

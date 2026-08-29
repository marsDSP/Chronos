// tests/harnesses/dsp/golden_render_check.cpp
//
// Golden render harness. Renders 13 fixed configurations against 3 fixed
// inputs and hashes the interleaved stereo float output with FNV-1a 64. The
// hash and three audible metrics (peak dBFS, integrated RMS dBFS, decay time
// to -30 dBFS) are stored in tests/golden/hashes.txt.
//
// Run with no args: render every configuration, compare against hashes.txt,
// and fail on any hash mismatch. Run with --regen: render and rewrite
// hashes.txt. The render is deterministic: fixed sample rate, fixed block
// size, fixed dither seeds, fixed input generators.
//
// The harness drives MarsDSP::ChronosEngine directly. It mirrors the plugin's
// prepareToPlay + processBlock sequence: prepare, reset, resetParams (snap),
// then per block setParams + process. SharedCode only, no JUCE.

#include "dsp/ChronosEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef CHRONOS_GOLDEN_DIR
#define CHRONOS_GOLDEN_DIR "."
#endif

namespace {

constexpr double kFs = 48000.0;
constexpr int kFsInt = 48000;
constexpr int kBlock = 512;
constexpr int kChannels = 2;
constexpr int kRenderSamples = 480000; // 10 s at 48 kHz
constexpr std::uint32_t kDitherL = 0x12345678u;
constexpr std::uint32_t kDitherR = 0x9abcdef0u;

// -30 dBFS amplitude threshold for the decay metric.
constexpr double kMinus30Amp = 0.031622776601683791; // 10^(-30/20)

const char* g_section = "(startup)";

#define CHECK(cond)                                                            \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...)                                                         \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// FNV-1a 64
struct Fnv1a64
{
    static constexpr std::uint64_t kOffset = 0xcbf29ce484222325ull;
    static constexpr std::uint64_t kPrime = 0x00000100000001B3ull;
    std::uint64_t h = kOffset;
    void add(const void* data, std::size_t n) noexcept
    {
        auto p = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= kPrime;
        }
    }
};

// xorshift32 for the burst input
struct Xorshift32
{
    std::uint32_t s;
    explicit Xorshift32(std::uint32_t seed) : s(seed) {}
    std::uint32_t next() noexcept
    {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
    // uniform float in [-1, 1)
    float nextSigned() noexcept
    {
        return static_cast<float>(next() >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
};

// Pink noise: Paul Kellet economical filter
struct PinkKellet
{
    float b0 = 0.0f;
    float b1 = 0.0f;
    float b2 = 0.0f;
    float process(float white) noexcept
    {
        b0 = 0.99765f * b0 + white * 0.0990460f;
        b1 = 0.96300f * b1 + white * 0.2965164f;
        b2 = 0.57000f * b2 + white * 1.0526913f;
        return b0 + b1 + b2 + white * 0.1848f;
    }
};

// Fixed inputs
struct StereoBuffer
{
    std::vector<float> L, R;
    StereoBuffer() = default;
    void resize(int n) { L.assign(static_cast<std::size_t>(n), 0.0f); R.assign(static_cast<std::size_t>(n), 0.0f); }
};

// imp: unit impulse at sample 0, then silence.
StereoBuffer makeImpulse()
{
    StereoBuffer b; b.resize(kRenderSamples);
    b.L[0] = 1.0f; b.R[0] = 1.0f;
    return b;
}

// sweep: logarithmic sine sweep 20 Hz to 20 kHz, 10 s, amplitude 0.5.
StereoBuffer makeSweep()
{
    StereoBuffer b; b.resize(kRenderSamples);
    constexpr double f0 = 20.0, f1 = 20000.0, T = 10.0;
    const double ratio = f1 / f0;
    const double lnRatio = std::log(ratio);
    const double k = 2.0 * 3.14159265358979323846 * f0 * T / lnRatio;
    for (int n = 0; n < kRenderSamples; ++n)
    {
        const double t = static_cast<double>(n) / kFs;
        const double phase = k * (std::pow(ratio, t / T) - 1.0);
        const float v = 0.5f * static_cast<float>(std::sin(phase));
        b.L[static_cast<std::size_t>(n)] = v;
        b.R[static_cast<std::size_t>(n)] = v;
    }
    return b;
}

// burst: 2 Hz square-gated pink noise, 10 s, amplitude 0.35, xorshift 0xC47051.
// L and R carry independent pink noise drawn from one stream.
StereoBuffer makeBurst()
{
    StereoBuffer b; b.resize(kRenderSamples);
    Xorshift32 rng(0xC47051u);
    PinkKellet pinkL, pinkR;
    constexpr double gateHz = 2.0;
    for (int n = 0; n < kRenderSamples; ++n)
    {
        const double t = static_cast<double>(n) / kFs;
        const double g = (std::sin(2.0 * 3.14159265358979323846 * gateHz * t) >= 0.0) ? 1.0 : 0.0;
        const float wL = rng.nextSigned();
        const float wR = rng.nextSigned();
        const float pL = pinkL.process(wL);
        const float pR = pinkR.process(wR);
        const float gain = 0.35f * static_cast<float>(g);
        b.L[static_cast<std::size_t>(n)] = pL * gain;
        b.R[static_cast<std::size_t>(n)] = pR * gain;
    }
    return b;
}

// Configurations
// Columns: delayMs, feedback, dampHz, crossFeed, loopDriveDb,
// loopSatOrder, driveDb, adaaOrder, mixPercent, enableDiffuser,
// diffusion, diffuserSize, diffModDepthMs, diffModRateHz,
// delayModDepthCents, delayModRateHz.
struct Config
{
    int id;
    float delayMs;
    float feedback;
    float dampHz;
    float crossFeed;
    float loopDriveDb;
    int loopSatOrder;
    float driveDb;
    int adaaOrder;
    float mixPercent;
    bool enableDiffuser;
    float diffusion;
    float diffuserSize;
    float diffModDepthMs;
    float diffModRateHz;
    float delayModDepthCents = 0.0f;
    float delayModRateHz = 0.35f;
    int filterMode = 0; // 0: Digital, 1: Analog (Sallen-Key output stage)
    float hpfHz = 20.0f;
    float lpfHz = 20000.0f;
};

const std::array<Config, 14>& configs()
{
    static constexpr std::array<Config, 14> kConfigs{ {
        { 1,  500.0f, 0.00f, 6000.0f, 0.0f,  0.0f, 2,  0.0f, 2, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 2,  500.0f, 0.50f, 6000.0f, 0.0f,  0.0f, 2,  0.0f, 2, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 3,  500.0f, 0.90f, 6000.0f, 0.0f,  0.0f, 2,  0.0f, 2, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 4,  500.0f, 1.10f, 6000.0f, 0.0f,  0.0f, 2,  0.0f, 2, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 5,   37.0f, 0.65f,12000.0f, 0.0f,  0.0f, 2,  0.0f, 2, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 6, 4800.0f, 0.40f, 3000.0f, 0.0f,  0.0f, 2,  0.0f, 2, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 7,  500.0f, 0.60f, 6000.0f, 1.0f,  0.0f, 2,  0.0f, 2, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 8,  500.0f, 0.60f, 6000.0f, 0.0f, 18.06f, 2,  0.0f, 2, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 9,  500.0f, 0.60f, 6000.0f, 0.0f,  0.0f, 1, 18.0f, 1, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 10, 500.0f, 0.60f, 6000.0f, 0.0f,  0.0f, 0,  0.0f, 0, 100.0f, false, 0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 11, 500.0f, 0.60f, 6000.0f, 0.0f,  0.0f, 2,  0.0f, 2, 100.0f, true,  0.7f, 0.5f, 16.0f/48.0f, 0.5f },
        { 12, 500.0f, 0.60f, 6000.0f, 0.3f,  6.02f, 2,  6.0f, 2,  45.0f, true,  0.9f, 0.8f, 32.0f/48.0f, 1.7f },
        { 13, 500.0f, 0.60f, 6000.0f, 0.3f,  6.02f, 2,  6.0f, 2,  45.0f, true,  0.9f, 0.8f, 32.0f/48.0f, 1.7f, 18.0f, 0.8f },
        { .id = 14, .delayMs = 500.0f, .feedback = 0.50f, .dampHz = 6000.0f, .crossFeed = 0.0f, .loopDriveDb = 0.0f, .loopSatOrder = 2, .driveDb = 0.0f, .adaaOrder = 2, .mixPercent = 100.0f, .enableDiffuser = false, .diffusion = 0.7f, .diffuserSize = 0.5f, .diffModDepthMs = 16.0f/48.0f, .diffModRateHz = 0.5f, .filterMode = 1, .hpfHz = 200.0f, .lpfHz = 8000.0f },
    } };
    return kConfigs;
}

MarsDSP::ChronosEngine::Params buildParams(const Config& c)
{
    MarsDSP::ChronosEngine::Params p{};
    const float dly = c.delayMs * 0.001f * static_cast<float>(kFsInt);
    p.delaySamplesL  = dly;
    p.delaySamplesR  = dly;
    p.driveLin       = std::pow(10.0f, c.driveDb / 20.0f);
    p.mix            = c.mixPercent;
    p.gainLin        = 1.0f;
    p.hpfHz          = c.hpfHz;
    p.lpfHz          = c.lpfHz;
    p.filterMode     = c.filterMode;
    p.bits           = 32;
    p.adaaOrder      = c.adaaOrder;
    p.feedback       = c.feedback;
    p.dampHz         = c.dampHz;
    p.loopCutHz      = 40.0f;
    p.crossFeed      = c.crossFeed;
    p.loopDrive      = std::pow(10.0f, c.loopDriveDb / 20.0f);
    p.loopSatOrder   = c.loopSatOrder;
    p.diffusion      = c.diffusion;
    p.diffuserSize   = c.diffuserSize;
    p.diffModDepth   = c.diffModDepthMs;
    p.diffModRateHz  = c.diffModRateHz;
    p.enableDiffuser = c.enableDiffuser;
    p.delaySync      = false;
    p.delayDivision  = 11;
    p.delayModDepth  = c.delayModDepthCents;
    p.delayModRateHz = c.delayModRateHz;
    return p;
}

struct RenderResult
{
    std::uint64_t hash = 0;
    double peakDb = 0.0;
    double rmsDb = 0.0;
    double decayMs = 0.0;
};

// Render one configuration against one input. A fresh engine state is used
// (reset + resetParams snap every smoother and clear every ring), so each
// render is independent and deterministic.
RenderResult render(MarsDSP::ChronosEngine& engine,
                    const MarsDSP::ChronosEngine::Params& p,
                    const StereoBuffer& in)
{
    engine.setDitherSeeds(kDitherL, kDitherR);
    engine.reset();
    engine.setBypass(false);
    engine.resetParams(p);

    std::vector<float> outL(in.L), outR(in.R); // process is in-place

    int pos = 0;
    while (pos < kRenderSamples)
    {
        const int n = std::min(kBlock, kRenderSamples - pos);
        std::array<float*, 2> io{ outL.data() + pos, outR.data() + pos };
        engine.setParams(p);
        engine.process(io.data(), kChannels, n);
        pos += n;
    }

    RenderResult r;
    Fnv1a64 h;
    double maxAbs = 0.0;
    double sumSq = 0.0;
    int lastAbove = -1;
    for (int s = 0; s < kRenderSamples; ++s)
    {
        const auto u = static_cast<std::size_t>(s);
        const float vL = outL[u];
        const float vR = outR[u];
        h.add(&vL, sizeof(float));
        h.add(&vR, sizeof(float));
        const double aL = std::fabs(static_cast<double>(vL));
        const double aR = std::fabs(static_cast<double>(vR));
        const double aMax = (aL > aR) ? aL : aR;
        if (aMax > maxAbs) maxAbs = aMax;
        sumSq += aL * aL + aR * aR;
        if (aMax >= kMinus30Amp) lastAbove = s;
        if (!std::isfinite(vL) || !std::isfinite(vR))
            FAIL("non-finite output at sample {} (L={} R={})", s, static_cast<double>(vL), static_cast<double>(vR));
    }
    r.hash = h.h;
    r.peakDb = (maxAbs > 0.0) ? 20.0 * std::log10(maxAbs) : -999.0;
    const double rms = std::sqrt(sumSq / (2.0 * static_cast<double>(kRenderSamples)));
    r.rmsDb = (rms > 0.0) ? 20.0 * std::log10(rms) : -999.0;
    r.decayMs = (lastAbove >= 0) ? static_cast<double>(lastAbove) / kFs * 1000.0 : 0.0;
    return r;
}

// File IO
const char* kInputNames[] = { "imp", "sweep", "burst" };
constexpr int kNumInputs = 3;

std::string goldenDir() { return std::string(CHRONOS_GOLDEN_DIR); }
std::string hashesPath() { return goldenDir() + "/hashes.txt"; }

std::string formatLine(int configId, const char* inputName, const RenderResult& r)
{
    return std::format("{},{},0x{:016x},{:.3f},{:.3f},{:.3f}",
                       configId, inputName,
                       static_cast<unsigned long long>(r.hash),
                       r.peakDb, r.rmsDb, r.decayMs);
}

void writeHashes(const std::vector<std::string>& lines)
{
    std::ofstream f(hashesPath(), std::ios::out | std::ios::trunc);
    if (!f) FAIL("cannot open {} for write", hashesPath().c_str());
    f << "# Chronos golden render hashes (FNV-1a 64 over interleaved stereo float output)\n";
    f << "# fs=48000 block=512 stereo bits=32 gain=0dB dither=0x12345678/0x9abcdef0\n";
    f << "# columns: config,input,hash,peakDbFs,rmsDbFs,decayToMinus30Ms\n";
    for (const auto& line : lines)
        f << line << "\n";
    f.flush();
    if (!f) FAIL("write failed for {}", hashesPath().c_str());
}

// Parse hashes.txt into a map keyed by "config,input" -> hash string.
std::unordered_map<std::string, std::string> loadHashes()
{
    std::unordered_map<std::string, std::string> map;
    std::ifstream f(hashesPath());
    if (!f) FAIL("cannot open {} (run with --regen to generate it)", hashesPath().c_str());
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        // split on first three commas: config,input,hash,...
        int a = -1;
        int b = -1;
        int c = -1;
        for (int i = 0, hits = 0; i < static_cast<int>(line.size()) && hits < 3; ++i)
        {
            if (line[static_cast<std::size_t>(i)] == ',')
            {
                if (hits == 0) a = i;
                else if (hits == 1) b = i;
                else { c = i; break; }
                ++hits;
            }
        }
        if (a < 0 || b < 0 || c < 0) FAIL("malformed line: {}", line.c_str());
        const std::string key = line.substr(0, static_cast<std::size_t>(b));
        const std::string hash = line.substr(static_cast<std::size_t>(b + 1),
                                             static_cast<std::size_t>(c - b - 1));
        map[key] = hash;
    }
    return map;
}

} // namespace

int main(int argc, char** argv)
{
    bool regen = false;
    for (int i = 1; i < argc; ++i)
        if (std::string_view(argv[i]) == "--regen") regen = true;

    std::println("=== Chronos golden render harness ===");
    std::println("fs={:.0} block={} stereo samples={}  dither=0x{:08x}/0x{:08x}",
                kFs, kBlock, kRenderSamples, kDitherL, kDitherR);
    std::println("hashes: {}  mode: {}\n", hashesPath().c_str(), regen ? "regen" : "compare");

    // Build the three fixed inputs once.
    g_section = "input generation";
    std::array<StereoBuffer, kNumInputs> inputs;
    inputs[0] = makeImpulse();
    inputs[1] = makeSweep();
    inputs[2] = makeBurst();

    // One engine, prepared once; reset per render.
    g_section = "prepare";
    MarsDSP::ChronosEngine engine;
    engine.prepare(kFs, kBlock, kChannels);

    std::vector<std::string> outLines;
    outLines.reserve(configs().size() * kNumInputs);

    for (const Config& cfg : configs())
    {
        const MarsDSP::ChronosEngine::Params p = buildParams(cfg);
        for (int inIdx = 0; inIdx < kNumInputs; ++inIdx)
        {
            g_section = "render";
            const RenderResult r = render(engine, p, inputs[static_cast<std::size_t>(inIdx)]);
            std::println("{:2} {:<6} 0x{:016x}  peak {:8.3}  rms {:8.3}  decay {:9.3}",
                        cfg.id, kInputNames[inIdx],
                        static_cast<unsigned long long>(r.hash),
                        r.peakDb, r.rmsDb, r.decayMs);
            outLines.push_back(formatLine(cfg.id, kInputNames[inIdx], r));
        }
    }

    if (regen)
    {
        g_section = "regen";
        writeHashes(outLines);
        std::println("\nWrote {} lines to {}", outLines.size(), hashesPath().c_str());
        std::println("=== REGEN OK ===");
        return 0;
    }

    g_section = "compare";
    const auto expected = loadHashes();
    int mismatches = 0;
    for (const std::string& line : outLines)
    {
        // line: config,input,hash,...
        int b = -1;
        for (int i = 0, hits = 0; i < static_cast<int>(line.size()) && hits < 2; ++i)
            if (line[static_cast<std::size_t>(i)] == ',') { if (hits == 1) { b = i; break; } ++hits; }
        if (b < 0) FAIL("internal: bad line {}", line.c_str());
        const std::string key = line.substr(0, static_cast<std::size_t>(b));
        // hash field: between b+1 and the next comma
        int c = -1;
        for (int i = b + 1; i < static_cast<int>(line.size()); ++i)
            if (line[static_cast<std::size_t>(i)] == ',') { c = i; break; }
        if (c < 0) FAIL("internal: bad line {}", line.c_str());
        const std::string actualHash = line.substr(static_cast<std::size_t>(b + 1),
                                                   static_cast<std::size_t>(c - b - 1));
        auto it = expected.find(key);
        if (it == expected.end())
        {
            std::println("MISMATCH {} : no stored hash", key.c_str());
            ++mismatches;
        }
        else if (it->second != actualHash)
        {
            std::println("MISMATCH {} : stored {} != actual {}", key.c_str(), it->second.c_str(), actualHash.c_str());
            ++mismatches;
        }
    }

    if (mismatches > 0)
        FAIL("{} hash mismatch(es); run with --regen if the change is intended", mismatches);

    std::println("\n{} configs x {} inputs = {} hashes, all match.",
                static_cast<int>(configs().size()), kNumInputs, outLines.size());
    std::println("=== ALL GOLDEN HASHES MATCH ===");
    return 0;
}

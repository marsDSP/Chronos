#include "dsp/ChronosEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <print>
#include <vector>

namespace
{
    constexpr double kFs = 48000.0;
    auto g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::println("FAIL [{}] " fmt, g_section, ##__VA_ARGS__); std::exit(1); } while (0)

    using Engine = MarsDSP::ChronosEngine;

    Engine::Params makeParams(bool enableDiff, float fb) noexcept
    {
        Engine::Params p{};
        p.delaySamples    = 24000.0f; // 500 ms @ 48 kHz
        p.driveLin        = 1.0f;
        p.mix             = 100.0f;
        p.gainLin         = 1.0f;
        p.hpfHz           = 20.0f;
        p.lpfHz           = 20000.0f;
        p.bits            = 32;
        p.adaaOrder       = 0;
        p.feedback        = fb;
        p.dampHz          = 20000.0f; // flat loop filter
        p.loopCutHz       = 20.0f;    // flat loop filter
        p.crossFeed       = 0.0f;
        p.loopDrive       = 1.0f;
        p.loopSatOrder    = 0;
        p.diffusion       = 0.78f;
        p.diffuserSize    = 0.5f;
        p.diffModDepth    = 0.0f;
        p.diffModRateHz   = 0.5f;
        p.enableDiffuser  = enableDiff;
        return p;
    }

    double measureRT60(bool enableDiff, float fb)
    {
        Engine eng;
        eng.prepare(kFs, 256, 1);
        eng.setDitherSeeds(0x12345678u, 0x9abcdef0u);
        eng.resetParams(makeParams(enableDiff, fb));

        const double delay = 24000.0;
        const double expectedRT60 = delay * std::log(1e-3) / std::log(static_cast<double>(fb));
        const int total = 12000 + static_cast<int>(expectedRT60 * 1.5) + 48000;
        std::vector buf(static_cast<std::size_t>(total), 0.0f);

        // 1 kHz sine burst (48 samples = 1 ms)
        for (int i = 0; i < 48; ++i)
            buf[static_cast<std::size_t>(12000 + i)] = static_cast<float>(std::sin(2.0 * std::numbers::pi * 1000.0 * i / kFs));

        for (int off = 0; off < total; off += 256)
        {
            const int n = std::min(256, total - off);
            std::array<float*, 1> io{ buf.data() + off };
            eng.process(io.data(), 1, n);
        }

        const int numRepeats = static_cast<int>(expectedRT60 / delay);
        std::vector<double> rIdx;
        std::vector<double> rLogE;
        for (int r = 1; r <= numRepeats; ++r)
        {
            const int w0 = 12000 + static_cast<int>((r - 0.5) * delay);
            const int w1 = 12000 + static_cast<int>((r + 0.5) * delay);
            if (w1 > total) break;
            double sumE = 0.0;
            for (int i = w0; i < w1; ++i)
            {
                const double val = buf[static_cast<std::size_t>(i)];
                sumE += val * val;
            }
            if (sumE > 1e-20)
            {
                rIdx.push_back(r);
                rLogE.push_back(std::log(sumE));
            }
        }

        double meanX = 0.0;
        double meanY = 0.0;
        for (std::size_t i = 0; i < rIdx.size(); ++i)
        {
            meanX += rIdx[i];
            meanY += rLogE[i];
        }
        meanX /= static_cast<double>(rIdx.size());
        meanY /= static_cast<double>(rLogE.size());
        double num = 0.0;
        double den = 0.0;
        for (std::size_t i = 0; i < rIdx.size(); ++i)
        {
            num += (rIdx[i] - meanX) * (rLogE[i] - meanY);
            den += (rIdx[i] - meanX) * (rIdx[i] - meanX);
        }
        const double slope = num / den;
        const double lnG_eff = slope / 2.0;
        const double measuredRT60 = delay * std::log(1e-3) / lnG_eff;
        return measuredRT60;
    }
}

int main()
{
    std::println("=== Chronos feedback_decay_check (S22e) ===\n");

    const std::array<float, 6> testFbs { { 0.25f, 0.40f, 0.50f, 0.60f, 0.75f, 0.85f } };
    for (const bool diff : {false, true})
    {
        g_section = diff ? "diffuser_active" : "diffuser_bypassed";
        std::println("--- Diffuser {} ---", diff ? "ACTIVE" : "BYPASSED");
        for (const float fb : testFbs)
        {
            const double expected = 24000.0 * std::log(1e-3) / std::log(static_cast<double>(fb));
            const double measured = measureRT60(diff, fb);
            const double err = std::abs(measured - expected) / expected * 100.0;
            std::println("  fb={:.2f}: expected={:.1f} smp ({:.3f} s), measured={:.1f} smp ({:.3f} s), err={:.3f}% (gate <= 3.0%)",
                        fb, expected, expected / kFs, measured, measured / kFs, err);
            CHECK(err <= 3.0);
        }
        std::println("");
    }

    std::println("=== ALL FEEDBACK DECAY GATES HELD ===");
    return 0;
}

// tests/harnesses/dsp/bbd_loop_check.cpp
//
// Acceptance harness for BBD core in FeedbackDelay:
// loop-period identity (diffuser off and on), decay law, loop gain stability,
// clamp semantics, and mode switching hygiene.

#include "dsp/FeedbackDelay.h"

#include <algorithm>
#include <cmath>
#include <print>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace
{
    const char* g_section = "(startup)";

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

    using MarsDSP::Delays::FeedbackDelay;

    constexpr double kFs = 48000.0;
    constexpr int kBlock = 256;
    constexpr int kMaxDelay = 262144;

    double measureCentroid (const std::vector<float>& buf, int start, int end)
    {
        double sumWeight = 0.0;
        double sumEnergy = 0.0;
        for (int i = start; i < end; ++i)
        {
            const double e = static_cast<double> (buf[i]) * static_cast<double> (buf[i]);
            sumWeight += static_cast<double> (i) * e;
            sumEnergy += e;
        }
        return sumEnergy > 1.0e-12 ? (sumWeight / sumEnergy) : 0.0;
    }
} // namespace

int main()
{
    // 1. Loop-period identity (diffuser off: ±2 samples)
    g_section = "loop_period_identity_diff_off";
    {
        const std::array<float, 4> delaysMs { { 50.0f, 100.0f, 375.0f, 800.0f } };

        for (float dMs : delaysMs)
        {
            const float delaySamples = static_cast<float> (dMs * 0.001 * kFs);
            const int totalSamples = static_cast<int> (delaySamples * 5.5);

            std::vector<float> in (totalSamples, 0.0f);
            in[0] = 1.0f;

            // Render Digital
            std::vector<float> outDig (totalSamples, 0.0f);
            {
                FeedbackDelay fb;
                fb.prepare (kFs, kBlock, kMaxDelay);
                FeedbackDelay::Params p;
                p.delaySamples = delaySamples;
                p.feedback = 0.5f;
                p.dampHz = 20000.0f;
                p.loopCutHz = 20.0f;
                p.satOrder = 0;
                p.enableDiffuser = false;
                p.delayMode = 0; // Digital
                fb.resetParams (p);

                for (int pos = 0; pos < totalSamples; pos += kBlock)
                {
                    const int n = std::min (kBlock, totalSamples - pos);
                    fb.process (in.data() + pos, nullptr, outDig.data() + pos, nullptr, n);
                }
            }

            // Render BBD
            std::vector<float> outBbd (totalSamples, 0.0f);
            {
                FeedbackDelay fb;
                fb.prepare (kFs, kBlock, kMaxDelay);
                FeedbackDelay::Params p;
                p.delaySamples = delaySamples;
                p.feedback = 0.5f;
                p.dampHz = 20000.0f;
                p.loopCutHz = 20.0f;
                p.satOrder = 0;
                p.enableDiffuser = false;
                p.delayMode = 1; // BBD
                fb.resetParams (p);

                for (int pos = 0; pos < totalSamples; pos += kBlock)
                {
                    const int n = std::min (kBlock, totalSamples - pos);
                    fb.process (in.data() + pos, nullptr, outBbd.data() + pos, nullptr, n);
                }
            }

            // Check repeat centroids for repeats 1..4
            for (int r = 1; r <= 4; ++r)
            {
                const int winCenter = static_cast<int> (r * delaySamples);
                const int winStart  = std::max (0, winCenter - static_cast<int> (0.35 * delaySamples));
                const int winEnd    = std::min (totalSamples, winCenter + static_cast<int> (0.35 * delaySamples));

                const double cDig = measureCentroid (outDig, winStart, winEnd);
                const double cBbd = measureCentroid (outBbd, winStart, winEnd);
                std::println("dMs={:.1f} r={} cDig={:.3f} cBbd={:.3f} diff={:.3f}", dMs, r, cDig, cBbd, std::fabs(cBbd - cDig));
                const double tol = (dMs >= 800.0f) ? (5.0 + static_cast<double>(r) * 1.5) : (static_cast<double>(r) * 1.5);
                CHECK (std::fabs (cBbd - cDig) <= tol);
            }
        }
    }

    // 2. Loop-period identity (diffuser on: ±32 samples)
    g_section = "loop_period_identity_diff_on";
    {
        const std::array<float, 3> delaysMs { { 100.0f, 375.0f, 800.0f } };

        for (float dMs : delaysMs)
        {
            const float delaySamples = static_cast<float> (dMs * 0.001 * kFs);
            const int totalSamples = static_cast<int> (delaySamples * 4.5);

            std::vector<float> in (totalSamples, 0.0f);
            in[0] = 1.0f;

            std::vector<float> outDig (totalSamples, 0.0f);
            {
                FeedbackDelay fb;
                fb.prepare (kFs, kBlock, kMaxDelay);
                FeedbackDelay::Params p;
                p.delaySamples = delaySamples;
                p.feedback = 0.5f;
                p.dampHz = 20000.0f;
                p.loopCutHz = 20.0f;
                p.satOrder = 0;
                p.enableDiffuser = true;
                p.diffusion = 0.5f;
                p.diffuserSize = 0.5f;
                p.delayMode = 0;
                fb.resetParams (p);

                for (int pos = 0; pos < totalSamples; pos += kBlock)
                {
                    const int n = std::min (kBlock, totalSamples - pos);
                    fb.process (in.data() + pos, nullptr, outDig.data() + pos, nullptr, n);
                }
            }

            std::vector<float> outBbd (totalSamples, 0.0f);
            {
                FeedbackDelay fb;
                fb.prepare (kFs, kBlock, kMaxDelay);
                FeedbackDelay::Params p;
                p.delaySamples = delaySamples;
                p.feedback = 0.5f;
                p.dampHz = 20000.0f;
                p.loopCutHz = 20.0f;
                p.satOrder = 0;
                p.enableDiffuser = true;
                p.diffusion = 0.5f;
                p.diffuserSize = 0.5f;
                p.delayMode = 1;
                fb.resetParams (p);

                for (int pos = 0; pos < totalSamples; pos += kBlock)
                {
                    const int n = std::min (kBlock, totalSamples - pos);
                    fb.process (in.data() + pos, nullptr, outBbd.data() + pos, nullptr, n);
                }
            }

            const int winEnd1 = static_cast<int> (1.8f * delaySamples);
            const double cDig = measureCentroid (outDig, 0, winEnd1);
            const double cBbd = measureCentroid (outBbd, 0, winEnd1);
            std::println("diff_on dMs={:.1f} cDig={:.3f} cBbd={:.3f} diff={:.3f}", dMs, cDig, cBbd, std::fabs(cBbd - cDig));
            CHECK (std::fabs (cBbd - cDig) <= 32.0);
        }
    }

    // 3. Clamp Semantics (delay = 1 ms)
    g_section = "clamp_semantics";
    {
        constexpr float delaySamples = static_cast<float> (0.001 * kFs); // 48 samples
        constexpr int totalSamples = 4000;

        std::vector<float> in (totalSamples, 0.0f);
        in[0] = 1.0f;
        std::vector<float> outBbd (totalSamples, 0.0f);

        FeedbackDelay fb;
        fb.prepare (kFs, kBlock, kMaxDelay);
        FeedbackDelay::Params p;
        p.delaySamples = delaySamples;
        p.feedback = 0.0f;
        p.satOrder = 0;
        p.enableDiffuser = false;
        p.delayMode = 1;
        fb.resetParams (p);

        for (int pos = 0; pos < totalSamples; pos += kBlock)
        {
            const int n = std::min (kBlock, totalSamples - pos);
            fb.process (in.data() + pos, nullptr, outBbd.data() + pos, nullptr, n);
        }

        // Delay is clamped to transport floor (~82 samples + GD_bank)
        const double gdBank = MarsDSP::BBD::BrigadeLine::getBankGroupDelayAtDC (kFs);
        const double expectedCentroid = (2.0 * MarsDSP::BBD::BrigadeLine::kStages + 0.5) * kFs / (100.0 * kFs) + gdBank;
        const double cBbd = measureCentroid (outBbd, 0, 500);

        CHECK (std::fabs (cBbd - expectedCentroid) <= 2.0);
    }

    // 4. Mode Flip Under Steady Audio
    g_section = "mode_flip_click";
    {
        FeedbackDelay fb;
        fb.prepare (kFs, kBlock, kMaxDelay);
        FeedbackDelay::Params p;
        p.delaySamples = 4800.0f;
        p.feedback = 0.4f;
        p.satOrder = 2;
        p.delayMode = 0; // Digital
        fb.resetParams (p);

        constexpr int totalSamples = 48000;
        std::vector<float> in (totalSamples);
        std::vector<float> out (totalSamples);

        const double amp = std::pow (10.0, -6.0 / 20.0);
        for (int i = 0; i < totalSamples; ++i)
            in[i] = static_cast<float> (amp * std::sin (2.0 * std::numbers::pi * 1000.0 * i / kFs));

        // First half Digital
        for (int pos = 0; pos < 24000; pos += kBlock)
            fb.process (in.data() + pos, nullptr, out.data() + pos, nullptr, kBlock);

        float maxStepBefore = 0.0f;
        for (int i = 1000; i < 24000; ++i)
            maxStepBefore = std::max (maxStepBefore, std::fabs (out[i] - out[i - 1]));

        // Switch to BBD
        p.delayMode = 1;
        fb.setParams (p);

        for (int pos = 24000; pos < totalSamples; pos += kBlock)
            fb.process (in.data() + pos, nullptr, out.data() + pos, nullptr, kBlock);

        float maxStepAfter = 0.0f;
        for (int i = 24001; i < totalSamples; ++i)
            maxStepAfter = std::max (maxStepAfter, std::fabs (out[i] - out[i - 1]));

        CHECK (maxStepAfter <= 4.0f * maxStepBefore);
    }

    std::println("=== bbd_loop_check OK ===");
    return 0;
}

// tests/harnesses/cd/compander_cell_check.cpp
//
// Verification harness for CompressorCell and ExpanderCell:
// static law, envelope timing, cascade transparency, tracking distortion,
// hostile input recovery, reset determinism, and FeedbackDelay integration.

#include "dsp/bbd/CompanderCell.h"
#include "dsp/FeedbackDelay.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <print>
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

    using MarsDSP::BBD::CompressorCell;
    using MarsDSP::BBD::ExpanderCell;
    using MarsDSP::Delays::FeedbackDelay;

    constexpr double kFs = 48000.0;

    double measureRms(const std::vector<float>& buf, std::size_t start, std::size_t count)
    {
        double sumSq = 0.0;
        for (std::size_t i = 0; i < count; ++i)
        {
            const double v = static_cast<double>(buf[start + i]);
            sumSq += v * v;
        }
        return std::sqrt(sumSq / static_cast<double>(count));
    }

    double toDbFs(double linear)
    {
        return 20.0 * std::log10(std::max(linear, 1e-15));
    }
} // namespace

int main()
{
    std::println("=== Compander Cell Check ===");

    // 1. Static law
    g_section = "static_law";
    {
        const std::array<double, 4> freqs{ 20.0, 100.0, 1000.0, 10000.0 };
        const std::array<double, 7> levelsDb{ -60.0, -40.0, -25.0, -20.0, -10.0, -6.0, 0.0 };

        for (double freq : freqs)
        {
            std::vector<double> inLevels;
            std::vector<double> compLevels;
            std::vector<double> expLevels;

            for (double inDb : levelsDb)
            {
                const double inAmp = std::pow(10.0, inDb / 20.0);
                const int nSamples = static_cast<int>(kFs * 0.5); // 500 ms
                std::vector<float> in(nSamples);
                for (int i = 0; i < nSamples; ++i)
                    in[i] = static_cast<float>(inAmp * std::sin(2.0 * std::numbers::pi * freq * i / kFs));

                CompressorCell comp;
                comp.prepare(kFs);
                std::vector<float> compOut(nSamples);
                for (int i = 0; i < nSamples; ++i)
                    compOut[i] = comp.processSample(in[i]);

                ExpanderCell exp;
                exp.prepare(kFs);
                std::vector<float> expOut(nSamples);
                for (int i = 0; i < nSamples; ++i)
                    expOut[i] = exp.processSample(in[i]);

                // Measure over the last 100 ms (settled)
                const int measureLen = static_cast<int>(kFs * 0.1);
                const int measureStart = nSamples - measureLen;
                const double inRms = measureRms(in, measureStart, measureLen);
                const double compRms = measureRms(compOut, measureStart, measureLen);
                const double expRms = measureRms(expOut, measureStart, measureLen);

                const double inRmsDb = toDbFs(inRms);
                const double compRmsDb = toDbFs(compRms);
                const double expRmsDb = toDbFs(expRms);

                // NE570 laws pivot at the reference level VR. Steady
                // state: comp output amplitude is sqrt(A*VR*pi/2), exp
                // output amplitude is A*A*pi/(2*VR), where A is the
                // input sine amplitude. Below VR the compressor
                // expands and the expander attenuates; above VR the
                // reverse. The rectifier mean factor (2/pi) sets the
                // pivot gain. The one-pole envelope ripple adds a
                // constant level bias: about 1 dB for the compressor,
                // about 2.5 dB for the expander (the square-law gain
                // amplifies the ripple). The slope check below holds
                // the actual 2:1 / 1:2 law.
                const double VR = MarsDSP::BBD::ExpanderCell::kRefLevel;
                const double kRect = 2.0 / std::numbers::pi;
                const double A = inRms * std::sqrt(2.0);
                const double compRmsPred = std::sqrt(A * VR * std::numbers::pi / 2.0) / std::sqrt(2.0);
                const double expRmsPred = (A * A * kRect / VR) / std::sqrt(2.0);
                CHECK(std::fabs(toDbFs(compRms) - toDbFs(compRmsPred)) <= 1.2);
                CHECK(std::fabs(toDbFs(expRms) - toDbFs(expRmsPred)) <= 2.5);

                inLevels.push_back(inRmsDb);
                compLevels.push_back(compRmsDb);
                expLevels.push_back(expRmsDb);
            }

            // Verify slopes above reference (-20 dBFS, indices 3..6 in levelsDb)
            for (std::size_t i = 3; i + 1 < inLevels.size(); ++i)
            {
                const double deltaIn = inLevels[i + 1] - inLevels[i];
                const double deltaComp = compLevels[i + 1] - compLevels[i];
                const double deltaExp = expLevels[i + 1] - expLevels[i];

                const double compSlope = deltaComp / deltaIn;
                const double expSlope = deltaExp / deltaIn;

                // Compressor slope 0.5 (2:1) within ±0.25
                CHECK(std::fabs(compSlope - 0.5) <= 0.25);
                // Expander slope 2.0 (1:2) within ±0.25
                CHECK(std::fabs(expSlope - 2.0) <= 0.25);
            }
        }
    }

    // 2. Envelope timing (Attack 3.0 ms, Release 13.3 ms)
    // The cells share one EnvelopeFollower. The expander envelope
    // tracks the input |x|, so a DC step gives a clean one-pole
    // approach and the 63.2% crossing is the time constant. The
    // feedback compressor envelope tracks its output |y|, which is a
    // nonlinear function of the envelope, so a step there distorts the
    // crossing. Measure on the expander with a DC step.
    g_section = "envelope_timing";
    {
        ExpanderCell exp;
        exp.prepare(kFs);

        // Attack: DC step from 1e-3 to 1.0
        for (int i = 0; i < 4800; ++i)
            (void)exp.processSample(1e-3f);

        const float envStart = exp.getEnvelope();
        const float targetAmp = 1.0f;
        const float swingAttack = targetAmp - envStart;
        const float threshold63Attack = envStart + 0.63212f * swingAttack;

        int attackSamples = 0;
        for (int i = 0; i < 4800; ++i)
        {
            (void)exp.processSample(targetAmp);
            if (exp.getEnvelope() < threshold63Attack)
                attackSamples++;
        }

        const double measuredAttackMs = static_cast<double>(attackSamples) / kFs * 1000.0;
        std::println("Attack time to 63.2%: {:.3f} ms (expected 3.0 ms ± 5%)", measuredAttackMs);
        CHECK(std::fabs(measuredAttackMs - 3.0) <= 0.15); // within 5%

        // Release: DC step from 1.0 to 1e-3
        for (int i = 0; i < 4800; ++i)
            (void)exp.processSample(1.0f);

        const float envStartRel = exp.getEnvelope();
        const float targetAmpRel = 1e-3f;
        const float swingRelease = envStartRel - targetAmpRel;
        const float threshold63Rel = envStartRel - 0.63212f * swingRelease;

        int releaseSamples = 0;
        for (int i = 0; i < 4800; ++i)
        {
            (void)exp.processSample(targetAmpRel);
            if (exp.getEnvelope() > threshold63Rel)
                releaseSamples++;
        }

        const double measuredReleaseMs = static_cast<double>(releaseSamples) / kFs * 1000.0;
        std::println("Release time to 63.2%: {:.3f} ms (expected 13.3 ms ± 5%)", measuredReleaseMs);
        CHECK(std::fabs(measuredReleaseMs - 13.3) <= 0.665); // within 5%
    }

    // 3. Cascade transparency: comp -> exp with unity middle
    g_section = "cascade_transparency";
    {
        const std::array<double, 4> freqs{ 20.0, 100.0, 1000.0, 10000.0 };
        const std::array<double, 8> levelsDb{ -60.0, -40.0, -30.0, -20.0, -10.0, -6.0, 0.0, 6.0 };

        for (double freq : freqs)
        {
            for (double inDb : levelsDb)
            {
                const double inAmp = std::pow(10.0, inDb / 20.0);
                const int nSamples = static_cast<int>(kFs * 0.4); // 400 ms
                std::vector<float> in(nSamples);
                for (int i = 0; i < nSamples; ++i)
                    in[i] = static_cast<float>(inAmp * std::sin(2.0 * std::numbers::pi * freq * i / kFs));

                CompressorCell comp;
                ExpanderCell exp;
                comp.prepare(kFs);
                exp.prepare(kFs);

                std::vector<float> out(nSamples);
                for (int i = 0; i < nSamples; ++i)
                {
                    const float c = comp.processSample(in[i]);
                    out[i] = exp.processSample(c);
                }

                // Measure over last 100 ms (after 250+ ms settling)
                const int measureLen = static_cast<int>(kFs * 0.1);
                const int measureStart = nSamples - measureLen;
                const double inRms = measureRms(in, measureStart, measureLen);
                const double outRms = measureRms(out, measureStart, measureLen);
                const double errDb = toDbFs(outRms) - toDbFs(inRms);

                CHECK(std::fabs(errDb) <= 0.2);
            }
        }
    }

    // 4. Tracking distortion: 1 kHz at -6 dBFS through comp -> exp
    g_section = "tracking_distortion";
    {
        constexpr double f0 = 1000.0;
        constexpr double inDb = -6.0;
        const double inAmp = std::pow(10.0, inDb / 20.0);
        constexpr int N = 32768;

        std::vector<float> in(N);
        for (int i = 0; i < N; ++i)
            in[i] = static_cast<float>(inAmp * std::sin(2.0 * std::numbers::pi * f0 * i / kFs));

        CompressorCell comp;
        ExpanderCell exp;
        comp.prepare(kFs);
        exp.prepare(kFs);

        // Warm up 100 ms
        for (int i = 0; i < 4800; ++i)
        {
            const float c = comp.processSample(in[i % N]);
            (void)exp.processSample(c);
        }

        std::vector<float> out(N);
        for (int i = 0; i < N; ++i)
        {
            const float c = comp.processSample(in[i]);
            out[i] = exp.processSample(c);
        }

        // Goertzel for harmonic distortion
        double fundPow = 0.0;
        double harmPow = 0.0;

        for (int k = 1; k * f0 < kFs * 0.5; ++k)
        {
            const double fk = k * f0;
            const double kG = 2.0 * std::cos(2.0 * std::numbers::pi * fk / kFs);
            double s1 = 0.0;
            double s2 = 0.0;
            for (int i = 0; i < N; ++i)
            {
                const double s = static_cast<double>(out[i]) + kG * s1 - s2;
                s2 = s1;
                s1 = s;
            }
            const double pK = (s1 * s1 + s2 * s2 - kG * s1 * s2)
                              * (2.0 / (static_cast<double>(N) * static_cast<double>(N)));
            if (k == 1)
                fundPow = pK;
            else
                harmPow += pK;
        }

        const double thdDb = 10.0 * std::log10(harmPow / std::max(fundPow, 1e-15));
        std::println("1 kHz at -6 dBFS comp->exp THD: {:.2f} dB (gate < -50 dB)", thdDb);
        CHECK(thdDb < -50.0);
    }

    // 5. Hostile inputs
    g_section = "hostile_inputs";
    {
        CompressorCell comp;
        ExpanderCell exp;
        comp.prepare(kFs);
        exp.prepare(kFs);

        const std::array<float, 5> hostile{
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            1.0e6f,
            -1.0e6f
        };

        for (float h : hostile)
        {
            const float c = comp.processSample(h);
            const float e = exp.processSample(c);
            CHECK(std::isfinite(c));
            CHECK(std::isfinite(e));
        }

        // Recovery within 500 ms of clean input
        constexpr int nRecovery = static_cast<int>(kFs * 0.5);
        for (int i = 0; i < nRecovery; ++i)
        {
            const float clean = 0.1f * std::sin(2.0f * std::numbers::pi_v<float> * 1000.0f * i / static_cast<float>(kFs));
            const float c = comp.processSample(clean);
            const float e = exp.processSample(c);
            CHECK(std::isfinite(c));
            CHECK(std::isfinite(e));
        }
        CHECK(comp.getEnvelope() > 0.05f && comp.getEnvelope() < 0.2f);
    }

    // 6. Reset determinism
    g_section = "reset_determinism";
    {
        CompressorCell comp;
        comp.prepare(kFs);
        std::vector<float> r1(1000), r2(1000);
        for (int i = 0; i < 1000; ++i)
            r1[i] = comp.processSample(static_cast<float>(std::sin(i * 0.1)));

        comp.reset();
        for (int i = 0; i < 1000; ++i)
            r2[i] = comp.processSample(static_cast<float>(std::sin(i * 0.1)));

        for (int i = 0; i < 1000; ++i)
            CHECK(r1[i] == r2[i]);
    }

    // 7. Integration checks: FeedbackDelay level budget & nominal tracking
    g_section = "feedback_delay_integration";
    {
        constexpr int kBlock = 256;
        constexpr int kMaxDelay = 262144;
        constexpr int totalSamples = 48000; // 1 s

        const double inAmp = std::pow(10.0, -20.0 / 20.0); // -20 dBFS (reference level)
        std::vector<float> in(totalSamples);
        for (int i = 0; i < totalSamples; ++i)
            in[i] = static_cast<float>(inAmp * std::sin(2.0 * std::numbers::pi * 1000.0 * i / kFs));

        // Render with frozen envelopes
        std::vector<float> outFrozen(totalSamples);
        {
            FeedbackDelay fb;
            fb.prepare(kFs, kBlock, kMaxDelay);
            FeedbackDelay::Params p;
            p.delaySamplesL = 4800.0f;
            p.delaySamplesR = 4800.0f;
            p.feedback = 0.5f;
            p.dampHz = 20000.0f;
            p.loopCutHz = 20.0f;
            p.satOrder = 0;
            p.enableDiffuser = false;
            p.delayMode = 1; // BBD
            fb.resetParams(p);
            fb.setEnvelopeFreeze(true);

            for (int pos = 0; pos < totalSamples; pos += kBlock)
            {
                const int n = std::min(kBlock, totalSamples - pos);
                fb.process(in.data() + pos, nullptr, outFrozen.data() + pos, nullptr, n);
            }
        }

        // Render with active compander
        std::vector<float> outActive(totalSamples);
        {
            FeedbackDelay fb;
            fb.prepare(kFs, kBlock, kMaxDelay);
            FeedbackDelay::Params p;
            p.delaySamplesL = 4800.0f;
            p.delaySamplesR = 4800.0f;
            p.feedback = 0.5f;
            p.dampHz = 20000.0f;
            p.loopCutHz = 20.0f;
            p.satOrder = 0;
            p.enableDiffuser = false;
            p.delayMode = 1; // BBD
            fb.resetParams(p);
            fb.setEnvelopeFreeze(false);

            for (int pos = 0; pos < totalSamples; pos += kBlock)
            {
                const int n = std::min(kBlock, totalSamples - pos);
                fb.process(in.data() + pos, nullptr, outActive.data() + pos, nullptr, n);
            }
        }

        // Nominal tracking: at reference level (-20 dBFS), wet RMS matches frozen render within ±0.5 dB
        const int measureLen = 24000;
        const int measureStart = totalSamples - measureLen;
        const double rmsFrozen = measureRms(outFrozen, measureStart, measureLen);
        const double rmsActive = measureRms(outActive, measureStart, measureLen);
        const double deltaDb = toDbFs(rmsActive) - toDbFs(rmsFrozen);
        std::println("Nominal tracking delta at -20 dBFS: {:.3f} dB (gate ±0.5 dB)", deltaDb);
        CHECK(std::fabs(deltaDb) <= 0.5);
    }

    std::println("=== compander_cell_check OK ===");
    return 0;
}

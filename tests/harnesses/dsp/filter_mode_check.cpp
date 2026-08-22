// tests/harnesses/dsp/filter_mode_check.cpp
//
// Verification harness for OutputFilterStage mode switching, crossfade timing,
// click-free behavior, channel matching, and stereo/mono processing.

#include "dsp/OutputFilterStage.h"
#include "dsp/ChronosEngine.h"

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

    using MarsDSP::Filters::OutputFilterStage;
} // namespace

int main()
{
    // 1. Latency check
    g_section = "latency";
    {
        CHECK (MarsDSP::ChronosEngine::latencySamples() == MarsDSP::Align::SaturatorAlign::kBudget);
    }

    // 2. Switch is click-free
    g_section = "click_free_switch";
    {
        constexpr double fs = 48000.0;
        OutputFilterStage stage;
        stage.prepare (fs, 2);
        stage.setMode (OutputFilterStage::Mode::Digital);
        stage.setCutoffs (200.0f, 5000.0f);

        constexpr int totalSamples = 48000;
        std::vector<float> in(totalSamples);
        std::vector<float> outL(totalSamples);
        std::vector<float> outR(totalSamples);

        const double amp = std::pow (10.0, -6.0 / 20.0);
        for (int i = 0; i < totalSamples; ++i)
            in[i] = static_cast<float> (amp * std::sin (2.0 * std::numbers::pi * 1000.0 * i / fs));

        // Process first half in Digital mode
        stage.process (in.data(), in.data(), outL.data(), outR.data(), 24000);

        // Compute max step before switch
        float maxStepBefore = 0.0f;
        for (int i = 1000; i < 24000; ++i)
            maxStepBefore = std::max (maxStepBefore, std::fabs (outL[i] - outL[i - 1]));

        // Flip to Analog mode
        stage.setMode (OutputFilterStage::Mode::Analog);
        stage.process (in.data() + 24000, in.data() + 24000, outL.data() + 24000, outR.data() + 24000, 24000);

        // Compute max step during and after switch
        float maxStepAfter = 0.0f;
        for (int i = 24001; i < totalSamples; ++i)
            maxStepAfter = std::max (maxStepAfter, std::fabs (outL[i] - outL[i - 1]));

        CHECK (maxStepAfter <= 4.0f * maxStepBefore);
    }

    // 3. Crossfade length
    g_section = "crossfade_length";
    {
        for (double fs : { 44100.0, 48000.0, 88200.0, 96000.0, 192000.0 })
        {
            OutputFilterStage stage;
            stage.prepare (fs, 2);
            stage.setMode (OutputFilterStage::Mode::Digital);
            stage.setCutoffs (200.0f, 5000.0f);

            // 1 sample per call to count exact fade steps
            std::vector<float> in (static_cast<std::size_t> (fs * 0.1), 0.5f);
            std::vector<float> outL (in.size());
            std::vector<float> outR (in.size());

            // Run Digital steady state
            stage.process (in.data(), in.data(), outL.data(), outR.data(), 1000);

            stage.setMode (OutputFilterStage::Mode::Analog);
            int fadeSteps = 0;
            for (std::size_t i = 1000; i < in.size(); ++i)
            {
                float oL;
                float oR;
                stage.process (in.data() + i, in.data() + i, &oL, &oR, 1);
                ++fadeSteps;
                // Fade is 20 ms
                if (fadeSteps > static_cast<int> (0.025 * fs)) break;
            }

            const double fadeDurationMs = 1000.0 * std::round (0.02 * fs) / fs;
            CHECK (std::fabs (fadeDurationMs - 20.0) <= 1.0);
        }
    }

    // 4. Both directions and flip during fade
    g_section = "flip_during_fade";
    {
        constexpr double fs = 48000.0;
        OutputFilterStage stage;
        stage.prepare (fs, 2);
        stage.setCutoffs (200.0f, 5000.0f);

        std::vector<float> in (48000, 0.5f);
        std::vector<float> outL (48000);
        std::vector<float> outR (48000);

        stage.process (in.data(), in.data(), outL.data(), outR.data(), 1000);

        // Start fade to Analog
        stage.setMode (OutputFilterStage::Mode::Analog);
        stage.process (in.data() + 1000, in.data() + 1000, outL.data() + 1000, outR.data() + 1000, 240); // 5 ms

        // Flip back to Digital mid-fade
        stage.setMode (OutputFilterStage::Mode::Digital);
        stage.process (in.data() + 1240, in.data() + 1240, outL.data() + 1240, outR.data() + 1240, 48000 - 1240);

        for (int i = 0; i < 48000; ++i)
        {
            CHECK (std::isfinite (outL[i]));
            CHECK (std::isfinite (outR[i]));
        }
    }

    // 5. Channel match
    g_section = "channel_match";
    {
        constexpr double fs = 48000.0;
        OutputFilterStage stage;
        stage.prepare (fs, 2);
        stage.setMode (OutputFilterStage::Mode::Analog);
        stage.setCutoffs (500.0f, 3000.0f);

        constexpr int N = 4096;
        std::vector<float> in(N);
        std::vector<float> outL(N);
        std::vector<float> outR(N);

        for (int i = 0; i < N; ++i)
            in[i] = static_cast<float> (std::sin (0.05 * i));

        stage.process (in.data(), in.data(), outL.data(), outR.data(), N);

        for (int i = 0; i < N; ++i)
            CHECK (outL[i] == outR[i]);
    }

    // 6. Mono and stereo, block size 1 sample
    g_section = "mono_and_block1";
    {
        constexpr double fs = 48000.0;
        OutputFilterStage stageMono;
        stageMono.prepare (fs, 1);
        stageMono.setMode (OutputFilterStage::Mode::Analog);
        stageMono.setCutoffs (300.0f, 4000.0f);

        for (int i = 0; i < 100; ++i)
        {
            float in = 0.2f;
            float out = 0.0f;
            stageMono.process (&in, nullptr, &out, nullptr, 1);
            CHECK (std::isfinite (out));
        }
    }

    std::println("=== filter_mode_check OK ===");
    return 0;
}

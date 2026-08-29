#include "TapSimulation.h"
#include "utils/helpers/TempoSync.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace MarsDSP::GUI::TapSim {

SimulationResult Engine::simulate(const Parameters& params)
{
    SimulationResult result;

    float timeL = std::clamp(params.timeLSeconds, 0.001f, 5.0f);
    float timeR = std::clamp(params.timeRSeconds, 0.001f, 5.0f);

    if (params.delaySync)
    {
        const double bpm = (params.secondsPerBeat > 0.0f) ? (60.0 / static_cast<double>(params.secondsPerBeat)) : 120.0;
        const double ms = Utils::Helpers::TempoSync::convertChoiceIndexToMilliseconds(params.delayDivision, bpm);
        const float syncedSec = static_cast<float>(std::clamp(ms, 1.0, 5000.0) * 0.001);
        timeL = syncedSec;
        timeR = syncedSec;
    }

    constexpr float kPi = std::numbers::pi_v<float>;
    const float thetaMix = std::clamp(params.mix * 0.01f, 0.0f, 1.0f) * (kPi * 0.5f);
    const float dryGain = std::cos(thetaMix);
    const float wetGain = std::sin(thetaMix);

    // Add dry taps at time zero.
    result.left.push_back(Tap{ .empty = false, .dry = true, .timeSeconds = 0.0f, .gain = dryGain });
    result.right.push_back(Tap{ .empty = false, .dry = true, .timeSeconds = 0.0f, .gain = dryGain });

    const float feedback = std::clamp(params.feedback, 0.0f, 1.2f);
    const float crossFeed = std::clamp(params.crossFeed, 0.0f, 1.0f);
    const float thetaCross = crossFeed * (kPi * 0.5f);
    const float cosCross = std::cos(thetaCross);
    const float sinCross = std::sin(thetaCross);

    const float maxWindow = std::max(0.5f, params.maxWindowSeconds);

    float sL = 1.0f;
    float sR = 1.0f - crossFeed;
    float curTimeL = timeL;
    float curTimeR = timeR;

    constexpr int kMaxRepeats = 120;
    for (int n = 1; n <= kMaxRepeats; ++n)
    {
        const bool outOfWindowL = curTimeL > maxWindow;
        const bool outOfWindowR = curTimeR > maxWindow;
        if (outOfWindowL && outOfWindowR)
            break;

        if (!outOfWindowL && std::fabs(sL) > 1e-6f)
        {
            result.left.push_back(Tap{
                .empty = false,
                .dry = false,
                .timeSeconds = curTimeL,
                .gain = sL * wetGain
            });
        }

        if (!outOfWindowR && std::fabs(sR) > 1e-6f)
        {
            result.right.push_back(Tap{
                .empty = false,
                .dry = false,
                .timeSeconds = curTimeR,
                .gain = sR * wetGain
            });
        }

        if (feedback <= 0.0f)
            break;

        const float nextL = feedback * (cosCross * sL + sinCross * sR);
        const float nextR = feedback * (cosCross * sR + sinCross * sL);

        if (std::max(std::fabs(nextL), std::fabs(nextR)) < 1e-5f)
            break;

        sL = nextL;
        sR = nextR;
        curTimeL += timeL;
        curTimeR += timeR;
    }

    const float lastL = result.left.empty() ? 0.0f : result.left.back().timeSeconds;
    const float lastR = result.right.empty() ? 0.0f : result.right.back().timeSeconds;
    result.totalTimeSeconds = std::max(0.25f, std::max(lastL, lastR));

    return result;
}

} // namespace MarsDSP::GUI::TapSim

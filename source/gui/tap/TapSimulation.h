#pragma once

#ifndef CHRONOS_TAP_SIMULATION_H
#define CHRONOS_TAP_SIMULATION_H

#include <cmath>
#include <vector>

namespace MarsDSP::GUI::TapSim {

// One tap event in the simulated tap train.
struct Tap {
    bool empty = false;
    bool dry = false;
    float timeSeconds = 0.0f;
    float gain = 0.0f;
};

// Parameters for the tap train simulation.
struct Parameters {
    float timeLSeconds = 0.375f;
    float timeRSeconds = 0.375f;
    float feedback = 0.42f;
    float crossFeed = 0.0f;
    float mix = 35.0f;
    bool delaySync = false;
    int delayDivision = 11;
    float secondsPerBeat = 0.5f;
    float maxWindowSeconds = 2.0f;
};

// The result of one simulation run.
struct SimulationResult {
    std::vector<Tap> left;
    std::vector<Tap> right;
    float totalTimeSeconds = 0.25f;
};

// The simulation engine that computes the tap train.
class Engine {
public:
    // Simulate the tap train for the given parameters.
    [[nodiscard]] static SimulationResult simulate(const Parameters& params);
};

} // namespace MarsDSP::GUI::TapSim

#endif

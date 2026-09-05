#pragma once

#ifndef CHRONOS_FACTORY_PRESETS_H
#define CHRONOS_FACTORY_PRESETS_H

#include <cstddef>

namespace MarsDSP::Presets {

// One parameter override in a factory preset.
// A parameter absent from the table takes its default.
struct FactoryPresetPair {
    const char* paramID;
    float value;
};

// One compiled-in factory preset.
struct FactoryPreset {
    const char* name;
    const char* bank;
    const FactoryPresetPair* pairs;
    int numPairs;
};

// Basics / Init.
inline constexpr FactoryPresetPair kInitPairs[] = {};

// Basics / Slapback.
inline constexpr FactoryPresetPair kSlapbackPairs[] = {
    { "delayTime", 110.0f },
    { "delayTimeR", 110.0f },
    { "timeLink", 1.0f },
    { "delaySync", 0.0f },
    { "feedback", 0.12f },
    { "dampHz", 6500.0f },
    { "loopCutHz", 90.0f },
    { "crossFeed", 0.0f },
    { "hpfFreq", 60.0f },
    { "lpfFreq", 12000.0f },
    { "mix", 28.0f },
};

// Sync / Quarter Note.
inline constexpr FactoryPresetPair kQuarterNotePairs[] = {
    { "delaySync", 1.0f },
    { "delayDivision", 11.0f },
    { "feedback", 0.45f },
    { "crossFeed", 0.15f },
    { "dampHz", 8000.0f },
    { "mix", 35.0f },
};

// Sync / Dotted Eighth.
inline constexpr FactoryPresetPair kDottedEighthPairs[] = {
    { "delaySync", 1.0f },
    { "delayDivision", 10.0f },
    { "feedback", 0.5f },
    { "crossFeed", 0.25f },
    { "dampHz", 7000.0f },
    { "mix", 32.0f },
};

// Analog / BBD Wash.
inline constexpr FactoryPresetPair kBbdWashPairs[] = {
    { "delayMode", 1.0f },
    { "delayTime", 420.0f },
    { "delayTimeR", 420.0f },
    { "feedback", 0.62f },
    { "dampHz", 3200.0f },
    { "loopCutHz", 120.0f },
    { "crossFeed", 0.4f },
    { "loopDrive", 4.0f },
    { "enableDiffuser", 1.0f },
    { "diffusion", 0.7f },
    { "diffuserSize", 0.6f },
    { "diffModDepth", 0.5f },
    { "diffModRateHz", 0.3f },
    { "lpfFreq", 6500.0f },
    { "mix", 45.0f },
};

// Analog / Tape Drift.
inline constexpr FactoryPresetPair kTapeDriftPairs[] = {
    { "delayMode", 1.0f },
    { "timeLink", 0.0f },
    { "delayTime", 300.0f },
    { "delayTimeR", 305.0f },
    { "feedback", 0.55f },
    { "delayModDepth", 12.0f },
    { "delayModRateHz", 0.28f },
    { "dampHz", 4500.0f },
    { "loopDrive", 6.0f },
    { "loopSatOrder", 2.0f },
    { "drive", 6.0f },
    { "bits", 12.0f },
    { "mix", 40.0f },
};

// Space / Ping Pong.
inline constexpr FactoryPresetPair kPingPongPairs[] = {
    { "timeLink", 0.0f },
    { "delayTime", 250.0f },
    { "delayTimeR", 375.0f },
    { "crossFeed", 0.85f },
    { "feedback", 0.5f },
    { "dampHz", 9000.0f },
    { "mix", 38.0f },
};

// Space / Ambient Bloom.
inline constexpr FactoryPresetPair kAmbientBloomPairs[] = {
    { "delayTime", 700.0f },
    { "delayTimeR", 700.0f },
    { "feedback", 0.78f },
    { "enableDiffuser", 1.0f },
    { "diffusion", 0.9f },
    { "diffuserSize", 0.85f },
    { "diffModDepth", 0.8f },
    { "diffModRateHz", 0.2f },
    { "dampHz", 5000.0f },
    { "loopCutHz", 150.0f },
    { "hpfFreq", 120.0f },
    { "lpfFreq", 9000.0f },
    { "mix", 55.0f },
};

// The compiled-in factory bank. Read-only, listed first, never written to disk.
inline constexpr FactoryPreset kFactoryPresets[] = {
    { "Init",            "Basics",      kInitPairs,    0 },
    { "Slapback",        "Basics",      kSlapbackPairs,   11 },
    { "Quarter Note",    "Sync",        kQuarterNotePairs,    6 },
    { "Dotted Eighth",   "Sync",        kDottedEighthPairs,    6 },
    { "BBD Wash",        "Analog",      kBbdWashPairs,   15 },
    { "Tape Drift",      "Analog",      kTapeDriftPairs,   13 },
    { "Ping Pong",       "Space",       kPingPongPairs,    7 },
    { "Ambient Bloom",   "Space",       kAmbientBloomPairs,   13 },
};

inline constexpr int kNumFactoryPresets =
    static_cast<int>(std::size(kFactoryPresets));

} // namespace MarsDSP::Presets

#endif

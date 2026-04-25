#include <iostream>
#include <fstream>
#include <cmath>
#include <numbers>
#include <vector>
#include <iomanip>
#include <filesystem>
#include "dsp/dynamics/bridge_ducker.h"

// Smoke test for the WDF-bridge sidechain ducker.
//
// Topology:
//   - Sidechain (dry): repeating "kick" - decaying sine at 80 Hz every 250 ms.
//   - Wet path:         sustained 440 Hz sine (think: a steady delay tail).
//
// We drive the ducker block-by-block, get a per-block gain, and apply it
// to the wet samples for that block via the BlockLerpSIMD ramp the ducker
// produces. The kick should crash the gain to a low floor; the gap
// between kicks should let it swell back toward unity.
//
// Block size is 512 (a power-of-two so the BlockLerpSIMD line is exact).

int main()
{
    namespace D = MarsDSP::DSP::Dynamics;

    constexpr double sampleRate     = 48000.0;
    constexpr int    durationSamples= 48000; // 1 s
    constexpr int    kBlockSize     = 512;
    constexpr float  kickPeriodSec  = 0.25f;
    constexpr float  kickFreqHz     = 80.0f;
    constexpr float  kickDecayMs    = 60.0f;
    constexpr float  wetFreqHz      = 440.0f;

    std::filesystem::create_directories ("tests/simd_harness/logs");

    std::vector<float> dryL (durationSamples, 0.0f);
    std::vector<float> dryR (durationSamples, 0.0f);
    std::vector<float> wetL (durationSamples, 0.0f);
    std::vector<float> wetR (durationSamples, 0.0f);

    // Dry sidechain: repeating decaying-sine kicks.
    {
        const int   periodSamples = static_cast<int>(kickPeriodSec * sampleRate);
        const float decayCoef     = std::exp (-1.0f / (kickDecayMs * 1.0e-3f * static_cast<float>(sampleRate)));
        for (int n = 0; n < durationSamples; ++n)
        {
            const int   nIntoKick = n % periodSamples;
            const float t         = static_cast<float>(nIntoKick) / static_cast<float>(sampleRate);
            const float envD      = std::pow (decayCoef, static_cast<float>(nIntoKick));
            const float s         = envD * std::sin (2.0f * std::numbers::pi_v<float> * kickFreqHz * t);
            dryL[n] = s;
            dryR[n] = s;
        }
    }

    // Wet: sustained sine at 70% of full scale.
    {
        for (int n = 0; n < durationSamples; ++n)
        {
            const float t = static_cast<float>(n) / static_cast<float>(sampleRate);
            const float s = 0.7f * std::sin (2.0f * std::numbers::pi_v<float> * wetFreqHz * t);
            wetL[n] = s;
            wetR[n] = s;
        }
    }

    std::vector<float> wetLOriginal = wetL;

    // Build the ducker. -24 dB threshold, 90% amount, fast attack, gentle release.
    D::BridgeDucker<kBlockSize, float> ducker;
    ducker.prepare (sampleRate, kBlockSize, /* numChannels */ 2);
    ducker.setBypassed (false);
    ducker.prepareBlock (/* thresholdDb */  -24.0f,
                         /* amount */         0.9f,
                         /* attackMs */       2.0f,
                         /* releaseMs */    100.0f);

    struct BlockTelemetry
    {
        int   sampleIndex;
        float envelope;
        float gain;
    };
    std::vector<BlockTelemetry> telemetry;
    telemetry.reserve (durationSamples / kBlockSize + 1);

    int processed = 0;
    while (processed + kBlockSize <= durationSamples)
    {
        const int n = kBlockSize;

        // 1) Resolve the WDF bridge for this block from the dry sidechain.
        ducker.resolveBlock (dryL.data() + processed,
                             dryR.data() + processed,
                             n);

        // 2) Apply the per-quad gain ramp to wetL/wetR in place.
        const auto& line = ducker.getGainLine();
        for (int q = 0; q < n / 4; ++q)
        {
            const auto vGain = line.quad (q);
            const auto vWetL = SIMD_MM(loadu_ps)(wetL.data() + processed + q * 4);
            const auto vWetR = SIMD_MM(loadu_ps)(wetR.data() + processed + q * 4);
            SIMD_MM(storeu_ps)(wetL.data() + processed + q * 4, SIMD_MM(mul_ps)(vWetL, vGain));
            SIMD_MM(storeu_ps)(wetR.data() + processed + q * 4, SIMD_MM(mul_ps)(vWetR, vGain));
        }

        telemetry.push_back ({
            processed + n - 1,
            ducker.getBlockEndEnvelope(),
            ducker.getBlockEndGain()
        });

        processed += n;
    }

    // Per-sample audio CSV.
    {
        std::ofstream csv ("tests/simd_harness/logs/simd_ducker_smoke_audio.csv");
        if (!csv.is_open()) { std::cerr << "open audio csv failed\n"; return 1; }
        csv << "n,t,dryL,wetL_in,wetL_out\n";
        csv << std::fixed << std::setprecision (8);
        for (int n = 0; n < durationSamples; ++n)
        {
            csv << n << ","
                << (static_cast<double>(n) / sampleRate) << ","
                << dryL[n] << ","
                << wetLOriginal[n] << ","
                << wetL[n] << "\n";
        }
    }

    // Per-block control CSV (drive column kept for backwards-compat with
    // the existing matplotlib viz; we leave it equal to (1 - gain) for now).
    {
        std::ofstream csv ("tests/simd_harness/logs/simd_ducker_smoke_control.csv");
        if (!csv.is_open()) { std::cerr << "open control csv failed\n"; return 1; }
        csv << "block_end_n,t,envelope,drive,gain\n";
        csv << std::fixed << std::setprecision (8);
        for (const auto& s : telemetry)
        {
            csv << s.sampleIndex << ","
                << (static_cast<double>(s.sampleIndex) / sampleRate) << ","
                << s.envelope << ","
                << (1.0f - s.gain) << ","
                << s.gain << "\n";
        }
    }

    float minGain = 1.0f;
    for (const auto& s : telemetry) minGain = std::min (minGain, s.gain);

    const bool gainSane = (minGain < 0.6f) && (minGain > 0.0f);
    std::cout << "Generated tests/simd_harness/logs/simd_ducker_smoke_audio.csv ("
              << durationSamples << " samples, " << telemetry.size() << " blocks).\n"
              << "  block size       = " << kBlockSize << " samples\n"
              << "  threshold        = -24 dB,  amount = 0.9\n"
              << "  attack/release   = 2 / 100 ms\n"
              << "  min block-end gain = " << minGain
              << (gainSane ? "  [OK]" : "  [WARN: ducking did not engage strongly]")
              << std::endl;

    return gainSane ? 0 : 1;
}

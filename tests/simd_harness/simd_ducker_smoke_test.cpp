#include <iostream>
#include <fstream>
#include <cmath>
#include <numbers>
#include <vector>
#include <iomanip>
#include <filesystem>
#include "dsp/dynamics/sinh_ducker.h"

// Smoke test for the diode-bridge sinh ducker.
//
// Scenario:
//   - Sidechain (dry): a synthesized "kick" - decaying sine at 80 Hz - retriggered every 250 ms.
//   - Wet path: a sustained 440 Hz sine (think of it as a steady delay tail).
//   - The ducker should pull the wet signal down on each kick and let it
//     swell back during the gaps.
//
// We deliberately invoke the ducker block-by-block (kBlockSize) to mimic
// the delay engine's process() call pattern; per-block prepareBlock and
// per-block processBlockStereo with non-quad-aligned tail handling is
// exercised by deliberately picking a block size that isn't a multiple
// of the inner SIMD stride (we use 511 = 4*127 + 3 → 3 scalar tail
// samples per block).

int main()
{
    namespace D = MarsDSP::DSP::Dynamics;

    constexpr double sampleRate     = 48000.0;
    constexpr int    durationSamples= 48000; // 1 s
    constexpr int    kBlockSize     = 511;   // intentionally non-quad-multiple
    constexpr float  kickPeriodSec  = 0.25f; // 4 hits/s
    constexpr float  kickFreqHz     = 80.0f;
    constexpr float  kickDecayMs    = 60.0f;
    constexpr float  wetFreqHz      = 440.0f;

    std::filesystem::create_directories ("tests/simd_harness/logs");

    // --- Build the input signals ---
    std::vector<float> dryL (durationSamples, 0.0f);
    std::vector<float> dryR (durationSamples, 0.0f);
    std::vector<float> wetL (durationSamples, 0.0f);
    std::vector<float> wetR (durationSamples, 0.0f);

    // Sidechain: repeating kick. Phase resets every period so each hit
    // looks like a fresh transient to the envelope follower.
    {
        const int    periodSamples = static_cast<int>(kickPeriodSec * sampleRate);
        const float  decayCoef     = std::exp (-1.0f / (kickDecayMs * 1.0e-3f * static_cast<float>(sampleRate)));
        for (int n = 0; n < durationSamples; ++n)
        {
            const int    nIntoKick = n % periodSamples;
            const float  t         = static_cast<float>(nIntoKick) / static_cast<float>(sampleRate);
            const float  envD      = std::pow (decayCoef, static_cast<float>(nIntoKick));
            const float  s         = envD * std::sin (2.0f * std::numbers::pi_v<float> * kickFreqHz * t);
            dryL[n] = s;
            dryR[n] = s;
        }
    }

    // Wet: sustained sine, normalised to ±0.7 so we have headroom for the
    // bridge's saturation stage to be visible in the output.
    {
        for (int n = 0; n < durationSamples; ++n)
        {
            const float t = static_cast<float>(n) / static_cast<float>(sampleRate);
            const float s = 0.7f * std::sin (2.0f * std::numbers::pi_v<float> * wetFreqHz * t);
            wetL[n] = s;
            wetR[n] = s;
        }
    }

    // Stash an unprocessed copy of the wet for plotting reference.
    std::vector<float> wetLOriginal = wetL;

    // --- Configure and run the ducker block by block ---
    D::SinhDucker<float> ducker;
    ducker.prepare (sampleRate, kBlockSize, /* numChannels */ 2);

    // Threshold deliberately low (0.05) so the ducker latches quickly on
    // the kick; amount near 1 to make the gain reduction obvious;
    // attack 1 ms / release 80 ms for a clean punch-then-swell.
    ducker.prepareBlock (/* threshold */        0.05f,
                         /* amount */           0.95f,
                         /* attackMs */         1.0f,
                         /* releaseMs */        80.0f,
                         /* saturationAmount */ 0.4f,
                         /* seriesScale */      1.5f);

    // Telemetry: per-block snapshots of (envelope, drive, gain) so the
    // visualisation can overlay them with the audio waveforms.
    struct BlockTelemetry
    {
        int   sampleIndex;
        float envelope;
        float drive;
        float gainL;
    };
    std::vector<BlockTelemetry> telemetry;
    telemetry.reserve (durationSamples / kBlockSize + 1);

    int processed = 0;
    while (processed < durationSamples)
    {
        const int n = std::min (kBlockSize, durationSamples - processed);

        // The host typically would call prepareBlock here too if any
        // params changed - we leave them static for the smoke test.
        ducker.processBlockStereo (dryL.data() + processed,
                                   dryR.data() + processed,
                                   wetL.data() + processed,
                                   wetR.data() + processed,
                                   n);

        telemetry.push_back ({
            processed + n - 1,
            ducker.getBlockEndEnvelope(),
            ducker.getBlockEndDrive(),
            ducker.getBlockEndGain (0)
        });

        processed += n;
    }

    // --- Per-sample CSV: wet_in, wet_out for the L channel + dry ---
    {
        std::ofstream csv ("tests/simd_harness/logs/simd_ducker_smoke_audio.csv");
        if (!csv.is_open())
        {
            std::cerr << "Failed to open simd_ducker_smoke_audio.csv\n";
            return 1;
        }
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

    // --- Per-block CSV: control signals ---
    {
        std::ofstream csv ("tests/simd_harness/logs/simd_ducker_smoke_control.csv");
        if (!csv.is_open())
        {
            std::cerr << "Failed to open simd_ducker_smoke_control.csv\n";
            return 1;
        }
        csv << "block_end_n,t,envelope,drive,gain\n";
        csv << std::fixed << std::setprecision (8);
        for (const auto& s : telemetry)
        {
            csv << s.sampleIndex << ","
                << (static_cast<double>(s.sampleIndex) / sampleRate) << ","
                << s.envelope << ","
                << s.drive    << ","
                << s.gainL    << "\n";
        }
    }

    // Quick numeric sanity: the kick peaks should drive gain well below 1.
    float minGain = 1.0f;
    for (const auto& s : telemetry) minGain = std::min (minGain, s.gainL);

    const bool gainSane = minGain < 0.6f && minGain > 0.0f;
    std::cout << "Generated tests/simd_harness/logs/simd_ducker_smoke_audio.csv ("
              << durationSamples << " samples, " << telemetry.size() << " blocks).\n"
              << "  block size = " << kBlockSize << " samples (deliberately non-quad)\n"
              << "  min block-end gain = " << minGain
              << (gainSane ? "  [OK]" : "  [WARN: ducking did not engage strongly]")
              << std::endl;

    return gainSane ? 0 : 1;
}

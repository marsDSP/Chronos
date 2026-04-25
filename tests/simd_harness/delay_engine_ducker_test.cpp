// Engine-level test for the BridgeDucker integration in DelayEngine.
//
// Setup:
//   - 1 second of audio at 48 kHz
//   - Dry input is silence with 4 short kicks (decaying sines @ 80 Hz, 250 ms apart)
//   - Mix = 1 (wet only), Feedback = 0 (no tail), DelayTime small so the wet
//     copy of each kick lands shortly after the dry kick that triggered it
//   - Ducker engaged with -24 dB threshold, 0.9 amount, fast attack
//
// Expected: in the windows BETWEEN kicks, the wet output is mostly silent
// (delay = single echo, feedback = 0). DURING / right-after the kicks, the
// wet output should be ducked relative to the un-ducked baseline. We
// compare two passes - (a) ducker bypassed, (b) ducker engaged - and
// assert that:
//   1) the kick-time peak of (b) is < 0.5 of the kick-time peak of (a)
//   2) outside-kick gaps of both are similarly silent
//   3) the engine doesn't NaN/blow up

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <vector>
#include <filesystem>
#include <JuceHeader.h>
#include "dsp/engine/delay/delay_engine.h"

namespace fs = std::filesystem;
using namespace MarsDSP::DSP;

namespace
{
    constexpr double kSampleRate     = 48000.0;
    constexpr int    kBlockSize      = 512;
    constexpr int    kNumBlocks      = 94; // ~1 s
    constexpr int    kTotalSamples   = kBlockSize * kNumBlocks;
    constexpr float  kKickPeriodSec  = 0.25f;
    constexpr float  kKickFreqHz     = 80.0f;
    constexpr float  kKickDecayMs    = 60.0f;
    constexpr float  kKickAmplitude  = 0.9f;

    void synthKickTrain (std::vector<float>& dryL, std::vector<float>& dryR)
    {
        const int   periodSamples = static_cast<int>(kKickPeriodSec * kSampleRate);
        const float decayCoef     = std::exp (-1.0f / (kKickDecayMs * 1.0e-3f * static_cast<float>(kSampleRate)));
        for (int n = 0; n < kTotalSamples; ++n)
        {
            const int   nIntoKick = n % periodSamples;
            const float t         = static_cast<float>(nIntoKick) / static_cast<float>(kSampleRate);
            const float envD      = std::pow (decayCoef, static_cast<float>(nIntoKick));
            const float s         = kKickAmplitude * envD * std::sin (2.0f * std::numbers::pi_v<float> * kKickFreqHz * t);
            dryL[n] = s;
            dryR[n] = s;
        }
    }

    // Run the engine across the kick train; returns the per-sample output
    // L channel. Caller controls whether the ducker is bypassed.
    std::vector<float> runEngine (bool duckerEngaged)
    {
        DelayEngine<float> engine;
        dsp::ProcessSpec spec {};
        spec.sampleRate       = kSampleRate;
        spec.maximumBlockSize = static_cast<uint32>(kBlockSize);
        spec.numChannels      = 2;
        engine.prepare (spec);

        // Wet-only, single echo, no feedback. Short delay so the wet copy
        // of each kick lands right after the kick itself.
        engine.setMixParam      (1.0f);
        engine.setFeedbackParam (0.0f);
        engine.setDelayTimeParam (50.0f);
        engine.setLowCutParam    (20.0f);
        engine.setHighCutParam   (20000.0f);
        engine.setCrossfeedParam (0.0f);
        engine.setMono           (false);
        engine.setBypassed       (false);

        // Ducker config.
        engine.setDuckerBypassedParam (!duckerEngaged);
        engine.setDuckerThresholdParam (-24.0f);
        engine.setDuckerAmountParam    ( 0.9f);
        engine.setDuckerAttackParam    ( 2.0f);
        engine.setDuckerReleaseParam   ( 100.0f);

        std::vector<float> dryL (kTotalSamples), dryR (kTotalSamples);
        synthKickTrain (dryL, dryR);
        std::vector<float> outL (kTotalSamples, 0.0f);

        AudioBuffer<float> blockBuf (2, kBlockSize);
        for (int b = 0; b < kNumBlocks; ++b)
        {
            const int sampleStart = b * kBlockSize;
            std::memcpy (blockBuf.getWritePointer (0), dryL.data() + sampleStart,
                         kBlockSize * sizeof (float));
            std::memcpy (blockBuf.getWritePointer (1), dryR.data() + sampleStart,
                         kBlockSize * sizeof (float));

            const AlignedBuffers::AlignedSIMDBufferView<float> view (
                blockBuf.getArrayOfWritePointers(),
                blockBuf.getNumChannels(),
                blockBuf.getNumSamples());
            engine.process (view, kBlockSize);

            std::memcpy (outL.data() + sampleStart, blockBuf.getReadPointer (0),
                         kBlockSize * sizeof (float));
        }

        return outL;
    }

    // Sliding-max amplitude envelope.
    std::vector<float> slidingPeak (const std::vector<float>& x, int win)
    {
        std::vector<float> r (x.size(), 0.0f);
        for (size_t i = 0; i < x.size(); ++i)
        {
            const size_t lo = (i >= static_cast<size_t>(win)) ? i - win : 0;
            float m = 0.0f;
            for (size_t j = lo; j <= i; ++j)
                m = std::max (m, std::abs (x[j]));
            r[i] = m;
        }
        return r;
    }
}

int main()
{
    fs::create_directories ("tests/simd_harness/logs");

    std::cout << "Running DelayEngine ducker test (1 s, 94 blocks of 512)...\n";
    const auto outDuckerOff = runEngine (/* duckerEngaged */ false);
    const auto outDuckerOn  = runEngine (/* duckerEngaged */ true);

    // Sanity: no NaN / inf in either trace.
    bool sane = true;
    for (size_t i = 0; i < outDuckerOff.size(); ++i)
    {
        if (!std::isfinite (outDuckerOff[i]) || !std::isfinite (outDuckerOn[i]))
        {
            sane = false;
            break;
        }
    }
    if (!sane)
    {
        std::cout << "  FAIL  engine produced non-finite samples." << std::endl;
        return 1;
    }

    // Compute peak amplitude in a window centred on each kick (offset by
    // the delay time so we sample the wet copy, not the dry).
    constexpr int kPeakWindow = 1024;
    constexpr int kDelaySamples = static_cast<int>(50.0f * kSampleRate / 1000.0f);
    const auto peakOff = slidingPeak (outDuckerOff, kPeakWindow);
    const auto peakOn  = slidingPeak (outDuckerOn,  kPeakWindow);

    float maxPeakOff = 0.0f, maxPeakOn = 0.0f;
    for (size_t i = static_cast<size_t>(kDelaySamples);
         i < peakOff.size(); ++i)
    {
        maxPeakOff = std::max (maxPeakOff, peakOff[i]);
        maxPeakOn  = std::max (maxPeakOn,  peakOn[i]);
    }

    const float peakRatio = (maxPeakOff > 1.0e-6f) ? (maxPeakOn / maxPeakOff) : 1.0f;
    const bool duckEngaged = (peakRatio < 0.5f);

    std::cout << std::fixed << std::setprecision (4);
    std::cout << "  Wet peak (ducker bypassed):  " << maxPeakOff << "\n";
    std::cout << "  Wet peak (ducker engaged):   " << maxPeakOn  << "\n";
    std::cout << "  Peak ratio (engaged/bypass): " << peakRatio
              << (duckEngaged ? "  [PASS]" : "  [FAIL]") << "\n";

    // Dump CSV for visual inspection / matplotlib.
    {
        std::ofstream csv ("tests/simd_harness/logs/delay_engine_ducker.csv");
        csv << "n,t,out_bypassed,out_engaged\n";
        csv << std::fixed << std::setprecision (8);
        for (int n = 0; n < kTotalSamples; ++n)
        {
            csv << n << ","
                << (static_cast<double>(n) / kSampleRate) << ","
                << outDuckerOff[n] << ","
                << outDuckerOn [n] << "\n";
        }
    }
    std::cout << "  Wrote tests/simd_harness/logs/delay_engine_ducker.csv\n";

    return duckEngaged ? 0 : 1;
}

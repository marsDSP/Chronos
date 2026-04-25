#pragma once

#ifndef CHRONOS_BRIDGE_DUCKER_H
#define CHRONOS_BRIDGE_DUCKER_H

// ============================================================================
//  bridge_ducker.h
// ----------------------------------------------------------------------------
//  Sidechain ducker for the delay taps. Three stages chained together:
//
//   1) Asymmetric one-pole peak follower runs on the (mono) sum of the dry
//      input. Per-sample, gives a smoothed envelope that tracks transients
//      sharply on the way up and decays gently on the way down.
//
//   2) The block-end envelope value is mapped to a saturation current Is
//      on a WDF antiparallel diode pair wired as a shunt in a
//      voltage-divider attenuator. As env grows, Is grows, the diode
//      dynamic resistance r_d = V_T / I_S drops, more signal is shunted
//      to ground, and the bridge's V_out / V_in ratio drops with the
//      natural soft-knee shape of the diode I/V curve.
//
//   3) The resulting block-end gain is linearly interpolated across the
//      audio block via BlockLerpSIMD so the per-sample multiply is a
//      single quad-load against a precomputed line. This keeps the WDF
//      cost O(1) per block while still giving per-sample-smooth gain.
//
//  Topology (signal-flow):
//
//      wet --- R_series ---+--- ducked_wet
//                          |
//                       (DiodePair shunt to GND, Is modulated by env)
//                          |
//                         GND
//
//  We probe the WDF at a small test voltage (well below V_T so the diode
//  stays in its linear region), measure the V_out / V_in ratio, and that
//  ratio IS the ducker gain we apply to the wet path. The wet path
//  itself never goes through the WDF - we just use it as a static
//  characteriser of "what would a bridge with this Is do to a unit
//  signal at this loudness". That keeps the audio inner loop a single
//  multiply per sample.
// ============================================================================

#include <algorithm>
#include <cmath>
#include "dsp/dynamics/envelope_follower.h"
#include "dsp/dynamics/block_lerp.h"
#include "dsp/wdf/wdf_core.h"
#include "dsp/wdf/wdf_diode_pair.h"

namespace MarsDSP::DSP::Dynamics
{
    template <int MaxBlockSize, typename SampleType = float>
    class BridgeDucker
    {
    public:
        BridgeDucker()
            : voltageSource (kSeriesResistanceOhms),
              polarityInverter (voltageSource),
              diode (polarityInverter, kIsMin, kVt, 1.0f)
        {
        }

        void prepare (double sampleRateHz, int maxBlockSize, int numChannels)
        {
            sampleRate    = static_cast<float>(sampleRateHz);
            audioChannels = std::max (numChannels, 1);

            envFollower.prepare (sampleRateHz, maxBlockSize, /* numChannels */ 1);

            // BlockLerpSIMD requires power-of-two; round down to the
            // largest pow2 ≤ min(maxBlockSize, MaxBlockSize).
            const int safeBlock = std::min (maxBlockSize, MaxBlockSize);
            int powTwoBlock = 4;
            while ((powTwoBlock << 1) <= safeBlock) powTwoBlock <<= 1;
            gainLerp.setBlockSize (powTwoBlock);
            gainLerp.instantize (1.0f);

            blockEndGain = 1.0f;
            blockEndEnv  = 0.0f;
            voltageSource.setResistanceValue (kSeriesResistanceOhms);
            diode.setDiodeParameters (kIsMin, kVt, 1.0f);
        }

        void prepareBlock (float thresholdDb,
                           float amountNorm,
                           float attackMs,
                           float releaseMs) noexcept
        {
            thresholdLin = std::pow (10.0f, std::clamp (thresholdDb, -60.0f, 0.0f) * 0.05f);
            amount       = std::clamp (amountNorm, 0.0f, 1.0f);
            envFollower.prepareBlock (attackMs, releaseMs);
        }

        void setBypassed (bool b) noexcept { bypassed = b; }

        // Run env follower on the dry sidechain, resolve the WDF bridge
        // once at the block-end envelope value, then push the resulting
        // gain into the BlockLerpSIMD line for the audio loop to consume.
        // dryR may be the same pointer as dryL for mono sidechain.
        void resolveBlock (const SampleType* __restrict dryL,
                           const SampleType* __restrict dryR,
                           int numSamples) noexcept
        {
            if (numSamples <= 0) return;

            // Per-sample env follower; we keep the latched block-end value.
            for (int i = 0; i < numSamples; ++i)
            {
                const SampleType mono = SampleType{0.5} * (dryL[i] + dryR[i]);
                envFollower.processSample (mono, /*channel*/ 0);
            }
            envFollower.latchBlockEnd();
            blockEndEnv = static_cast<float>(envFollower.getBlockEndEnv (0));

            if (bypassed || amount < 1.0e-4f)
            {
                blockEndGain = 1.0f;
                gainLerp.setTarget (1.0f);
                return;
            }

            // Map env to saturation current. Below threshold: Is = Is_min
            // (diode is essentially off, r_d -> infinity, bridge invisible).
            // Well above threshold: Is approaches Is_max (r_d -> small,
            // bridge shorts to GND, gain -> 0).
            const float overshoot = std::max (0.0f, blockEndEnv - thresholdLin);

            // Drive in the [0, ~10] band; 1 - exp(-drive) gives a
            // perceptually pleasant compressive curve in [0, 1).
            const float drive    = overshoot * amount * kAmountToDriveScale;
            const float biasNorm = 1.0f - std::exp (-drive);

            // Log-linear interpolation between Is_min and Is_max.
            const float Is = kIsMin * std::exp (biasNorm * kLogIsRange);
            diode.setDiodeParameters (Is, kVt, 1.0f);

            // Probe the bridge at a small test voltage. testV << V_T so
            // the diode stays in the linear region of its I/V; the
            // V_out/V_in ratio is then a pure function of (R_series, r_d)
            // and equals the gain we want to apply to the wet path.
            // The PolarityInverter between the source and the diode
            // flips the wave-domain sign convention so the probe voltage
            // reads positive on the diode-load side of the divider.
            voltageSource.setVoltage (kProbeVoltage);
            diode.incident (polarityInverter.reflected());
            polarityInverter.incident (diode.reflected());

            // Read the voltage across the diode (load) port: this is the
            // bridge's V_out (we read at the load, not the source).
            const float vOut = MarsDSP::DSP::WDF::voltage<float> (diode);
            const float gain = std::clamp (std::abs (vOut) / kProbeVoltage, 0.0f, 1.0f);

            blockEndGain = gain;
            gainLerp.setTarget (gain);
        }

        // The audio inner loop consumes the per-quad gain line.
        const BlockLerpSIMD<MaxBlockSize>& getGainLine() const noexcept
        {
            return gainLerp;
        }

        // Evaluate the gain ramp at sample n (scalar tail helper).
        float gainAt (int n) const noexcept
        {
            const int q = n >> 2;
            const int lane = n & 3;
            alignas(16) float lanes[4];
            SIMD_MM(store_ps)(lanes, gainLerp.quad (q));
            return lanes[lane];
        }

        void reset() noexcept
        {
            envFollower.reset();
            voltageSource.setResistanceValue (kSeriesResistanceOhms);
            diode.setDiodeParameters (kIsMin, kVt, 1.0f);
            gainLerp.instantize (1.0f);
            blockEndGain = 1.0f;
            blockEndEnv  = 0.0f;
        }

        [[nodiscard]] float getBlockEndGain()     const noexcept { return blockEndGain; }
        [[nodiscard]] float getBlockEndEnvelope() const noexcept { return blockEndEnv; }
        [[nodiscard]] float getThresholdLin()     const noexcept { return thresholdLin; }
        [[nodiscard]] bool  isBypassed()          const noexcept { return bypassed; }

    private:
        // ---- Topology constants ----//
        // Series resistor in the divider; 4.7 kΩ is a typical
        // small-signal analog audio attenuator R.
        static constexpr float kSeriesResistanceOhms = 4.7e3f;

        // Diode saturation current bounds. At Is = 1e-12 A the diode is
        // essentially infinite-impedance (no ducking). At Is = 1e-4 A
        // the dynamic resistance r_d = V_T / I_S = ~260 Ω forms a
        // divider with the 4.7 kΩ series resistor for ~-24 dB ducking,
        // which is a strong-but-not-silent floor that suits a creative
        // ducker (a true brick-wall feel can be reached by stacking
        // amount + threshold).
        static constexpr float kIsMin = 1.0e-12f;
        static constexpr float kIsMax = 1.0e-4f;
        static constexpr float kLogIsRange = 18.4206810f; // ln(Is_max / Is_min)

        // Thermal voltage at room temperature.
        static constexpr float kVt = 0.02585f;

        // Test voltage used to probe the bridge each block. 10 mV is
        // well below V_T (~26 mV) so the diode stays in the linear
        // region of its I/V curve and we get a pure ratio gain.
        static constexpr float kProbeVoltage = 0.01f;

        // Scaling factor mapping (env - threshold) into the bias drive
        // domain. Tuned so that env at full scale with amount = 1
        // saturates the curve close to Is_max (i.e. hits the -24 dB
        // floor), and amount = 0 produces no ducking regardless of
        // envelope.
        static constexpr float kAmountToDriveScale = 5.0f;

        // ---- State ----//
        EnvelopeFollower<SampleType> envFollower;

        WDF::ResistiveVoltageSource<float> voltageSource;
        WDF::PolarityInverter<float, WDF::ResistiveVoltageSource<float>> polarityInverter;
        WDF::DiodePair<float, WDF::PolarityInverter<float, WDF::ResistiveVoltageSource<float>>, WDF::DiodeQuality::Fast> diode;

        BlockLerpSIMD<MaxBlockSize> gainLerp;

        float sampleRate    { 48000.0f };
        int   audioChannels { 2 };
        float thresholdLin  { 0.0631f };  // -24 dB
        float amount        { 0.0f };
        float blockEndGain  { 1.0f };
        float blockEndEnv   { 0.0f };
        bool  bypassed      { true };

        BridgeDucker (const BridgeDucker&)            = delete;
        BridgeDucker& operator= (const BridgeDucker&) = delete;
    };
}
#endif

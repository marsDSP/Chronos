#pragma once

#ifndef CHRONOS_SINH_DIODE_BRIDGE_H
#define CHRONOS_SINH_DIODE_BRIDGE_H

// ============================================================================
//  sinh_diode_bridge.h
// ----------------------------------------------------------------------------
//  Models a 4-diode bridge wired as a shunt voltage-divider attenuator,
//  whose dynamic shunt resistance is set by a control-voltage that maps
//  through sinh().
//
//  Physical sketch (one channel):
//
//      x[n] ─── R_series ──┬── y[n]
//                          │
//                       (bridge)        ← shunt to ground
//                          │
//                         GND
//
//  Diode small-signal conductance scales with bias current:
//
//      g_shunt(I_bias) = I_bias / V_T
//
//  and the bias current is what comes out of the sinh control law fed
//  by the (smoothed) sidechain envelope:
//
//      I_bias_norm = sinh(drive)        with drive ∈ [0, ~π]
//
//  Voltage-divider gain in normalised units (R_series == 1):
//
//      gain = R_shunt / (R_series + R_shunt)
//           = 1 / (1 + g_shunt · seriesScale)
//           = 1 / (1 + seriesScale · sinh(drive))
//
//  drive = 0 ⇒ gain = 1, drive grows ⇒ gain → 0. The mapping is concave
//  for small drive (gentle compression around threshold) and aggressive
//  past it (sinh's exponential tail), which gives a satisfying "thump"
//  on transients.
//
//  Saturation. The audio passing *through* the bridge slightly modulates
//  the diodes' operating point, so as a tap is ducked it is also softly
//  clipped. We model this with the project's ADAA-tanh waveshaper whose
//  drive scales with the (normalised) bias level - quiet sidechain ⇒
//  no saturation, slammed sidechain ⇒ noticeable analog colour.
//
//  Block / sample APIs match the project's wow_engine / flutter_engine
//  pattern: prepare() once, processQuad() consumes a SIMD_M128 of audio
//  plus a SIMD_M128 of *normalised* bias drive values produced by the
//  ducker's control law. The bridge does not own the drive smoothing -
//  that is the upstream envelope follower's job.
// ============================================================================

#include <algorithm>
#include <vector>
#include "dsp/math/fastermath.h"
#include "dsp/math/simd/simd_config.h"
#include "dsp/engine/delay/waveshaper.h"

namespace MarsDSP::DSP::Dynamics
{
    template <typename SampleType = float>
    class SinhDiodeBridge
    {
    public:
        SinhDiodeBridge() = default;

        void prepare (double sampleRateHz, int maxBlockSize, int numChannels)
        {
            (void) sampleRateHz; (void) maxBlockSize;
            saturators.assign (static_cast<size_t>(std::max (numChannels, 1)),
                               MarsDSP::Waveshapers::ADAATanh{});
            blockEndGain.assign (saturators.size(), SampleType{1});
        }
        // Normalised series resistance (R_series / R_shunt0). Larger values
        // make the ducker bite harder for the same bias drive. Sensible
        // range [0.25, 4.0]; default 1.0 reproduces the canonical
        // analog-bridge gain curve.
        void setSeriesScale (float v) noexcept
        {
            seriesScale = std::max (0.001f, v);
        }

        // Saturation amount in [0, 1]. Scales how aggressively the ADAA
        // tanh kicks in as the bias rises. 0 = clean attenuation, 1 =
        // full vintage colour at peak ducking.
        void setSaturationAmount (float v) noexcept
        {
            saturationAmount = std::clamp (v, 0.0f, 1.0f);
        }

        // Maximum effective drive for the ADAA tanh (in tanh-input units).
        // Higher values clip harder per unit of bias. 4 gives a Neve-ish
        // softclip at peak ducking without going into hard-rail.
        void setMaxSaturationDrive (float v) noexcept
        {
            maxSaturationDrive = std::max (0.5f, v);
        }

        // ---------------- per-quad ----------------
        // wet     : 4 audio samples
        // drive   : 4 sinh-input drive values; should already be clamped to
        //           [0, ~π] by the upstream control law.
        //
        // Returns 4 ducked + softly saturated samples. seriesScale,
        // saturationAmount and maxSaturationDrive are read with their
        // current values (intentionally lock-free / lightly atomic - the
        // host should set them block-rate from prepareBlock).
        SIMD_M128 processQuad (SIMD_M128 wet, SIMD_M128 drive, size_t channelIndex) noexcept
        {
            // 1) attenuation gain per lane: 1 / (1 + seriesScale·sinh(drive))
            const auto biasI    = fasterSinh (drive);
            const auto vSeries  = SIMD_MM(set1_ps)(seriesScale);
            const auto vOne     = SIMD_MM(set1_ps)(1.0f);
            const auto denom    = SIMD_MM(add_ps)(vOne, SIMD_MM(mul_ps)(vSeries, biasI));
            const auto gain     = SIMD_MM(div_ps)(vOne, denom);

            // 2) attenuated audio, the post-divider voltage at the bridge.
            const auto y_pre    = SIMD_MM(mul_ps)(wet, gain);

            // 3) saturation drive scales with normalised bias level. Use
            // 1 - gain as a cheap stand-in for "how hard is the bridge
            // working" - it's 0 when not ducking and approaches 1 as the
            // gain crashes to 0.
            const auto biasNorm = SIMD_MM(sub_ps)(vOne, gain);
            const auto vSatAmt  = SIMD_MM(set1_ps)(saturationAmount * maxSaturationDrive);
            // satDrive = 1 + saturationAmount·maxDrive·biasNorm
            const auto satDrive = SIMD_MM(add_ps)(vOne, SIMD_MM(mul_ps)(vSatAmt, biasNorm));
            const auto driven   = SIMD_MM(mul_ps)(y_pre, satDrive);

            // 4) ADAA tanh, then divide back out by satDrive so the
            // small-signal response stays unity at zero bias.
            const auto shaped   = saturators[channelIndex].processQuad (driven);
            const auto y_sat    = SIMD_MM(div_ps)(shaped, satDrive);

            // 5) blend the linear and saturated branches by saturationAmount
            // so a 0% saturation knob is exactly the linear voltage divider.
            const auto vBlend   = SIMD_MM(set1_ps)(saturationAmount);
            const auto vKeep    = SIMD_MM(sub_ps)(vOne, vBlend);
            const auto y_out    = SIMD_MM(add_ps)(SIMD_MM(mul_ps)(vKeep, y_pre), SIMD_MM(mul_ps)(vBlend, y_sat));

            // Latch the lane-3 gain so the engine has a "block-end gain" to
            // visualise / cross-reference when debugging.
            alignas(16) float lanes[4];
            SIMD_MM(store_ps)(lanes, gain);
            blockEndGain[channelIndex] = static_cast<SampleType>(lanes[3]);

            return y_out;
        }

        // ---------------- per-sample ----------------
        // Convenience scalar fallback when a non-quad-aligned tail needs
        // to be processed. Routes through the SIMD overload to keep
        // outputs bit-identical with the quad-aligned section.
        SampleType processSample (SampleType wet, SampleType drive, size_t channelIndex) noexcept
        {
            const auto vWet = SIMD_MM(set1_ps)(static_cast<float>(wet));
            const auto vDrv = SIMD_MM(set1_ps)(static_cast<float>(drive));
            const auto vRes = processQuad (vWet, vDrv, channelIndex);
            alignas(16) float lanes[4];
            SIMD_MM(store_ps)(lanes, vRes);
            return static_cast<SampleType>(lanes[0]);
        }

        [[nodiscard]] SampleType getBlockEndGain (size_t channelIndex) const noexcept
        {
            return blockEndGain[channelIndex];
        }

        void reset() noexcept
        {
            for (auto& s : saturators) s.reset();
            std::fill (blockEndGain.begin(), blockEndGain.end(), SampleType{1});
        }

    private:
        float seriesScale        { 1.0f };
        float saturationAmount   { 0.5f };
        float maxSaturationDrive { 4.0f };

        std::vector<MarsDSP::Waveshapers::ADAATanh> saturators;
        std::vector<SampleType>                     blockEndGain;

        SinhDiodeBridge (const SinhDiodeBridge&)            = delete;
        SinhDiodeBridge& operator= (const SinhDiodeBridge&) = delete;
    };
}
#endif

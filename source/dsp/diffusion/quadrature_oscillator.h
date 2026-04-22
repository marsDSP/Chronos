#pragma once

#ifndef CHRONOS_QUADRATURE_OSCILLATOR_H
#define CHRONOS_QUADRATURE_OSCILLATOR_H

// ============================================================================
//  quadrature_oscillator.h
// ----------------------------------------------------------------------------
//  Stateful sine/cosine pair that emits two 90-degree-offset samples per
//  process() tick, recovered from a single 2x2 complex rotation matrix
//  applied to the running phase vector.
//
//  ChronosReverb uses this to modulate each of its four reverb blocks'
//  delay tap positions with a quartet of 90-degree-offset LFO waveforms
//  (cos, sin, -cos, -sin), which decorrelates the modulation across
//  blocks and produces a smoother, richer late-tail texture than a single
//  LFO would.
//
//  Algorithm: we keep a unit-length 2D state vector (cosinePhaseSample,
//  sinePhaseSample) and multiply it by the rotation matrix
//
//      [ cos(dθ)  -sin(dθ) ]
//      [ sin(dθ)   cos(dθ) ]
//
//  every call to advancePhaseByOneSample(). The rotation's trigonometric
//  constants are precomputed once per setRateInRadiansPerSample() call,
//  so the per-sample cost is four multiplies and two adds.
//
//  Drift over time: after N rotations the length of (r, i) is nominally
//  still 1, but floating-point accumulation will bleed a small amount in
//  either direction. The drift is acceptable because the ChronosReverb
//  modulation multiplier (~5 ms of delay at full amount) is robust to a
//  few percent of amplitude error. If a downstream caller ever needs
//  stricter invariance it can call snapToCosineSineUnitPhasor()
//  periodically.
// ============================================================================

#include <cmath>
#include <type_traits>

namespace MarsDSP::DSP::ChronosReverb
{
    // ----------------------------------------------------------------------------
    //  ChronosReverbQuadratureModulationOscillator<SampleType>
    // ----------------------------------------------------------------------------
    template <typename SampleType>
    class ChronosReverbQuadratureModulationOscillator
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "ChronosReverbQuadratureModulationOscillator requires a floating-point sample type.");

        // ------------------------------------------------------------------
        //  Reset the phasor to (1, 0) - i.e. cosine = 1, sine = 0 - which
        //  is the phase angle 0. Useful at plugin instantiation / transport
        //  restart so two instances of the reverb produce identical LFO
        //  positions in lockstep when their oscillators were previously
        //  running at different speeds.
        // ------------------------------------------------------------------
        void snapToCosineSineUnitPhasor() noexcept
        {
            cosinePhaseSample = static_cast<SampleType>(1);
            sinePhaseSample   = static_cast<SampleType>(0);
        }

        // ------------------------------------------------------------------
        //  setRateInRadiansPerSample: set the rotation increment dθ per
        //  advancePhaseByOneSample() call. Precomputes cos(dθ) and sin(dθ)
        //  so the inner loop doesn't have to call std::cos/std::sin.
        //
        //  Typical usage from ChronosReverb:
        //      lfo.setRateInRadiansPerSample(
        //          2.0 * M_PI * lfoFrequencyHz / sampleRateHz);
        //  For ChronosReverb's 2^-2 = 0.25 Hz default, at 48 kHz this is
        //  dθ ≈ 3.27e-5 radians per sample.
        // ------------------------------------------------------------------
        void setRateInRadiansPerSample(SampleType rotationIncrementRadians) noexcept
        {
            cachedCosineOfRotationIncrement =
                static_cast<SampleType>(std::cos(static_cast<double>(rotationIncrementRadians)));
            cachedSineOfRotationIncrement =
                static_cast<SampleType>(std::sin(static_cast<double>(rotationIncrementRadians)));
        }

        // ------------------------------------------------------------------
        //  advancePhaseByOneSample: rotate the internal phasor by one
        //  dθ step. After the call, getCosineSample() and getSineSample()
        //  return the new values.
        //
        //  Derivation: multiplying the current (cos(θ), sin(θ)) by the
        //  rotation matrix above gives (cos(θ+dθ), sin(θ+dθ)), so long
        //  as (cos(dθ), sin(dθ)) have been precomputed.
        // ------------------------------------------------------------------
        void advancePhaseByOneSample() noexcept
        {
            const SampleType newCosine =
                cosinePhaseSample * cachedCosineOfRotationIncrement
                - sinePhaseSample  * cachedSineOfRotationIncrement;
            const SampleType newSine =
                cosinePhaseSample * cachedSineOfRotationIncrement
                + sinePhaseSample  * cachedCosineOfRotationIncrement;

            cosinePhaseSample = newCosine;
            sinePhaseSample   = newSine;
        }

        [[nodiscard]] SampleType getCosineSample() const noexcept { return cosinePhaseSample; }
        [[nodiscard]] SampleType getSineSample()   const noexcept { return sinePhaseSample;   }

    private:
        // The running quadrature phasor. On reset, initialised to the
        // angle-0 state (1, 0); advancePhaseByOneSample() then rotates it
        // by the cached increment.
        SampleType cosinePhaseSample { static_cast<SampleType>(1) };
        SampleType sinePhaseSample   { static_cast<SampleType>(0) };

        // Precomputed rotation-matrix constants. Updated only when the
        // caller sets a new rate; the inner loop never recomputes them.
        SampleType cachedCosineOfRotationIncrement { static_cast<SampleType>(1) };
        SampleType cachedSineOfRotationIncrement   { static_cast<SampleType>(0) };
    };
}
#endif

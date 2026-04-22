#pragma once

#ifndef CHRONOS_SPLANE_CURVEFIT_CORE_H
#define CHRONOS_SPLANE_CURVEFIT_CORE_H

// ============================================================================
//  splane_curvefit_core.h
// ----------------------------------------------------------------------------
//  S-plane parallel complex-pole filter section, discretised via the
//  BILINEAR TRANSFORM with cutoff pre-warping. Designed so an analog
//  prototype (Butterworth, measured curve-fit, etc.) can be expressed as
//  a partial-fraction decomposition
//
//      H(s) = directThroughGain + sum_k [ residue_k / (s - pole_k) ]
//
//  and realized as a bank of independent first-order BILINEAR sections.
//  Each analog term r_k / (s - p_k) becomes a first-order z-domain
//  section with one complex pole and a zero at z = -1:
//
//      zPole_k        = (2/T + p_k) / (2/T - p_k)
//      sectionGain_k  = r_k / (2/T - p_k)
//      y_k[n]         = zPole_k * y_k[n-1]
//                     + sectionGain_k * (u[n] + u[n-1])
//
//  Output per sample is then
//
//      outputSample[n] = directThroughGain * u[n] + Re(sum_k y_k[n])
//
//  Conjugate-pair poles / conjugate-pair residues produce a real output.
//
//  Why bilinear rather than impulse-invariance?
//  --------------------------------------------
//  Impulse-invariance (z-pole = exp(s-pole * T), gain = residue * T) is
//  numerically inexpensive and maps the analog impulse response exactly but
//  ALIASES the frequency response of any prototype that doesn't roll off
//  to zero at infinity, which includes every highpass and bandpass, and
//  any lowpass whose cutoff is a non-trivial fraction of Nyquist. In a
//  feedback loop the aliased passband gain can easily exceed 1 and blow
//  the loop up.
//
//  Bilinear maps the entire s-plane imaginary axis onto the unit circle
//  without aliasing and preserves the analog magnitude response exactly
//  (after pre-warping the cutoff), so for Butterworth prototypes (max
//  flat, |H| <= 1) we are guaranteed |H_discrete(e^jw)| <= 1 across all
//  frequencies. Stable in feedback for any fb < 1.
//
//  Pre-warping
//  -----------
//  Bilinear warps the frequency axis nonlinearly; to hit the
//  intended -3dB corner we pre-warp the desired angular cutoff via
//
//      w_analog = (2/T) * tan(w_target * T / 2)
//
//  before scaling the prototype poles / residues.
// ============================================================================

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

namespace MarsDSP::DSP::SPlaneCurveFit
{
    // ----------------------------------------------------------------------------
    //  Fixed section width: 4 complex poles / 4 complex residues per section.
    //  This matches the brigade's 4-wide SIMD width and is also exactly the
    //  partial-fraction decomposition of a 4th-order analog prototype
    //  (2 conjugate pairs). If higher-order sections are ever needed, the code
    //  below can be templates on kNumParallelComplexPoles; keeping it fixed
    //  for now to avoid bloating the header.
    // ----------------------------------------------------------------------------
    inline constexpr std::size_t kNumParallelComplexPoles = 4;

    // ----------------------------------------------------------------------------
    //  Packed container for the analog-domain prototype coefficients of one
    //  filter section. The caller is responsible for ordering poles / residues
    //  so that conjugate pairs appear consecutively; only convention, the
    //  discretization does not assume anything about ordering.
    // ----------------------------------------------------------------------------
    struct AnalogPrototypeCoefficients
    {
        // Poles of the analog prototype, normalized so that the effective
        // cutoff lies at 1 rad/s. They will be scaled by (2*pi*cutoffHz) at
        // configuration time.
        std::array<std::complex<float>, kNumParallelComplexPoles> analogPoles{};

        // Residues of the partial-fraction decomposition, normalized at the
        // same unit cutoff. They will also be scaled by (2*pi*cutoffHz).
        std::array<std::complex<float>, kNumParallelComplexPoles> analogResidues{};

        // Direct (s -> infinity) constant term of the partial-fraction
        // decomposition. Nonzero for highpass prototypes whose numerator
        // degree equals denominator degree; zero for strict lowpass.
        float directThroughGain{0.0f};
    };

    // ----------------------------------------------------------------------------
    //  ParallelComplexPoleSection
    //
    //  Realises a bank of kNumParallelComplexPoles first-order complex IIRs in
    //  parallel. Coefficients are recomputed only when the user calls
    //  configureForCutoff(); per-sample cost is dominated by
    //  kNumParallelComplexPoles complex multiply-adds.
    // ----------------------------------------------------------------------------
    class ParallelComplexPoleSection
    {
    public:
        ParallelComplexPoleSection() = default;

        // ------------------------------------------------------------------
        //  Configure the section from a prototype + desired cutoff. Must be
        //  called after the sample rate has been set via prepare().
        // ------------------------------------------------------------------
        void configureForCutoff(const AnalogPrototypeCoefficients &prototype,
                                float desiredCutoffInHz) noexcept
        {
            activePrototype = prototype;
            currentCutoffInHz = desiredCutoffInHz;
            cachedDirectThroughGain = prototype.directThroughGain;
            recomputeDiscreteCoefficients();
        }

        // ------------------------------------------------------------------
        //  Change only the cutoff; keeps the currently cached prototype.
        // ------------------------------------------------------------------
        void setCutoffInHz(float desiredCutoffInHz) noexcept
        {
            currentCutoffInHz = desiredCutoffInHz;
            recomputeDiscreteCoefficients();
        }

        // ------------------------------------------------------------------
        //  Set sample rate and (re)derive discrete coefficients if we already
        //  have a prototype configured.
        // ------------------------------------------------------------------
        void prepare(double hostSampleRateInHz) noexcept
        {
            hostSampleRate = hostSampleRateInHz;
            hostSampleInterval = 1.0 / hostSampleRateInHz;
            recomputeDiscreteCoefficients();
            resetInternalState();
        }

        // Clears all complex section states to zero; does not touch coefficient tables
        void resetInternalState() noexcept
        {
            for (auto &singleSectionState: perSectionComplexState)
                singleSectionState = std::complex(0.0f, 0.0f);

            previousInputSample = 0.0f;
        }

        // ------------------------------------------------------------------
        //  Process one real-valued input sample, produce one real-valued
        //  output sample.
        // ------------------------------------------------------------------
        [[nodiscard]] float processSingleSample(float inputSample) noexcept
        {
            // Bilinear's first-order section needs the sum of current and
            // previous input samples (from the zero at z = -1).
            const float bilinearInputSum = inputSample + previousInputSample;
            const std::complex bilinearInputSumComplex(bilinearInputSum, 0.0f);

            float accumulatedRealOutput = cachedDirectThroughGain * inputSample;

            for (std::size_t poleIndex = 0;
                 poleIndex < kNumParallelComplexPoles;
                 ++poleIndex)
            {
                // First-order bilinear recursion:
                //   y_k[n] = zPole_k * y_k[n-1] + sectionGain_k * (u[n] + u[n-1])
                perSectionComplexState[poleIndex] = discretePolesInZDomain[poleIndex]
                                                    * perSectionComplexState[poleIndex]
                                                    + discreteResiduesInZDomain[poleIndex]
                                                    * bilinearInputSumComplex;

                accumulatedRealOutput += perSectionComplexState[poleIndex].real();
            }

            previousInputSample = inputSample;
            return accumulatedRealOutput;
        }

        // Convenience wrappers to inspect current state (tests, GUI probes).
        [[nodiscard]] double getHostSampleRateInHz() const noexcept { return hostSampleRate; }
        [[nodiscard]] float getCurrentCutoffInHz() const noexcept { return currentCutoffInHz; }

    private:
        // Recompute bilinear coefficients from the current prototype,
        // cutoff and sample rate. Applies cutoff pre-warping so the
        // discretised filter's -3 dB point matches the user's requested
        // cutoff even when the cutoff is a significant fraction of
        // Nyquist.
        void recomputeDiscreteCoefficients() noexcept
        {
            if (hostSampleRate <= 0.0)
                return;

            // Clamp cutoff: keep away from DC (degenerate) and strictly
            // below Nyquist (pre-warp tan() goes to infinity exactly at
            // Nyquist).
            const float safeCutoffInHz = std::clamp(currentCutoffInHz, 0.01f, static_cast<float>(0.499 * hostSampleRate));

            const float sampleIntervalFloat = static_cast<float>(hostSampleInterval);

            // Pre-warp: convert the desired discrete cutoff to the
            // equivalent analog angular frequency that bilinear will map
            // back to the intended discrete one.
            const float angularCutoffTarget = static_cast<float>(2.0 * M_PI) * safeCutoffInHz;

            const float angularCutoffPrewarped = (2.0f / sampleIntervalFloat)
                                                 * std::tan(angularCutoffTarget
                                                 * sampleIntervalFloat * 0.5f);

            // Pre-computed bilinear scale factor (2/T). Appears in every
            // per-section coefficient.
            const std::complex bilinearScaleFactor(2.0f / sampleIntervalFloat, 0.0f);

            for (std::size_t poleIndex = 0;
                 poleIndex < kNumParallelComplexPoles;
                 ++poleIndex)
            {
                // Scale the normalized analog pole to the pre-warped cutoff.
                const std::complex<float> scaledAnalogPole = activePrototype.analogPoles[poleIndex]
                                                                * angularCutoffPrewarped;

                // Residues of both LPF and HPF Butterworth prototypes in
                // this file scale linearly with the cutoff, so a single
                // multiply suffices. (If a higher-order numerator
                // prototype is ever used, recompute residues analytically
                // instead of relying on linear scaling.)
                const std::complex<float> scaledAnalogResidue = activePrototype.analogResidues[poleIndex]
                                                                * angularCutoffPrewarped;

                // Bilinear per-section denominator: (2/T) - p_k.
                const std::complex<float> bilinearDenominator = bilinearScaleFactor - scaledAnalogPole;

                // z-plane pole: (2/T + p_k) / (2/T - p_k).
                discretePolesInZDomain[poleIndex] = (bilinearScaleFactor + scaledAnalogPole)
                                                        / bilinearDenominator;

                // Per-section gain applied to (u[n] + u[n-1]):
                //     r_k / (2/T - p_k).
                discreteResiduesInZDomain[poleIndex] = scaledAnalogResidue / bilinearDenominator;
            }
        }

        // Active analog prototype (the "what to fit" target: curve-fit
        // coefficients, Butterworth, Chebyshev, etc.).
        AnalogPrototypeCoefficients activePrototype{};

        // Host / discrete domain parameters.
        double hostSampleRate{44100.0};
        double hostSampleInterval{1.0 / 44100.0};
        float currentCutoffInHz{1000.0f};

        // Cached direct-through (s -> infinity) constant from the prototype.
        // Folded into processSingleSample() as a per-sample multiply-add.
        float cachedDirectThroughGain{0.0f};

        // Discretised coefficients (recomputed by recomputeDiscreteCoefficients).
        std::array<std::complex<float>, kNumParallelComplexPoles>
        discretePolesInZDomain{};

        std::array<std::complex<float>, kNumParallelComplexPoles>
        discreteResiduesInZDomain{};

        // Per-section complex IIR state. Reset on prepare() / resetInternalState().
        std::array<std::complex<float>, kNumParallelComplexPoles>
        perSectionComplexState{};

        // Previous input sample, shared across all sections (the zero at
        // z = -1 introduced by bilinear needs u[n-1]).
        float previousInputSample{0.0f};
    };

    // ----------------------------------------------------------------------------
    //  Helpers for constructing analog prototypes. Kept here rather than in
    //  the HPF / LPF headers because both share the "Butterworth poles at
    //  unit cutoff" primitive, and keeping it in one place means the unit
    //  tests only ever touch one copy of the pole table.
    // ----------------------------------------------------------------------------
    namespace ButterworthPrototypeBuilder
    {
        // 4 left-half-plane poles of the 4th-order Butterworth polynomial at
        // unit normalized cutoff (wc = 1 rad/s). Two conjugate pairs.
        //
        //   pole_k = exp( j * pi * (2k + 3) / 8 ) for k = 0..3, restricted to Re(pole) < 0.
        //
        // This is equivalent to:
        //
        //     p = -sin(pi/8) +/- j*cos(pi/8)
        //     p = -sin(3*pi/8) +/- j*cos(3*pi/8)
        inline std::array<std::complex<float>, kNumParallelComplexPoles>
        computeButterworthLeftHalfPlanePolesAtUnitCutoff() noexcept
        {
            const float sineOfPiOverEight        = std::sin(static_cast<float>(M_PI) / 8.0f);
            const float cosineOfPiOverEight      = std::cos(static_cast<float>(M_PI) / 8.0f);
            const float sineOfThreePiOverEight   = std::sin(3.0f * static_cast<float>(M_PI) / 8.0f);
            const float cosineOfThreePiOverEight = std::cos(3.0f * static_cast<float>(M_PI) / 8.0f);

            return
            {
                std::complex(-sineOfPiOverEight,      +cosineOfPiOverEight),
                std::complex(-sineOfPiOverEight,      -cosineOfPiOverEight),
                std::complex(-sineOfThreePiOverEight, +cosineOfThreePiOverEight),
                std::complex(-sineOfThreePiOverEight, -cosineOfThreePiOverEight),
            };
        }

        // Partial-fraction residues for the normalized-cutoff lowpass
        //
        //     H_lp(s) = 1 / prod_k (s - pole_k)
        //
        // Residue at pole_k is r_k = 1 / prod_{j != k}(pole_k - pole_j).
        inline std::array<std::complex<float>, kNumParallelComplexPoles>
        computeLowpassResiduesAtUnitCutoff(const std::array<std::complex<float>,
            kNumParallelComplexPoles> &analogPoles) noexcept
        {
            std::array<std::complex<float>, kNumParallelComplexPoles> residues {};

            for (std::size_t k = 0; k < kNumParallelComplexPoles; ++k)
            {
                std::complex denominatorProduct(1.0f, 0.0f);

                for (std::size_t j = 0; j < kNumParallelComplexPoles; ++j)
                    if (j != k)
                        denominatorProduct *= (analogPoles[k] - analogPoles[j]);

                residues[k] = std::complex(1.0f, 0.0f) / denominatorProduct;
            }

            return residues;
        }

        // Partial-fraction residues for the normalized-cutoff highpass
        //
        //     H_hp(s) = s^N / prod_k (s - pole_k) (where N = number of poles)
        //            = 1 + sum_k r_k / (s - pole_k)
        //
        // Residue at pole_k for this equal-degree case is
        //
        //     r_k = pole_k^N / prod_{j != k}(pole_k - pole_j).
        //
        inline std::array<std::complex<float>, kNumParallelComplexPoles>
        computeHighpassResiduesAtUnitCutoff(
            const std::array<std::complex<float>, kNumParallelComplexPoles> &analogPoles) noexcept
        {
            std::array<std::complex<float>, kNumParallelComplexPoles> residues {};

            for (std::size_t k = 0; k < kNumParallelComplexPoles; ++k)
            {
                std::complex<float> numeratorPower = analogPoles[k];

                for (std::size_t power = 1; power < kNumParallelComplexPoles; ++power)
                    numeratorPower *= analogPoles[k];

                std::complex denominatorProduct(1.0f, 0.0f);

                for (std::size_t j = 0; j < kNumParallelComplexPoles; ++j)
                    if (j != k)
                        denominatorProduct *= (analogPoles[k] - analogPoles[j]);

                residues[k] = numeratorPower / denominatorProduct;
            }

            return residues;
        }
    }
}
#endif

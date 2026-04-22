#pragma once

#ifndef CHRONOS_SPLANE_CURVEFIT_HIGHPASS_H
#define CHRONOS_SPLANE_CURVEFIT_HIGHPASS_H

// ============================================================================
//  splane_curvefit_highpass.h
// ----------------------------------------------------------------------------
//  4th-order curve-fit highpass filter. Uses a Butterworth analog prototype
//  as its default "target curve"; because the core section applies the
//  bilinear transform (see splane_curvefit_core.h), the analog response is
//  preserved exactly up to frequency pre-warping, and the discrete
//  magnitude response is bounded by 1 everywhere - safe to use inside a
//  feedback loop.
//
//  If you want to drop in measured curve-fit coefficients (e.g., a sampled
//  passive network response), replace the contents of buildAnalogPrototype()
//  with the fitted pole/residue tables; the rest of the pipeline is
//  prototype-agnostic.
//
//  Used on the feedback path of the delay engine in place of the old RBJ
//  biquad highpass. The feedback-path HPF keeps DC / subsonic build-up out
//  of the feedback loop.
// ============================================================================

#include "splane_curvefit_core.h"

namespace MarsDSP::DSP::SPlaneCurveFit
{
    class SPlaneCurveFitHighpassFilter
    {
    public:
        SPlaneCurveFitHighpassFilter() noexcept
        {
            // Build the analog prototype once at construction
            // subsequent prepare / setCutoff calls only rescale it.
            internalSection.configureForCutoff(buildAnalogPrototype(), currentCutoffInHz);
        }

        // ------------------------------------------------------------------
        //  Set sample rate. Call once before audio starts or when the
        //  host's sample rate changes.
        // ------------------------------------------------------------------
        void prepare(double hostSampleRateInHz) noexcept
        {
            internalSection.prepare(hostSampleRateInHz);
        }

        // Flush filter state; does not disturb coefficients.
        void reset() noexcept
        {
            internalSection.resetInternalState();
        }

        // Assign a new cutoff in Hz. Cheap; only recomputes discrete
        // coefficients, not residues/poles (prototype fixes those).
        void setCutoffFrequencyInHz(float desiredCutoffInHz) noexcept
        {
            currentCutoffInHz = desiredCutoffInHz;
            internalSection.setCutoffInHz(desiredCutoffInHz);
        }

        // Drop-in replacement for juce::dsp::IIR::Filter<float>::processSample()
        // style APIs: scalar sample-by-sample.
        [[nodiscard]] float processSample(float inputSample) noexcept
        {
            return internalSection.processSingleSample(inputSample);
        }

        [[nodiscard]] float getCutoffFrequencyInHz() const noexcept
        {
            return currentCutoffInHz;
        }

    private:
        // Construct the default Butterworth analog prototype packed into the
        // parallel complex-pole format the core section consumes.
        //
        // H_hp(s) = s^N / B_N(s) = 1 + sum_k r_k / (s - p_k)
        // with r_k = p_k^N / prod_{j!=k}(p_k - p_j), directThroughGain = 1.
        static AnalogPrototypeCoefficients buildAnalogPrototype() noexcept
        {
            AnalogPrototypeCoefficients prototype;

            prototype.analogPoles =
                ButterworthPrototypeBuilder::computeButterworthLeftHalfPlanePolesAtUnitCutoff();

            prototype.analogResidues =
                ButterworthPrototypeBuilder::computeHighpassResiduesAtUnitCutoff(prototype.analogPoles);

            // For a proper highpass with equal numerator / denominator
            // degree, the partial-fraction expansion has a constant
            // direct-through term equal to the leading-coefficient ratio,
            // which for a monic numerator s^N and monic denominator is 1.
            prototype.directThroughGain = 1.0f;

            return prototype;
        }

        // Default chosen to match the previous biquad default (20 Hz low
        // cut) so that ChronosProcessor continues to behave identically at
        // parameter defaults.
        float currentCutoffInHz{20.0f};

        ParallelComplexPoleSection internalSection {};
    };
}
#endif

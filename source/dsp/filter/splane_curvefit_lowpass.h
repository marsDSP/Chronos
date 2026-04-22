#pragma once

#ifndef CHRONOS_SPLANE_CURVEFIT_LOWPASS_H
#define CHRONOS_SPLANE_CURVEFIT_LOWPASS_H

// ============================================================================
//  splane_curvefit_lowpass.h
// ----------------------------------------------------------------------------
//  4th-order curve-fit lowpass filter. Uses a Butterworth analog
//  prototype as its default "target curve" - replace
//  buildAnalogPrototype() with fitted coefficients to model a measured
//  analog response.
//
//  Used on the feedback path of the delay engine: the feedback-path LPF
//  softens the high-frequency content of successive repeats and avoids
//  ultrasonic build-up.
// ============================================================================

#include "splane_curvefit_core.h"

namespace MarsDSP::DSP::SPlaneCurveFit
{
    class SPlaneCurveFitLowpassFilter
    {
    public:
        SPlaneCurveFitLowpassFilter() noexcept
        {
            internalSection.configureForCutoff(buildAnalogPrototype(), currentCutoffInHz);
        }

        void prepare(double hostSampleRateInHz) noexcept
        {
            internalSection.prepare(hostSampleRateInHz);
        }

        void reset() noexcept
        {
            internalSection.resetInternalState();
        }

        void setCutoffFrequencyInHz(float desiredCutoffInHz) noexcept
        {
            currentCutoffInHz = desiredCutoffInHz;
            internalSection.setCutoffInHz(desiredCutoffInHz);
        }

        [[nodiscard]] float processSample(float inputSample) noexcept
        {
            return internalSection.processSingleSample(inputSample);
        }

        [[nodiscard]] float getCutoffFrequencyInHz() const noexcept
        {
            return currentCutoffInHz;
        }

    private:
        static AnalogPrototypeCoefficients buildAnalogPrototype() noexcept
        {
            AnalogPrototypeCoefficients prototype;

            prototype.analogPoles =
                ButterworthPrototypeBuilder::computeButterworthLeftHalfPlanePolesAtUnitCutoff();

            prototype.analogResidues =
                ButterworthPrototypeBuilder::computeLowpassResiduesAtUnitCutoff(prototype.analogPoles);

            // Strict lowpass: numerator degree < denominator degree, so the
            // direct-through term is zero.
            prototype.directThroughGain = 0.0f;

            return prototype;
        }

        // Default chosen to match previous biquad default (20 kHz high cut).
        float currentCutoffInHz{20000.0f};

        ParallelComplexPoleSection internalSection{};
    };
}
#endif

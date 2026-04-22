#pragma once

#ifndef CHRONOS_FEEDBACK_SHELF_FILTER_H
#define CHRONOS_FEEDBACK_SHELF_FILTER_H

// ============================================================================
//  feedback_shelf_filter.h
// ----------------------------------------------------------------------------
//  First-order analog-prototype shelving filter (as in Abel & Berners
//  "DSP for Digital Audio Effects", ch. 5). Given:
//
//     lowFrequencyGain  - target gain at DC
//     highFrequencyGain - target gain at Nyquist
//     crossoverFrequencyHz - transition corner
//
//  the filter produces first-order numerator + denominator coefficients via
//  a pre-warped bilinear transform, then runs a transposed direct-form I
//  recursion per sample. Used per-channel in the feedback delay network to
//  bias long-time energy toward either end of the spectrum.
// ============================================================================

#include <cmath>
#include <type_traits>

namespace MarsDSP::DSP::Diffusion
{
    template<typename SampleType>
    class FeedbackShelfFilter
    {
    public:
        static_assert(std::is_floating_point_v<SampleType>,
                      "FeedbackShelfFilter requires a floating-point sample type.");

        FeedbackShelfFilter() noexcept { reset(); }

        void reset() noexcept
        {
            previousInputSample = SampleType{};
            previousOutputSample = SampleType{};
            numeratorCoefficientZero = SampleType{1};
            numeratorCoefficientOne = SampleType{};
            denominatorCoefficientOne = SampleType{};
        }

        // Set coefficients from target low / high gains and crossover freq.
        // Uses the standard pre-warped bilinear derivation: pole at the
        // crossover frequency, zero positioned to hit the specified gains.
        void computeCoefficientsForTargetGains(
            SampleType lowFrequencyGain,
            SampleType highFrequencyGain,
            SampleType crossoverFrequencyHz,
            double hostSampleRateInHz) noexcept
        {
            const double sampleIntervalSeconds = 1.0 / hostSampleRateInHz;

            // Pre-warp the crossover so bilinear lands it at the intended
            // discrete angular frequency.
            const double angularCutoffRadPerSec = 2.0 * M_PI * static_cast<double>(crossoverFrequencyHz);
            const double angularCutoffPrewarped = (2.0 / sampleIntervalSeconds) * std::tan(
                                                      angularCutoffRadPerSec * sampleIntervalSeconds * 0.5);

            // First-order shelf in analog domain:
            //    H(s) = lowGain * (s + a) / (s + b)
            //  with b = wc and a = wc * (lowGain / highGain) so that
            //  |H(0)| = lowGain and |H(inf)| = highGain.
            const double denominatorPole = angularCutoffPrewarped;
            const double numeratorZero = (highFrequencyGain == SampleType{})
                                            ? denominatorPole
                                            : denominatorPole
                                              * (static_cast<double>(lowFrequencyGain)
                                              / static_cast<double>(highFrequencyGain));

            // Bilinear: s = (2/T) (z - 1)/(z + 1).
            const double twoOverT = 2.0 / sampleIntervalSeconds;
            const double numA = static_cast<double>(highFrequencyGain) * (twoOverT + numeratorZero);
            const double numB = static_cast<double>(highFrequencyGain) * (-twoOverT + numeratorZero);
            const double denA = twoOverT + denominatorPole;
            const double denB = -twoOverT + denominatorPole;

            numeratorCoefficientZero = static_cast<SampleType>(numA / denA);
            numeratorCoefficientOne = static_cast<SampleType>(numB / denA);
            denominatorCoefficientOne = static_cast<SampleType>(denB / denA);
        }

        SampleType processSingleSample(SampleType inputSample) noexcept
        {
            // Direct-form I, 1st-order:
            //   y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
            const SampleType outputSample = numeratorCoefficientZero * inputSample
                                                + numeratorCoefficientOne   * previousInputSample
                                                - denominatorCoefficientOne * previousOutputSample;

            previousInputSample = inputSample;
            previousOutputSample = outputSample;
            return outputSample;
        }

    private:
        SampleType numeratorCoefficientZero{static_cast<SampleType>(1)};
        SampleType numeratorCoefficientOne{static_cast<SampleType>(0)};
        SampleType denominatorCoefficientOne{static_cast<SampleType>(0)};

        SampleType previousInputSample{static_cast<SampleType>(0)};
        SampleType previousOutputSample{static_cast<SampleType>(0)};
    };
}
#endif

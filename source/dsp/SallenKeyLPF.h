#pragma once

#ifndef CHRONOS_SALLEN_KEY_LPF_H
#define CHRONOS_SALLEN_KEY_LPF_H

#include "SallenKeyJunction.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace MarsDSP::Filters
{
    /** Second-order wave-digital Sallen-Key low-pass filter.
     *  An R-type junction models the op-amp follower. The output is the
     *  negated voltage of the port-3 capacitor.
     */
    class SallenKeyLPF
    {
    public:
        SallenKeyLPF() = default;

        void prepare (double sampleRate)
        {
            fs_ = sampleRate;
            c1.prepare (static_cast<float> (sampleRate));
            c2.prepare (static_cast<float> (sampleRate));
            rType.propagateImpedanceChange();
            reset();
        }

        void reset()
        {
            c1.reset();
            c2.reset();
            rType.clearWaveState();
        }

        void setParams (float freqHz, float q)
        {
            const float fClamped = std::clamp (freqHz, 10.0f, static_cast<float> (0.49 * fs_));
            const float wa = static_cast<float> (2.0 * fs_ * std::tan (std::numbers::pi_v<double> * static_cast<double> (fClamped) / fs_));
            const float Rv = 1.0f / (wa * capVal_);
            const float qClamped = std::clamp (q, 0.05f, 10.999f);
            const float disc = std::max (0.0f, capRatio_ * capRatio_ - 4.0f * qClamped * qClamped);
            const float sp = 2.0f * qClamped / (capRatio_ + std::sqrt (disc));

            r1.setResistanceValue (Rv * sp);
            r2.setResistanceValue (Rv / sp);
        }

        inline float processSample (float x) noexcept
        {
            source.setVoltage (x);
            source.incident (series.reflected());
            series.incident (source.reflected());
            return -WDF::voltage<float> (c2);
        }

    private:
        double fs_ { 48000.0 };

        static constexpr float capVal_ = 1.0e-8f;
        static constexpr float capRatio_ = 22.0f;

        // Port 1: feedback capacitor
        WDF::CapacitorT<float> c1 { capVal_ * capRatio_, 48000.0f };

        // Port 2: series resistor
        WDF::ResistorT<float> r2 { 1.0e3f };

        // Port 3: shunt capacitor
        WDF::CapacitorT<float> c2 { capVal_ / capRatio_, 48000.0f };

        // The R-type adaptor
        WDF::RtypeAdaptor<float, 0, SallenKeyImpedanceCalc,
                          decltype (c1), decltype (r2), decltype (c2)> rType { c1, r2, c2 };

        // Resistor above the junction
        WDF::ResistorT<float> r1 { 1.0e6f };

        // Series adaptor connecting the input resistor and the R-type block
        WDF::WDFSeriesT<float, decltype (rType), decltype (r1)> series { rType, r1 };

        // Root voltage source
        WDF::IdealVoltageSourceT<float, decltype (series)> source { series };
    };
}
#endif

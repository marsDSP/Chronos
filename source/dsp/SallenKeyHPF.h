#pragma once

#ifndef CHRONOS_SALLEN_KEY_HPF_H
#define CHRONOS_SALLEN_KEY_HPF_H

#include "SallenKeyJunction.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace MarsDSP::Filters
{
    /**
     *  Second-order wave-digital Sallen-Key high-pass filter.
     *  An R-type junction models the op-amp follower. The output is the
     *  negated voltage of the port-3 resistor. Equal capacitors keep the
     *  resistor spread at 22 for Q = 11.
     */
    class SallenKeyHPF
    {
    public:
        SallenKeyHPF() = default;

        void prepare(double sampleRate)
        {
            fs_ = sampleRate;
            c1.prepare(static_cast<float>(sampleRate));
            c2.prepare(static_cast<float>(sampleRate));
            rType.propagateImpedanceChange();
            reset();
        }

        void reset()
        {
            c1.reset();
            c2.reset();
            rType.clearWaveState();
        }

        void setParams(float freqHz, float q)
        {
            const float fClamped = std::clamp(freqHz, 10.0f, static_cast<float>(0.49 * fs_));
            const auto  wa = static_cast<float>(2.0 * fs_ * std::tan(std::numbers::pi_v<double> * static_cast<double>(fClamped) / fs_));
            const float Rv = 1.0f / (wa * capVal_);
            const float qClamped = std::clamp(q, 0.05f, 11.0f);
            const float sp = 2.0f * qClamped;

            r1.setResistanceValue(Rv / sp);
            r2.setResistanceValue(Rv * sp);
        }

        inline float processSample(float x) noexcept
        {
            source.setVoltage(x);
            source.incident(series.reflected());
            series.incident(source.reflected());
            return -WDF::voltage<float>(r2);
        }

    private:
        double fs_{48000.0};

        static constexpr float capVal_ = 1.0e-8f;

        // Port 1: feedback resistor
        WDF::ResistorT<float> r1{1.0e3f};

        // Port 2: series capacitor
        WDF::CapacitorT<float> c2{capVal_, 48000.0f};

        // Port 3: shunt resistor to ground
        WDF::ResistorT<float> r2{1.0e3f};

        // The R-type adaptor
        WDF::RtypeAdaptor<float, 0, SallenKeyImpedanceCalc, decltype (r1), decltype (c2), decltype (r2)> rType{
            r1, c2, r2
        };

        // Input series capacitor above the junction
        WDF::CapacitorT<float> c1{capVal_, 48000.0f};

        // Series adaptor connecting the input capacitor and the R-type block
        WDF::WDFSeriesT<float, decltype (rType), decltype (c1)> series{rType, c1};

        // Root voltage source
        WDF::IdealVoltageSourceT<float, decltype (series)> source{series};
    };
}
#endif

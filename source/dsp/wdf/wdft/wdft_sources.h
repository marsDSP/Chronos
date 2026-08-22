#pragma once

#ifndef CHRONOS_WDF_SOURCES_H
#define CHRONOS_WDF_SOURCES_H

#include "wdft_base.h"

namespace MarsDSP::WDF
{
    /** WDF Ideal Voltage source (non-adaptable) */
    template <typename T, typename Next>
    class IdealVoltageSourceT final : public RootWDF
    {
    public:
        explicit IdealVoltageSourceT (Next& next)
        {
            next.connectToParent (this);
            calcImpedance();
        }

        void calcImpedance() override {}

        /** Sets the voltage of the voltage source, in Volts */
        void setVoltage (T newV) { Vs = newV; }

        /** Accepts an incident wave into a WDF ideal voltage source. */
        inline void incident (T x) noexcept
        {
            wdf.a = x;
        }

        /** Propagates a reflected wave from a WDF ideal voltage source. */
        inline T reflected() noexcept
        {
            wdf.b = -wdf.a + static_cast<T>(2.0) * Vs;
            return wdf.b;
        }

        WDFMembers<T> wdf;

    private:
        T Vs = static_cast<T>(0.0);
    };

    /** WDF Voltage source with series resistance */
    template <typename T>
    class ResistiveVoltageSourceT final : public BaseWDF
    {
    public:
        /** Creates a new resistive voltage source.
         * @param value: initial resistance value, in Ohms
         */
        explicit ResistiveVoltageSourceT (T value = static_cast<NumericType<T>>(1.0e-9)) : R_value (value)
        {
            calcImpedance();
        }

        /** Sets the resistance value of the series resistor, in Ohms. */
        void setResistanceValue (T newR)
        {
            if (all (newR == R_value))
                return;

            R_value = newR;
            propagateImpedanceChange();
        }

        /** Computes the impedance for a WDF resistive voltage source: Z_Vr = Z_R */
        inline void calcImpedance() override
        {
            wdf.R = R_value;
            wdf.G = static_cast<T>(1.0) / wdf.R;
        }

        /** Sets the voltage of the voltage source, in Volts */
        void setVoltage (T newV) { Vs = newV; }

        /** Accepts an incident wave into a WDF resistive voltage source. */
        inline void incident (T x) noexcept
        {
            wdf.a = x;
        }

        /** Propagates a reflected wave from a WDF resistive voltage source. */
        inline T reflected() noexcept
        {
            wdf.b = Vs;
            return wdf.b;
        }

        WDFMembers<T> wdf;

    private:
        T Vs = static_cast<T>(0.0);
        T R_value = static_cast<T>(1.0e-9);
    };
}
#endif

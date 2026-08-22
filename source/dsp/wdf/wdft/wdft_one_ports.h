#pragma once

#ifndef CHRONOS_WDF_ONE_PORTS_H
#define CHRONOS_WDF_ONE_PORTS_H

#include "wdft_base.h"

namespace MarsDSP::WDF
{
    /** WDF Resistor Node */
    template <typename T>
    class ResistorT final : public BaseWDF
    {
    public:
        /** Creates a new WDF Resistor with a given resistance.
         * @param value: resistance in Ohms
         */
        explicit ResistorT (T value) : R_value (value)
        {
            calcImpedance();
        }

        /** Sets the resistance value of the WDF resistor, in Ohms. */
        void setResistanceValue (T newR)
        {
            if (all (newR == R_value))
                return;

            R_value = newR;
            propagateImpedanceChange();
        }

        /** Computes the impedance of the WDF resistor, Z_R = R. */
        inline void calcImpedance() override
        {
            wdf.R = R_value;
            wdf.G = static_cast<T>(1.0) / wdf.R;
        }

        /** Accepts an incident wave into a WDF resistor. */
        inline void incident (T x) noexcept
        {
            wdf.a = x;
        }

        /** Propagates a reflected wave from a WDF resistor. */
        inline T reflected() noexcept
        {
            wdf.b = static_cast<T>(0.0);
            return wdf.b;
        }

        WDFMembers<T> wdf;

    private:
        T R_value = static_cast<T>(1.0e-9);
    };

    /** WDF Capacitor Node */
    template <typename T>
    class CapacitorT final : public BaseWDF
    {
    public:
        /** Creates a new WDF Capacitor.
         * @param value: Capacitance value in Farads
         * @param _fs: WDF sample rate
         */
        explicit CapacitorT (T value, T _fs = static_cast<T>(48000.0)) : C_value (value),
                                                             fs (_fs)
        {
            calcImpedance();
        }

        /** Prepares the capacitor to operate at a new sample rate */
        void prepare (T sampleRate)
        {
            fs = sampleRate;
            propagateImpedanceChange();

            reset();
        }

        /** Resets the capacitor state */
        void reset()
        {
            z = static_cast<T>(0.0);
        }

        /** Sets the capacitance value of the WDF capacitor, in Farads. */
        void setCapacitanceValue (T newC)
        {
            if (all (newC == C_value))
                return;

            C_value = newC;
            propagateImpedanceChange();
        }

        /** Computes the impedance of the WDF capacitor,
         *             1
         * Z_C = --------------
         *        2 * f_s * C
         */
        inline void calcImpedance() override
        {
            wdf.R = static_cast<T>(1.0) / (static_cast<T>(2.0) * C_value * fs);
            wdf.G = static_cast<T>(1.0) / wdf.R;
        }

        /** Accepts an incident wave into a WDF capacitor. */
        inline void incident (T x) noexcept
        {
            wdf.a = x;
            z = wdf.a;
        }

        /** Propagates a reflected wave from a WDF capacitor. */
        inline T reflected() noexcept
        {
            wdf.b = z;
            return wdf.b;
        }

        WDFMembers<T> wdf;

    private:
        T C_value = static_cast<T>(1.0e-6);
        T z = static_cast<T>(0.0);

        T fs;
    };

    /** WDF Inductor Node */
    template <typename T>
    class InductorT final : public BaseWDF
    {
    public:
        /** Creates a new WDF Inductor.
         * @param value: Inductance value in Henries
         * @param _fs: WDF sample rate
         */
        explicit InductorT (T value, T _fs = static_cast<T>(48000.0)) : L_value (value),
                                                            fs (_fs)
        {
            calcImpedance();
        }

        /** Prepares the inductor to operate at a new sample rate */
        void prepare (T sampleRate)
        {
            fs = sampleRate;
            propagateImpedanceChange();
            reset();
        }

        /** Resets the inductor state */
        void reset()
        {
            z = static_cast<T>(0.0);
        }

        /** Sets the inductance value of the WDF inductor, in Henries. */
        void setInductanceValue (T newL)
        {
            if (all (newL == L_value))
                return;

            L_value = newL;
            propagateImpedanceChange();
        }

        /** Computes the impedance of the WDF inductor,
         * Z_L = 2 * f_s * L
         */
        inline void calcImpedance() override
        {
            wdf.R = static_cast<T>(2.0) * L_value * fs;
            wdf.G = static_cast<T>(1.0) / wdf.R;
        }

        /** Accepts an incident wave into a WDF inductor. */
        inline void incident (T x) noexcept
        {
            wdf.a = x;
            z = wdf.a;
        }

        /** Propagates a reflected wave from a WDF inductor. */
        inline T reflected() noexcept
        {
            wdf.b = -z;
            return wdf.b;
        }

        WDFMembers<T> wdf;

    private:
        T L_value = static_cast<T>(1.0e-6);
        T z = static_cast<T>(0.0);
        T fs;
    };
}
#endif

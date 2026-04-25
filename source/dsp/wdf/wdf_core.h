#pragma once

#ifndef CHRONOS_WDF_CORE_H
#define CHRONOS_WDF_CORE_H

// ============================================================================
//  wdf_core.h
// ----------------------------------------------------------------------------
//  Minimal wave-digital-filter primitives needed to assemble a diode bridge
//  attenuator. We only carry what's required by the bridge topology:
//
//    Voltage source ── R_series ── Junction ── DiodePair (root, in wdf_diode_pair.h)
//                                      │
//                                      └── R_load (or capacitor for filtering)
//
//  Each node holds a WDFPort (R, G, a, b) and exposes incident() / reflected()
//  using the standard wave equations:
//
//    voltage  = (a + b) * 0.5
//    current  = (a - b) * 0.5 * G
//
//  A topology is built bottom-up by composition, then wave reflections are
//  pumped through it once per audio sample. Impedance changes (e.g. from a
//  setResistanceValue() call) propagate up the connection chain via
//  propagateImpedanceChange().
//
//  All nodes are templated on a sample type T so callers can build either a
//  scalar graph (T = float) or a SIMD-quad graph (T = SIMD_M128) - the
//  arithmetic is identical because both types support the same operators.
// ============================================================================

namespace MarsDSP::DSP::WDF
{
    // ------------------------------------------------------------------ //
    //  Wave-port state. Holds the impedance R, its inverse G, plus the
    //  incident wave a and reflected wave b for a single port.
    // ------------------------------------------------------------------ //
    template <typename T>
    struct WDFPort
    {
        T R = static_cast<T>(1.0e-9);   // port impedance
        T G = static_cast<T>(1.0e9);    // port admittance (= 1 / R)
        T a = static_cast<T>(0);        // incident wave
        T b = static_cast<T>(0);        // reflected wave
    };

    // ------------------------------------------------------------------ //
    //  WDFNode: any non-root WDF element. Knows its parent so impedance
    //  changes can bubble up the tree.
    // ------------------------------------------------------------------ //
    class WDFNode
    {
    public:
        virtual ~WDFNode() = default;

        void connectToParent (WDFNode* p) noexcept { parent = p; }

        virtual void calcImpedance() = 0;

        inline virtual void propagateImpedanceChange()
        {
            calcImpedance();
            if (parent != nullptr)
                parent->propagateImpedanceChange();
        }

    protected:
        WDFNode* parent = nullptr;
    };

    // Root nodes terminate impedance propagation - they have no parent above.
    class WDFRoot : public WDFNode
    {
    public:
        inline void propagateImpedanceChange() override { calcImpedance(); }

    private:
        void connectToParent (WDFNode*) noexcept {}
    };

    // ------------------------------------------------------------------ //
    //  Probes
    // ------------------------------------------------------------------ //
    template <typename T, typename N>
    inline T voltage (const N& node) noexcept
    {
        return (node.wdf.a + node.wdf.b) * static_cast<T>(0.5);
    }

    template <typename T, typename N>
    inline T current (const N& node) noexcept
    {
        return (node.wdf.a - node.wdf.b) * (static_cast<T>(0.5) * node.wdf.G);
    }

    // ================================================================== //
    //  One-port elements
    // ================================================================== //

    // Plain resistor: Z_R = R, b is always zero (a perfect resistor doesn't
    // store energy so the reflected wave never has memory).
    template <typename T>
    class WDFResistor final : public WDFNode
    {
    public:
        explicit WDFResistor (T value) : R_value (value) { calcImpedance(); }

        void setResistanceValue (T v) noexcept
        {
            if (v == R_value) return;
            R_value = v;
            propagateImpedanceChange();
        }

        inline void calcImpedance() override
        {
            wdf.R = R_value;
            wdf.G = static_cast<T>(1) / wdf.R;
        }

        inline void incident (T x) noexcept { wdf.a = x; }
        inline T    reflected() noexcept    { wdf.b = static_cast<T>(0); return wdf.b; }

        WDFPort<T> wdf;

    private:
        T R_value;
    };

    // Capacitor: Z_C = 1 / (2*fs*C). Trapezoidal integration via "b = z" trick.
    template <typename T>
    class WDFCapacitor final : public WDFNode
    {
    public:
        WDFCapacitor (T value, T fs = static_cast<T>(48000)) : C_value (value), sampleRate (fs)
        {
            calcImpedance();
        }

        void prepare (T fs) noexcept
        {
            sampleRate = fs;
            propagateImpedanceChange();
            reset();
        }

        void reset() noexcept { z = static_cast<T>(0); }

        void setCapacitanceValue (T v) noexcept
        {
            if (v == C_value) return;
            C_value = v;
            propagateImpedanceChange();
        }

        inline void calcImpedance() override
        {
            wdf.R = static_cast<T>(1) / (static_cast<T>(2) * C_value * sampleRate);
            wdf.G = static_cast<T>(1) / wdf.R;
        }

        inline void incident (T x) noexcept { wdf.a = x; z = wdf.a; }
        inline T    reflected() noexcept    { wdf.b = z; return wdf.b; }

        WDFPort<T> wdf;

    private:
        T C_value;
        T sampleRate;
        T z = static_cast<T>(0);
    };

    // ================================================================== //
    //  Sources
    // ================================================================== //

    // Resistive voltage source: an ideal Thevenin source with series R.
    template <typename T>
    class ResistiveVoltageSource final : public WDFNode
    {
    public:
        explicit ResistiveVoltageSource (T value = static_cast<T>(1.0e-9)) : R_value (value)
        {
            calcImpedance();
        }

        void setResistanceValue (T v) noexcept
        {
            if (v == R_value) return;
            R_value = v;
            propagateImpedanceChange();
        }

        inline void calcImpedance() override
        {
            wdf.R = R_value;
            wdf.G = static_cast<T>(1) / wdf.R;
        }

        void setVoltage (T newV) noexcept { Vs = newV; }

        inline void incident (T x) noexcept { wdf.a = x; }
        inline T    reflected() noexcept    { wdf.b = Vs; return wdf.b; }

        WDFPort<T> wdf;

    private:
        T Vs = static_cast<T>(0);
        T R_value;
    };

    // Ideal voltage source (root): forces the port voltage regardless of load.
    template <typename T, typename Next>
    class IdealVoltageSource final : public WDFRoot
    {
    public:
        explicit IdealVoltageSource (Next& next)
        {
            next.connectToParent (this);
            calcImpedance();
        }

        void calcImpedance() override {}

        void setVoltage (T newV) noexcept { Vs = newV; }

        inline void incident (T x) noexcept { wdf.a = x; }
        inline T    reflected() noexcept    { wdf.b = -wdf.a + static_cast<T>(2) * Vs; return wdf.b; }

        WDFPort<T> wdf;

    private:
        T Vs = static_cast<T>(0);
    };

    // ================================================================== //
    //  Adaptors
    // ================================================================== //

    // Three-port series adaptor: Z_s = Z_1 + Z_2.
    template <typename T, typename P1, typename P2>
    class WDFSeries final : public WDFNode
    {
    public:
        WDFSeries (P1& p1, P2& p2) : port1 (p1), port2 (p2)
        {
            port1.connectToParent (this);
            port2.connectToParent (this);
            calcImpedance();
        }

        inline void calcImpedance() override
        {
            wdf.R         = port1.wdf.R + port2.wdf.R;
            wdf.G         = static_cast<T>(1) / wdf.R;
            port1Reflect  = port1.wdf.R / wdf.R;
        }

        inline void incident (T x) noexcept
        {
            const auto b1 = port1.wdf.b - port1Reflect * (x + port1.wdf.b + port2.wdf.b);
            port1.incident (b1);
            port2.incident (-(x + b1));
            wdf.a = x;
        }

        inline T reflected() noexcept
        {
            wdf.b = -(port1.reflected() + port2.reflected());
            return wdf.b;
        }

        P1& port1;
        P2& port2;
        WDFPort<T> wdf;

    private:
        T port1Reflect = static_cast<T>(1);
    };

    // Three-port parallel adaptor: 1/Z_p = 1/Z_1 + 1/Z_2.
    template <typename T, typename P1, typename P2>
    class WDFParallel final : public WDFNode
    {
    public:
        WDFParallel (P1& p1, P2& p2) : port1 (p1), port2 (p2)
        {
            port1.connectToParent (this);
            port2.connectToParent (this);
            calcImpedance();
        }

        inline void calcImpedance() override
        {
            wdf.G         = port1.wdf.G + port2.wdf.G;
            wdf.R         = static_cast<T>(1) / wdf.G;
            port1Reflect  = port1.wdf.G / wdf.G;
        }

        inline void incident (T x) noexcept
        {
            const auto b2 = wdf.b - port2.wdf.b + x;
            port1.incident (b2 + bDiff);
            port2.incident (b2);
            wdf.a = x;
        }

        inline T reflected() noexcept
        {
            port1.reflected();
            port2.reflected();
            bDiff = port2.wdf.b - port1.wdf.b;
            wdf.b = port2.wdf.b - port1Reflect * bDiff;
            return wdf.b;
        }

        P1& port1;
        P2& port2;
        WDFPort<T> wdf;

    private:
        T port1Reflect = static_cast<T>(1);
        T bDiff        = static_cast<T>(0);
    };

    // Polarity inverter: identical impedance, flips the wave sign.
    template <typename T, typename P>
    class PolarityInverter final : public WDFNode
    {
    public:
        explicit PolarityInverter (P& p) : port1 (p)
        {
            port1.connectToParent (this);
            calcImpedance();
        }

        inline void calcImpedance() override
        {
            wdf.R = port1.wdf.R;
            wdf.G = static_cast<T>(1) / wdf.R;
        }

        inline void incident (T x) noexcept { wdf.a = x; port1.incident (-x); }
        inline T    reflected() noexcept    { wdf.b = -port1.reflected(); return wdf.b; }

        WDFPort<T> wdf;

    private:
        P& port1;
    };

    // Factory helpers (CTAD-style) so callers don't have to spell template params.
    template <typename T, typename P1, typename P2>
    inline auto makeSeries (P1& p1, P2& p2) { return WDFSeries<T, P1, P2> (p1, p2); }

    template <typename T, typename P1, typename P2>
    inline auto makeParallel (P1& p1, P2& p2) { return WDFParallel<T, P1, P2> (p1, p2); }

    template <typename T, typename P>
    inline auto makeInverter (P& p) { return PolarityInverter<T, P> (p); }
}
#endif

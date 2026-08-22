#pragma once

#ifndef CHRONOS_WDF_BASE_H
#define CHRONOS_WDF_BASE_H

#include <type_traits>

namespace MarsDSP::WDF
{
    /** Element type helper: returns T for floats, and value_type otherwise. */
    namespace SampleTypeHelpers
    {
        template <typename T, bool = std::is_floating_point_v<T>>
        struct ElementType
        {
            using Type = T;
        };

        template <typename T>
        struct ElementType<T, false>
        {
            using Type = T::value_type;
        };
    } // namespace SampleTypeHelpers

    /** Numeric (scalar) element type underlying T (T itself for scalars). */
    template <typename T>
    using NumericType = SampleTypeHelpers::ElementType<T>::Type;

    /** True if all elements of a SIMD bool vector are equal; scalar bool path. */
    inline bool all (bool x) noexcept { return x; }

    /** Scalar select helper used by the SIMD-aware code paths. */
    template <typename T>
    inline T select (bool b, const T& t, const T& f)
    {
        return b ? t : f;
    }

    /** Base WDF class for propagating impedance changes between elements */
    class BaseWDF
    {
    public:
        virtual ~BaseWDF() = default;

        void connectToParent (BaseWDF* p) { parent = p; }

        virtual void calcImpedance() = 0;

        inline virtual void propagateImpedanceChange()
        {
            if (dontPropagateImpedance)
                return;

            calcImpedance();

            if (parent != nullptr)
                parent->propagateImpedanceChange();
        }

    protected:
        BaseWDF* parent = nullptr;

    private:
        bool dontPropagateImpedance = false;
    };

    /** Base class for propagating impedance changes into root WDF elements */
    class RootWDF : public BaseWDF
    {
    public:
        inline void propagateImpedanceChange() override { calcImpedance(); }

    private:
        void connectToParent (BaseWDF*) {}
    };

    /** Helper struct for common WDF member variables */
    template <typename T>
    struct WDFMembers
    {
        T R = static_cast<NumericType<T>>(1.0e-9); /* impedance */
        T G = static_cast<T>(1.0) / R;             /* admittance */
        T a = static_cast<T>(0.0);                 /* incident wave */
        T b = static_cast<T>(0.0);                 /* reflected wave */
    };

    /** Probe the voltage across this circuit element. */
    template <typename T, typename WDFType>
    inline T voltage (const WDFType& wdf) noexcept
    {
        return (wdf.wdf.a + wdf.wdf.b) * static_cast<T>(0.5);
    }

    /** Probe the current through this circuit element. */
    template <typename T, typename WDFType>
    inline T current (const WDFType& wdf) noexcept
    {
        return (wdf.wdf.a - wdf.wdf.b) * (static_cast<T>(0.5) * wdf.wdf.G);
    }
}
#endif

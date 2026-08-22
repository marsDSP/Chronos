#pragma once

#ifndef CHRONOS_WDF_RTYPE_H
#define CHRONOS_WDF_RTYPE_H

#include "wdft_base.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>

namespace MarsDSP::WDF
{
    namespace rtype_detail
    {
        /** Apply Fn(elem, index) to each tuple element. */
        template <typename Fn, typename Tuple, std::size_t... Ix>
        constexpr void forEachInTuple (Fn&& fn, Tuple&& tuple, std::index_sequence<Ix...>) noexcept (
            noexcept (std::initializer_list<int> { (fn (std::get<Ix> (tuple), Ix), 0)... }))
        {
            (void) std::initializer_list<int> { ((void) fn (std::get<Ix> (tuple), Ix), 0)... };
        }

        template <typename T>
        using TupleIndexSequence = std::make_index_sequence<
            std::tuple_size<std::remove_cv_t<std::remove_reference_t<T>>>::value>;

        template <typename Fn, typename Tuple>
        constexpr void forEachInTuple (Fn&& fn, Tuple&& tuple) noexcept (
            noexcept (forEachInTuple (std::forward<Fn> (fn),
                                      std::forward<Tuple> (tuple),
                                      TupleIndexSequence<Tuple> {})))
        {
            forEachInTuple (std::forward<Fn> (fn),
                            std::forward<Tuple> (tuple),
                            TupleIndexSequence<Tuple> {});
        }

        /** Lightweight aligned-array shim with operator[], data(), clear(),
         *  and size(). Alignment is fixed at 16 bytes.
         */
        template <typename ElementType, int arraySize, int alignment = 16>
        struct AlignedArray
        {
            template <typename IntType>
            ElementType& operator[] (IntType index) noexcept
            {
                return array[static_cast<std::size_t> (index)];
            }

            template <typename IntType>
            const ElementType& operator[] (IntType index) const noexcept
            {
                return array[static_cast<std::size_t> (index)];
            }

            ElementType* data() noexcept { return array.data(); }
            const ElementType* data() const noexcept { return array.data(); }

            void clear()
            {
                array.fill (ElementType {});
            }

            static constexpr int size() noexcept { return arraySize; }

        private:
            alignas (alignment) std::array<ElementType, static_cast<std::size_t> (arraySize)> array {};
        };

        template <typename T, int nRows, int nCols = nRows, int alignment = 16>
        using Matrix = std::array<AlignedArray<T, nRows, alignment>, static_cast<std::size_t> (nCols)>;

        /** Scalar S-matrix scatter: b = S * a. */
        template <typename T, int numPorts>
        constexpr typename std::enable_if_t<std::is_floating_point<T>::value, void>
        RtypeScatter (const Matrix<T, numPorts>& S_,
                      const AlignedArray<T, numPorts>& a_,
                      AlignedArray<T, numPorts>& b_)
        {
            for (int c = 0; c < numPorts; ++c)
            {
                b_[c] = S_[0][c] * a_[0];
                for (int r = 1; r < numPorts; ++r)
                    b_[c] += S_[r][c] * a_[r];
            }
        }

        // The adapted port resistance makes the diagonal entry of the scattering
        // matrix zero. The value at that index of the incident vector therefore has
        // no effect on the wave that goes to the parent.
        template <typename T, int numPorts>
        constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
            RtypeScatterRow (const Matrix<T, numPorts>& S_,
                             const AlignedArray<T, numPorts>& a_,
                             int portIndex)
        {
            T sum = S_[0][portIndex] * a_[0];
            for (int r = 1; r < numPorts; ++r)
                sum += S_[r][portIndex] * a_[r];
            return sum;
        }
    } // namespace rtype_detail

    /**
     *  Adaptable R-type adaptor with a fixed compile-time scattering matrix.
     */
    template <typename T, int upPortIndex, typename ImpedanceCalculator, typename... PortTypes>
    class RtypeAdaptor : public BaseWDF
    {
    public:
        /** Number of ports connected to RtypeAdaptor (including the up port). */
        static constexpr auto numPorts = int (sizeof...(PortTypes) + 1);

        explicit RtypeAdaptor (PortTypes&... dps) : downPorts (std::tie (dps...))
        {
            b_vec.clear();
            a_vec.clear();

            rtype_detail::forEachInTuple (
                [&] (auto& port, std::size_t) { port.connectToParent (this); }, downPorts);
        }

        /** Re-computes the port impedance at the adapted upward-facing port. */
        void calcImpedance() override
        {
            wdf.R = ImpedanceCalculator::calcImpedance (*this);
            wdf.G = static_cast<T>(1) / wdf.R;
        }

        /** Returns the impedances of the down-facing ports, in declaration order. */
        constexpr auto getPortImpedances()
        {
            std::array<T, numPorts - 1> portImpedances {};
            rtype_detail::forEachInTuple (
                [&] (auto& port, std::size_t i) { portImpedances[i] = port.wdf.R; }, downPorts);
            return portImpedances;
        }

        /** Sets the scattering matrix data (numPorts x numPorts, row-major). */
        void setSMatrixData (const std::array<std::array<T, numPorts>, numPorts>& mat)
        {
            for (int i = 0; i < numPorts; ++i)
                for (int j = 0; j < numPorts; ++j)
                    S_matrix[static_cast<std::size_t> (j)][static_cast<std::size_t> (i)] =
                        mat[static_cast<std::size_t> (i)][static_cast<std::size_t> (j)];
        }

        /** Computes the incident wave coming down from the parent. */
        inline void incident (T downWave) noexcept
        {
            wdf.a = downWave;
            a_vec[upPortIndex] = wdf.a;

            rtype_detail::RtypeScatter (S_matrix, a_vec, b_vec);
            rtype_detail::forEachInTuple (
                [&] (auto& port, std::size_t i)
                {
                    auto portIndex = getPortIndex (static_cast<int>(i));
                    port.incident (b_vec[portIndex]);
                },
                downPorts);
        }

        /** Computes the reflected wave going up to the parent. */
        inline T reflected() noexcept
        {
            rtype_detail::forEachInTuple (
                [&] (auto& port, std::size_t i)
                {
                    auto portIndex = getPortIndex (static_cast<int>(i));
                    a_vec[portIndex] = port.reflected();
                },
                downPorts);

            wdf.b = rtype_detail::RtypeScatterRow (S_matrix, a_vec, upPortIndex);
            return wdf.b;
        }

        /** Wipes the wave-flow state on this adaptor. */
        inline void clearWaveState() noexcept
        {
            wdf.a = static_cast<T>(0);
            wdf.b = static_cast<T>(0);
            a_vec.clear();
            b_vec.clear();
        }

        WDFMembers<T> wdf;

    private:
        constexpr auto getPortIndex (int tupleIndex)
        {
            return tupleIndex < upPortIndex ? tupleIndex : tupleIndex + 1;
        }

        std::tuple<PortTypes&...> downPorts;

        rtype_detail::Matrix<T, numPorts> S_matrix;
        rtype_detail::AlignedArray<T, numPorts> a_vec;
        rtype_detail::AlignedArray<T, numPorts> b_vec;
    };
}
#endif

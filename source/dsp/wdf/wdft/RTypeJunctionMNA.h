#pragma once

#ifndef CHRONOS_WDF_RTYPE_JUNCTION_MNA_H
#define CHRONOS_WDF_RTYPE_JUNCTION_MNA_H

#include <array>
#include <cassert>
#include <cmath>

namespace MarsDSP::WDF
{
    template<int NumNodes, int NumPorts>
    class RJunctionMNA
    {
    public:
        using NodeMatrix = std::array<std::array<double, NumNodes>, NumNodes>;
        using NodeVector = std::array<double, NumNodes>;
        using PortVector = std::array<double, NumPorts>;
        using PivotVector = std::array<int, NumNodes>;
        template<typename TOut>
        using SMatrix = std::array<std::array<TOut, NumPorts>, NumPorts>;

        static constexpr int ground = -1;

        RJunctionMNA() noexcept { clear(); }

        void clear() noexcept
        {
            for (int i = 0; i < NumNodes; ++i)
                for (int j = 0; j < NumNodes; ++j)
                    Gint[i][j] = 0.0;

            for (int p = 0; p < NumPorts; ++p)
                ports[p] = {ground, ground};
        }

        // Conductance g between nodes n1 and n2 (either may be `ground`).
        void stampConductance(int n1, int n2, double g) noexcept
        {
            addG(n1, n1, g);
            addG(n2, n2, g);
            addG(n1, n2, -g);
            addG(n2, n1, -g);
        }

        // Op-amp / VCVS: source of value gain * (v[nPlus] - v[nMinus]),
        // referenced to ground, driving node nOut through a series output
        // resistance Rout.
        void stampOpAmp(int nPlus, int nMinus, int nOut, double gain, double Rout) noexcept
        {
            const double go = 1.0 / Rout;
            addG(nOut, nOut, go);
            addG(nOut, nPlus, -go * gain);
            addG(nOut, nMinus, go * gain);
        }

        // Declare port p between nodes hot and cold: v_p = v[hot] - v[cold].
        void setPort(int port, int hot, int cold) noexcept
        {
            ports[port] = {hot, cold};
        }

        // Compute the adapted resistance at `adaptedPort` and the full
        // NumPorts x NumPorts scattering matrix (row-major, b = S a), given the
        // resistances of the other ports. portRes[adaptedPort] is ignored.
        // Returns the adapted port resistance.
        template<typename TOut>
        double solveScattering(const PortVector &portRes, int adaptedPort,
                               SMatrix<TOut> &S) const noexcept
        {
            // System with every port except the adapted one terminated.
            NodeMatrix M = Gint;

            for (int p = 0; p < NumPorts; ++p)
                if (p != adaptedPort)
                    addPortConductance(M, p, 1.0 / portRes[p]);

            // Thevenin resistance at the adapted port: inject a unit current,
            // read the port voltage.
            NodeMatrix A = M;
            PivotVector piv{};

            [[maybe_unused]] bool ok = luFactor(A, piv);
            assert(ok);

            NodeVector x{};
            inject(adaptedPort, 1.0, x);
            luSolve(A, piv, x);

            const double Rad = portVoltage(adaptedPort, x);
            assert(std::isfinite (Rad) && Rad > 0.0);

            // Full system including the adapted termination; factor once, then
            // one back-substitution per scattering column.
            addPortConductance(M, adaptedPort, 1.0 / Rad);
            ok = luFactor(M, piv);
            assert(ok);

            for (int q = 0; q < NumPorts; ++q)
            {
                const double Gq = 1.0 / (q == adaptedPort ? Rad : portRes[q]);

                NodeVector f{};
                inject(q, Gq, f);
                luSolve(M, piv, f);

                for (int p = 0; p < NumPorts; ++p)
                    S[p][q] = static_cast<TOut> (2.0 * portVoltage(p, f) - (p == q ? 1.0 : 0.0));
            }

            // Guaranteed analytically by the choice of Rad; pin it exactly.
            S[adaptedPort][adaptedPort] = static_cast<TOut> (0);

            return Rad;
        }

    private:
        struct Port
        {
            int hot;
            int cold;
        };

        NodeMatrix Gint{};
        std::array<Port, NumPorts> ports{};

        void addG(int i, int j, double g) noexcept
        {
            if (i >= 0 && j >= 0)
                Gint[i][j] += g;
        }

        static void addTo(NodeMatrix &M, int i, int j, double g) noexcept
        {
            if (i >= 0 && j >= 0)
                M[i][j] += g;
        }

        void addPortConductance(NodeMatrix &M, int p, double g) const noexcept
        {
            const auto [hot, cold] = ports[p];
            addTo(M, hot, hot, g);
            addTo(M, cold, cold, g);
            addTo(M, hot, cold, -g);
            addTo(M, cold, hot, -g);
        }

        void inject(int p, double amount, NodeVector &rhs) const noexcept
        {
            const auto [hot, cold] = ports[p];
            if (hot >= 0) rhs[hot] += amount;
            if (cold >= 0) rhs[cold] -= amount;
        }

        double portVoltage(int p, const NodeVector &x) const noexcept
        {
            const auto [hot, cold] = ports[p];
            return (hot >= 0 ? x[hot] : 0.0) - (cold >= 0 ? x[cold] : 0.0);
        }

        // In-place LU with partial pivoting (LAPACK-style ipiv semantics).
        static bool luFactor(NodeMatrix &A, PivotVector &piv) noexcept
        {
            for (int k = 0; k < NumNodes; ++k)
            {
                int p = k;
                double amax = std::fabs(A[k][k]);
                for (int i = k + 1; i < NumNodes; ++i)
                    if (std::fabs(A[i][k]) > amax)
                    {
                        amax = std::fabs(A[i][k]);
                        p = i;
                    }

                piv[k] = p;
                if (p != k)
                    for (int j = 0; j < NumNodes; ++j)
                    {
                        const double t = A[k][j];
                        A[k][j] = A[p][j];
                        A[p][j] = t;
                    }

                if (A[k][k] == 0.0)
                    return false; // singular

                const double inv = 1.0 / A[k][k];
                for (int i = k + 1; i < NumNodes; ++i)
                {
                    A[i][k] *= inv;
                    const double lik = A[i][k];
                    for (int j = k + 1; j < NumNodes; ++j)
                        A[i][j] -= lik * A[k][j];
                }
            }
            return true;
        }

        static void luSolve(const NodeMatrix &A, const PivotVector &piv,
                            NodeVector &b) noexcept
        {
            for (int k = 0; k < NumNodes; ++k)
                if (piv[k] != k)
                {
                    const double t = b[k];
                    b[k] = b[piv[k]];
                    b[piv[k]] = t;
                }

            for (int k = 0; k < NumNodes; ++k)
                for (int i = k + 1; i < NumNodes; ++i)
                    b[i] -= A[i][k] * b[k];

            for (int k = NumNodes - 1; k >= 0; --k)
            {
                double s = b[k];
                for (int j = k + 1; j < NumNodes; ++j)
                    s -= A[k][j] * b[j];
                b[k] = s / A[k][k];
            }
        }
    };
}
#endif

#pragma once

#ifndef CHRONOS_WDF_RTYPE_JUNCTION_FAST_H
#define CHRONOS_WDF_RTYPE_JUNCTION_FAST_H

#include <array>
#include <cmath>

namespace MarsDSP::WDF
{
    template<int NumNodes, int NumPorts>
    class RJunctionFast
    {
        static_assert(NumNodes == 3, "analytic inverse path is specialised for 3 nodes");

    public:
        using NodeMatrix = std::array<std::array<double, NumNodes>, NumNodes>;
        using PortVector = std::array<double, NumPorts>;
        template<typename TOut>
        using SMatrix = std::array<std::array<TOut, NumPorts>, NumPorts>;

        static constexpr int ground = -1;

        RJunctionFast() noexcept { clear(); }

        void clear() noexcept
        {
            for (int i = 0; i < NumNodes; ++i)
                for (int j = 0; j < NumNodes; ++j)
                    Gint[i][j] = 0.0;

            for (int p = 0; p < NumPorts; ++p)
                for (int i = 0; i < NumNodes; ++i)
                    P[i][p] = 0.0;
        }

        void stampConductance(int n1, int n2, double g) noexcept
        {
            addG(n1, n1, g);
            addG(n2, n2, g);
            addG(n1, n2, -g);
            addG(n2, n1, -g);
        }

        // VCVS gain*(v[nPlus] - v[nMinus]) driving nOut through series Rout.
        void stampOpAmp(int nPlus, int nMinus, int nOut, double gain, double Rout) noexcept
        {
            const double go = 1.0 / Rout;
            addG(nOut, nOut, go);
            addG(nOut, nPlus, -go * gain);
            addG(nOut, nMinus, go * gain);
        }

        void setPort(int port, int hot, int cold) noexcept
        {
            for (int i = 0; i < NumNodes; ++i)
                P[i][port] = 0.0;
            if (hot >= 0) P[hot][port] += 1.0;
            if (cold >= 0) P[cold][port] -= 1.0;
        }

        // Returns the adapted port resistance and fills S (row-major, b = S a).
        template<typename TOut>
        double solveScattering(const PortVector &portRes, int adaptedPort, SMatrix<TOut> &S) const noexcept
        {
            // ---- M0: internal stamps + every port except the adapted one ----
            NodeMatrix M = {{
                {Gint[0][0], Gint[0][1], Gint[0][2]},
                {Gint[1][0], Gint[1][1], Gint[1][2]},
                {Gint[2][0], Gint[2][1], Gint[2][2]}
            }};

            for (int q = 0; q < NumPorts; ++q)
            {
                if (q == adaptedPort) continue;
                const double g = 1.0 / portRes[q];
                for (int i = 0; i < 3; ++i)
                {
                    const double pi = P[i][q];
                    if (pi == 0.0) continue;
                    for (int j = 0; j < 3; ++j)
                        M[i][j] += g * pi * P[j][q];
                }
            }

            // ---- analytic 3x3 inverse: one division, no branches ----
            const double c00 = (M[1][1] * M[2][2] - M[1][2] * M[2][1]);
            const double c01 = -(M[0][1] * M[2][2] - M[0][2] * M[2][1]);
            const double c02 = (M[0][1] * M[1][2] - M[0][2] * M[1][1]);
            const double c10 = -(M[1][0] * M[2][2] - M[1][2] * M[2][0]);
            const double c11 = (M[0][0] * M[2][2] - M[0][2] * M[2][0]);
            const double c12 = -(M[0][0] * M[1][2] - M[0][2] * M[1][0]);
            const double c20 = (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
            const double c21 = -(M[0][0] * M[2][1] - M[0][1] * M[2][0]);
            const double c22 = (M[0][0] * M[1][1] - M[0][1] * M[1][0]);

            const double invDet = 1.0 / (M[0][0] * c00 + M[0][1] * c10 + M[0][2] * c20);
            const NodeMatrix Mi = {{
                {c00 * invDet, c01 * invDet, c02 * invDet},
                {c10 * invDet, c11 * invDet, c12 * invDet},
                {c20 * invDet, c21 * invDet, c22 * invDet}
            }};

            // ---- W = M0^-1 P  (3 x NumPorts), then A = P^T W ----
            std::array<std::array<double, NumPorts>, 3> W{};
            for (int i = 0; i < 3; ++i)
                for (int q = 0; q < NumPorts; ++q)
                    W[i][q] = Mi[i][0] * P[0][q] + Mi[i][1] * P[1][q] + Mi[i][2] * P[2][q];

            SMatrix<double> A{};
            for (int p = 0; p < NumPorts; ++p)
                for (int q = 0; q < NumPorts; ++q)
                    A[p][q] = P[0][p] * W[0][q] + P[1][p] * W[1][q] + P[2][p] * W[2][q];

            // ---- rank-1 correction + scaling ----
            const double Rad = A[adaptedPort][adaptedPort];
            const double half = 0.5 / Rad;

            PortVector G{};
            for (int q = 0; q < NumPorts; ++q)
                G[q] = (q == adaptedPort) ? (1.0 / Rad) : (1.0 / portRes[q]);

            for (int p = 0; p < NumPorts; ++p)
            {
                const double Apa = A[p][adaptedPort];
                for (int q = 0; q < NumPorts; ++q)
                {
                    const double B = A[p][q] - Apa * A[adaptedPort][q] * half;
                    S[p][q] = static_cast<TOut>(2.0 * G[q] * B - (p == q ? 1.0 : 0.0));
                }
            }

            S[adaptedPort][adaptedPort] = static_cast<TOut>(0); // exact by construction
            return Rad;
        }

    private:
        NodeMatrix Gint{};
        std::array<std::array<double, NumPorts>, NumNodes> P{};

        void addG(int i, int j, double g) noexcept
        {
            if (i >= 0 && j >= 0)
                Gint[i][j] += g;
        }
    };
}
#endif

#pragma once

#ifndef CHRONOS_TESTS_WDF_DD_ORACLE_H
#define CHRONOS_TESTS_WDF_DD_ORACLE_H

#include "f2_dd_oracle.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace WdfOracle
{
    using DD = F2Oracle::DD;
    using Matrix3 = std::array<std::array<DD, 3>, 3>;
    using Vector3 = std::array<DD, 3>;

    struct OracleResult
    {
        DD Ra;
        std::array<std::array<DD, 4>, 4> S;
    };

    struct Port
    {
        int hot;
        int cold;
    };

    inline void addConductance (Matrix3 &M, int n1, int n2, DD g) noexcept
    {
        if (n1 >= 0) M[n1][n1] = F2Oracle::dd_add (M[n1][n1], g);
        if (n2 >= 0) M[n2][n2] = F2Oracle::dd_add (M[n2][n2], g);
        if (n1 >= 0 && n2 >= 0)
        {
            M[n1][n2] = F2Oracle::dd_sub (M[n1][n2], g);
            M[n2][n1] = F2Oracle::dd_sub (M[n2][n1], g);
        }
    }

    inline void addPortConductance (Matrix3 &M, const Port& port, DD g) noexcept
    {
        addConductance (M, port.hot, port.cold, g);
    }

    inline bool gaussJordanSolve3 (const Matrix3 &M_in, const Vector3 &rhs_in, Vector3 &sol) noexcept
    {
        Matrix3 A;
        Vector3 b;
        for (int i = 0; i < 3; ++i)
        {
            b[i] = rhs_in[i];
            for (int j = 0; j < 3; ++j)
                A[i][j] = M_in[i][j];
        }
        std::array<int, 3> col_perm { 0, 1, 2 };

        for (int k = 0; k < 3; ++k)
        {
            int piv_r = k;
            int piv_c = k;
            double max_val = F2Oracle::dd_abs_hi (A[k][k]);
            for (int i = k; i < 3; ++i)
            {
                for (int j = k; j < 3; ++j)
                {
                    double v = F2Oracle::dd_abs_hi (A[i][j]);
                    if (v > max_val)
                    {
                        max_val = v;
                        piv_r = i;
                        piv_c = j;
                    }
                }
            }

            if (max_val == 0.0)
                return false;

            if (piv_r != k)
            {
                for (int j = 0; j < 3; ++j)
                    std::swap (A[k][j], A[piv_r][j]);
                std::swap (b[k], b[piv_r]);
            }

            if (piv_c != k)
            {
                for (int i = 0; i < 3; ++i)
                    std::swap (A[i][k], A[i][piv_c]);
                std::swap (col_perm[k], col_perm[piv_c]);
            }

            const DD piv = A[k][k];
            for (int j = k; j < 3; ++j)
                A[k][j] = F2Oracle::dd_div (A[k][j], piv);
            b[k] = F2Oracle::dd_div (b[k], piv);

            for (int i = 0; i < 3; ++i)
            {
                if (i != k)
                {
                    const DD factor = A[i][k];
                    for (int j = k; j < 3; ++j)
                        A[i][j] = F2Oracle::dd_sub (A[i][j], F2Oracle::dd_mul (factor, A[k][j]));
                    b[i] = F2Oracle::dd_sub (b[i], F2Oracle::dd_mul (factor, b[k]));
                }
            }
        }

        for (int i = 0; i < 3; ++i)
            sol[col_perm[i]] = b[i];

        return true;
    }

    // Solve scattering in double-double arithmetic using Gauss-Jordan elimination.
    // Port 0 is adapted (nA -> ground).
    inline OracleResult solveScatteringDD (const std::array<double, 4> &portRes, int adaptedPort = 0) noexcept
    {
        constexpr int nA = 0;
        constexpr int nB = 1;
        constexpr int nO = 2;
        constexpr int ground = -1;

        const std::array<Port, 4> ports {
            Port { nA, ground }, // port 0
            Port { nO, nA },     // port 1
            Port { nB, nA },     // port 2
            Port { nB, ground }  // port 3
        };

        Matrix3 M0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                M0[i][j] = F2Oracle::dd_from (0.0);

        // Input resistance between nB and nO
        const DD gInput = F2Oracle::dd_div (F2Oracle::dd_one(), F2Oracle::dd_from (1.0e9));
        addConductance (M0, nB, nO, gInput);

        // Op-amp VCVS stamp: gain 100, Rout -0.1
        const DD go = F2Oracle::dd_div (F2Oracle::dd_one(), F2Oracle::dd_from (-1.0e-1));
        const DD gain = F2Oracle::dd_from (100.0);
        M0[nO][nO] = F2Oracle::dd_add (M0[nO][nO], go);
        M0[nO][nB] = F2Oracle::dd_sub (M0[nO][nB], F2Oracle::dd_mul (go, gain));
        M0[nO][nO] = F2Oracle::dd_add (M0[nO][nO], F2Oracle::dd_mul (go, gain));

        // Terminate all ports except adapted
        for (int p = 0; p < 4; ++p)
        {
            if (p != adaptedPort)
            {
                const DD gp = F2Oracle::dd_div (F2Oracle::dd_one(), F2Oracle::dd_from (portRes[p]));
                addPortConductance (M0, ports[p], gp);
            }
        }

        // Thevenin resistance: inject unit current into adapted port
        Vector3 rhs { F2Oracle::dd_from (0.0), F2Oracle::dd_from (0.0), F2Oracle::dd_from (0.0) };
        if (ports[adaptedPort].hot >= 0)
            rhs[ports[adaptedPort].hot] = F2Oracle::dd_add (rhs[ports[adaptedPort].hot], F2Oracle::dd_one());
        if (ports[adaptedPort].cold >= 0)
            rhs[ports[adaptedPort].cold] = F2Oracle::dd_sub (rhs[ports[adaptedPort].cold], F2Oracle::dd_one());

        Vector3 v0;
        [[maybe_unused]] bool ok = gaussJordanSolve3 (M0, rhs, v0);

        DD vAdapted = F2Oracle::dd_from (0.0);
        if (ports[adaptedPort].hot >= 0)
            vAdapted = F2Oracle::dd_add (vAdapted, v0[ports[adaptedPort].hot]);
        if (ports[adaptedPort].cold >= 0)
            vAdapted = F2Oracle::dd_sub (vAdapted, v0[ports[adaptedPort].cold]);

        const DD Rad = vAdapted;

        // Full system with adapted port terminated
        Matrix3 M = M0;

        const DD gRad = F2Oracle::dd_div (F2Oracle::dd_one(), Rad);
        addPortConductance (M, ports[adaptedPort], gRad);

        OracleResult res {};
        res.Ra = Rad;

        for (int q = 0; q < 4; ++q)
        {
            const DD Gq = (q == adaptedPort) ? gRad : F2Oracle::dd_div (F2Oracle::dd_one(), F2Oracle::dd_from (portRes[q]));
            Vector3 rhs_q { F2Oracle::dd_from (0.0), F2Oracle::dd_from (0.0), F2Oracle::dd_from (0.0) };
            if (ports[q].hot >= 0)
                rhs_q[ports[q].hot] = F2Oracle::dd_add (rhs_q[ports[q].hot], Gq);
            if (ports[q].cold >= 0)
                rhs_q[ports[q].cold] = F2Oracle::dd_sub (rhs_q[ports[q].cold], Gq);

            Vector3 v_q;
            gaussJordanSolve3 (M, rhs_q, v_q);

            for (int p = 0; p < 4; ++p)
            {
                DD vp = F2Oracle::dd_from (0.0);
                if (ports[p].hot >= 0)
                    vp = F2Oracle::dd_add (vp, v_q[ports[p].hot]);
                if (ports[p].cold >= 0)
                    vp = F2Oracle::dd_sub (vp, v_q[ports[p].cold]);

                res.S[p][q] = F2Oracle::dd_sub (F2Oracle::dd_mul_d (vp, 2.0), (p == q ? F2Oracle::dd_one() : F2Oracle::dd_from (0.0)));
            }
        }

        res.S[adaptedPort][adaptedPort] = F2Oracle::dd_from (0.0);
        return res;
    }
}
#endif

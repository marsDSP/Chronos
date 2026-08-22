#pragma once

#ifndef CHRONOS_SALLEN_KEY_JUNCTION_H
#define CHRONOS_SALLEN_KEY_JUNCTION_H

#include "wdf/wdft/wdft.h"
#include "wdf/wdft/RTypeJunctionFast.h"

#include <array>

namespace MarsDSP::Filters
{
    /**
     *  Op-amp constants for the Sallen-Key junction.
     *  The negative output resistance gives a non-inverting voltage follower.
     */
    static constexpr double opAmpGain = 100.0;
    static constexpr double opAmpInputRes = 1.0e9;
    static constexpr double opAmpOutputRes = -1.0e-1;

    enum
    {
        nA = 0,
        nB = 1,
        nO = 2,
        numNodes = 3,
        numPorts = 4
    };

    /**
     *  Stamps the op-amp and declares the four ports of the Sallen-Key junction.
     *  Port 0 faces the parent and is adapted. The port map is identical for
     *  the low-pass and the high-pass; only the element types differ.
     */
    struct SallenKeyImpedanceCalc
    {
        using Solver = WDF::RJunctionFast<numNodes, numPorts>;

        template <typename RTypeAdaptor>
        static float calcImpedance (RTypeAdaptor& R)
        {
            const auto [Rb, Rc, Rd] = R.getPortImpedances();

            Solver mna;
            mna.stampConductance (nB, nO, 1.0 / opAmpInputRes);
            mna.stampOpAmp (nB, nO, nO, opAmpGain, opAmpOutputRes);

            mna.setPort (0, nA, Solver::ground);
            mna.setPort (1, nO, nA);
            mna.setPort (2, nB, nA);
            mna.setPort (3, nB, Solver::ground);

            const std::array<double, numPorts> portRes { 0.0, static_cast<double> (Rb),
                                                         static_cast<double> (Rc),
                                                         static_cast<double> (Rd) };

            std::array<std::array<float, numPorts>, numPorts> S {};
            const double Ra = mna.solveScattering (portRes, 0, S);

            R.setSMatrixData (S);
            return static_cast<float> (Ra);
        }
    };
}
#endif

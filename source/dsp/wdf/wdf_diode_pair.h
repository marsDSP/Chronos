#pragma once

#ifndef CHRONOS_WDF_DIODE_PAIR_H
#define CHRONOS_WDF_DIODE_PAIR_H

// ============================================================================
//  wdf_diode_pair.h
// ----------------------------------------------------------------------------
//  Antiparallel diode pair as a non-adaptable WDF root. Two diodes wired
//  back-to-back (cathode-to-anode and anode-to-cathode) shunt large positive
//  and large negative excursions to ground symmetrically - the same shape
//  as a balanced 4-diode bridge where the two output nodes are tied
//  together. Hooked up as the shunt of a series-resistor voltage divider
//  it gives the analog ducking behaviour described in the design notes:
//  attenuation grows with bias-current drive while preserving even+odd
//  harmonic colour from the diode I-V curve.
//
//  Werner et al. derived an explicit reflection equation using the Wright-
//  Omega function W(x):
//
//      b = a − 2·V_T · sgn(a) · ( W(p + λ·a/V_T) − W(p − λ·a/V_T) )
//
//  with p = log(R·I_s / V_T) and λ = sgn(a). We use wrightOmega4 from
//  wdf_math.h as a fast polynomial-plus-Halley approximation. The diode
//  parameters (saturation current I_s, thermal voltage V_T, series-diode
//  count) can be retuned at runtime; setDiodeParameters() recomputes the
//  pre-cached log() so the inner loop stays constant-time.
//
//  Two qualities are exposed via a template parameter so callers can pick
//  between "good enough" (single-cubic Wright-Omega3, ~0.1% error) and
//  "essentially exact" (Wright-Omega4, near machine epsilon).
// ============================================================================

#include "dsp/wdf/wdf_core.h"
#include "dsp/wdf/wdf_math.h"
#include <cmath>

namespace MarsDSP::DSP::WDF
{
    enum class DiodeQuality
    {
        Fast,       // wrightOmega3 only
        Accurate    // wrightOmega3 + one Halley step (= wrightOmega4)
    };

    template <typename T, typename Next, DiodeQuality Quality = DiodeQuality::Accurate>
    class DiodePair final : public WDFRoot
    {
    public:
        // n        : the WDF subnetwork that feeds this root
        // Is       : reverse saturation current (typical 1N4148: 2.52e-9 A)
        // Vt       : thermal voltage at room temperature
        // nDiodes  : number of diodes in series in each branch (raises Vt)
        DiodePair (Next& n,
                   T Is,
                   T Vt      = static_cast<T>(25.85e-3),
                   T nDiodes = static_cast<T>(1)) : next (n)
        {
            n.connectToParent (this);
            setDiodeParameters (Is, Vt, nDiodes);
        }

        // Recomputes pre-cached helpers used by reflected().
        void setDiodeParameters (T newIs, T newVt, T nDiodes) noexcept
        {
            Is        = newIs;
            Vt        = nDiodes * newVt;
            twoVt     = static_cast<T>(2) * Vt;
            oneOverVt = static_cast<T>(1) / Vt;
            calcImpedance();
        }

        inline void calcImpedance() override
        {
            R_Is           = next.wdf.R * Is;
            R_Is_overVt    = R_Is * oneOverVt;
            logR_Is_overVt = std::log (R_Is_overVt);
        }

        inline void incident (T x) noexcept
        {
            wdf.a = x;
        }

        inline T reflected() noexcept
        {
            const T lambda           = signum (wdf.a);
            const T lambda_a_over_Vt = lambda * wdf.a * oneOverVt;

            if constexpr (Quality == DiodeQuality::Accurate)
            {
                wdf.b = wdf.a - twoVt * lambda
                                * (wrightOmega4 (logR_Is_overVt + lambda_a_over_Vt)
                                 - wrightOmega4 (logR_Is_overVt - lambda_a_over_Vt));
            }
            else
            {
                wdf.b = wdf.a - twoVt * lambda
                                * (wrightOmega3 (logR_Is_overVt + lambda_a_over_Vt)
                                 - wrightOmega3 (logR_Is_overVt - lambda_a_over_Vt));
            }
            return wdf.b;
        }

        WDFPort<T> wdf;

    private:
        T Is             = static_cast<T>(2.52e-9);
        T Vt             = static_cast<T>(25.85e-3);
        T twoVt          = static_cast<T>(0);
        T oneOverVt      = static_cast<T>(0);
        T R_Is           = static_cast<T>(0);
        T R_Is_overVt    = static_cast<T>(0);
        T logR_Is_overVt = static_cast<T>(0);

        const Next& next;
    };
}
#endif

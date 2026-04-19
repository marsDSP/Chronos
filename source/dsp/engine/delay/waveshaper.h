// Antiderivative-antialiased (ADAA) tanh waveshaper.
//
// First-order ADAA using the bounded Padé tanh and its analytical integral
// from fastermath.h. Intended for reuse at any softclip call site where we
// want stateful, aliasing-reduced nonlinearity - each instance carries its
// own (x_{n-1}, F(x_{n-1})) pair so two simultaneous streams cannot pollute
// each other's history.
//
// Theory recap. For a nonlinearity f with antiderivative F:
//
//     y[n] = ( F(x[n]) - F(x[n-1]) ) / ( x[n] - x[n-1] )
//
// converges to f(x) as (x[n]-x[n-1]) → 0 but aliases ~6 dB/oct less than
// the direct f(x). The 0/0 case is handled by falling back to the direct
// nonlinearity evaluated at the midpoint of x[n] and x[n-1].
//
// SIMD strategy. A single quad carries four consecutive samples; lane 0's
// "previous sample" comes from the stashed carry, lanes 1..3 read from the
// quad's own earlier lanes via a one-lane left-shift trick inside
// `FasterMath::fasterTanhADAA`. Carries then persist lane-3 of the quad for
// the next call.
#pragma once
#ifndef CHRONOS_WAVESHAPER_H
#define CHRONOS_WAVESHAPER_H

#include "dsp/math/fastermath.h"

namespace MarsDSP::Waveshapers
{
    // First-order ADAA tanh. Stateful: seed via reset() (also the default
    // post-construction state). Safe to use per-channel / per-softclip-site.
    class ADAATanh
    {
    public:
        // Seed state so the first sample starts from x=0, F=logcosh(0)=0.
        // Should be called from the engine's own reset() / prepare() path.
        void reset() noexcept
        {
            carryX = carryF = 0.0f;
        }

        // Process one SIMD quad of four consecutive samples.
        // The four lanes are implicitly paired with lanes [carry, x0, x1, x2]
        // so the ADAA recursion picks up exactly where the previous call left
        // off. Updates internal carry state on return.
        [[nodiscard]] SIMD_M128 processQuad(const SIMD_M128 x) noexcept
        {
            return fasterTanhADAA(x, carryX, carryF);
        }

        // Process one scalar sample. Produces a result that is bit-identical
        // to the corresponding lane of a SIMD quad containing the same
        // sequence of inputs (scalar and SIMD paths share fastTanhIntegral
        // and fasterTanhBounded primitives).
        [[nodiscard]] float processSample(const float x) noexcept
        {
            return fasterTanhADAA(x, carryX, carryF);
        }

        // Direct accessors for engines that want to pull the carry pair into
        // local floats across a tight SIMD loop (keeps state in registers
        // instead of round-tripping through memory every quad).
        [[nodiscard]] float getCarryX() const noexcept
        {
            return carryX;
        }

        [[nodiscard]] float getCarryF() const noexcept
        {
            return carryF;
        }

        void setCarry(const float cx, const float cf) noexcept
        {
            carryX = cx;
            carryF = cf;
        }

    private:
        float carryX = 0.0f;
        float carryF = 0.0f;
    };
}
#endif

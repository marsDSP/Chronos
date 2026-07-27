// tests/harnesses/cd/nonlinearity_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Correctness harness for the ADAA nonlinearity policies (MarsDSP::Nonlinear::
// TanhNL, AlgebraicNL). Each policy supplies f, F1 = integral(f), F2 = integral(F1).
// These three are the whole point: if F1' != f or F2' != F1, nothing downstream
// (ADAA1/ADAA2, the alias harness) can be correct.
//
//   1. F1' = f   – central difference (h = 1e-5) over [-20, 20], combined
//                  abs/rel tolerance 1e-6.
//   2. F2' = F1  – likewise.
//   3. Parity    – F1(-x) == F1(x) and F2(-x) == -F2(x), bit-exact (==),
//                  1e5 points. F1 is even, F2 is odd for both curves.
//   4. Origin    – F2(0) == 0.0 exactly; F1(0) == the policy's defining value
//                  exactly (0.0 for TanhNL, 1.0 for AlgebraicNL — whose
//                  F1 = sqrt(1+x^2), so F1(0) = 1, NOT 0. "F1(0)==0 for each
//                  policy" would be inconsistent with AlgebraicNL's definition,
//                  so the per-policy value is used here).
//   5. Overflow  – F1(+-700), F2(+-700) all finite. Catches a naive log(cosh).
//   6. Tail      – TanhNL only: |F1(8) - (8 - ln2)| = log1p(e^-16) ≈ 1.13e-7,
//                  checked < 2e-7 (6e-8 is the *G*-tail bound,
//                  |G(8)-pi^2/24|<6e-8, verified separately here — the F1 tail
//                  is log1p(exp(-2|x|)), ~1.9x larger than the G tail at |x|=8).
//
// Conventions (matching ring_buffer_check / dilog_check): plain main(), exit
// code, printf, always-live CHECK/FAIL (NOT assert). Links SharedCode only;
// no JUCE. No forced -O2 so assert preconditions stay armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/nonlinear/Nonlinearities.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

using MarsDSP::Nonlinear::AlgebraicNL;
using MarsDSP::Nonlinear::TanhNL;
using MarsDSP::Nonlinear::kLn2;
using MarsDSP::Nonlinear::kPiSqOver24;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kCDh = 1e-5;          // central-difference step
constexpr double kDerivTol = 1e-6;     // combined abs/rel derivative tolerance

// Combined abs/rel tolerance: passes when absErr <= tol * max(|ref|, 1.0).
// This is well-scaled across the whole [-20, 20] range (|f| from 0 to ~1,
// |F1| from 0 to ~19) and avoids a div-by-zero at f(0)=0. Central-difference
// truncation is O(h^2) ~ 1e-10, far under this bar.
inline bool derivOk(double absErr, double ref) noexcept
{
    return absErr <= kDerivTol * std::fmax(std::fabs(ref), 1.0);
}

// 1 & 2: F1' = f, F2' = F1 by central difference over [-xMax, xMax].
template <typename NL>
void checkDerivatives(double xMax)
{
    constexpr int N = 40001;
    double maxF1err = 0.0, maxF2err = 0.0, worstF1x = 0.0, worstF2x = 0.0;
    for (int i = 0; i < N; ++i)
    {
        const double x = -xMax + 2.0 * xMax * static_cast<double>(i) / static_cast<double>(N - 1);
        const double f1cd = (NL::F1(x + kCDh) - NL::F1(x - kCDh)) / (2.0 * kCDh);
        const double f2cd = (NL::F2(x + kCDh) - NL::F2(x - kCDh)) / (2.0 * kCDh);
        const double e1 = std::fabs(f1cd - NL::f(x));
        const double e2 = std::fabs(f2cd - NL::F1(x));
        if (e1 > maxF1err) { maxF1err = e1; worstF1x = x; }
        if (e2 > maxF2err) { maxF2err = e2; worstF2x = x; }
        if (!derivOk(e1, NL::f(x)))
            FAIL("F1'!=f at x=%.4f: cd=%.16f f=%.16f |err|=%.3e", x, f1cd, NL::f(x), e1);
        if (!derivOk(e2, NL::F1(x)))
            FAIL("F2'!=F1 at x=%.4f: cd=%.16f F1=%.16f |err|=%.3e", x, f2cd, NL::F1(x), e2);
    }
    std::printf("  F1'=f  max |err| = %.3e at x=%.4f (tol %.0e)\n", maxF1err, worstF1x, kDerivTol);
    std::printf("  F2'=F1 max |err| = %.3e at x=%.4f (tol %.0e)\n", maxF2err, worstF2x, kDerivTol);
}

// 3: parity, bit-exact (==). F1 even, F2 odd.
template <typename NL>
void checkParity()
{
    constexpr int N = 100000;
    double xMax = 20.0;
    for (int i = 0; i < N; ++i)
    {
        const double x = -xMax + 2.0 * xMax * static_cast<double>(i) / static_cast<double>(N - 1);
        const double f1p = NL::F1(-x), f1n = NL::F1(x);
        const double f2p = NL::F2(-x), f2n = NL::F2(x);
        if (f1p != f1n)
            FAIL("F1 not even at x=%.6f: F1(-x)=%.17g F1(x)=%.17g", x, f1p, f1n);
        if (f2p != -f2n)
            FAIL("F2 not odd at x=%.6f: F2(-x)=%.17g -F2(x)=%.17g", x, f2p, -f2n);
    }
    std::printf("  parity F1 even / F2 odd, bit-exact over %d pts: PASS\n", N);
}

// 4, 5: origin values + no overflow.
template <typename NL>
void checkOriginAndOverflow(double f1AtZero)
{
    CHECK(NL::F2(0.0) == 0.0);
    CHECK(NL::F1(0.0) == f1AtZero);
    std::printf("  F1(0)=%.17g  F2(0)=%.17g (expected F1(0)=%g, F2(0)=0)\n",
                NL::F1(0.0), NL::F2(0.0), f1AtZero);

    const double xs[4] = { 700.0, -700.0, 1000.0, -1000.0 };
    for (double x : xs)
    {
        if (!std::isfinite(NL::F1(x))) FAIL("F1(%.0f) not finite: %.17g", x, NL::F1(x));
        if (!std::isfinite(NL::F2(x))) FAIL("F2(%.0f) not finite: %.17g", x, NL::F2(x));
    }
    std::printf("  F1,F2 at +-700/+1000 all finite: PASS\n");
}

// 6: TanhNL tail only.
void checkTanhTail()
{
    const double tailF1 = std::fabs(TanhNL::F1(8.0) - (8.0 - kLn2));
    // F1 tail = log1p(exp(-2|x|)); at |x|=8 that is log1p(e^-16) ≈ 1.125e-7.
    // Verify against the directly-computed value (formula check) and a sane
    // magnitude bound (asymptote check).
    const double tailExact = std::log1p(std::exp(-16.0));
    const double eFormula = std::fabs(tailF1 - tailExact);
    std::printf("  F1 tail @ |x|=8: |F1(8)-(8-ln2)| = %.4e  (log1p(e^-16) = %.4e, |diff| = %.2e)\n",
                tailF1, tailExact, eFormula);
    CHECK(eFormula <= 1e-15);        // F1 is exactly the overflow-safe identity
    CHECK(tailF1 < 2e-7);            // asymptote; 6e-8 is the G bound, see below

    // The 6e-8 bound is on G, not F1: |G(8) - pi^2/24| < 6e-8,
    // G(a) = 1/2 Li2(-e^-2a) + pi^2/24.
    const double g8 = 0.5 * MarsDSP::Math::dilogNeg(std::exp(-16.0)) + kPiSqOver24;
    const double gTail = std::fabs(g8 - kPiSqOver24);
    std::printf("  G tail @ |x|=8: |G(8)-pi^2/24| = %.4e (bound 6e-8)\n", gTail);
    CHECK(gTail < 6e-8);
    std::printf("  tail checks: PASS\n");
}

template <typename NL>
int runPolicy(const char* label, double f1AtZero)
{
    std::printf("\n[%s]\n", label);
    g_section = label;
    std::printf("derivatives over [-20, 20] (h=%.0e):\n", kCDh);
    checkDerivatives<NL>(20.0);
    checkParity<NL>();
    checkOriginAndOverflow<NL>(f1AtZero);
    if constexpr (std::is_same_v<NL, TanhNL>)
        checkTanhTail();
    return 0;
}

} // namespace

int main()
{
    std::printf("=== Chronos nonlinearity (ADAA policy) correctness harness ===\n");
    std::printf("ln2 = %.16f  pi^2/24 = %.16f\n", kLn2, kPiSqOver24);

    runPolicy<TanhNL>("TanhNL", 0.0);
    runPolicy<AlgebraicNL>("AlgebraicNL", 1.0);

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

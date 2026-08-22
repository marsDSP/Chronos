/**
 * Correctness harness for the ADAA nonlinearity policies TanhNL and AlgebraicNL.
 * Each policy supplies f, F1, and F2. Plain main(), exit code,
 * always-live CHECK/FAIL.
 */

#include "dsp/nonlinear/Nonlinearities.h"

#include <cmath>
#include <cstdlib>
#include <print>

namespace {

using MarsDSP::Nonlinear::AlgebraicNL;
using MarsDSP::Nonlinear::TanhNL;
using MarsDSP::Nonlinear::kLn2;
// pi^2/24, used only by the G-tail check. Local so the harness does not
// depend on the header exposing it.
constexpr double kPiSqOver24 = 0.4112335167120566;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kCDh = 1e-5;          // central-difference step
constexpr double kDerivTol = 1e-6;     // combined abs/rel derivative tolerance

/// Combined abs/rel tolerance: passes when absErr <= tol * max(|ref|, 1.0).
inline bool derivOk(double absErr, double ref) noexcept
{
    return absErr <= kDerivTol * std::fmax(std::fabs(ref), 1.0);
}

// 1 and 2: F1' = f, F2' = F1 by central difference over [-xMax, xMax].
template <typename NL>
void checkDerivatives(double xMax)
{
    constexpr int N = 40001;
    double maxF1err = 0.0;
    double maxF2err = 0.0;
    double worstF1x = 0.0;
    double worstF2x = 0.0;
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
            FAIL("F1'!=f at x={{:.4f}}: cd={{:.16f}} f={{:.16f}} |err|={{:.3e}}", x, f1cd, NL::f(x), e1);
        if (!derivOk(e2, NL::F1(x)))
            FAIL("F2'!=F1 at x={{:.4f}}: cd={{:.16f}} F1={{:.16f}} |err|={{:.3e}}", x, f2cd, NL::F1(x), e2);
    }
    std::println("  F1'=f  max |err| = {:.3e} at x={:.4f} (tol {:.0e})", maxF1err, worstF1x, kDerivTol);
    std::println("  F2'=F1 max |err| = {:.3e} at x={:.4f} (tol {:.0e})", maxF2err, worstF2x, kDerivTol);
}

// 3: parity, bit-exact. F1 even, F2 odd.
template <typename NL>
void checkParity()
{
    constexpr int N = 100000;
    double xMax = 20.0;
    for (int i = 0; i < N; ++i)
    {
        const double x = -xMax + 2.0 * xMax * static_cast<double>(i) / static_cast<double>(N - 1);
        const double f1p = NL::F1(-x);
        const double f1n = NL::F1(x);
        const double f2p = NL::F2(-x);
        const double f2n = NL::F2(x);
        if (f1p != f1n)
            FAIL("F1 not even at x={{:.6f}}: F1(-x)={{:.17g}} F1(x)={{:.17g}}", x, f1p, f1n);
        if (f2p != -f2n)
            FAIL("F2 not odd at x={{:.6f}}: F2(-x)={{:.17g}} -F2(x)={{:.17g}}", x, f2p, -f2n);
    }
    std::println("  parity F1 even / F2 odd, bit-exact over {} pts: PASS", N);
}

// 4 and 5: origin values and no overflow.
template <typename NL>
void checkOriginAndOverflow(double f1AtZero)
{
    CHECK(NL::F2(0.0) == 0.0);
    CHECK(NL::F1(0.0) == f1AtZero);
    std::println("  F1(0)={:.17g}  F2(0)={:.17g} (expected F1(0)={}, F2(0)=0)",
                NL::F1(0.0), NL::F2(0.0), f1AtZero);

    const std::array<double, 4> xs = { 700.0, -700.0, 1000.0, -1000.0 };
    for (double x : xs)
    {
        if (!std::isfinite(NL::F1(x))) FAIL("F1({{:.0f}}) not finite: {{:.17g}}", x, NL::F1(x));
        if (!std::isfinite(NL::F2(x))) FAIL("F2({{:.0f}}) not finite: {{:.17g}}", x, NL::F2(x));
    }
    std::println("  F1,F2 at +-700/+1000 all finite: PASS");
}

// 6: TanhNL tail only.
void checkTanhTail()
{
    const double tailF1 = std::fabs(TanhNL::F1(8.0) - (8.0 - kLn2));
    // F1 tail = log1p(exp(-2|x|)) at |x|=8.
    const double tailExact = std::log1p(std::exp(-16.0));
    const double eFormula = std::fabs(tailF1 - tailExact);
    std::println("  F1 tail @ |x|=8: |F1(8)-(8-ln2)| = {:.4e}  (log1p(e^-16) = {:.4e}, |diff| = {:.2e})",
                tailF1, tailExact, eFormula);
    CHECK(eFormula <= 1e-15);
    CHECK(tailF1 < 2e-7);

    // The 6e-8 bound is on G, not F1: |G(8) - pi^2/24| < 6e-8.
    const double g8 = 0.5 * MarsDSP::Math::dilogNeg(std::exp(-16.0)) + kPiSqOver24;
    const double gTail = std::fabs(g8 - kPiSqOver24);
    std::println("  G tail @ |x|=8: |G(8)-pi^2/24| = {:.4e} (bound 6e-8)", gTail);
    CHECK(gTail < 6e-8);
    std::println("  tail checks: PASS");
}

template <typename NL>
int runPolicy(const char* label, double f1AtZero)
{
    std::println();
    std::println("[{}]", label);
    g_section = label;
    std::println("derivatives over [-20, 20] (h={:.0e}):", kCDh);
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
    std::println("=== Chronos nonlinearity (ADAA policy) correctness harness ===");
    std::println("ln2 = {:.16f}", kLn2);

    runPolicy<TanhNL>("TanhNL", 0.0);
    runPolicy<AlgebraicNL>("AlgebraicNL", 1.0);

    std::println();
    std::println("=== ALL PROPERTIES HELD ===");
    return 0;
}

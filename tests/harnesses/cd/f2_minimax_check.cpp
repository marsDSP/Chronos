// tests/harnesses/cd/f2_minimax_check.cpp
// Correctness harness for MarsDSP::Math::f1Tanh / f2Tanh, the regional
// minimax kernels for the tanh antiderivatives.
//
//   1. Oracle agreement   – the two oracles in f2_dd_oracle.h agree to
//                           < 1e-25 relative. If they do not, every later
//                           section measures against a broken reference.
//   2. Relative error F2  – f2Tanh vs the oracle, log-spaced over
//                           |x| in [1e-12, 1000]. Gate <= 3.0 ulp, an
//                           a-priori budget: the derivation script measures
//                           the Estrin polynomial error at <= 1.6 ulp, the
//                           assembly adds three roundings (u = x^2, x*u,
//                           *P) at <= 1.5 ulp worst case, and the fit is
//                           0.004 ulp. The 2.5 figure seen elsewhere was
//                           calibrated against a Horner assembly (0.72 ulp
//                           polynomial error); the header evaluates with
//                           the Estrin split for dependency depth, and its
//                           polynomial error is 1.58 ulp.
//   3. Relative error F1  – f1Tanh vs the oracle, same sweep. Gate <= 2 ulp
//                           (a-priori budget: 1/2 ulp from u = x^2, 1/2 ulp
//                           from the final multiply, <= 0.9 ulp polynomial,
//                           ~0.1 ulp fit. A flat 1.5 figure had no budget
//                           behind it; the derivation script carries the
//                           same 2.0 gate).
//   4. Near zero          – |x| in [1e-12, 1e-3], f2Tanh <= 3 ulp relative.
//                           Same region-I Estrin assembly as section 2, so
//                           the same a-priori budget applies (<= 1.6 ulp
//                           polynomial + <= 1.5 ulp assembly + fit). The
//                           dilogarithm-era implementation (Ref::) is
//                           printed alongside: its relative error is
//                           unbounded here. This section is the regression
//                           this work exists to fix.
//   5. Seam at a0         – value jump <= 1 ulp between the region
//                           expressions evaluated explicitly at a0. Then a
//                           finite-difference slope check: (F2(a0+e) -
//                           F2(a0-e))/2e against F1(a0). A value-continuous
//                           but slope-broken seam passes the jump and fails
//                           here.
//   6. Seam at a1         – same treatment.
//   7. Structural         – F2(0) == 0.0 and F1(0) == 0.0 exactly. Parity:
//                           F2(-x) == -F2(x) and F1(-x) == F1(x) over 1e6
//                           points including both seams and denormals. For
//                           denormal x, u = x*x flushes to +0, so F2 returns
//                           +-0.0; == does not distinguish signed zeros, so
//                           the denormal case passes. It is asserted
//                           deliberately, not by luck.
//   8. Monotonicity       – F1 non-decreasing on [0, 1000], F2 strictly
//                           increasing on [-1000, 1000], 1e6 points crossing
//                           both seams.
//   9. Derivatives        – F2' = F1 and F1' = tanh by central difference
//                           (h = 1e-5), tol 1e-6 combined abs/rel. Broad
//                           sweep plus fine grids around both seams.
//  10. Extremes           – finite at +-700, +-1000, +-1e6; no NaN, no inf,
//                           no denormal output. F1(1000) within 1 ulp of
//                           1000 - ln2.
//  11. Basis conditioning – the region-I monomial condition number,
//                           measured at runtime, matches the value the
//                           derivation script committed to tests/logs,
//                           within 10%. Catches a coefficient transcription
//                           error that still looks accurate on the fit grid.
//  12. Transcription      – coefficient counts match the declared degrees.
//                           The python scripts re-derive the values and
//                           exit non-zero on drift; they run in CI
//                           alongside these harnesses.
//
// Usage: f2_minimax_check [--full] [--points N]
//   Default: CI-sized sweep counts. --full: the dense local counts.
//   --points N: override the section 2/3 sweep density.
//
// Conventions (matching dilog_check / f2_oracle_check): plain main(),
// printf, exit code, always-live CHECK/FAIL (NOT assert). Links SharedCode
// only; no JUCE. No forced -O2 so the header assert preconditions stay
// armed in a Debug configure.

#include "math/TanhAntiderivatives.h"

#include "f2_dd_oracle.h"

#include <cmath>
#include <cstdio>
#include <print>
#include <cstdlib>
#include <cstring>

namespace {

using MarsDSP::Math::f1Tanh;
using MarsDSP::Math::f2Tanh;
using MarsDSP::Math::kF1RegionI;
using MarsDSP::Math::kF1RegionIIL;
using MarsDSP::Math::kF2RegionI;
using MarsDSP::Math::kF2RegionIIPsi;
using MarsDSP::Math::kTanA0;
using MarsDSP::Math::kTanA1;
using MarsDSP::Math::kTanC2;
using MarsDSP::Math::kTanLn2Hi;
using MarsDSP::Math::kTanLn2Lo;
using namespace F2Oracle;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kUlp = 2.220446049250313e-16;   // 2^-52, the relative ulp

int gPoints = 100000;   // section 2/3 sweep density (CI default)
int gAgree  = 300;      // section 1 agreement grid

double ulpOf(double e) { return e / kUlp; }

// Explicit region expressions, used by the seam jump checks. These mirror
// the header's dispatch branches one at a time.
double f2RegionI(double x)
{
    const double u = x * x;
    return x * u * MarsDSP::Math::detail::estrin15(kF2RegionI, u);
}

double f2RegionII(double a)
{
    const double h = (a - kTanLn2Hi) - kTanLn2Lo;
    const double t = std::exp(-2.0 * a);
    const double e = 0.5 * t * MarsDSP::Math::detail::estrin11(kF2RegionIIPsi, t);
    return std::fma(0.5 * h, h, kTanC2) - e;
}

double f2RegionIII(double a)
{
    const double h = (a - kTanLn2Hi) - kTanLn2Lo;
    return std::fma(0.5 * h, h, kTanC2);
}

// 1. Oracle agreement
void sectionOracleAgreement()
{
    g_section = "oracle agreement";
    double worst = 0.0;
    double worstX = 0.0;
    for (int i = 0; i < gAgree; ++i)
    {
        const double x = 1e-12 * std::pow(1e15, static_cast<double>(i) / (gAgree - 1));
        const double r = ddRelDiff(f2DD(x), quadDD(x));
        if (r > worst) { worst = r; worstX = x; }
    }
    std::println("closed form vs quadrature, {} points: max rel diff = {:.3} at x = {:.4} (gate 1e-25)",
                gAgree, worst, worstX);
    CHECK(worst < 1e-25);
    std::println("oracle agreement (1e-25): PASS");
}

// 2/3. Relative error sweeps
void sectionRelErr(bool f1)
{
    g_section = f1 ? "relative error F1" : "relative error F2";
    double worst = 0.0;
    double worstX = 0.0;
    std::array<double, 3> regWorst = {{ 0.0, 0.0, 0.0 }};
    std::array<double, 3> regX = {{ 0.0, 0.0, 0.0 }};
    for (int i = 0; i < gPoints; ++i)
    {
        const double x = 1e-12 * std::pow(1e15, static_cast<double>(i) / (gPoints - 1));
        const double got = f1 ? f1Tanh(x) : f2Tanh(x);
        const double ref = f1 ? toDouble(f1DD(x)) : toDouble(f2DD(x));
        const double rel = std::fabs(got - ref) / std::fabs(ref);
        if (rel > worst) { worst = rel; worstX = x; }
        const int reg = x <= kTanA0 ? 0 : (x < kTanA1 ? 1 : 2);
        if (rel > regWorst[reg]) { regWorst[reg] = rel; regX[reg] = x; }
    }
    std::array<const char*, 3> names = {{ "I [1e-12, 1]", "II (1, 19)", "III [19, 1000]" }};
    for (int r = 0; r < 3; ++r)
        std::println("    region {:<16} max rel err = {:.3} = {:.3} ulp at x = {:.6}",
                    names[r], regWorst[r], ulpOf(regWorst[r]), regX[r]);
    const double gate = f1 ? 2.0 : 3.0;
    std::println("{}: max rel err {:.3} ulp at x = {:.6} (gate <= {:.1} ulp): {}",
                f1 ? "F1" : "F2", ulpOf(worst), worstX, gate,
                ulpOf(worst) <= gate ? "PASS" : "FAIL");
    CHECK(ulpOf(worst) <= gate);
}

// 4. Near-zero relative accuracy
void sectionNearZero()
{
    g_section = "near zero";
    double worst = 0.0;
    double worstX = 0.0;
    double worstRef = 0.0;
    double worstRefX = 0.0;
    const int n = gPoints < 10000 ? 10000 : gPoints / 10;
    for (int i = 0; i < n; ++i)
    {
        const double x = 1e-12 * std::pow(1e9, static_cast<double>(i) / (n - 1));
        const double ref = toDouble(f2DD(x));
        const double rel = std::fabs(f2Tanh(x) - ref) / std::fabs(ref);
        if (rel > worst) { worst = rel; worstX = x; }
        const double relOld = std::fabs(MarsDSP::Math::Ref::f2TanhRef(x) - ref) / std::fabs(ref);
        if (relOld > worstRef) { worstRef = relOld; worstRefX = x; }
    }
    std::println("    new kernel: max rel err = {:.3} = {:.3} ulp at x = {:.4} (gate <= 3 ulp)",
                worst, ulpOf(worst), worstX);
    std::println("    dilog era : max rel err = {:.3} at x = {:.4} (unbounded, for the record)",
                worstRef, worstRefX);
    CHECK(ulpOf(worst) <= 3.0);
    std::println("near-zero F2 relative accuracy (3 ulp, {} points in [1e-12, 1e-3]): PASS", n);
}

// 5/6. Seam continuity
// The jump gate compares the two region expressions evaluated explicitly at
// the seam. The slope check gates the central finite difference of F2
// against F1 at the seam, with a bound that carries the FD truncation
// (0.5 e^2), the kernel's own evaluation rounding (3 ulp of F2 / e), and a
// small floor. A slope jump of size S shows up as a constant offset S/2 at
// small e and is caught where the bound is tightest.
void seamCheck(double seam, double f2I, double f2O, const char* tag)
{
    const double jump = std::fabs(f2I - f2O);
    const double jumpUlp = jump / (std::fabs(f2I) * kUlp);
    std::println("    seam {}: |F2_left - F2_right| = {:.3} = {:.3} ulp (gate <= 1 ulp)",
                tag, jump, jumpUlp);
    CHECK(jumpUlp <= 1.0);

    const double f1AtSeam = f1Tanh(seam);
    const double f2Mag = std::fabs(f2Tanh(seam));
    double worstRel = 0.0;
    double worstEps = 0.0;
    for (int i = 0; i < 30; ++i)
    {
        const double eps = 1e-12 * std::pow(1e11, static_cast<double>(i) / 29.0);
        const double fd = (f2Tanh(seam + eps) - f2Tanh(seam - eps)) / (2.0 * eps);
        const double bound = 0.5 * eps * eps
                           + (3.0 * f2Mag * kUlp) / eps
                           + 1e-12 * std::fabs(f1AtSeam);
        const double err = std::fabs(fd - f1AtSeam);
        const double rel = err / bound;
        if (rel > worstRel) { worstRel = rel; worstEps = eps; }
        if (err > bound)
            FAIL("seam {} slope: eps={:.3} fd={:.17} F1={:.17} err={:.3} > bound={:.3}",
                 tag, eps, fd, f1AtSeam, err, bound);
    }
    std::println("    seam {} slope: max err/bound = {:.3} at eps = {:.3} (gate <= 1): PASS",
                tag, worstRel, worstEps);
}

void sectionSeams()
{
    g_section = "seam a0";
    seamCheck(kTanA0, f2RegionI(kTanA0), f2RegionII(kTanA0), "a0");
    g_section = "seam a1";
    seamCheck(kTanA1, f2RegionII(kTanA1), f2RegionIII(kTanA1), "a1");
}

// 7. Structural exactness
void sectionStructural()
{
    g_section = "structural exactness";

    CHECK(f2Tanh(0.0) == 0.0);
    CHECK(f1Tanh(0.0) == 0.0);
    std::println("F2(0) == 0.0 and F1(0) == 0.0 exactly: PASS");

    // Parity over 1e6 points, log-spaced from the denormal range up, plus
    // ulp steps around both seams. For denormal x the F2 path returns a
    // signed zero on both sides; == holds for -0.0 == +0.0, and the
    // deliberate assertion below pins that behaviour.
    int denormals = 0;
    for (int i = 0; i < 1000000; ++i)
    {
        // [1e-310, 1e3]. The decade range 1e313 cannot be held in a double,
        // so the exponent is interpolated directly.
        const double x = std::pow(10.0, -310.0 + 313.0 * static_cast<double>(i) / 999999.0);
        if (x < 2.2250738585072014e-308) ++denormals;
        if (!(f2Tanh(-x) == -f2Tanh(x)))
            FAIL("F2 parity broken at x = {:.17}: {:.17} vs {:.17}",
                 x, f2Tanh(-x), -f2Tanh(x));
        if (!(f1Tanh(-x) == f1Tanh(x)))
            FAIL("F1 parity broken at x = {:.17}: {:.17} vs {:.17}",
                 x, f1Tanh(-x), f1Tanh(x));
    }
    const double seamPts[8] = {
        std::nextafter(kTanA0, 0.0), kTanA0, std::nextafter(kTanA0, 2.0),
        kTanA0,
        std::nextafter(kTanA1, 0.0), kTanA1, std::nextafter(kTanA1, 100.0),
        kTanA1,
    };
    for (double x : seamPts)
    {
        CHECK(f2Tanh(-x) == -f2Tanh(x));
        CHECK(f1Tanh(-x) == f1Tanh(x));
    }
    // Deliberate denormal check: x*x flushes to +0, so F2 returns +-0.0.
    const double dn = 1e-310;
    CHECK(f2Tanh(dn) == 0.0 && f2Tanh(-dn) == 0.0);
    CHECK(f1Tanh(dn) == 0.0 && f1Tanh(-dn) == 0.0);
    std::println("parity bit-exact over 1e6 points ({} denormals) + both seams: PASS",
                denormals);
}

// 8. Monotonicity
void sectionMonotonicity()
{
    g_section = "monotonicity";
    constexpr int n = 1000000;
    double prev = f1Tanh(0.0);
    for (int i = 1; i <= n; ++i)
    {
        const double x = 1000.0 * static_cast<double>(i) / n;
        const double v = f1Tanh(x);
        if (!(v >= prev))
            FAIL("F1 not non-decreasing at x={:.6} prev={:.17} v={:.17}", x, prev, v);
        prev = v;
    }
    double prevF2 = f2Tanh(-1000.0);
    for (int i = 1; i <= n; ++i)
    {
        const double x = -1000.0 + 2000.0 * static_cast<double>(i) / n;
        const double v = f2Tanh(x);
        if (!(v > prevF2))
            FAIL("F2 not strictly increasing at x={:.6} prev={:.17} v={:.17}", x, prevF2, v);
        prevF2 = v;
    }
    std::println("F1 non-decreasing on [0, 1000], F2 strictly increasing on [-1000, 1000] ({} pts): PASS",
                n);
}

// 9. Derivative consistency
void sectionDerivatives()
{
    g_section = "derivative consistency";
    const double h = 1e-5;
    double worst1 = 0.0;
    double worst2 = 0.0;
    double worstX1 = 0.0;
    double worstX2 = 0.0;
    auto probe = [&](double x) {
        const double fd1 = (f2Tanh(x + h) - f2Tanh(x - h)) / (2.0 * h);
        const double t1 = f1Tanh(x);
        const double e1 = std::fabs(fd1 - t1);
        const double tol1 = 1e-6 * (std::fabs(t1) + 1e-3);
        if (e1 / tol1 > worst1) { worst1 = e1 / tol1; worstX1 = x; }
        if (e1 > tol1)
            FAIL("F2' != F1 at x={:.8}: fd={:.17} F1={:.17} err={:.3} tol={:.3}",
                 x, fd1, t1, e1, tol1);
        const double fd2 = (f1Tanh(x + h) - f1Tanh(x - h)) / (2.0 * h);
        const double t2 = std::tanh(x);
        const double e2 = std::fabs(fd2 - t2);
        const double tol2 = 1e-6 * (std::fabs(t2) + 1e-3);
        if (e2 / tol2 > worst2) { worst2 = e2 / tol2; worstX2 = x; }
        if (e2 > tol2)
            FAIL("F1' != tanh at x={:.8}: fd={:.17} tanh={:.17} err={:.3} tol={:.3}",
                 x, fd2, t2, e2, tol2);
    };
    for (int i = 0; i <= 4000; ++i)
        probe(-40.0 + 80.0 * static_cast<double>(i) / 4000.0);
    // Fine grids around both seams.
    for (int i = 0; i < 40; ++i)
    {
        const double d = 1e-9 * std::pow(0.5e9, static_cast<double>(i) / 39.0);
        probe(kTanA0 * (1.0 - d));
        probe(kTanA0 * (1.0 + d));
        probe(kTanA1 * (1.0 - d));
        probe(kTanA1 * (1.0 + d));
    }
    std::println("F2' = F1: max err/tol = {:.3} at x = {:.6}", worst1, worstX1);
    std::println("F1' = tanh: max err/tol = {:.3} at x = {:.6}", worst2, worstX2);
    std::println("derivative consistency (h = 1e-5, tol 1e-6): PASS");
}

// 10. Extremes
void sectionExtremes()
{
    g_section = "extremes";
    const std::array<double, 6> xs = {{ 700.0, -700.0, 1000.0, -1000.0, 1e6, -1e6 }};
    for (double x : xs)
    {
        const double v1 = f1Tanh(x);
        const double v2 = f2Tanh(x);
        CHECK(std::isfinite(v1) && std::isfinite(v2));
        CHECK((v1 == 0.0 || std::isnormal(v1)) && (v2 == 0.0 || std::isnormal(v2)));
    }
    std::println("finite, normal output at +-700, +-1000, +-1e6: PASS");

    // F1(1000) = 1000 - ln2 + log1p(e^-2000) = 1000 - ln2 exactly.
    const double got = f1Tanh(1000.0);
    const double ref = 999.3068528194401;              // nearest double to 1000 - ln2
    const double ulpAt = 1.1368683772161603e-13;       // ulp of 999.3
    const double err = std::fabs(got - ref);
    std::println("F1(1000) = {:.17} vs 1000 - ln2 = {:.17}: err = {:.3} (gate 1 ulp = {:.3})",
                got, ref, err, ulpAt);
    CHECK(err <= ulpAt);
}

// 11. Basis conditioning
// Monomial basis condition number: max over u in [0, 1] of
// sum_k |c_k| u^k / |p(u)|. Sensitive to coefficient transcription errors
// even when the fit grid still looks accurate. The committed reference
// values come from the derivation scripts' logs in tests/logs/baseline/.
template <size_t N>
double basisCond(const std::array<double, N>& c)
{
    double worst = 0.0;
    for (int i = 0; i <= 10000; ++i)
    {
        const double u = static_cast<double>(i) / 10000.0;
        double num = 0.0;
        double den = c[N - 1];
        for (int k = static_cast<int>(N) - 2; k >= 0; --k)
        {
            num = num * u + std::fabs(c[static_cast<size_t>(k)]);
            den = den * u + c[static_cast<size_t>(k)];
        }
        const double r = num / std::fabs(den);
        if (r > worst) worst = r;
    }
    return worst;
}

void sectionBasisCond()
{
    g_section = "basis conditioning";
    const double pCond = basisCond(kF2RegionI);
    const double sCond = basisCond(kF1RegionI);
    const double pRef = 1.22911268079;   // tests/logs/baseline/f2_region1_basis_condition.txt
    const double sRef = 1.41920804845;   // tests/logs/baseline/f1_region1_basis_condition.txt
    std::println("    P(u): measured {:.6} vs committed {:.6} (gate: 10%)", pCond, pRef);
    std::println("    S(u): measured {:.6} vs committed {:.6} (gate: 10%)", sCond, sRef);
    CHECK(std::fabs(pCond / pRef - 1.0) <= 0.10);
    CHECK(std::fabs(sCond / sRef - 1.0) <= 0.10);
    std::println("basis conditioning matches the committed derivation logs: PASS");
}

// 12. Coefficient transcription
void sectionTranscription()
{
    g_section = "coefficient transcription";
    static_assert(kF2RegionI.size() == 15,   "P(u) must be degree 14");
    static_assert(kF1RegionI.size() == 15,   "S(u) must be degree 14");
    static_assert(kF2RegionIIPsi.size() == 11, "psi(t) must be degree 10");
    static_assert(kF1RegionIIL.size() == 11,   "L(t) must be degree 10");
    std::println("coefficient counts: P/S degree 14, psi/L degree 10: PASS");
    std::println("    (the python derivation scripts re-derive these values and\n     exit non-zero on drift; they run in CI alongside this harness)");
}

int runAll()
{
    F2Oracle::init();

    sectionOracleAgreement();
    sectionRelErr(false);
    sectionRelErr(true);
    sectionNearZero();
    sectionSeams();
    sectionStructural();
    sectionMonotonicity();
    sectionDerivatives();
    sectionExtremes();
    sectionBasisCond();
    sectionTranscription();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--full") == 0)
        {
            gPoints = 10000000;
            gAgree = 20000;
        }
        else if (std::strcmp(argv[i], "--points") == 0 && i + 1 < argc)
        {
            gPoints = std::atoi(argv[++i]);
        }
        else
        {
            std::println("usage: f2_minimax_check [--full] [--points N]");
            return 2;
        }
    }

    std::println("=== Chronos TanhNL regional minimax kernel harness ===");
    std::println("kernels: f1Tanh / f2Tanh (three regions, Estrin, two-part ln2)");
    std::println("sweep density: {} points, agreement grid {} points\n",
                gPoints, gAgree);

    int r = runAll();

    std::println("\n=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

/**
 * High-precision oracle harness for the TanhNL antiderivatives F1 and F2.
 * Certifies the accuracy of the current implementation and provides the
 * oracle that later kernels are measured against. Plain main(), exit
 * code, always-live CHECK/FAIL.
 */

#include "dsp/nonlinear/Nonlinearities.h"

#include "f2_dd_oracle.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <print>

namespace {

using MarsDSP::Nonlinear::TanhNL;
using namespace F2Oracle;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kU = 2.220446049250313e-16;   // DBL_EPSILON

int gPoints = 200;    // region sweep density (default: CI-sized)
int gAgree = 300;     // oracle-agreement grid size

void sectionSelfTest()
{
    g_section = "dd self-test";

    // twoSum / twoProd exactness identities.
    {
        double s = 0.0;
        double e = 0.0;
        twoSum(1.0, kU, s, e);
        CHECK(s == 1.0 + kU && e == 0.0);
        twoSum(1.0, kU * 0.5, s, e);
        CHECK(s == 1.0 && e == kU * 0.5);  // 2^-53 falls off the sum
        double p = 0.0;
        double ep = 0.0;
        twoProd(1.0 + kU, 1.0 + kU, p, ep);
        CHECK(p == 1.0 + 2.0 * kU);
        CHECK(ep == kU * kU);              // the dropped cross term
        std::println("twoSum/twoProd exactness: PASS");
    }

    // dd division by an integer round-trips.
    {
        const DD v = dd_from(0.875);
        const DD q = dd_div_int(dd_mul_int(v, 7), 7);
        CHECK(q.hi == v.hi && q.lo == v.lo);
        std::println("dd mul/div int round-trip (bit-exact): PASS");
    }

    // tanh series anchors: T_2 = -1/3, T_3 = 2/15, T_4 = -17/315.
    {
        const DD t2 = dd_div(dd_from(-1.0), dd_from(3.0));
        const DD t3 = dd_div(dd_from(2.0), dd_from(15.0));
        const DD t4 = dd_div(dd_from(-17.0), dd_from(315.0));
        CHECK(dd_abs_hi(dd_sub(gT[2], t2)) < 1e-30);
        CHECK(dd_abs_hi(dd_sub(gT[3], t3)) < 1e-30);
        CHECK(dd_abs_hi(dd_sub(gT[4], t4)) < 1e-30);
        std::println("tanh series coefficient anchors (DD recurrence): PASS");
    }

    // F2 series anchors: p_0 = 1/6, p_1 = -1/60.
    {
        const DD p0ref = dd_div(dd_from(1.0), dd_from(6.0));
        const DD p1ref = dd_div(dd_from(-1.0), dd_from(60.0));
        CHECK(dd_abs_hi(dd_sub(gP2[0], p0ref)) < 1e-30);
        CHECK(dd_abs_hi(dd_sub(gP2[1], p1ref)) < 1e-30);
        std::println("F2 series coefficient anchors: PASS");
    }

    // Gauss rule: weights sum to 2, first node matches the known value.
    {
        DD sum = dd_from(0.0);
        for (int k = 0; k < GaussDD::N; ++k)
            sum = dd_add(sum, g_gl.w[k]);
        CHECK(dd_abs_hi(dd_sub(sum, dd_from(2.0))) < 1e-28);
        CHECK(std::fabs(g_gl.x[GaussDD::N - 1].hi - 0.9894009349916499) < 1e-13);
        std::println("Gauss-Legendre-16 rule in DD (sum w = 2): PASS");
    }
}

void sectionAnchors()
{
    g_section = "dd transcendentals";

    const DD t = ddExpNeg2(dd_one());
    const DD dt = dd_sub(t, kAnchorExpNeg2);
    std::println("exp(-2)   vs 60-digit anchor: |diff| = {:.3e} (gate 1e-28)",
                dd_abs_hi(dt));
    CHECK(dd_abs_hi(dt) < 1e-28);

    const DD li = ddDilogNegDirect(kAnchorExpNeg2);
    const DD dl = dd_sub(li, kAnchorLi2);
    std::println("Li2(-e^-2) vs 60-digit anchor: |diff| = {:.3e} (gate 1e-28)",
                dd_abs_hi(dl));
    CHECK(dd_abs_hi(dl) < 1e-28);

    const DD f1 = ddLnCosh(dd_one());
    const DD df = dd_sub(f1, kAnchorF1at1);
    std::println("ln cosh(1) vs 60-digit anchor: |diff| = {:.3e} (gate 1e-26)",
                dd_abs_hi(df));
    CHECK(dd_abs_hi(df) < 1e-26);

    const DD f2 = f2DD(1.0);
    const DD dg = dd_sub(f2, kAnchorF2at1);
    std::println("F2(1)     vs 60-digit anchor: |diff| = {:.3e} (gate 1e-26)",
                dd_abs_hi(dg));
    CHECK(dd_abs_hi(dg) < 1e-26);

    // The series route must agree with the closed route at the a = 0.5 switch.
    const DD atHalf = f2DD(0.5);
    const DD quad = quadDD(0.5);
    const double r = ddRelDiff(atHalf, quad);
    std::println("series-vs-quadrature at a = 0.5: rel diff = {:.3e} (gate 1e-25)", r);
    CHECK(r < 1e-25);

    std::println("dd transcendentals vs 60-digit anchors: PASS");
}

void sectionAgreement()
{
    g_section = "oracle agreement";

    double worst = 0.0;
    double worstX = 0.0;
    int n = gAgree;
    for (int i = 0; i < n; ++i)
    {
        const double x = 1e-12 * std::pow(1e15, static_cast<double>(i) / (n - 1));
        const double r = ddRelDiff(f2DD(x), quadDD(x));
        if (r > worst) { worst = r; worstX = x; }
    }
    // The a = 0.5 switch of the closed oracle, straddled by one ulp steps.
    const std::array<double, 3> edge = { std::nextafter(0.5, 0.0), 0.5, std::nextafter(0.5, 1.0) };
    for (double x : edge)
    {
        const double r = ddRelDiff(f2DD(x), quadDD(x));
        if (r > worst) { worst = r; worstX = x; }
    }
    std::println("closed form vs quadrature, {} log-spaced points in [1e-12, 1000]",
                "    plus the a = 0.5 switch: max rel diff = {:.3e} at x = {:.6e} (gate 1e-25)",
                n, worst, worstX);
    CHECK(worst < 1e-25);
    std::println("oracle agreement (1e-25): PASS");
}

// One row of the status-quo single-point reference gates.
struct RefPoint { double x; double refRel; double refAbs; };

void sectionStatusQuoF2()
{
    g_section = "status quo F2";

    // Deterministic single-point rows. Independently measured with mpmath
    // at 60 digits against the exact float64 code path. Factor-2 gate:
    // the last ulp of the platform exp/log1p can move these values.
    const std::array<RefPoint, 8> relPts = {{
        { 1e-8,   4.6e8,  0.0 },
        { 4e-6,   1.70,   0.0 },
        { 6e-6,   0.83,   0.0 },
        { 8e-6,   0.37,   0.0 },
        { 1e-5,   0.13,   0.0 },
        { 1.8e-5,  0.0055, 0.0 },
        { 3e-5,   0.0081, 0.0 },
        { 1e-4,   2.5e-5, 0.0 },
    }};
    std::println("F2 single-point relative error vs reference (factor-2 gate):");
    for (const auto& rp : relPts)
    {
        const double got = TanhNL::F2(rp.x);
        const double refd = toDouble(f2DD(rp.x));
        const double rel = std::fabs(got - refd) / std::fabs(refd);
        const double ratio = rel / rp.refRel;
        std::println("    x = {:8.1e} : rel err = {:.3e}   ref {:.3e}   ratio {:.2f} {}",
                    rp.x, rel, rp.refRel, ratio, ratio <= 2.0 ? "" : "<-- FAIL");
        CHECK(ratio <= 2.0);
    }

    const std::array<RefPoint, 2> absPts = {{
        { 17.6, 0.0, 2.97e-14 },
        { 520.0, 0.0, 3.87e-11 },
    }};
    std::println("F2 single-point absolute error vs reference (factor-2 gate):");
    for (const auto& rp : absPts)
    {
        const double got = TanhNL::F2(rp.x);
        const double refd = toDouble(f2DD(rp.x));
        const double absE = std::fabs(got - refd);
        const double ratio = absE / rp.refAbs;
        std::println("    x = {:8.1f} : abs err = {:.3e}   ref {:.3e}   ratio {:.2f} {}",
                    rp.x, absE, rp.refAbs, ratio, ratio <= 2.0 ? "" : "<-- FAIL");
        CHECK(ratio <= 2.0);
    }
    std::println("status-quo F2 single-point gates: PASS");
}

struct Region { double lo; double hi; double refRel; double refAbs; const char* note; };

void sweepRegion(const Region& rg, bool f1)
{
    double maxRel = 0.0;
    double maxAbs = 0.0;
    double argRel = 0.0;
    double argAbs = 0.0;
    for (int i = 0; i < gPoints; ++i)
    {
        const double x = rg.lo * std::pow(rg.hi / rg.lo,
                                          static_cast<double>(i) / (gPoints - 1));
        const double got = f1 ? TanhNL::F1(x) : TanhNL::F2(x);
        const double refd = f1 ? toDouble(f1DD(x)) : toDouble(f2DD(x));
        const double absE = std::fabs(got - refd);
        const double rel = absE / std::fabs(refd);
        if (rel > maxRel) { maxRel = rel; argRel = x; }
        if (absE > maxAbs) { maxAbs = absE; argAbs = x; }
    }
    std::println("  [{:7.0e}, {:7.0e}]  rel {:9.2e} @ {:9.3e} ({})   abs {:9.2e} @ {:9.3e} ({})",
                rg.lo, rg.hi, maxRel, argRel, rg.note, maxAbs, argAbs,
                rg.refAbs > 0.0 ? "ref-bounded" : "no ref");
    // Alarm gates against the independently measured maxima.
    if (rg.refRel > 0.0)
        CHECK(maxRel <= 5.0 * rg.refRel);
    if (rg.refAbs > 0.0)
        CHECK(maxAbs <= 5.0 * rg.refAbs);
}

void sectionStatusQuoTables()
{
    // Region maxima of the current implementation. References from the
    // independent 60-digit measurement. Row 1 has no finite rel reference:
    // the relative error is unbounded there by design of the defect.
    g_section = "status quo F2 table";
    std::println("F2 status-quo error table ({} points per region):", gPoints);
    const std::array<Region, 7> regionsF2 = {{
        { 1e-9, 1e-3, 0.0,     1.09e-16, "unbounded" },
        { 1e-3, 1e-1, 1.4e-7,  1.44e-16, "ref 1.4e-7" },
        { 1e-1, 5e-1, 5.47e-13, 0.0,     "ref 5.5e-13" },
        { 5e-1, 1.0,  3.20e-15, 1.06e-16, "ref 3.2e-15" },
        { 1.0,  3.0,  7.58e-16, 0.0,     "ref 7.6e-16" },
        { 3.0, 19.0,  3.12e-16, 2.97e-14, "ref 3.1e-16" },
        { 19.0, 700.0, 3.01e-16, 3.87e-11, "ref 3.0e-16" },
    }};
    for (const auto& rg : regionsF2)
        sweepRegion(rg, false);
    std::println("status-quo F2 region table (factor-5 alarm): PASS");

    g_section = "status quo F1 table";
    std::println("F1 status-quo error table ({} points per region, informational):",
                gPoints);
    const std::array<Region, 7> regionsF1 = {{
        { 1e-9, 1e-3, 0.0, 0.0, "-" }, { 1e-3, 1e-1, 0.0, 0.0, "-" },
        { 1e-1, 5e-1, 0.0, 0.0, "-" }, { 5e-1, 1.0,  0.0, 0.0, "-" },
        { 1.0,  3.0,  0.0, 0.0, "-" }, { 3.0, 19.0,  0.0, 0.0, "-" },
        { 19.0, 700.0, 0.0, 0.0, "-" },
    }};
    for (const auto& rg : regionsF1)
        sweepRegion(rg, true);
    std::println("status-quo F1 region table: printed (no gates)");
}

int runAll()
{
    F2Oracle::init();

    sectionSelfTest();
    sectionAnchors();
    sectionAgreement();
    sectionStatusQuoF2();
    sectionStatusQuoTables();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--full") == 0)
        {
            gPoints = 3000;
            gAgree = 20000;
        }
        else if (std::strcmp(argv[i], "--points") == 0 && i + 1 < argc)
        {
            gPoints = std::atoi(argv[++i]);
        }
        else
        {
            std::println("usage: f2_oracle_check [--full] [--points N]");
            return 2;
        }
    }

    std::println("=== Chronos TanhNL antiderivative oracle harness ===");
    std::println("oracle: double-double closed form vs DD Gauss-Legendre-16 quadrature");
    std::println("sweep density: {} points/region, agreement grid {} points",
                gPoints, gAgree);
    std::println();

    const int r = runAll();

    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

// tests/harnesses/cd/f2_oracle_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// High-precision oracle harness for the TanhNL antiderivatives F1 and F2.
// It certifies the accuracy of the current implementation before anything
// changes, and it provides the oracle that later kernels are measured
// against.
//
// Why double-double (DD) arithmetic: long double is 64-bit on arm64, so it
// is bit-identical to the code under test. DD carries two doubles as an
// unevaluated sum. It gives roughly 31 significant decimal digits.
//
// Two independent oracles are implemented and cross-checked:
//   1. Closed form. The same formula the production code evaluates, but in
//      DD with a 100-term dilogarithm series. Near x = 0 the closed form
//      cancels badly, so there it switches to the Taylor series of F2. The
//      series coefficients come from a recurrence on the tanh series, which
//      follows from tanh' = 1 - tanh^2. The recurrence is stable and needs
//      no transcribed constants.
//   2. Quadrature. F2(x) = integral_0^x ln cosh(u) du by composite
//      Gauss-Legendre-16 with panels of width <= 0.5. Nodes and weights
//      come from a Newton solve on the Legendre recurrence, computed in DD.
//      Double nodes and weights would cap the oracle at about 1e-16, which
//      would void the 1e-25 agreement gate. The integrand ln cosh(u) uses
//      log1p(2*sinh^2(u/2)) for u <= 0.5. That form has only positive
//      terms, so it cannot cancel.
//
// The two oracles must agree to < 1e-25 relative. Then the harness measures
// the current TanhNL::F1 / TanhNL::F2 against the closed-form oracle and
// prints the error tables. Reference values were measured independently
// with mpmath at 60 digits. The single-point rows gate at a factor of 2.
// The region maxima print a factor-5 alarm gate: the maximum of a rounding
// error curve moves with the sweep grid, so the region gate only catches a
// broken oracle.
//
// Two properties of the current implementation are on record here:
//   - The relative error of F2 is UNBOUNDED as x -> 0. It is about 4.6e8
//     at x = 1e-8. This is the defect this file documents.
//   - The absolute error near zero floors at about 1.1e-16 (half a ulp of
//     the ~0.4-sized terms that cancel), not at 4e-17.
//
// Usage: f2_oracle_check [--full] [--points N]
//   Default: small sweep counts for CI (runs in a few seconds at -O0).
//   --full: dense sweeps for local use. --points N: region sweep density.
//
// Conventions (matching dilog_check / nonlinearity_check): plain main(),
// printf, exit code, always-live CHECK/FAIL (NOT assert). Links SharedCode
// only; no JUCE. No forced -O2 so the header assert preconditions stay
// armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/nonlinear/Nonlinearities.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using MarsDSP::Nonlinear::TanhNL;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kU   = 2.220446049250313e-16;  // DBL_EPSILON
constexpr double kLn2 = 0.6931471805599453;

// ── Double-double arithmetic ─────────────────────────────────────────────
// A DD value x represents hi + lo, with |lo| <= half a ulp of |hi|.
// The algorithms are the standard Dekker/Knuth pair. Their error is about
// 2^-105 times the largest operand magnitude. Every use below keeps the
// result within a small condition number of the operands, so the relative
// error stays near 2^-105.

struct DD { double hi, lo; };

DD dd_from(double v) noexcept { return { v, 0.0 }; }
DD dd_one() noexcept { return { 1.0, 0.0 }; }
DD dd_neg(DD a) noexcept { return { -a.hi, -a.lo }; }

// s + e == a + b exactly (Knuth 2Sum, branch-free).
void twoSum(double a, double b, double& s, double& e) noexcept
{
    s = a + b;
    const double v = s - a;
    e = (a - (s - v)) + (b - v);
}

// p + e == a * b exactly (Dekker two-product, fused multiply-add).
void twoProd(double a, double b, double& p, double& e) noexcept
{
    p = a * b;
    e = std::fma(a, b, -p);
}

// Normalize s + e into a DD pair. Requires |e| <= about |s|.
DD quickTwoSum(double s, double e) noexcept
{
    const double hi = s + e;
    return { hi, e - (hi - s) };
}

DD dd_add(DD a, DD b) noexcept
{
    double s, e;
    twoSum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    return quickTwoSum(s, e);
}

DD dd_add_d(DD a, double b) noexcept
{
    double s, e;
    twoSum(a.hi, b, s, e);
    e += a.lo;
    return quickTwoSum(s, e);
}

DD dd_sub(DD a, DD b) noexcept { return dd_add(a, dd_neg(b)); }

DD dd_mul(DD a, DD b) noexcept
{
    double p, e;
    twoProd(a.hi, b.hi, p, e);
    e += a.hi * b.lo + a.lo * b.hi;
    return quickTwoSum(p, e);
}

DD dd_mul_d(DD a, double s) noexcept
{
    double p, e;
    twoProd(a.hi, s, p, e);
    e += a.lo * s;
    return quickTwoSum(p, e);
}

// Multiply by a small exact integer.
DD dd_mul_int(DD a, int k) noexcept { return dd_mul_d(a, static_cast<double>(k)); }

DD dd_div(DD a, DD b) noexcept
{
    const double q1 = a.hi / b.hi;
    const DD r = dd_sub(a, dd_mul_d(b, q1));
    const double q2 = r.hi / b.hi;
    return quickTwoSum(q1, q2);
}

// Divide by a small exact integer. A correctly rounded reciprocal would
// inject about 1e-16 of relative error and void the DD precision, so this
// uses the full DD division path.
DD dd_div_int(DD a, int k) noexcept { return dd_div(a, dd_from(static_cast<double>(k))); }

double dd_abs_hi(DD a) noexcept { return std::fabs(a.hi); }

// ── DD constants ─────────────────────────────────────────────────────────

// ln(2), split hi/lo. The pair sums to ln(2) with a residual of 5.7e-34.
constexpr DD kLn2DD { 0.6931471805599453, 2.3190468138462996e-17 };
// pi, split hi/lo.
constexpr DD kPiDD { 3.141592653589793, 1.2246467991473532e-16 };

// Reference anchors, measured with mpmath at 60 digits and split hi/lo.
constexpr DD kAnchorExpNeg2 { 0.1353352832366127, -1.042381423288669e-17 };   // e^-2
constexpr DD kAnchorLi2     { -0.13101248471442378, 1.1246570985943699e-17 }; // Li2(-e^-2)
constexpr DD kAnchorF1at1   { 0.4337808304830272, 7.081895146469789e-18 };    // ln cosh(1)
constexpr DD kAnchorF2at1   { 0.15258009379489942, -9.965501769494956e-18 };  // F2(1)

// pi^2 / 24, computed once in DD from kPiDD.
DD gPiSq24;

// ── DD transcendentals ───────────────────────────────────────────────────

// e^(-2a) for a >= 0. Reduction r = y - k*ln2 with the hi/lo ln2 pair,
// then a Taylor series for exp(r), then ldexp for 2^k. For large a the
// result underflows to exactly +0.0. That is a contract, not an accident.
DD ddExpNeg2(DD a) noexcept
{
    assert(a.hi >= 0.0);
    const DD y = dd_mul_d(a, -2.0);                 // exact: power of two
    const long k = std::lround(y.hi / kLn2);
    const DD r = dd_sub(y, dd_mul_d(kLn2DD, static_cast<double>(k)));
    // |r| <= ln2/2 + a rounding whisker. exp(r) by Horner on r^m/m!.
    // Divisions stay exact through dd_div_int.
    DD acc = dd_one();
    for (int m = 26; m >= 1; --m)
        acc = dd_add_d(dd_div_int(dd_mul(r, acc), m), 1.0);
    // 0.35^26 / 26! ~ 3e-41, far below the DD floor.
    const int ki = static_cast<int>(k);
    return { std::ldexp(acc.hi, ki), std::ldexp(acc.lo, ki) };
}

// log1p(y) for 0 <= y <= 1, through log1p(y) = 2*atanh(z), z = y/(2+y).
// z <= 1/3, so the odd series converges fast. All terms are positive for
// y >= 0, so there is no cancellation.
DD ddLog1p(DD y) noexcept
{
    assert(y.hi >= 0.0 && y.hi <= 1.0);
    if (y.hi == 0.0) return dd_from(0.0);
    const DD z = dd_div(y, dd_add_d(y, 2.0));
    const DD v = dd_mul(z, z);
    // atanh(z)/z = sum_m v^m / (2m+1), Horner from the top.
    constexpr int N = 40;                            // (1/3)^(2N+1) ~ 4e-39
    DD acc = dd_div_int(dd_one(), 2 * N + 1);
    for (int m = N - 1; m >= 0; --m)
        acc = dd_add(dd_div_int(dd_one(), 2 * m + 1), dd_mul(v, acc));
    return dd_mul_d(dd_mul(z, acc), 2.0);
}

// sinh(y) for 0 <= y <= 0.25, Horner on y^2 with factorial coefficients
// from an exact integer recurrence. All terms positive.
DD ddSinhSmall(DD y) noexcept
{
    assert(y.hi >= 0.0 && y.hi <= 0.25);
    const DD v = dd_mul(y, y);
    constexpr int N = 13;                            // v^14/29! ~ 2e-47
    DD c[N + 1];
    c[0] = dd_one();                                 // c_m = 1/(2m+1)!
    for (int m = 1; m <= N; ++m)
        c[m] = dd_div_int(c[m - 1], 2 * m * (2 * m + 1));
    DD acc = c[N];
    for (int m = N - 1; m >= 0; --m)
        acc = dd_add(c[m], dd_mul(v, acc));
    return dd_mul(y, acc);
}

// ln cosh(a) for a >= 0. Two routes, both free of cancellation:
//   a <= 0.5: log1p(2*sinh^2(a/2)). cosh(a) - 1 = 2*sinh^2(a/2) is an
//             exact identity, and every term is positive.
//   a >  0.5: a - ln2 + log1p(e^-2a). The terms cancel at most a factor
//             of about 6 at a = 0.5, which DD absorbs.
DD ddLnCosh(DD a) noexcept
{
    assert(a.hi >= 0.0);
    if (a.hi <= 0.5)
    {
        const DD w = ddSinhSmall(dd_mul_d(a, 0.5));
        const DD v = dd_mul_d(dd_mul(w, w), 2.0);    // cosh(a) - 1
        return ddLog1p(v);                           // v <= 0.128
    }
    const DD t = ddExpNeg2(a);
    return dd_add(dd_sub(a, kLn2DD), ddLog1p(t));
}

// Li2(-t) for 0 <= t <= 0.5, alternating Horner, 100 terms. Mirrors the
// structure of the production dilogSeries, lifted to DD. The Landen fold
// is not needed: the closed-form F2 below only runs for a > 0.5, where
// t < e^-1 < 0.5 always.
DD ddDilogNegDirect(DD t) noexcept
{
    assert(t.hi >= 0.0 && t.hi <= 0.5);
    if (t.hi == 0.0) return dd_from(0.0);
    constexpr int kTerms = 100;                      // 0.5^100/100^2 ~ 8e-35
    DD acc = dd_div_int(dd_one(), kTerms * kTerms);
    for (int k = kTerms - 1; k >= 1; --k)
        acc = dd_sub(dd_div_int(dd_one(), k * k), dd_mul(t, acc));
    return dd_neg(dd_mul(t, acc));
}

// ── F2 Taylor series coefficients (region near zero) ─────────────────────
// F2(x) = x*u*sum_k p_k u^k, u = x^2. Since F2'' = tanh' = 1 - tanh^2,
// the tanh coefficients T_k in tanh(x) = sum_k T_k x^(2k-1) satisfy
//   T_1 = 1,   T_{m+1} = -(sum_{i=1}^{m} T_i T_{m+1-i}) / (2m+1).
// Integrating twice gives p_k = T_{k+1} / ((2k+2)(2k+3)).
// The recurrence sums products of same-scale terms, so it is stable in DD.
// No transcribed constants. T_k decays like (2/pi)^(2k), so 37 terms give
// a truncation of about 0.101^37 ~ 1e-37 at u = 0.25.

constexpr int kP2N = 36;                             // terms used at u <= 0.25
DD gT[kP2N + 2];                                     // tanh coefficients T_1..T_37
DD gP2[kP2N];                                        // F2 series coefficients

void buildF2Series() noexcept
{
    gT[1] = dd_one();
    for (int m = 1; m <= kP2N; ++m)
    {
        DD s = dd_from(0.0);
        for (int i = 1; i <= m; ++i)
            s = dd_add(s, dd_mul(gT[i], gT[m + 1 - i]));
        gT[m + 1] = dd_neg(dd_div_int(s, 2 * m + 1));
    }
    for (int k = 0; k < kP2N; ++k)
        gP2[k] = dd_div_int(dd_div_int(gT[k + 1], 2 * k + 2), 2 * k + 3);
}

// ── Closed-form oracle ───────────────────────────────────────────────────

DD ddF2Closed(double x) noexcept
{
    const double a = std::fabs(x);
    if (a == 0.0) return dd_from(0.0);
    DD mag;
    if (a <= 0.5)
    {
        const DD u = dd_mul(dd_from(a), dd_from(a)); // exact two-product
        DD acc = gP2[kP2N - 1];
        for (int k = kP2N - 2; k >= 0; --k)
            acc = dd_add(gP2[k], dd_mul(u, acc));
        mag = dd_mul(dd_from(a), dd_mul(u, acc));
    }
    else
    {
        const DD aa = dd_from(a);
        const DD t = ddExpNeg2(aa);
        const DD li = ddDilogNegDirect(t);
        const DD g = dd_add(dd_mul_d(li, 0.5), gPiSq24);
        const DD a2 = dd_mul(aa, aa);
        // The production closed form, in DD: a^2/2 - a*ln2 + g.
        mag = dd_add(dd_sub(dd_mul_d(a2, 0.5), dd_mul(aa, kLn2DD)), g);
    }
    return x < 0.0 ? dd_neg(mag) : mag;              // F2 is odd
}

// ── Quadrature oracle ────────────────────────────────────────────────────
// Composite Gauss-Legendre-16 over panels of width 0.5, prefix-summed once
// up to a = 1000. A partial panel at the top covers the remainder. The
// panel centre and half-width of the partial panel are exact in DD:
// a - b is a Sterbenz-exact subtraction and the centre uses a DD add.

struct GaussDD
{
    static constexpr int N = 16;
    DD x[N], w[N];

    void build() noexcept
    {
        for (int i = 0; i < (N + 1) / 2; ++i)
        {
            const double seed = std::cos(3.141592653589793 * (i + 0.75) / (N + 0.5));
            DD z = dd_from(seed);
            DD pp = dd_one();
            for (int it = 0; it < 60; ++it)
            {
                DD p1 = dd_one(), p2 = dd_from(0.0);
                for (int j = 0; j < N; ++j)
                {
                    const DD p3 = p2;
                    p2 = p1;
                    p1 = dd_div_int(dd_sub(dd_mul_int(dd_mul(z, p2), 2 * j + 1),
                                           dd_mul_int(p3, j)), j + 1);
                }
                pp = dd_mul_int(dd_div(dd_sub(dd_mul(z, p1), p2),
                                       dd_sub(dd_mul(z, z), dd_one())), N);
                const DD dz = dd_div(p1, pp);
                z = dd_sub(z, dz);
                if (dd_abs_hi(dz) < 1e-30) break;
            }
            x[i] = dd_neg(z);
            x[N - 1 - i] = z;
            const DD wk = dd_div(dd_from(2.0),
                                 dd_mul(dd_sub(dd_one(), dd_mul(z, z)),
                                        dd_mul(pp, pp)));
            w[i] = wk;
            w[N - 1 - i] = wk;
        }
    }
};

GaussDD g_gl;
constexpr int kMaxPanels = 2000;                     // covers a <= 1000
std::vector<DD> g_prefix;                            // prefix sums, size 2001

void buildPanels() noexcept
{
    g_prefix.assign(kMaxPanels + 1, dd_from(0.0));
    for (int p = 0; p < kMaxPanels; ++p)
    {
        const DD c = dd_from(0.5 * static_cast<double>(p) + 0.25); // exact
        const DD r = dd_from(0.25);                                // exact
        DD s = dd_from(0.0);
        for (int k = 0; k < GaussDD::N; ++k)
            s = dd_add(s, dd_mul(g_gl.w[k],
                                 ddLnCosh(dd_add(c, dd_mul(r, g_gl.x[k])))));
        g_prefix[p + 1] = dd_add(g_prefix[p], dd_mul(r, s));
    }
}

DD ddF2Quad(double x) noexcept
{
    const double a = std::fabs(x);
    if (a == 0.0) return dd_from(0.0);
    assert(a <= 1000.0);
    const int full = static_cast<int>(std::floor(2.0 * a)); // 2a is exact
    DD total = g_prefix[full];
    const double b = 0.5 * static_cast<double>(full);
    if (a > b)
    {
        // b <= a <= 2b for b >= 0.5, so a - b is exact (Sterbenz).
        // b == 0 gives a - b = a, also exact.
        const DD r = dd_mul_d(dd_from(a - b), 0.5);
        const DD c = dd_mul_d(dd_add(dd_from(a), dd_from(b)), 0.5);
        DD s = dd_from(0.0);
        for (int k = 0; k < GaussDD::N; ++k)
            s = dd_add(s, dd_mul(g_gl.w[k],
                                 ddLnCosh(dd_add(c, dd_mul(r, g_gl.x[k])))));
        total = dd_add(total, dd_mul(r, s));
    }
    return x < 0.0 ? dd_neg(total) : total;
}

// ── Harness body ─────────────────────────────────────────────────────────

int gPoints = 200;    // region sweep density (default: CI-sized)
int gAgree = 300;     // oracle-agreement grid size

// Relative agreement between two DD values, measured on the hi parts.
// The hi part of a DD difference carries magnitudes down to ~1e-32.
double ddRelDiff(DD a, DD b) noexcept
{
    const DD d = dd_sub(a, b);
    return dd_abs_hi(d) / std::fabs(b.hi);
}

void sectionSelfTest()
{
    g_section = "dd self-test";

    // twoSum / twoProd exactness identities.
    {
        double s, e;
        twoSum(1.0, kU, s, e);
        CHECK(s == 1.0 + kU && e == 0.0);
        twoSum(1.0, kU * 0.5, s, e);
        CHECK(s == 1.0 && e == kU * 0.5);  // 2^-53 falls off the sum
        double p, ep;
        twoProd(1.0 + kU, 1.0 + kU, p, ep);
        CHECK(p == 1.0 + 2.0 * kU);
        CHECK(ep == kU * kU);              // the dropped cross term
        std::printf("twoSum/twoProd exactness: PASS\n");
    }

    // dd division by an integer round-trips.
    {
        const DD v = dd_from(0.875);
        const DD q = dd_div_int(dd_mul_int(v, 7), 7);
        CHECK(q.hi == v.hi && q.lo == v.lo);
        std::printf("dd mul/div int round-trip (bit-exact): PASS\n");
    }

    // tanh series anchors: T_2 = -1/3, T_3 = 2/15, T_4 = -17/315.
    {
        const DD t2 = dd_div(dd_from(-1.0), dd_from(3.0));
        const DD t3 = dd_div(dd_from(2.0), dd_from(15.0));
        const DD t4 = dd_div(dd_from(-17.0), dd_from(315.0));
        CHECK(dd_abs_hi(dd_sub(gT[2], t2)) < 1e-30);
        CHECK(dd_abs_hi(dd_sub(gT[3], t3)) < 1e-30);
        CHECK(dd_abs_hi(dd_sub(gT[4], t4)) < 1e-30);
        std::printf("tanh series coefficient anchors (DD recurrence): PASS\n");
    }

    // F2 series anchors: p_0 = 1/6, p_1 = -1/60.
    {
        const DD p0ref = dd_div(dd_from(1.0), dd_from(6.0));
        const DD p1ref = dd_div(dd_from(-1.0), dd_from(60.0));
        CHECK(dd_abs_hi(dd_sub(gP2[0], p0ref)) < 1e-30);
        CHECK(dd_abs_hi(dd_sub(gP2[1], p1ref)) < 1e-30);
        std::printf("F2 series coefficient anchors: PASS\n");
    }

    // Gauss rule: weights sum to 2, first node matches the known value.
    {
        DD sum = dd_from(0.0);
        for (int k = 0; k < GaussDD::N; ++k)
            sum = dd_add(sum, g_gl.w[k]);
        CHECK(dd_abs_hi(dd_sub(sum, dd_from(2.0))) < 1e-28);
        CHECK(std::fabs(g_gl.x[GaussDD::N - 1].hi - 0.9894009349916499) < 1e-13);
        std::printf("Gauss-Legendre-16 rule in DD (sum w = 2): PASS\n");
    }
}

void sectionAnchors()
{
    g_section = "dd transcendentals";

    const DD t = ddExpNeg2(dd_one());
    const DD dt = dd_sub(t, kAnchorExpNeg2);
    std::printf("exp(-2)   vs 60-digit anchor: |diff| = %.3e (gate 1e-28)\n",
                dd_abs_hi(dt));
    CHECK(dd_abs_hi(dt) < 1e-28);

    const DD li = ddDilogNegDirect(kAnchorExpNeg2);
    const DD dl = dd_sub(li, kAnchorLi2);
    std::printf("Li2(-e^-2) vs 60-digit anchor: |diff| = %.3e (gate 1e-28)\n",
                dd_abs_hi(dl));
    CHECK(dd_abs_hi(dl) < 1e-28);

    const DD f1 = ddLnCosh(dd_one());
    const DD df = dd_sub(f1, kAnchorF1at1);
    std::printf("ln cosh(1) vs 60-digit anchor: |diff| = %.3e (gate 1e-26)\n",
                dd_abs_hi(df));
    CHECK(dd_abs_hi(df) < 1e-26);

    const DD f2 = ddF2Closed(1.0);
    const DD dg = dd_sub(f2, kAnchorF2at1);
    std::printf("F2(1)     vs 60-digit anchor: |diff| = %.3e (gate 1e-26)\n",
                dd_abs_hi(dg));
    CHECK(dd_abs_hi(dg) < 1e-26);

    // The series route must agree with the closed route at the a = 0.5
    // switch. Both are computed explicitly here.
    const DD atHalf = ddF2Closed(0.5);
    const DD quad = ddF2Quad(0.5);
    const double r = ddRelDiff(atHalf, quad);
    std::printf("series-vs-quadrature at a = 0.5: rel diff = %.3e (gate 1e-25)\n", r);
    CHECK(r < 1e-25);

    std::printf("dd transcendentals vs 60-digit anchors: PASS\n");
}

void sectionAgreement()
{
    g_section = "oracle agreement";

    double worst = 0.0, worstX = 0.0;
    int n = gAgree;
    for (int i = 0; i < n; ++i)
    {
        const double x = 1e-12 * std::pow(1e15, static_cast<double>(i) / (n - 1));
        const double r = ddRelDiff(ddF2Closed(x), ddF2Quad(x));
        if (r > worst) { worst = r; worstX = x; }
    }
    // The a = 0.5 switch of the closed oracle, straddled by one ulp steps.
    const double edge[3] = { std::nextafter(0.5, 0.0), 0.5, std::nextafter(0.5, 1.0) };
    for (double x : edge)
    {
        const double r = ddRelDiff(ddF2Closed(x), ddF2Quad(x));
        if (r > worst) { worst = r; worstX = x; }
    }
    std::printf("closed form vs quadrature, %d log-spaced points in [1e-12, 1000]\n"
                "    plus the a = 0.5 switch: max rel diff = %.3e at x = %.6e (gate 1e-25)\n",
                n, worst, worstX);
    CHECK(worst < 1e-25);
    std::printf("oracle agreement (1e-25): PASS\n");
}

// One row of the status-quo single-point reference gates.
struct RefPoint { double x, refRel, refAbs; };

void sectionStatusQuoF2()
{
    g_section = "status quo F2";

    // Deterministic single-point rows. Independently measured with mpmath
    // at 60 digits against the exact float64 code path. Factor-2 gate:
    // the last ulp of the platform exp/log1p can move these values.
    const RefPoint relPts[] = {
        { 1e-8,   4.6e8,  0.0 },
        { 4e-6,   1.70,   0.0 },
        { 6e-6,   0.83,   0.0 },
        { 8e-6,   0.37,   0.0 },
        { 1e-5,   0.13,   0.0 },
        { 1.8e-5, 0.0055, 0.0 },
        { 3e-5,   0.0081, 0.0 },
        { 1e-4,   2.5e-5, 0.0 },
        { 1e-3,   1.4e-7, 0.0 },
    };
    std::printf("F2 single-point relative error vs reference (factor-2 gate):\n");
    for (const auto& rp : relPts)
    {
        const double got = TanhNL::F2(rp.x);
        const DD ref = ddF2Closed(rp.x);
        const double refd = ref.hi + ref.lo;
        const double rel = std::fabs(got - refd) / std::fabs(refd);
        const double ratio = rel / rp.refRel;
        std::printf("    x = %8.1e : rel err = %.3e   ref %.3e   ratio %.2f %s\n",
                    rp.x, rel, rp.refRel, ratio, ratio <= 2.0 ? "" : "<-- FAIL");
        CHECK(ratio <= 2.0);
    }

    const RefPoint absPts[] = {
        { 17.6, 0.0, 2.97e-14 },
        { 520.0, 0.0, 3.87e-11 },
    };
    std::printf("F2 single-point absolute error vs reference (factor-2 gate):\n");
    for (const auto& rp : absPts)
    {
        const double got = TanhNL::F2(rp.x);
        const DD ref = ddF2Closed(rp.x);
        const double refd = ref.hi + ref.lo;
        const double absE = std::fabs(got - refd);
        const double ratio = absE / rp.refAbs;
        std::printf("    x = %8.1f : abs err = %.3e   ref %.3e   ratio %.2f %s\n",
                    rp.x, absE, rp.refAbs, ratio, ratio <= 2.0 ? "" : "<-- FAIL");
        CHECK(ratio <= 2.0);
    }
    std::printf("status-quo F2 single-point gates: PASS\n");
}

struct Region { double lo, hi; double refRel; double refAbs; const char* note; };

void sweepRegion(const Region& rg, bool f1)
{
    double maxRel = 0.0, maxAbs = 0.0, argRel = 0.0, argAbs = 0.0;
    for (int i = 0; i < gPoints; ++i)
    {
        const double x = rg.lo * std::pow(rg.hi / rg.lo,
                                          static_cast<double>(i) / (gPoints - 1));
        const double got = f1 ? TanhNL::F1(x) : TanhNL::F2(x);
        const DD ref = ddF2Closed(x);
        const DD refF1 = ddLnCosh(dd_from(x));
        const double refd = f1 ? (refF1.hi + refF1.lo) : (ref.hi + ref.lo);
        const double absE = std::fabs(got - refd);
        const double rel = absE / std::fabs(refd);
        if (rel > maxRel) { maxRel = rel; argRel = x; }
        if (absE > maxAbs) { maxAbs = absE; argAbs = x; }
    }
    std::printf("  [%7.0e, %7.0e]  rel %9.2e @ %9.3e (%s)   abs %9.2e @ %9.3e (%s)\n",
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
    std::printf("F2 status-quo error table (%d points per region):\n", gPoints);
    const Region regionsF2[] = {
        { 1e-9, 1e-3, 0.0,     1.09e-16, "unbounded" },
        { 1e-3, 1e-1, 1.4e-7,  1.44e-16, "ref 1.4e-7" },
        { 1e-1, 5e-1, 5.47e-13, 0.0,     "ref 5.5e-13" },
        { 5e-1, 1.0,  3.20e-15, 1.06e-16, "ref 3.2e-15" },
        { 1.0,  3.0,  7.58e-16, 0.0,     "ref 7.6e-16" },
        { 3.0, 19.0,  3.12e-16, 2.97e-14, "ref 3.1e-16" },
        { 19.0, 700.0, 3.01e-16, 3.87e-11, "ref 3.0e-16" },
    };
    for (const auto& rg : regionsF2)
        sweepRegion(rg, false);
    std::printf("status-quo F2 region table (factor-5 alarm): PASS\n");

    g_section = "status quo F1 table";
    std::printf("F1 status-quo error table (%d points per region, informational):\n",
                gPoints);
    const Region regionsF1[] = {
        { 1e-9, 1e-3, 0.0, 0.0, "-" }, { 1e-3, 1e-1, 0.0, 0.0, "-" },
        { 1e-1, 5e-1, 0.0, 0.0, "-" }, { 5e-1, 1.0,  0.0, 0.0, "-" },
        { 1.0,  3.0,  0.0, 0.0, "-" }, { 3.0, 19.0,  0.0, 0.0, "-" },
        { 19.0, 700.0, 0.0, 0.0, "-" },
    };
    for (const auto& rg : regionsF1)
        sweepRegion(rg, true);
    std::printf("status-quo F1 region table: printed (no gates)\n");
}

int runAll()
{
    gPiSq24 = dd_div_int(dd_mul(kPiDD, kPiDD), 24);
    buildF2Series();
    g_gl.build();
    buildPanels();

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
            std::printf("usage: f2_oracle_check [--full] [--points N]\n");
            return 2;
        }
    }

    std::printf("=== Chronos TanhNL antiderivative oracle harness ===\n");
    std::printf("oracle: double-double closed form vs DD Gauss-Legendre-16 quadrature\n");
    std::printf("sweep density: %d points/region, agreement grid %d points\n\n",
                gPoints, gAgree);

    int r = runAll();

    std::printf("\n=== %s ===\n", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

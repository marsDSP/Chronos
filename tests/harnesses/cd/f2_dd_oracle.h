// tests/harnesses/cd/f2_dd_oracle.h
// ──────────────────────────────────────────────────────────────────────────
// Shared double-double (DD) oracle for the TanhNL antiderivatives F1 and
// F2. Test-only: included by f2_oracle_check.cpp and f2_minimax_check.cpp.
// It must never ship in production code.
//
// Why DD arithmetic: long double is 64-bit on arm64, so it is bit-identical
// to the code under test. DD carries two doubles as an unevaluated sum and
// gives roughly 31 significant decimal digits.
//
// Two independent oracles, cross-checked against each other:
//   1. f2DD: the same closed form the production code evaluates, but in DD
//      with a 100-term dilogarithm series. Near x = 0 the closed form
//      cancels badly, so there it switches to the Taylor series of F2. The
//      series coefficients come from a recurrence on the tanh series, which
//      follows from tanh' = 1 - tanh^2. The recurrence is stable and needs
//      no transcribed constants.
//   2. quadDD: F2(x) = integral_0^x ln cosh(u) du by composite
//      Gauss-Legendre-16 with panels of width <= 0.5. Nodes and weights
//      come from a Newton solve on the Legendre recurrence, computed in DD.
//      Double nodes and weights would cap the oracle at about 1e-16, which
//      would void the 1e-25 agreement gate. The integrand ln cosh(u) uses
//      log1p(2*sinh^2(u/2)) for u <= 0.5. That form has only positive
//      terms, so it cannot cancel.
//
// f1DD is ln cosh in DD (the small-|x| route is again cancellation-free).
// F1 needs no second oracle beyond this.
//
// Usage: call F2Oracle::init() once before any oracle call. init() is
// idempotent. Everything here is inline so each harness binary keeps its
// own copy; there is no shared state across processes.
// ──────────────────────────────────────────────────────────────────────────

#pragma once

#ifndef CHRONOS_TESTS_F2_DD_ORACLE_H
#define CHRONOS_TESTS_F2_DD_ORACLE_H

#include <cassert>
#include <cmath>
#include <vector>

namespace F2Oracle {

// ── Double-double arithmetic ─────────────────────────────────────────────
// A DD value x represents hi + lo, with |lo| <= half a ulp of |hi|.
// The algorithms are the standard Dekker/Knuth pair. Their error is about
// 2^-105 times the largest operand magnitude. Every use below keeps the
// result within a small condition number of the operands, so the relative
// error stays near 2^-105.

struct DD { double hi, lo; };

inline DD dd_from(double v) noexcept { return { v, 0.0 }; }
inline DD dd_one() noexcept { return { 1.0, 0.0 }; }
inline DD dd_neg(DD a) noexcept { return { -a.hi, -a.lo }; }

// s + e == a + b exactly (Knuth 2Sum, branch-free).
inline void twoSum(double a, double b, double& s, double& e) noexcept
{
    s = a + b;
    const double v = s - a;
    e = (a - (s - v)) + (b - v);
}

// p + e == a * b exactly (Dekker two-product, fused multiply-add).
inline void twoProd(double a, double b, double& p, double& e) noexcept
{
    p = a * b;
    e = std::fma(a, b, -p);
}

// Normalize s + e into a DD pair. Requires |e| <= about |s|.
inline DD quickTwoSum(double s, double e) noexcept
{
    const double hi = s + e;
    return { hi, e - (hi - s) };
}

inline DD dd_add(DD a, DD b) noexcept
{
    double s, e;
    twoSum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    return quickTwoSum(s, e);
}

inline DD dd_add_d(DD a, double b) noexcept
{
    double s, e;
    twoSum(a.hi, b, s, e);
    e += a.lo;
    return quickTwoSum(s, e);
}

inline DD dd_sub(DD a, DD b) noexcept { return dd_add(a, dd_neg(b)); }

inline DD dd_mul(DD a, DD b) noexcept
{
    double p, e;
    twoProd(a.hi, b.hi, p, e);
    e += a.hi * b.lo + a.lo * b.hi;
    return quickTwoSum(p, e);
}

inline DD dd_mul_d(DD a, double s) noexcept
{
    double p, e;
    twoProd(a.hi, s, p, e);
    e += a.lo * s;
    return quickTwoSum(p, e);
}

// Multiply by a small exact integer.
inline DD dd_mul_int(DD a, int k) noexcept { return dd_mul_d(a, static_cast<double>(k)); }

inline DD dd_div(DD a, DD b) noexcept
{
    const double q1 = a.hi / b.hi;
    const DD r = dd_sub(a, dd_mul_d(b, q1));
    const double q2 = r.hi / b.hi;
    return quickTwoSum(q1, q2);
}

// Divide by a small exact integer. A correctly rounded reciprocal would
// inject about 1e-16 of relative error and void the DD precision, so this
// uses the full DD division path.
inline DD dd_div_int(DD a, int k) noexcept { return dd_div(a, dd_from(static_cast<double>(k))); }

inline double dd_abs_hi(DD a) noexcept { return std::fabs(a.hi); }

// Relative difference between two DD values, measured on the hi parts.
// The hi part of a DD difference carries magnitudes down to ~1e-32.
inline double ddRelDiff(DD a, DD b) noexcept
{
    const DD d = dd_sub(a, b);
    return dd_abs_hi(d) / std::fabs(b.hi);
}

// ── DD constants ─────────────────────────────────────────────────────────

// ln(2), split hi/lo. The pair sums to ln(2) with a residual of 5.7e-34.
inline constexpr DD kLn2DD { 0.6931471805599453, 2.3190468138462996e-17 };
// pi, split hi/lo.
inline constexpr DD kPiDD { 3.141592653589793, 1.2246467991473532e-16 };

// Reference anchors, measured with mpmath at 60 digits and split hi/lo.
inline constexpr DD kAnchorExpNeg2 { 0.1353352832366127, -1.042381423288669e-17 };   // e^-2
inline constexpr DD kAnchorLi2     { -0.13101248471442378, 1.1246570985943699e-17 }; // Li2(-e^-2)
inline constexpr DD kAnchorF1at1   { 0.4337808304830272, 7.081895146469789e-18 };    // ln cosh(1)
inline constexpr DD kAnchorF2at1   { 0.15258009379489942, -9.965501769494956e-18 };  // F2(1)

constexpr double kLn2 = 0.6931471805599453;

// ── Internal state (built by init()) ─────────────────────────────────────

inline DD gPiSq24;            // pi^2 / 24, from kPiDD in DD
inline bool gReady = false;

// tanh coefficients T_1..T_37 and the F2 series coefficients p_k.
// F2(x) = x*u*sum_k p_k u^k, u = x^2. Since F2'' = tanh' = 1 - tanh^2,
// the tanh coefficients T_k in tanh(x) = sum_k T_k x^(2k-1) satisfy
//   T_1 = 1,   T_{m+1} = -(sum_{i=1}^{m} T_i T_{m+1-i}) / (2m+1).
// Integrating twice gives p_k = T_{k+1} / ((2k+2)(2k+3)).
// The recurrence sums products of same-scale terms, so it is stable in DD.
// No transcribed constants. T_k decays like (2/pi)^(2k), so 37 terms give
// a truncation of about 0.101^37 ~ 1e-37 at u = 0.25.
constexpr int kP2N = 36;                              // terms used at u <= 0.25
inline DD gT[kP2N + 2];
inline DD gP2[kP2N];

// ── DD transcendentals ───────────────────────────────────────────────────

// e^(-2a) for a >= 0. Reduction r = y - k*ln2 with the hi/lo ln2 pair,
// then a Taylor series for exp(r), then ldexp for 2^k. For large a the
// result underflows to exactly +0.0. That is a contract, not an accident.
inline DD ddExpNeg2(DD a) noexcept
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
inline DD ddLog1p(DD y) noexcept
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
inline DD ddSinhSmall(DD y) noexcept
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
inline DD ddLnCosh(DD a) noexcept
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
inline DD ddDilogNegDirect(DD t) noexcept
{
    assert(t.hi >= 0.0 && t.hi <= 0.5);
    if (t.hi == 0.0) return dd_from(0.0);
    constexpr int kTerms = 100;                      // 0.5^100/100^2 ~ 8e-35
    DD acc = dd_div_int(dd_one(), kTerms * kTerms);
    for (int k = kTerms - 1; k >= 1; --k)
        acc = dd_sub(dd_div_int(dd_one(), k * k), dd_mul(t, acc));
    return dd_neg(dd_mul(t, acc));
}

// ── Gauss-Legendre-16 in DD ──────────────────────────────────────────────

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

inline GaussDD g_gl;
constexpr int kMaxPanels = 2000;                     // covers a <= 1000
inline std::vector<DD> g_prefix;                     // prefix sums, size 2001

// ── init: idempotent one-time build ──────────────────────────────────────

inline void init() noexcept
{
    if (gReady) return;
    gPiSq24 = dd_div_int(dd_mul(kPiDD, kPiDD), 24);

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

    g_gl.build();

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

    gReady = true;
}

// ── Closed-form oracle ───────────────────────────────────────────────────

inline DD f2DD(double x) noexcept
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

// F1 = ln cosh, in DD.
inline DD f1DD(double x) noexcept
{
    return ddLnCosh(dd_from(std::fabs(x)));
}

// ── Quadrature oracle ────────────────────────────────────────────────────
// Composite Gauss-Legendre-16 over panels of width 0.5, prefix-summed once
// up to a = 1000. A partial panel at the top covers the remainder. The
// panel centre and half-width of the partial panel are exact in DD:
// a - b is a Sterbenz-exact subtraction and the centre uses a DD add.

inline DD quadDD(double x) noexcept
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

// Double view of a DD value (nearest double to hi + lo).
inline double toDouble(DD a) noexcept { return a.hi + a.lo; }

} // namespace F2Oracle

#endif

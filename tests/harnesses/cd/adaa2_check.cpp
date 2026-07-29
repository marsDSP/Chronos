// tests/harnesses/cd/adaa2_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Correctness harness for MarsDSP::Nonlinear::ADAA2 (and, for the static
// curve, ADAA1). ADAA2 output is twice the second divided difference of F2
// over the last three input samples; the whole difficulty is that the two
// nested divisions cancel as 1/delta^2, so the kernel carries four branches
// and the harness has to certify each of them AND the seams between them.
//
//   1. Static curve      – constant input settles to f(x), |err| <= 1e-13,
//                          x in [-40, 40]; ADAA2/ADAA1 x TanhNL/AlgebraicNL.
//   2. Branch (b)        – confluent outer nodes x0 = x2 = a, x1 = b: the
//                          Nyquist "...a,b,a,b..." pattern heavy saturation
//                          produces. Log grid |a-b| in [1e-6, 20] x a in
//                          [-20, 20], both signs. THIS IS WHY THE HARNESS
//                          EXISTS - the generic formula divides by ~0 here.
//   3. Branch seams      – walk delta across kEpsInner and kEpsOuter in 1000
//                          steps each; the error must not jump, and the
//                          output step must not exceed the oracle's own step
//                          plus the local error bound. A seam discontinuity
//                          is a program-dependent switch between two transfer
//                          curves: it radiates exactly the broadband energy
//                          ADAA exists to remove, and no static-curve test
//                          can see it.
//   4. Error surface     – the physically relevant node family
//                          x_k = A sin(w(n-k)), A in [0.1, 40] x w in
//                          [1e-4, pi]. Reports max error with its argmax.
//                          GATES THE EXIT CODE.
//   5. Degenerate        – all 2^3 coincidence patterns plus all-zero and
//                          the +-700 extremes; every output must be finite.
//   6. Reset             – 100 samples, reset(), the same 100: bit-exact.
//   7. Parity            – odd curves: -x[n] in => -y[n] out, bit-exact.
//
// ──────────────────────────────────────────────────────────────
// NOT a long double implementation of the generic formula: long double is
// 64-bit on arm64 (the same note AGENTS.md records for dilog_check), so it
// would be bit-identical to the code under test and would certify nothing.
//
// Instead use Hermite-Genocchi, which turns the divided difference into an
// integral and so has no cancellation at all:
//
//     y = 2*F2[x0,x1,x2] = 2 * INT_simplex f(t0*x0 + t1*x1 + t2*x2) dt
//
// i.e. y is the mean of f over the triangle spanned by the three nodes -
// manifestly bounded by max|f| and manifestly continuous in the nodes, which
// is precisely what the branchy kernel is trying to reproduce. Two evaluated
// forms, both verified against f(x) = x (where y must equal (x0+x1+x2)/3):
//
//   wide nodes  (span >= 0.5): integrate the inner direction analytically via
//       F1, leaving one 1-D integral
//           y = 2/(x2-x1) INT_0^1 [F1(x0+(x2-x0)t) - F1(x0+(x1-x0)t)] dt
//       The residual 1/(x2-x1) is defused by permuting the (symmetric) node
//       list so the widest-separated pair lands in the denominator, which
//       bounds the cancellation by 2*u*max|F1|/span <= 4e-14.
//
//   tight nodes (span <  0.5): the 2-D tensor form in f directly,
//           y = 2 INT_0^1 t INT_0^1 f(x0 + q*t + c*t*s) ds dt
//       which averages bounded values and cannot cancel at all.
//
// Both are integrated with composite Gauss-Legendre-16, panels sized so each
// panel spans <= 1.0 in x. tanh's nearest poles sit at +-i*pi/2, giving a
// Bernstein parameter ~6 and a per-panel error ~6^-32; the quadrature is
// exact to rounding. Nodes come from a Newton solve on the Legendre
// recurrence rather than transcribed tables (no typo surface).
//
// ── Tolerances ────────────────────────────────────────────────────────────
// A fixed tolerance is meaningless here: the achievable accuracy swings over
// ten orders of magnitude with node spacing. Every comparison is therefore
// against a per-point error MODEL (errBound below) that predicts the branch's
// own truncation plus its cancellation, and the gate is on the ratio
// err/bound. A regression shows up as a ratio blow-up regardless of where in
// the node space it lands.
//
// Conventions (matching ring_buffer_check / dilog_check / nonlinearity_check):
// plain main(), exit code, printf, always-live CHECK/FAIL (NOT assert, which
// NDEBUG would void). Links SharedCode only; no JUCE. No forced -O2 so the
// header assert preconditions stay armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using MarsDSP::Nonlinear::ADAA1;
using MarsDSP::Nonlinear::ADAA2;
using MarsDSP::Nonlinear::AlgebraicNL;
using MarsDSP::Nonlinear::TanhNL;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kPi = 3.14159265358979323846;
constexpr double kU = 2.220446049250313e-16;   // DBL_EPSILON

// Mirrors of ADAA2<NL>::kEpsInner / kEpsOuter, which are private. Kept in
// sync by hand: if they change in the header they must change here, and
// sections 3 and 4 will complain loudly if they drift.
constexpr double kEpsInner = 1e-4;
constexpr double kEpsOuter = 1e-6;

// Slack on the error model. The model captures the leading term of each
// branch's truncation and cancellation; the constant absorbs the sub-leading
// terms it deliberately omits. Measured maxima sit well under this. see the
// printed ratios.
constexpr double kSlack = 40.0;

// Absolute gate on the section-4 error surface. The measured floor is 3.1e-4
// for TanhNL, reached at A ~ 1.4 with a near-Nyquist alternation: there the
// inner gaps sit just either side of kEpsInner while the outer gap C is ~60x
// smaller, so branch (c)'s O(A^2/24) midpoint truncation enters ONE of the
// two inner quotients and is then amplified by the outer 2/C. (Symmetric
// truncation would cancel; it is the asymmetry across the seam that costs.)
// The gate carries ~3x headroom over that.
constexpr double kSurfaceGate = 1e-3;

// ── Gauss-Legendre ────────────────────────────────────────────────────────

struct GaussRule
{
    static constexpr int N = 16;
    double x[N]{};
    double w[N]{};

    GaussRule() noexcept
    {
        for (int i = 0; i < (N + 1) / 2; ++i)
        {
            double z = std::cos(kPi * (static_cast<double>(i) + 0.75) / (N + 0.5));
            double pp = 0.0;
            for (int it = 0; it < 100; ++it)
            {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < N; ++j)
                {
                    const double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
                }
                pp = N * (z * p1 - p2) / (z * z - 1.0);
                const double dz = p1 / pp;
                z -= dz;
                if (std::fabs(dz) <= 1e-17)
                    break;
            }
            x[i] = -z;
            x[N - 1 - i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            w[N - 1 - i] = w[i];
        }
    }
};

const GaussRule g_gl{};

// Composite Gauss-Legendre-16 of g over [0, 1].
template <typename G>
double integrate01(G&& g, int panels) noexcept
{
    const double h = 1.0 / static_cast<double>(panels);
    double sum = 0.0;
    for (int p = 0; p < panels; ++p)
    {
        const double c = (static_cast<double>(p) + 0.5) * h;
        const double r = 0.5 * h;
        double s = 0.0;
        for (int k = 0; k < GaussRule::N; ++k)
            s += g_gl.w[k] * g(c + r * g_gl.x[k]);
        sum += r * s;
    }
    return sum;
}

// Hermite-Genocchi oracle: y = 2*F2[x0,x1,x2]
template <typename NL>
double oracleADAA2(double n0, double n1, double n2) noexcept
{
    const double lo = std::min({ n0, n1, n2 });
    const double hi = std::max({ n0, n1, n2 });
    const double span = hi - lo;

    if (span < 0.5)
    {
        const double q = n1 - n0;
        const double c = n2 - n1;
        return 2.0 * integrate01([&](double t) {
            return t * integrate01([&](double s) { return NL::f(n0 + q * t + c * t * s); }, 1);
        }, 1);
    }

    // The divided difference is symmetric in its nodes, so relabel to put the
    // widest-separated pair (== the full span) into the denominator.
    const double d01 = std::fabs(n0 - n1);
    const double d02 = std::fabs(n0 - n2);
    const double d12 = std::fabs(n1 - n2);
    double x0 = n0, x1 = n1, x2 = n2;
    if (d12 >= d01 && d12 >= d02)      { x0 = n0; x1 = n1; x2 = n2; }
    else if (d02 >= d01)               { x0 = n1; x1 = n0; x2 = n2; }
    else                               { x0 = n2; x1 = n0; x2 = n1; }

    const int panels = std::clamp(static_cast<int>(std::ceil(span)), 1, 128);
    const double I = integrate01([&](double t) {
        return NL::F1(x0 + (x2 - x0) * t) - NL::F1(x0 + (x1 - x0) * t);
    }, panels);
    return 2.0 * I / (x2 - x1);
}

// The absolute rounding error of an F1/F2 evaluation is set by the largest
// INTERMEDIATE, not by the magnitude of the result. TanhNL::F2 assembles
// sign(x)*(x^2/2 - |x|*ln2 + G) with G = Li2(-e^-2|x|)/2 + pi^2/24, so near
// x = 0 three ~0.4-sized terms cancel down to ~x^3/6: |F2| there is 1e-13
// while its error floor is still ~u. (This is the same cancellation Commit 12
// of the spec proposes to strip, seen from the small-|x| end instead of the
// large-|x| end.) These two scales bound the largest intermediate for both
// policies at every |x|.
template <typename NL>
double scaleF2(double x) noexcept { return std::fabs(NL::F2(x)) + MarsDSP::Nonlinear::kLn2 * std::fabs(x) + 1.0; }

template <typename NL>
double scaleF1(double x) noexcept { return std::fabs(NL::F1(x)) + std::fabs(x) + 1.0; }

// Per-point error model, branch by branch. Leading terms only:
//   (a) centroid truncation  |f''| * SUM(xi-s)^2 / 24, with max|f''| ~ 0.86
//       for both curves and SUM(xi-s)^2 = (A^2+B^2+C^2)/3.
//   (c) midpoint-derivative truncation of an inner quotient, A^2/24.
//   otherwise the cancellation 2*u*scaleF2 / gap in each inner quotient,
//   propagated through the outer division.
// The trailing term is the rounding of the final ops plus the oracle's own
// floor, which differs between its two quadrature forms.
template <typename NL>
double errBound(double x0, double x1, double x2, double y) noexcept
{
    const double A = std::fabs(x0 - x1);
    const double B = std::fabs(x1 - x2);
    const double C = std::fabs(x0 - x2);
    const double span = std::max({ x0, x1, x2 }) - std::min({ x0, x1, x2 });

    const double s1 = std::fmax(scaleF1<NL>(x0), std::fmax(scaleF1<NL>(x1), scaleF1<NL>(x2)));
    const double s2 = std::fmax(scaleF2<NL>(x0), std::fmax(scaleF2<NL>(x1), scaleF2<NL>(x2)));

    const double oracleFloor = (span < 0.5) ? 1e-15 : 8.0 * kU * s1 / span;
    const double round = 8.0 * kU * (std::fabs(y) + 1.0) + oracleFloor;

    if (A < kEpsInner && B < kEpsInner)
        return 0.05 * (A * A + B * B + C * C) + round;

    const double d1e = (A < kEpsInner) ? A * A / 24.0 : 2.0 * kU * s2 / A;

    if (C < kEpsOuter)
    {
        // Branch (b) evaluates F1 at m02 but reuses d1 = F2[x0,x1]. Once
        // x0 != x2 those disagree by F2[x0,x0,x1]*(m02-x0) = y*C/4, so the
        // branch truncates at O(C) - not O(C^2) - and that term dominates
        // everything else here. It is what sets kEpsOuter.
        const double m = std::fmax(std::fabs(0.5 * (x0 + x2) - x1), 0.5 * kEpsInner);
        return std::fabs(y) * C / (2.0 * m) + 2.0 * (d1e + 2.0 * kU * s1) / m + round;
    }

    const double d2e = (B < kEpsInner) ? B * B / 24.0 : 2.0 * kU * s2 / B;
    return 2.0 * (d1e + d2e) / C + round;
}

// Single-shot evaluation of the kernel at an explicit node triple. The state
// is private, so prime it by feeding x2 then x1: after those two calls
// x1_ = x1, x2_ = x2, F2x1_ = F2(x1) and d2_ = F2[x1,x2] -
// exactly (including any fallback it took) the state the third call needs.
template <typename NL>
double evalTriple(double x0, double x1, double x2) noexcept
{
    ADAA2<NL> s;
    s.reset();
    (void) s.process(x2);
    (void) s.process(x1);
    return s.process(x0);
}

// Running max of err/bound, carrying the argmax for the report.
struct Worst
{
    double ratio = 0.0;
    double err = 0.0;
    double bound = 0.0;
    double a = 0.0, b = 0.0, c = 0.0;   // argmax nodes (or A/omega in §4)

    void feed(double e, double bnd, double n0, double n1, double n2) noexcept
    {
        const double r = e / bnd;
        if (r > ratio) { ratio = r; err = e; bound = bnd; a = n0; b = n1; c = n2; }
    }
};

// ── 1. Static curve ───────────────────────────────────────────────────────

template <typename NL, typename Stage>
void checkStaticCurve(const char* what)
{
    constexpr double kTol = 1e-13;
    double worst = 0.0, worstX = 0.0;
    for (int i = 0; i <= 800; ++i)
    {
        const double x = -40.0 + 80.0 * static_cast<double>(i) / 800.0;
        Stage s;
        s.reset();
        double y = 0.0;
        for (int k = 0; k < 8; ++k)
            y = s.process(x);
        const double e = std::fabs(y - NL::f(x));
        if (e > worst) { worst = e; worstX = x; }
        if (e > kTol)
            FAIL("%s static curve at x=%.6f: y=%.17g f=%.17g |err|=%.3e", what, x, y, NL::f(x), e);
    }
    std::printf("  %-22s max |y - f(x)| = %.3e at x=%+.3f (tol %.0e)\n", what, worst, worstX, kTol);
}

// ── 2. Branch (b): confluent outer nodes ──────────────────────────────────

template <typename NL>
void checkConfluent(const char* what)
{
    Worst w;
    double maxAbs = 0.0, maxAbsA = 0.0, maxAbsD = 0.0;
    int nBranchB = 0, nBranchA = 0;

    for (int ia = 0; ia <= 20; ++ia)
    {
        const double a = -20.0 + 2.0 * static_cast<double>(ia);
        for (int id = 0; id < 25; ++id)
        {
            // log grid, 1e-6 .. 20
            const double d = std::pow(10.0, -6.0 + 7.30103 * static_cast<double>(id) / 24.0);
            for (int sgn = 0; sgn < 2; ++sgn)
            {
                const double b = (sgn == 0) ? a - d : a + d;
                const double y = evalTriple<NL>(a, b, a);
                const double ref = oracleADAA2<NL>(a, b, a);
                const double e = std::fabs(y - ref);
                const double bnd = errBound<NL>(a, b, a, y);
                (d < kEpsInner ? nBranchA : nBranchB) += 1;

                if (!std::isfinite(y))
                    FAIL("%s non-finite at a=%.6f b=%.6f", what, a, b);
                if (e > maxAbs) { maxAbs = e; maxAbsA = a; maxAbsD = b - a; }
                w.feed(e, bnd, a, b, a);
                if (e > kSlack * bnd)
                    FAIL("%s a=%+.4f delta=%+.3e: y=%.17g ref=%.17g |err|=%.3e bound=%.3e ratio=%.1f",
                         what, a, b - a, y, ref, e, bnd, e / bnd);
            }
        }
    }
    std::printf("  %-22s max |err| = %.3e at a=%+.1f delta=%+.3e\n", what, maxAbs, maxAbsA, maxAbsD);
    std::printf("  %-22s max err/bound = %.2f  (err %.3e, bound %.3e) at a=%+.4f b=%+.4f\n",
                "", w.ratio, w.err, w.bound, w.a, w.b);
    std::printf("  %-22s %d pts took branch (b), %d fell through to (a)\n", "", nBranchB, nBranchA);
}

// ── 3. Branch-seam continuity ─────────────────────────────────────────────
//
// Walk delta across a seam and demand two things at every step: the error
// stays inside the model, and the output STEP stays inside the oracle's own
// step plus the two local bounds. The second is the real test - it fails on a
// discontinuity even if both sides are individually accurate.

template <typename NL>
void checkSeam(const char* what, double centre, bool inner)
{
    constexpr int N = 1000;
    const double eps = inner ? kEpsInner : kEpsOuter;
    Worst w;
    double maxJump = 0.0, maxJumpAt = 0.0;
    double prevY = 0.0, prevRef = 0.0, prevBnd = 0.0;

    for (int i = 0; i <= N; ++i)
    {
        const double d = eps * (0.5 + 1.0 * static_cast<double>(i) / static_cast<double>(N));

        // inner seam: only A = |x0-x1| crosses eps (B and C stay ~1).
        // outer seam: only C = |x0-x2| crosses eps (A and B stay ~1).
        const double x0 = inner ? centre + d : centre;
        const double x1 = inner ? centre : centre - 1.0;
        const double x2 = inner ? centre - 1.0 : centre + d;

        const double y = evalTriple<NL>(x0, x1, x2);
        const double ref = oracleADAA2<NL>(x0, x1, x2);
        const double bnd = errBound<NL>(x0, x1, x2, y);
        const double e = std::fabs(y - ref);
        w.feed(e, bnd, x0, x1, x2);
        if (e > kSlack * bnd)
            FAIL("%s delta=%.6e: |err|=%.3e bound=%.3e ratio=%.1f", what, d, e, bnd, e / bnd);

        if (i > 0)
        {
            const double step = std::fabs(y - prevY);
            const double allow = std::fabs(ref - prevRef) + 10.0 * (bnd + prevBnd);
            if (step > allow)
                FAIL("%s seam jump at delta=%.6e: |dy|=%.3e allowed %.3e (oracle step %.3e)",
                     what, d, step, allow, std::fabs(ref - prevRef));
            const double excess = step - std::fabs(ref - prevRef);
            if (excess > maxJump) { maxJump = excess; maxJumpAt = d; }
        }
        prevY = y;
        prevRef = ref;
        prevBnd = bnd;
    }
    std::printf("  %-22s max err/bound = %5.2f | worst excess step = %.3e at delta=%.4e\n",
                what, w.ratio, maxJump, maxJumpAt);
}

// ── 4. Error surface over the physical node family ────────────────────────

template <typename NL>
double checkErrorSurface(const char* what)
{
    constexpr int kA = 10;
    constexpr int kW = 20;
    constexpr int kN = 64;

    Worst w;
    double maxAbs = 0.0, maxAbsAmp = 0.0, maxAbsOmega = 0.0;

    for (int ia = 0; ia < kA; ++ia)
    {
        const double amp = std::pow(10.0, -1.0 + 2.60206 * static_cast<double>(ia) / (kA - 1));   // 0.1 .. 40
        for (int iw = 0; iw < kW; ++iw)
        {
            const double omega = std::pow(10.0, -4.0 + 4.49715 * static_cast<double>(iw) / (kW - 1)); // 1e-4 .. pi

            ADAA2<NL> s;
            s.reset();
            double xm1 = 0.0, xm2 = 0.0;
            for (int n = 0; n < kN; ++n)
            {
                const double x = amp * std::sin(omega * static_cast<double>(n));
                const double y = s.process(x);
                if (!std::isfinite(y))
                    FAIL("%s non-finite at A=%.4f w=%.6f n=%d", what, amp, omega, n);
                if (n >= 2)
                {
                    const double ref = oracleADAA2<NL>(x, xm1, xm2);
                    const double e = std::fabs(y - ref);
                    const double bnd = errBound<NL>(x, xm1, xm2, y);
                    if (e > maxAbs) { maxAbs = e; maxAbsAmp = amp; maxAbsOmega = omega; }
                    // argmax carries (A, omega) here, not nodes.
                    w.feed(e, bnd, amp, omega, static_cast<double>(n));
                }
                xm2 = xm1;
                xm1 = x;
            }
        }
    }
    std::printf("  %-22s max |err| = %.3e at A=%.3f w=%.5f (%.1f Hz @ 48k)  [gate %.0e]\n",
                what, maxAbs, maxAbsAmp, maxAbsOmega, maxAbsOmega * 48000.0 / (2.0 * kPi), kSurfaceGate);
    std::printf("  %-22s max err/bound = %5.2f (err %.3e, bound %.3e) at A=%.3f w=%.5f n=%d\n",
                "", w.ratio, w.err, w.bound, w.a, w.b, static_cast<int>(w.c));
    if (w.ratio > kSlack)
        FAIL("%s error surface exceeds model: ratio %.1f > %.0f", what, w.ratio, kSlack);
    if (maxAbs > kSurfaceGate)
        FAIL("%s error surface floor regressed: %.3e > %.0e", what, maxAbs, kSurfaceGate);
    return maxAbs;
}

// ── 5. Degenerate node patterns ───────────────────────────────────────────

template <typename NL>
void checkDegenerate(const char* what)
{
    const double vals[] = { 0.0, 1e-18, 1e-9, kEpsInner, 0.5 * kEpsInner, 0.7, 3.0, 40.0, -40.0, 700.0, -700.0 };
    constexpr int nv = static_cast<int>(sizeof(vals) / sizeof(vals[0]));
    int n = 0;

    // All 2^3 coincidence patterns: each of x1, x2 either equals x0 or is
    // offset by one of the test values.
    for (int i = 0; i < nv; ++i)
        for (int j = 0; j < nv; ++j)
            for (int k = 0; k < nv; ++k)
            {
                const double x0 = vals[i];
                for (int p = 0; p < 8; ++p)
                {
                    const double x1 = (p & 1) ? x0 : x0 + vals[j];
                    const double x2 = (p & 2) ? x0 : x0 + vals[k];
                    const double x0b = (p & 4) ? x1 : x0;
                    const double y = evalTriple<NL>(x0b, x1, x2);
                    ++n;
                    if (!std::isfinite(y))
                        FAIL("%s non-finite at (%.17g, %.17g, %.17g) -> %.17g", what, x0b, x1, x2, y);
                    // The exact output is a mean of f over the node simplex,
                    // so |y| <= max|f| = 1. No oracle is run here (these node
                    // patterns are the ill-conditioned ones by construction),
                    // but the overshoot must still fit the error model - that
                    // is the accuracy statement this section can make.
                    const double slack = std::fmax(kSlack * errBound<NL>(x0b, x1, x2, y), 1e-12);
                    if (std::fabs(y) > 1.0 + slack)
                        FAIL("%s |y|>1 by more than the model at (%.17g, %.17g, %.17g) -> %.17g (allowed %.3e)",
                             what, x0b, x1, x2, y, 1.0 + slack);
                }
            }

    // The all-zero case, run as a stream rather than a triple.
    ADAA2<NL> s;
    s.reset();
    for (int i = 0; i < 16; ++i)
        CHECK(s.process(0.0) == 0.0);

    std::printf("  %-22s %d degenerate triples finite and |y| <= 1: PASS\n", what, n);
}

// ── 6. Reset determinism ──────────────────────────────────────────────────

template <typename NL>
void checkReset(const char* what)
{
    std::vector<double> in(100), first(100), second(100);
    unsigned seed = 12345u;
    for (int i = 0; i < 100; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        in[static_cast<size_t>(i)] = 40.0 * (static_cast<double>(seed >> 8) / 8388608.0 - 1.0);
    }

    ADAA2<NL> s;
    s.reset();
    for (int i = 0; i < 100; ++i) first[static_cast<size_t>(i)] = s.process(in[static_cast<size_t>(i)]);
    s.reset();
    for (int i = 0; i < 100; ++i) second[static_cast<size_t>(i)] = s.process(in[static_cast<size_t>(i)]);

    for (int i = 0; i < 100; ++i)
        if (first[static_cast<size_t>(i)] != second[static_cast<size_t>(i)])
            FAIL("%s reset not clean at n=%d: %.17g vs %.17g", what, i,
                 first[static_cast<size_t>(i)], second[static_cast<size_t>(i)]);

    std::printf("  %-22s reset() bit-exact over 100 samples: PASS\n", what);
}

// ── 7. Parity ─────────────────────────────────────────────────────────────

template <typename NL>
void checkParity(const char* what)
{
    ADAA2<NL> sp, sn;
    sp.reset();
    sn.reset();
    unsigned seed = 777u;
    for (int i = 0; i < 5000; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const double x = 40.0 * (static_cast<double>(seed >> 8) / 8388608.0 - 1.0);
        const double yp = sp.process(x);
        const double yn = sn.process(-x);
        if (yn != -yp)
            FAIL("%s parity broken at n=%d x=%.17g: y(+)=%.17g y(-)=%.17g", what, i, x, yp, yn);
    }
    std::printf("  %-22s odd symmetry bit-exact over 5000 samples: PASS\n", what);
}

// Sanity check on the oracle itself: for f(x) = x the second divided
// difference of F2 = x^3/6 is exactly (x0+x1+x2)/6, so y must be the node
// centroid. Both quadrature forms are exercised (tight and wide spans).
struct LinearNL
{
    static double f(double x) noexcept { return x; }
    static double F1(double x) noexcept { return 0.5 * x * x; }
    static double F2(double x) noexcept { return x * x * x / 6.0; }
};

void checkOracle()
{
    const double triples[][3] = {
        { 0.0, 0.0, 0.0 }, { 0.1, 0.2, 0.3 }, { -0.05, 0.02, 0.01 },
        { 3.0, -4.0, 7.0 }, { -20.0, 0.0, 20.0 }, { 5.0, 5.0, -9.0 },
        { 1.0, 1.0, 1.0 + 1e-9 }, { 40.0, -40.0, 0.5 },
    };
    double worst = 0.0;
    for (const auto& t : triples)
    {
        const double got = oracleADAA2<LinearNL>(t[0], t[1], t[2]);
        const double want = (t[0] + t[1] + t[2]) / 3.0;
        const double e = std::fabs(got - want) / std::fmax(std::fabs(want), 1.0);
        worst = std::fmax(worst, e);
        if (e > 1e-14)
            FAIL("oracle wrong on f(x)=x at (%g,%g,%g): got %.17g want %.17g", t[0], t[1], t[2], got, want);
    }
    // Gauss-Legendre self-check: weights sum to 2 on [-1,1].
    double ws = 0.0;
    for (int k = 0; k < GaussRule::N; ++k) ws += g_gl.w[k];
    CHECK(std::fabs(ws - 2.0) < 1e-14);
    std::printf("  GL-%d weights sum to 2 (err %.2e); oracle reproduces the centroid for f(x)=x (max rel err %.2e)\n",
                GaussRule::N, std::fabs(ws - 2.0), worst);
}

template <typename NL>
double runPolicy(const char* label)
{
    std::printf("\n[ADAA2<%s>]\n", label);
    g_section = label;

    std::printf(" 2. branch (b), confluent outer nodes (x0 = x2 = a, x1 = b):\n");
    checkConfluent<NL>("confluent");

    std::printf(" 3. branch-seam continuity (1000 steps across each eps):\n");
    checkSeam<NL>("inner seam @ x=0.7", 0.7, true);
    checkSeam<NL>("inner seam @ x=12", 12.0, true);
    checkSeam<NL>("outer seam @ x=0.7", 0.7, false);
    checkSeam<NL>("outer seam @ x=12", 12.0, false);

    std::printf(" 4. error surface, x_k = A sin(w(n-k)):\n");
    const double surf = checkErrorSurface<NL>("surface");

    std::printf(" 5-7. degenerate / reset / parity:\n");
    checkDegenerate<NL>("degenerate");
    checkReset<NL>("reset");
    checkParity<NL>("parity");
    return surf;
}

} // namespace

int main()
{
    std::printf("=== Chronos ADAA2 correctness harness ===\n");
    std::printf("kEpsInner = %.0e  kEpsOuter = %.0e  slack = %.0fx model\n", kEpsInner, kEpsOuter, kSlack);

    std::printf("\n[oracle self-check]\n");
    g_section = "oracle";
    checkOracle();

    std::printf("\n[1. static curve]\n");
    g_section = "static curve";
    checkStaticCurve<TanhNL, ADAA2<TanhNL>>("ADAA2<TanhNL>");
    checkStaticCurve<TanhNL, ADAA1<TanhNL>>("ADAA1<TanhNL>");
    checkStaticCurve<AlgebraicNL, ADAA2<AlgebraicNL>>("ADAA2<AlgebraicNL>");
    checkStaticCurve<AlgebraicNL, ADAA1<AlgebraicNL>>("ADAA1<AlgebraicNL>");

    const double sTanh = runPolicy<TanhNL>("TanhNL");
    const double sAlg = runPolicy<AlgebraicNL>("AlgebraicNL");

    std::printf("\nerror-surface maxima: TanhNL %.3e, AlgebraicNL %.3e\n", sTanh, sAlg);
    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

/**
 * Correctness harness for ADAA2 (and, for the static curve, ADAA1).
 * ADAA2 output is twice the second divided difference of F2 over the
 * last three input samples. Plain main(), exit code, always-live CHECK/FAIL.
 */

#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <print>
#include <vector>

namespace
{
    using MarsDSP::Nonlinear::ADAA1;
    using MarsDSP::Nonlinear::ADAA2;
    using MarsDSP::Nonlinear::AlgebraicNL;
    using MarsDSP::Nonlinear::TanhNL;

    const char *g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kU = 2.220446049250313e-16; // DBL_EPSILON

    // Mirrors of ADAA2<NL>::kEpsInner / kEpsOuter, which are private.
    constexpr double kEpsInner = 1e-4;
    constexpr double kEpsOuter = 1e-6;

    // Slack on the error model.
    constexpr double kSlack = 40.0;

    // Absolute gate on the section-4 error surface.
    constexpr double kSurfaceGate = 1e-3;

    /// Gauss-Legendre rule. Nodes come from a Newton solve on the Legendre recurrence.
    struct GaussRule
    {
        static constexpr int N = 16;
        std::array<double, N> x{};
        std::array<double, N> w{};

        GaussRule() noexcept
        {
            for (int i = 0; i < (N + 1) / 2; ++i)
            {
                double z = std::cos(kPi * (static_cast<double>(i) + 0.75) / (N + 0.5));
                double pp = 0.0;
                for (int it = 0; it < 100; ++it)
                {
                    double p1 = 1.0;
                    double p2 = 0.0;
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
                x[static_cast<std::size_t>(i)] = -z;
                x[static_cast<std::size_t>(N - 1 - i)] = z;
                w[static_cast<std::size_t>(i)] = 2.0 / ((1.0 - z * z) * pp * pp);
                w[static_cast<std::size_t>(N - 1 - i)] = w[static_cast<std::size_t>(i)];
            }
        }
    };

    const GaussRule g_gl{};

    /// Composite Gauss-Legendre-16 of g over [0, 1].
    template<typename G>
    double integrate01(G &&g, int panels) noexcept
    {
        const double h = 1.0 / static_cast<double>(panels);
        double sum = 0.0;
        for (int p = 0; p < panels; ++p)
        {
            const double c = (static_cast<double>(p) + 0.5) * h;
            const double r = 0.5 * h;
            double s = 0.0;
            for (int k = 0; k < GaussRule::N; ++k)
                s += g_gl.w[static_cast<std::size_t>(k)] * g(c + r * g_gl.x[static_cast<std::size_t>(k)]);
            sum += r * s;
        }
        return sum;
    }

    /// Hermite-Genocchi oracle: y = 2*F2[x0,x1,x2] = 2*integral over simplex f.
    template<typename NL>
    double oracleADAA2(double n0, double n1, double n2) noexcept
    {
        const double lo = std::min({n0, n1, n2});
        const double hi = std::max({n0, n1, n2});
        const double span = hi - lo;

        if (span < 0.5)
        {
            const double q = n1 - n0;
            const double c = n2 - n1;
            return 2.0 * integrate01([&](double t)
            {
                return t * integrate01([&](double s) { return NL::f(n0 + q * t + c * t * s); }, 1);
            }, 1);
        }

        // Relabel to put the widest-separated pair into the denominator.
        const double d01 = std::fabs(n0 - n1);
        const double d02 = std::fabs(n0 - n2);
        const double d12 = std::fabs(n1 - n2);
        double x0 = n0;
        double x1 = n1;
        double x2 = n2;
        if (d12 >= d01 && d12 >= d02)
        {
            x0 = n0;
            x1 = n1;
            x2 = n2;
        } else if (d02 >= d01)
        {
            x0 = n1;
            x1 = n0;
            x2 = n2;
        } else
        {
            x0 = n2;
            x1 = n0;
            x2 = n1;
        }

        const int panels = std::clamp(static_cast<int>(std::ceil(span)), 1, 128);
        const double I = integrate01([&](double t)
        {
            return NL::F1(x0 + (x2 - x0) * t) - NL::F1(x0 + (x1 - x0) * t);
        }, panels);
        return 2.0 * I / (x2 - x1);
    }

    // The absolute rounding error of an F1/F2 evaluation is set by the largest
    // intermediate, not by the magnitude of the result.
    template<typename NL>
    double scaleF2(double x) noexcept { return std::fabs(NL::F2(x)) + MarsDSP::Nonlinear::kLn2 * std::fabs(x) + 1.0; }

    template<typename NL>
    double scaleF1(double x) noexcept { return std::fabs(NL::F1(x)) + std::fabs(x) + 1.0; }

    /// Per-point error model, branch by branch. Leading terms only.
    template<typename NL>
    double errBound(double x0, double x1, double x2, double y) noexcept
    {
        const double A = std::fabs(x0 - x1);
        const double B = std::fabs(x1 - x2);
        const double C = std::fabs(x0 - x2);
        const double span = std::max({x0, x1, x2}) - std::min({x0, x1, x2});

        const double s1 = std::fmax(scaleF1<NL>(x0), std::fmax(scaleF1<NL>(x1), scaleF1<NL>(x2)));
        const double s2 = std::fmax(scaleF2<NL>(x0), std::fmax(scaleF2<NL>(x1), scaleF2<NL>(x2)));

        const double oracleFloor = (span < 0.5) ? 1e-15 : 8.0 * kU * s1 / span;
        const double round = 8.0 * kU * (std::fabs(y) + 1.0) + oracleFloor;

        if (A < kEpsInner && B < kEpsInner)
            return 0.05 * (A * A + B * B + C * C) + round;

        const double d1e = (A < kEpsInner) ? A * A / 24.0 : 2.0 * kU * s2 / A;

        if (C < kEpsOuter)
        {
            // Branch (b) reuses d1 = F2[x0,x1] and truncates at O(C), not O(C^2).
            const double m = std::fmax(std::fabs(0.5 * (x0 + x2) - x1), 0.5 * kEpsInner);
            return std::fabs(y) * C / (2.0 * m) + 2.0 * (d1e + 2.0 * kU * s1) / m + round;
        }

        const double d2e = (B < kEpsInner) ? B * B / 24.0 : 2.0 * kU * s2 / B;
        return 2.0 * (d1e + d2e) / C + round;
    }

    /// Single-shot evaluation of the kernel at an explicit node triple.
    template<typename NL>
    double evalTriple(double x0, double x1, double x2) noexcept
    {
        ADAA2<NL> s;
        s.reset();
        (void) s.process(x2);
        (void) s.process(x1);
        return s.process(x0);
    }

    /// Running max of err/bound, carrying the argmax for the report.
    struct Worst
    {
        double ratio = 0.0;
        double err = 0.0;
        double bound = 0.0;
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;

        void feed(double e, double bnd, double n0, double n1, double n2) noexcept
        {
            const double r = e / bnd;
            if (r > ratio)
            {
                ratio = r;
                err = e;
                bound = bnd;
                a = n0;
                b = n1;
                c = n2;
            }
        }
    };

    // 1. Static curve.
    template<typename NL, typename Stage>
    void checkStaticCurve(const char *what)
    {
        constexpr double kTol = 1e-13;
        double worst = 0.0;
        double worstX = 0.0;
        for (int i = 0; i <= 800; ++i)
        {
            const double x = -40.0 + 80.0 * static_cast<double>(i) / 800.0;
            Stage s;
            s.reset();
            double y = 0.0;
            for (int k = 0; k < 8; ++k)
                y = s.process(x);
            const double e = std::fabs(y - NL::f(x));
            if (e > worst)
            {
                worst = e;
                worstX = x;
            }
            if (e > kTol)
                FAIL("{{}} static curve at x={{:.6f}}: y={{:.17g}} f={{:.17g}} |err|={{:.3e}}", what, x, y, NL::f(x), e);
        }
        std::println("  {:<22} max |y - f(x)| = {:.3e} at x={:+.3f} (tol {:.0e})", what, worst, worstX, kTol);
    }

    // 2. Branch (b): confluent outer nodes.
    template<typename NL>
    void checkConfluent(const char *what)
    {
        Worst w;
        double maxAbs = 0.0;
        double maxAbsA = 0.0;
        double maxAbsD = 0.0;
        int nBranchB = 0;
        int nBranchA = 0;

        for (int ia = 0; ia <= 20; ++ia)
        {
            const double a = -20.0 + 2.0 * static_cast<double>(ia);
            for (int id = 0; id < 25; ++id)
            {
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
                        FAIL("{{}} non-finite at a={{:.6f}} b={{:.6f}}", what, a, b);
                    if (e > maxAbs)
                    {
                        maxAbs = e;
                        maxAbsA = a;
                        maxAbsD = b - a;
                    }
                    w.feed(e, bnd, a, b, a);
                    if (e > kSlack * bnd)
                        FAIL("{{}} a={{:+.4f}} delta={{:+.3e}}: y={{:.17g}} ref={{:.17g}} |err|={{:.3e}} bound={{:.3e}} ratio={{:.1f}}",
                         what, a, b - a, y, ref, e, bnd, e / bnd);
                }
            }
        }
        std::println("  {:<22} max |err| = {:.3e} at a={:+.1f} delta={:+.3e}", what, maxAbs, maxAbsA, maxAbsD);
        std::println("  {:<22} max err/bound = {:.2f}  (err {:.3e}, bound {:.3e}) at a={:+.4f} b={:+.4f}",
                     "", w.ratio, w.err, w.bound, w.a, w.b);
        std::println("  {:<22} {} pts took branch (b), {} fell through to (a)", "", nBranchB, nBranchA);
    }

    // 3. Branch-seam continuity.
    template<typename NL>
    void checkSeam(const char *what, double centre, bool inner)
    {
        constexpr int N = 1000;
        const double eps = inner ? kEpsInner : kEpsOuter;
        Worst w;
        double maxJump = 0.0;
        double maxJumpAt = 0.0;
        double prevY = 0.0;
        double prevRef = 0.0;
        double prevBnd = 0.0;

        for (int i = 0; i <= N; ++i)
        {
            const double d = eps * (0.5 + 1.0 * static_cast<double>(i) / static_cast<double>(N));

            const double x0 = inner ? centre + d : centre;
            const double x1 = inner ? centre : centre - 1.0;
            const double x2 = inner ? centre - 1.0 : centre + d;

            const double y = evalTriple<NL>(x0, x1, x2);
            const double ref = oracleADAA2<NL>(x0, x1, x2);
            const double bnd = errBound<NL>(x0, x1, x2, y);
            const double e = std::fabs(y - ref);
            w.feed(e, bnd, x0, x1, x2);
            if (e > kSlack * bnd)
                FAIL("{{}} delta={{:.6e}}: |err|={{:.3e}} bound={{:.3e}} ratio={{:.1f}}",
                 what, d, e, bnd, e / bnd);

            if (i > 0)
            {
                const double step = std::fabs(y - prevY);
                const double allow = std::fabs(ref - prevRef) + 10.0 * (bnd + prevBnd);
                if (step > allow)
                    FAIL("{{}} seam jump at delta={{:.6e}}: |dy|={{:.3e}} allowed {{:.3e}} (oracle step {{:.3e}})",
                     what, d, step, allow, std::fabs(ref - prevRef));
                const double excess = step - std::fabs(ref - prevRef);
                if (excess > maxJump)
                {
                    maxJump = excess;
                    maxJumpAt = d;
                }
            }
            prevY = y;
            prevRef = ref;
            prevBnd = bnd;
        }
        std::println("  {:<22} max err/bound = {:5.2f} | worst excess step = {:.3e} at delta={:.4e}",
                     what, w.ratio, maxJump, maxJumpAt);
    }

    // 4. Error surface over the physical node family.
    template<typename NL>
    double checkErrorSurface(const char *what)
    {
        constexpr int kA = 10;
        constexpr int kW = 20;
        constexpr int kN = 64;

        Worst w;
        double maxAbs = 0.0;
        double maxAbsAmp = 0.0;
        double maxAbsOmega = 0.0;

        for (int ia = 0; ia < kA; ++ia)
        {
            const double amp = std::pow(10.0, -1.0 + 2.60206 * static_cast<double>(ia) / (kA - 1));
            for (int iw = 0; iw < kW; ++iw)
            {
                const double omega = std::pow(10.0, -4.0 + 4.49715 * static_cast<double>(iw) / (kW - 1));

                ADAA2<NL> s;
                s.reset();
                double xm1 = 0.0;
                double xm2 = 0.0;
                for (int n = 0; n < kN; ++n)
                {
                    const double x = amp * std::sin(omega * static_cast<double>(n));
                    const double y = s.process(x);
                    if (!std::isfinite(y))
                        FAIL("{{}} non-finite at A={{:.4f}} w={{:.6f}} n={{}}", what, amp, omega, n);
                    if (n >= 2)
                    {
                        const double ref = oracleADAA2<NL>(x, xm1, xm2);
                        const double e = std::fabs(y - ref);
                        const double bnd = errBound<NL>(x, xm1, xm2, y);
                        if (e > maxAbs)
                        {
                            maxAbs = e;
                            maxAbsAmp = amp;
                            maxAbsOmega = omega;
                        }
                        w.feed(e, bnd, amp, omega, static_cast<double>(n));
                    }
                    xm2 = xm1;
                    xm1 = x;
                }
            }
        }
        std::println("  {:<22} max |err| = {:.3e} at A={:.3f} w={:.5f} ({:.1f} Hz @ 48k)  [gate {:.0e}]",
                     what, maxAbs, maxAbsAmp, maxAbsOmega, maxAbsOmega * 48000.0 / (2.0 * kPi), kSurfaceGate);
        std::println("  {:<22} max err/bound = {:5.2f} (err {:.3e}, bound {:.3e}) at A={:.3f} w={:.5f} n={}",
                     "", w.ratio, w.err, w.bound, w.a, w.b, static_cast<int>(w.c));
        if (w.ratio > kSlack)
            FAIL("{{}} error surface exceeds model: ratio {{:.1f}} > {{:.0f}}", what, w.ratio, kSlack);
        if (maxAbs > kSurfaceGate)
            FAIL("{{}} error surface floor regressed: {{:.3e}} > {{:.0e}}", what, maxAbs, kSurfaceGate);
        return maxAbs;
    }

    // 5. Degenerate node patterns.
    template<typename NL>
    void checkDegenerate(const char *what)
    {
        const std::array<double, 11> vals = {
            0.0, 1e-18, 1e-9, kEpsInner, 0.5 * kEpsInner, 0.7, 3.0, 40.0, -40.0, 700.0, -700.0
        };
        int n = 0;

        // All 2^3 coincidence patterns.
        for (int i = 0; i < 11; ++i)
            for (int j = 0; j < 11; ++j)
                for (int k = 0; k < 11; ++k)
                {
                    const double x0 = vals[static_cast<std::size_t>(i)];
                    for (int p = 0; p < 8; ++p)
                    {
                        const double x1 = (p & 1) ? x0 : x0 + vals[static_cast<std::size_t>(j)];
                        const double x2 = (p & 2) ? x0 : x0 + vals[static_cast<std::size_t>(k)];
                        const double x0b = (p & 4) ? x1 : x0;
                        const double y = evalTriple<NL>(x0b, x1, x2);
                        ++n;
                        if (!std::isfinite(y))
                            FAIL("{{}} non-finite at ({{:.17g}}, {{:.17g}}, {{:.17g}}) -> {{:.17g}}", what, x0b, x1, x2, y);
                        // The exact output is a mean of f over the node simplex, so |y| <= max|f| = 1.
                        const double slack = std::fmax(kSlack * errBound<NL>(x0b, x1, x2, y), 1e-12);
                        if (std::fabs(y) > 1.0 + slack)
                            FAIL("{{}} |y|>1 by more than the model at ({{:.17g}}, {{:.17g}}, {{:.17g}}) -> {{:.17g}} (allowed {{:.3e}})",
                            what, x0b, x1, x2, y, 1.0 + slack);
                    }
                }

        // The all-zero case, run as a stream rather than a triple.
        ADAA2<NL> s;
        s.reset();
        for (int i = 0; i < 16; ++i)
            CHECK(s.process(0.0) == 0.0);

        std::println("  {:<22} {} degenerate triples finite and |y| <= 1: PASS", what, n);
    }

    // 6. Reset determinism.
    template<typename NL>
    void checkReset(const char *what)
    {
        std::vector<double> in(100);
        std::vector<double> first(100);
        std::vector<double> second(100);
        unsigned seed = 12345u;
        for (int i = 0; i < 100; ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            in[static_cast<std::size_t>(i)] = 40.0 * (static_cast<double>(seed >> 8) / 8388608.0 - 1.0);
        }

        ADAA2<NL> s;
        s.reset();
        for (int i = 0; i < 100; ++i) first[static_cast<std::size_t>(i)] = s.process(in[static_cast<std::size_t>(i)]);
        s.reset();
        for (int i = 0; i < 100; ++i) second[static_cast<std::size_t>(i)] = s.process(in[static_cast<std::size_t>(i)]);

        for (int i = 0; i < 100; ++i)
            if (first[static_cast<std::size_t>(i)] != second[static_cast<std::size_t>(i)])
                FAIL("{{}} reset not clean at n={{}}: {{:.17g}} vs {{:.17g}}", what, i,
                 first[static_cast<std::size_t>(i)], second[static_cast<std::size_t>(i)]);

        std::println("  {:<22} reset() bit-exact over 100 samples: PASS", what);
    }

    // 7. Parity.
    template<typename NL>
    void checkParity(const char *what)
    {
        ADAA2<NL> sp;
        ADAA2<NL> sn;
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
                FAIL("{{}} parity broken at n={{}} x={{:.17g}}: y(+)={{:.17g}} y(-)={{:.17g}}", what, i, x, yp, yn);
        }
        std::println("  {:<22} odd symmetry bit-exact over 5000 samples: PASS", what);
    }

    // 8. a0-seam triples for TanhNL.
    void checkA0Seam()
    {
        constexpr double a0 = 1.0;
        constexpr int kN = 22;

        Worst w;
        double maxAbs = 0.0;
        std::array<int, 4> nBranches = {0, 0, 0, 0};
        int nTotal = 0;

        for (int i = 0; i < kN; ++i)
        {
            const double d1 = std::pow(10.0, -8.0 + 8.0 * static_cast<double>(i) / (kN - 1));
            for (int j = 0; j < kN; ++j)
            {
                const double d2 = std::pow(10.0, -8.0 + 8.0 * static_cast<double>(j) / (kN - 1));

                const std::array<std::array<double, 3>, 4> t = {
                    {
                        {a0 + d1, a0 - d2, a0 + d2},
                        {a0 + d1, a0 - d2, a0 + d1},
                        {a0 - d1, a0 + d2, a0 - d2},
                        {a0 - d1, a0 - d1, a0 + d2},
                    }
                };

                for (const auto &tr: t)
                {
                    const double x0 = tr[0];
                    const double x1 = tr[1];
                    const double x2 = tr[2];
                    const double y = evalTriple<TanhNL>(x0, x1, x2);

                    if (!std::isfinite(y))
                        FAIL("a0-seam non-finite at ({{:.6f}}, {{:.6f}}, {{:.6f}})", x0, x1, x2);

                    const double ref = oracleADAA2<TanhNL>(x0, x1, x2);
                    const double e = std::fabs(y - ref);
                    const double bnd = errBound<TanhNL>(x0, x1, x2, y);
                    w.feed(e, bnd, x0, x1, x2);

                    const double A = std::fabs(x0 - x1);
                    const double B = std::fabs(x1 - x2);
                    const double C = std::fabs(x0 - x2);
                    if (A < kEpsInner && B < kEpsInner) ++nBranches[0];
                    else if (C < kEpsOuter) ++nBranches[1];
                    else if (A < kEpsInner) ++nBranches[2];
                    else ++nBranches[3];

                    if (e > maxAbs) maxAbs = e;
                    ++nTotal;

                    if (e > kSlack * bnd)
                        FAIL("a0-seam ({{:.6f}}, {{:.6f}}, {{:.6f}}): err={{:.3e}} bound={{:.3e}} ratio={{:.1f}}",
                         x0, x1, x2, e, bnd, e / bnd);
                }
            }
        }
        std::println("  a0-seam ({} triples):  max |err| = {:.3e}  max err/bound = {:.3f}",
                     nTotal, maxAbs, w.ratio);
        std::println("  branches: (a) {}  (b) {}  (c) {}  (d) {}",
                     nBranches[0], nBranches[1], nBranches[2], nBranches[3]);
        if (maxAbs > kSurfaceGate)
            FAIL("a0-seam surface exceeded: {{:.3e}} > {{:.0e}}", maxAbs, kSurfaceGate);
        std::println("  a0-seam: PASS (gate kSurfaceGate = {:.0e})", kSurfaceGate);
    }

    // Sanity check on the oracle: for f(x) = x the second divided difference of
    // F2 = x^3/6 is exactly (x0+x1+x2)/6, so y must be the node centroid.
    struct LinearNL
    {
        static double f(double x) noexcept { return x; }
        static double F1(double x) noexcept { return 0.5 * x * x; }
        static double F2(double x) noexcept { return x * x * x / 6.0; }
    };

    void checkOracle()
    {
        const std::array<std::array<double, 3>, 8> triples = {
            {
                {0.0, 0.0, 0.0}, {0.1, 0.2, 0.3}, {-0.05, 0.02, 0.01},
                {3.0, -4.0, 7.0}, {-20.0, 0.0, 20.0}, {5.0, 5.0, -9.0},
                {1.0, 1.0, 1.0 + 1e-9}, {40.0, -40.0, 0.5},
            }
        };
        double worst = 0.0;
        for (const auto &t: triples)
        {
            const double got = oracleADAA2<LinearNL>(t[0], t[1], t[2]);
            const double want = (t[0] + t[1] + t[2]) / 3.0;
            const double e = std::fabs(got - want) / std::fmax(std::fabs(want), 1.0);
            worst = std::fmax(worst, e);
            if (e > 1e-14)
                FAIL("oracle wrong on f(x)=x at ({{{{}}}},{{{{}}}},{{{{}}}}): got {{:.17g}} want {{:.17g}}", t[0], t[1], t[2], got, want);
        }
        double ws = 0.0;
        for (int k = 0; k < GaussRule::N; ++k) ws += g_gl.w[static_cast<std::size_t>(k)];
        CHECK(std::fabs(ws - 2.0) < 1e-14);
        std::println("  GL-{} weights sum to 2 (err {:.2e}); oracle reproduces the centroid for f(x)=x (max rel err {:.2e})",
            GaussRule::N, std::fabs(ws - 2.0), worst);
    }

    template<typename NL>
    double runPolicy(const char *label)
    {
        std::println();
        std::println("[ADAA2<{}>]", label);
        g_section = label;

        std::println(" 2. branch (b), confluent outer nodes (x0 = x2 = a, x1 = b):");
        checkConfluent<NL>("confluent");

        std::println(" 3. branch-seam continuity (1000 steps across each eps):");
        checkSeam<NL>("inner seam @ x=0.7", 0.7, true);
        checkSeam<NL>("inner seam @ x=12", 12.0, true);
        checkSeam<NL>("outer seam @ x=0.7", 0.7, false);
        checkSeam<NL>("outer seam @ x=12", 12.0, false);

        std::println(" 4. error surface, x_k = A sin(w(n-k)):");
        const double surf = checkErrorSurface<NL>("surface");

        std::println(" 5-7. degenerate / reset / parity:");
        checkDegenerate<NL>("degenerate");
        checkReset<NL>("reset");
        checkParity<NL>("parity");
        return surf;
    }
} // namespace

int main()
{
    std::println("=== Chronos ADAA2 correctness harness ===");
    std::println("kEpsInner = {:.0e}  kEpsOuter = {:.0e}  slack = {:.0f}x model", kEpsInner, kEpsOuter, kSlack);

    std::println();
    std::println("[oracle self-check]");
    g_section = "oracle";
    checkOracle();

    std::println();
    std::println("[1. static curve]");
    g_section = "static curve";
    checkStaticCurve<TanhNL, ADAA2<TanhNL> >("ADAA2<TanhNL>");
    checkStaticCurve<TanhNL, ADAA1<TanhNL> >("ADAA1<TanhNL>");
    checkStaticCurve<AlgebraicNL, ADAA2<AlgebraicNL> >("ADAA2<AlgebraicNL>");
    checkStaticCurve<AlgebraicNL, ADAA1<AlgebraicNL> >("ADAA1<AlgebraicNL>");

    const double sTanh = runPolicy<TanhNL>("TanhNL");
    const double sAlg = runPolicy<AlgebraicNL>("AlgebraicNL");

    std::println();
    std::println("[8. a0-seam triples straddling TanhNL's region boundary]");
    g_section = "a0-seam";
    checkA0Seam();

    std::println();
    std::println("error-surface maxima: TanhNL {:.3e}, AlgebraicNL {:.3e}", sTanh, sAlg);
    std::println();
    std::println("=== ALL PROPERTIES HELD ===");
    return 0;
}

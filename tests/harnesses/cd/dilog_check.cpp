// tests/harnesses/cd/dilog_check.cpp
// ──────────────────────────────────────────────────────────────────────────
// Correctness harness for MarsDSP::Math::dilogNeg / dilogSeries, the
// Landen-folded Li2(-t) used by TanhNL::F2 (the second antiderivative of tanh
// has no elementary form and needs the dilogarithm).
//
//   1. Zero          – dilogNeg(0) == 0 exactly.
//   2. Li2(-1)       – dilogNeg(1) == -pi^2/12 to 1e-15.
//   3. Landen seam   – sweep t in [0.4, 0.6]; the direct-series branch and the
//                      Landen branch agree to 1e-15 across the t = 0.5 seam
//                      (both computed explicitly; the dispatch is checked too).
//   4. Known values  – Li2(-1/2), Li2(-0.1) to 1e-14.
//   5. Independent   – dilogNeg vs a Simpson-quadrature oracle of
//                      -integral_0^t ln(1+u)/u du (a genuinely different method
//                      from the series). Substitutes for the inversion FE
//                      Li2(-t)+Li2(-1/t) = -pi^2/6 - 1/2 ln^2 t, which the spec
//                      names but which is out of domain for the given t (none
//                      of {0.2,0.5,0.9} has both -t and -1/t in [0,1]); long
//                      double is 64-bit on arm64 so a higher-precision series
//                      would not help here. The domain of dilogNeg is not
//                      widened to make any check pass.
//   6. Monotonicity  – dilogNeg strictly decreasing on (0, 1], 1e5 points.
//
// Conventions (matching ring_buffer_check): plain main(), exit code, printf,
// always-live CHECK/FAIL (NOT assert — NDEBUG in Release would void every
// test). Links SharedCode only; no JUCE. No forced -O2 so the header's assert
// preconditions stay armed in a Debug configure.
// ──────────────────────────────────────────────────────────────────────────

#include "dsp/nonlinear/Dilogarithm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kPi         = std::numbers::pi_v<double>;
constexpr double kPiSqOver12 = (kPi * kPi) / 12.0;          // 0.8224670334241132
constexpr double kLi2NegHalf = -0.4484142069236462;         // Li2(-1/2)
// Li2(-0.1) = sum_{k>=1} (-0.1)^k / k^2 = -0.0976052352293216 (hand-summed:
// -0.1 + 0.0025 - 1.111e-4 + 6.25e-6 - 4e-7 + ...). A previous implementation
// listed -0.09760605976896, which is off by 8.2e-7 — that value is Li2(-0.10000086),
// typo in the trailing digits? The correct value is used here and is independently
// cross-confirmed by the Simpson oracle below (t=0.1 is in the single-point set)
constexpr double kLi2NegPt1  = -0.0976052352293216;         // Li2(-0.1)

// Simpson-quadrature oracle: Li2(-t) = -integral_0^t ln(1+u)/u du.
// The integrand has a removable singularity at 0 (limit = -1). nPanels must
// be even. Independent of the series: a different summation method entirely.
double li2NegSimpson(double t, long long nPanels)
{
    if (t <= 0.0) return 0.0;
    const double h = t / static_cast<double>(nPanels);
    auto f = [](double u) noexcept -> double {
        if (u <= 0.0) return -1.0;                 // limit of -ln(1+u)/u as u->0
        return -std::log1p(u) / u;
    };
    double s = f(0.0) + f(t);                       // f0 + fN
    for (long long k = 1; k < nPanels; k += 2)
        s += 4.0 * f(static_cast<double>(k) * h);   // odd nodes
    for (long long k = 2; k < nPanels; k += 2)
        s += 2.0 * f(static_cast<double>(k) * h);   // even nodes
    return s * (h / 3.0);
}

int runAll()
{
    using MarsDSP::Math::dilogNeg;
    using MarsDSP::Math::dilogSeries;

    // ── 1. Zero ────────────────────────────────────────────────────────────
    g_section = "zero";
    {
        CHECK(dilogNeg(0.0) == 0.0);
        std::printf("dilogNeg(0) == 0: PASS\n");
    }

    // ── 2. Li2(-1) == -pi^2/12 ─────────────────────────────────────────────
    g_section = "Li2(-1)";
    {
        const double got = dilogNeg(1.0);
        const double err = std::fabs(got - (-kPiSqOver12));
        std::printf("dilogNeg(1) = %.16f  exp = -%.16f  |err| = %.3e\n",
                    got, kPiSqOver12, err);
        CHECK(err <= 1e-15);
        std::printf("Li2(-1) == -pi^2/12 (1e-15): PASS\n");
    }

    // ── 3. Landen seam across t = 0.5 ──────────────────────────────────────
    g_section = "Landen seam";
    {
        constexpr int N = 10000;
        double maxBranchDiff = 0.0;      // direct vs Landen, t <= 0.5
        double maxDispatch   = 0.0;      // dilogNeg vs Landen form, all t
        double worstT = 0.0;
        for (int i = 0; i <= N; ++i)
        {
            const double t = 0.4 + 0.2 * static_cast<double>(i) / static_cast<double>(N);
            const double u = t / (1.0 + t);
            const double lt = std::log1p(t);
            const double landen = -0.5 * lt * lt - dilogSeries(u);   // valid on [0,1]

            const double disp = dilogNeg(t);
            maxDispatch = std::fmax(maxDispatch, std::fabs(disp - landen));

            if (t <= 0.5)
            {
                const double direct = dilogSeries(-t);              // valid (|−t|<=0.5)
                const double d = std::fabs(direct - landen);
                if (d > maxBranchDiff) { maxBranchDiff = d; worstT = t; }
            }
        }
        std::printf("seam [0.4,0.5] direct-vs-Landen max |diff| = %.3e at t=%.4f\n",
                    maxBranchDiff, worstT);
        std::printf("seam [0.4,0.6] dispatch-vs-Landen max |diff| = %.3e\n", maxDispatch);
        CHECK(maxBranchDiff <= 1e-15);
        CHECK(maxDispatch <= 1e-15);
        std::printf("Landen seam (1e-15): PASS\n");
    }

    // ── 4. Known values ────────────────────────────────────────────────────
    g_section = "known values";
    {
        const double e0 = std::fabs(dilogNeg(0.5) - kLi2NegHalf);
        const double e1 = std::fabs(dilogNeg(0.1) - kLi2NegPt1);
        std::printf("Li2(-0.5): got=%.16f exp=%.16f |err|=%.3e\n",
                    dilogNeg(0.5), kLi2NegHalf, e0);
        std::printf("Li2(-0.1): got=%.16f exp=%.16f |err|=%.3e\n",
                    dilogNeg(0.1), kLi2NegPt1, e1);
        CHECK(e0 <= 1e-14);
        CHECK(e1 <= 1e-14);
        std::printf("known values (1e-14): PASS\n");
    }

    // ── 5. Independent Simpson-quadrature oracle ───────────────────────────
    g_section = "independent oracle";
    {
        const double pts[4] = { 0.1, 0.2, 0.5, 0.9 };
        double maxSingle = 0.0;
        for (double t : pts)
        {
            const double got = dilogNeg(t);
            const double ref = li2NegSimpson(t, 2'000'000);
            const double e = std::fabs(got - ref);
            maxSingle = std::fmax(maxSingle, e);
            std::printf("oracle t=%.2f: dilogNeg=%.16f simpson=%.16f |err|=%.3e\n",
                        t, got, ref, e);
        }
        CHECK(maxSingle <= 1e-11);
        std::printf("independent oracle @ {0.2,0.5,0.9} (1e-11): PASS\n", maxSingle);

        // Dense sweep over (0, 1]; fewer panels per point but still well under
        // the double floor relative to a 1e-9 bar.
        constexpr int M = 1000;
        double maxSweep = 0.0, sweepWorst = 0.0;
        for (int i = 1; i <= M; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(M);
            const double got = dilogNeg(t);
            const double ref = li2NegSimpson(t, 200'000);
            const double e = std::fabs(got - ref);
            if (e > maxSweep) { maxSweep = e; sweepWorst = t; }
        }
        std::printf("oracle sweep (0,1] %d pts max |err| = %.3e at t=%.4f\n",
                    M, maxSweep, sweepWorst);
        CHECK(maxSweep <= 1e-9);
        std::printf("independent oracle sweep (1e-9): PASS\n");
    }

    // ── 6. Monotonicity: strictly decreasing on (0, 1] ─────────────────────
    g_section = "monotonicity";
    {
        constexpr int M = 100'000;
        double prev = dilogNeg(1.0 / static_cast<double>(M + 1)); // smallest t
        for (int i = 2; i <= M; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(M + 1);
            const double v = dilogNeg(t);
            if (!(v < prev))
                FAIL("not strictly decreasing at i=%d t=%.6f prev=%.16f v=%.16f",
                     i, t, prev, v);
            prev = v;
        }
        std::printf("monotonicity, strictly decreasing on (0,1] (%d pts): PASS\n", M);
    }

    return 0;
}

} // namespace

int main()
{
    std::printf("=== Chronos dilogarithm (Li2(-t)) correctness harness ===\n");
    std::printf("pi^2/12 = %.16f\n\n", kPiSqOver12);

    int r = runAll();

    std::printf("\n=== %s ===\n", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

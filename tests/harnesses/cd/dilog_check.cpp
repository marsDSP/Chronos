/**
 * Correctness harness for dilogNeg and dilogSeries, the Landen-folded Li2(-t)
 * used by the tanh antiderivatives. Plain main(), exit code,
 * always-live CHECK/FAIL.
 */

#include "../../../source/math/Dilogarithm.h"

#include <cmath>
#include <cstdlib>
#include <numbers>
#include <print>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

constexpr double kPi         = std::numbers::pi_v<double>;
constexpr double kPiSqOver12 = (kPi * kPi) / 12.0;
constexpr double kLi2NegHalf = -0.4484142069236462;
// Li2(-0.1): hand-summed series. Cross-confirmed by the Simpson oracle below.
constexpr double kLi2NegPt1  = -0.0976052352293216;

/// Simpson-quadrature oracle: Li2(-t) = -integral_0^t ln(1+u)/u du.
/// Independent of the series: a different summation method entirely.
double li2NegSimpson(double t, long long nPanels)
{
    if (t <= 0.0) return 0.0;
    const double h = t / static_cast<double>(nPanels);
    auto f = [](double u) noexcept -> double {
        if (u <= 0.0) return -1.0;
        return -std::log1p(u) / u;
    };
    double s = f(0.0) + f(t);
    for (long long k = 1; k < nPanels; k += 2)
        s += 4.0 * f(static_cast<double>(k) * h);
    for (long long k = 2; k < nPanels; k += 2)
        s += 2.0 * f(static_cast<double>(k) * h);
    return s * (h / 3.0);
}

int runAll()
{
    using MarsDSP::Math::dilogNeg;
    using MarsDSP::Math::dilogSeries;

    // 1. Zero.
    g_section = "zero";
    {
        CHECK(dilogNeg(0.0) == 0.0);
        std::println("dilogNeg(0) == 0: PASS");
    }

    // 2. Li2(-1) == -pi^2/12.
    g_section = "Li2(-1)";
    {
        const double got = dilogNeg(1.0);
        const double err = std::fabs(got - (-kPiSqOver12));
        std::println("dilogNeg(1) = {:.16f}  exp = -{:.16f}  |err| = {:.3e}",
                    got, kPiSqOver12, err);
        CHECK(err <= 1e-15);
        std::println("Li2(-1) == -pi^2/12 (1e-15): PASS");
    }

    // 3. Landen seam across t = 0.5.
    g_section = "Landen seam";
    {
        constexpr int N = 10000;
        double maxBranchDiff = 0.0;
        double maxDispatch   = 0.0;
        double worstT = 0.0;
        for (int i = 0; i <= N; ++i)
        {
            const double t = 0.4 + 0.2 * static_cast<double>(i) / static_cast<double>(N);
            const double u = t / (1.0 + t);
            const double lt = std::log1p(t);
            const double landen = -0.5 * lt * lt - dilogSeries(u);

            const double disp = dilogNeg(t);
            maxDispatch = std::fmax(maxDispatch, std::fabs(disp - landen));

            if (t <= 0.5)
            {
                const double direct = dilogSeries(-t);
                const double d = std::fabs(direct - landen);
                if (d > maxBranchDiff) { maxBranchDiff = d; worstT = t; }
            }
        }
        std::println("seam [0.4,0.5] direct-vs-Landen max |diff| = {:.3e} at t={:.4f}",
                    maxBranchDiff, worstT);
        std::println("seam [0.4,0.6] dispatch-vs-Landen max |diff| = {:.3e}", maxDispatch);
        CHECK(maxBranchDiff <= 1e-15);
        CHECK(maxDispatch <= 1e-15);
        std::println("Landen seam (1e-15): PASS");
    }

    // 4. Known values.
    g_section = "known values";
    {
        const double e0 = std::fabs(dilogNeg(0.5) - kLi2NegHalf);
        const double e1 = std::fabs(dilogNeg(0.1) - kLi2NegPt1);
        std::println("Li2(-0.5): got={:.16f} exp={:.16f} |err|={:.3e}",
                    dilogNeg(0.5), kLi2NegHalf, e0);
        std::println("Li2(-0.1): got={:.16f} exp={:.16f} |err|={:.3e}",
                    dilogNeg(0.1), kLi2NegPt1, e1);
        CHECK(e0 <= 1e-14);
        CHECK(e1 <= 1e-14);
        std::println("known values (1e-14): PASS");
    }

    // 5. Independent Simpson-quadrature oracle.
    g_section = "independent oracle";
    {
        const std::array<double, 4> pts = { 0.1, 0.2, 0.5, 0.9 };
        double maxSingle = 0.0;
        for (double t : pts)
        {
            const double got = dilogNeg(t);
            const double ref = li2NegSimpson(t, 2'000'000);
            const double e = std::fabs(got - ref);
            maxSingle = std::fmax(maxSingle, e);
            std::println("oracle t={:.2f}: dilogNeg={:.16f} simpson={:.16f} |err|={:.3e}",
                        t, got, ref, e);
        }
        CHECK(maxSingle <= 1e-11);
        std::println("independent oracle @ {{0.1,0.2,0.5,0.9}} (1e-11): max |err| = {:.3e} PASS", maxSingle);

        // Dense sweep over (0, 1].
        constexpr int M = 1000;
        double maxSweep = 0.0;
        double sweepWorst = 0.0;
        for (int i = 1; i <= M; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(M);
            const double got = dilogNeg(t);
            const double ref = li2NegSimpson(t, 200'000);
            const double e = std::fabs(got - ref);
            if (e > maxSweep) { maxSweep = e; sweepWorst = t; }
        }
        std::println("oracle sweep (0,1] {} pts max |err| = {:.3e} at t={:.4f}",
                    M, maxSweep, sweepWorst);
        CHECK(maxSweep <= 1e-9);
        std::println("independent oracle sweep (1e-9): PASS");
    }

    // 6. Monotonicity: strictly decreasing on (0, 1].
    g_section = "monotonicity";
    {
        constexpr int M = 100'000;
        double prev = dilogNeg(1.0 / static_cast<double>(M + 1));
        for (int i = 2; i <= M; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(M + 1);
            const double v = dilogNeg(t);
            if (!(v < prev))
                FAIL("not strictly decreasing at i={{}} t={{:.6f}} prev={{:.16f}} v={{:.16f}}",
                     i, t, prev, v);
            prev = v;
        }
        std::println("monotonicity, strictly decreasing on (0,1] ({} pts): PASS", M);
    }

    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos dilogarithm (Li2(-t)) correctness harness ===");
    std::println("pi^2/12 = {:.16f}", kPiSqOver12);
    std::println();

    const int r = runAll();

    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

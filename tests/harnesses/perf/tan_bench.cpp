// tests/harnesses/perf/tan_bench.cpp
// ──────────────────────────────────────────────────────────────────────────
// Microbenchmark + accuracy regression check for the mmTan kernel used by the
// MarsDSP::Filters SVF bilinear pre-warping, against std::tan as the reference.
//
// What it checks
//   1. Accuracy  – mmTanScalar(x) vs std::tan(x) over [-xMax, +xMax], where
//                  xMax = π·(0.49·fs)/fs ≈ 1.539 < 1.55 (the kernel's
//                  precondition, enforced by the 0.49·fs clamp in SVF::setParams).
//                  FAIL if max relative error > 1e-4.  The float32 kernel ceiling
//                  is ~4e-6 near the pole, so 1e-4 catches coefficient corruption
//                  without flaking on float32 rounding noise.
//   2. Tan perf  – ns/tan for std::tan, the mmTanScalar bridge SVF actually
//                  calls, and the raw mmTan(M128) 4-lane kernel.  FAIL (regression)
//                  if mmTanScalar is more than 5% slower than std::tan.
//   3. SVF perf  – SimdSVF block-ramp throughput (setCoeffForBlock +
//                  processBlockStep, M128 stereo in lanes 0,1).  Informational
//                  baseline (~6.3 ns/sample on arm64); not a pass/fail check.
//
// Build:  cmake -S . -B build -DBUILD_TEST_HARNESSES=ON
//         cmake --build build --target tan_bench
// Run:    ./build/tests/tan_bench
// Exit:   0 = no regression, non-zero = accuracy or performance regression.
// ──────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <numbers>
#include <vector>

#include "dsp/StateVariable.h"            // mmTanScalar (anon namespace; identical to SVF's path)
#include "math/Trigonometry.h"  // mmTan(M128), mmTan(float)

namespace {

constexpr double kPi   = std::numbers::pi_v<double>;
constexpr double kFs   = 48000.0;
constexpr double kFmin = 10.0;
constexpr double kFmax = 0.49 * kFs;               // matches SVF::setParams clamp
constexpr double kXmax = kPi * kFmax / kFs;        // ≈ 1.539 < 1.55 kernel precondition
constexpr double kRelErrLimit    = 1e-4;           // accuracy regression threshold
constexpr double kPerfRegression = 1.05;           // mmTanScalar may be up to 5% slower

using Clock = std::chrono::steady_clock;

// Keep the scalar loops scalar so they measure true per-call throughput of one
// std::tan / one mmTanScalar, instead of being auto-vectorized across iterations
// (which would conflate scalar and 4-wide SIMD throughput).  The M128 loop below
// is intentionally 4-wide and is left vectorizable.
#ifdef __clang__
#define CHRONOS_NO_VECTORIZE _Pragma("clang loop vectorize(disable)")
#else
#define CHRONOS_NO_VECTORIZE
#endif

// Compiler barriers (the Google Benchmark DoNotOptimize technique).  At -O2
// clang will otherwise hoist the deterministic accumulations above Clock::now()
// (the loop has no observable side effect, so it computes the sum whenever it
// likes - including before the timer) or dead-code-eliminate them, yielding
// 0 ns/tan.  doNotOptimize(v) emits `asm volatile("" : : "r,m"(v) : "memory")`,
// which forces v to be materialized each iteration and blocks reordering across
// the timer.  Applied identically to both scalar loops so the comparison stays
// fair; the absolute ns/tan is slightly inflated by the barrier but the ratio
// (which is what the regression check uses) is unaffected.
#if defined(__clang__) || defined(__GNUC__)
template <class T>
inline void doNotOptimize(T const& v) noexcept
{
    asm volatile("" : : "r,m"(v) : "memory");
}
#else
template <class T>
inline void doNotOptimize(T const& v) noexcept
{
    volatile T sink = v; (void)sink;   // MSVC fallback (weaker; no x64 inline asm)
}
#endif

// Precomputed argument tables so the per-iteration input varies and the
// compiler cannot hoist the tan out of the timed loop.  Values span the actual
// SVF pre-warping domain [π·10/fs, π·0.49].
std::vector<double> makeInputsD(std::size_t n)
{
    std::vector<double> xs(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        xs[i] = kPi * (kFmin + (kFmax - kFmin) * t) / kFs;
    }
    return xs;
}

std::vector<float> makeInputsF(std::size_t groups)
{
    std::vector<float> xs(4 * groups);
    for (std::size_t g = 0; g < groups; ++g)
    {
        const double t = static_cast<double>(g) / static_cast<double>(groups);
        const float  x = static_cast<float>(kPi * (kFmin + (kFmax - kFmin) * t) / kFs);
        for (int lane = 0; lane < 4; ++lane) xs[4 * g + lane] = x;
    }
    return xs;
}

// Run fn (which performs `ops` tan evaluations) `reps` times; return the best
// (min) ns/tan and sink the accumulators into sinkOut so the loop body stays
// live.  Min is the most representative throughput sample (least contention).
template <class Fn>
double benchNsPerOp(Fn fn, std::size_t ops, std::size_t reps, double& sinkOut)
{
    double best  = std::numeric_limits<double>::infinity();
    double total = 0.0;
    for (std::size_t r = 0; r < reps; ++r)
    {
        const auto t0  = Clock::now();
        const double a = fn();
        const auto t1  = Clock::now();
        total += a;
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    sinkOut = total;
    return best / static_cast<double>(ops);
}

struct AccResult { double maxAbs; double maxRel; double worstX; };

AccResult accuracySweep(std::size_t N)
{
    double maxAbs = 0.0, maxRel = 0.0, worstX = 0.0;
    for (std::size_t i = 0; i < N; ++i)
    {
        const double t   = static_cast<double>(i) / static_cast<double>(N - 1);
        const double x   = -kXmax + 2.0 * kXmax * t;          // [-kXmax, +kXmax]
        const double ref = std::tan(x);
        const double got = mmTanScalar(x);
        const double absErr = std::fabs(got - ref);
        const double denom  = std::fmax(std::fabs(ref), 1e-6);
        const double relErr = absErr / denom;
        if (relErr > maxRel) { maxRel = relErr; worstX = x; }
        maxAbs = std::max(maxAbs, absErr);
    }
    return {maxAbs, maxRel, worstX};
}

} // namespace

int main()
{
    constexpr std::size_t tableN = 4096;
    constexpr std::size_t groups = 1024;
    constexpr std::size_t mask   = tableN - 1;
    constexpr std::size_t mask4  = groups - 1;
    const std::vector<double> xs  = makeInputsD(tableN);
    const std::vector<float>  xs4 = makeInputsF(groups);

    constexpr std::size_t scalarOps = 20'000'000;
    constexpr std::size_t vecIters  = 5'000'000;             // ×4 lanes = 20M ops
    constexpr std::size_t vecOps    = vecIters * 4;
    constexpr std::size_t reps      = 5;

    std::printf("=== Chronos tan pre-warping harness ===\n");
    std::printf("fs=%.0f Hz  fMin=%.0f  fMax=%.1f (0.49*fs)  xMax=%.6f rad (< 1.55)\n",
                kFs, kFmin, kFmax, kXmax);
    std::printf("ops=%zu  reps=%zu  (sink accumulators keep loops live)\n\n",
                scalarOps, reps);

    // ---- accuracy ----
    constexpr std::size_t accN = 200001;
    const AccResult acc = accuracySweep(accN);
    const bool accOk = acc.maxRel <= kRelErrLimit;
    std::printf("[accuracy] mmTanScalar vs std::tan over %zu pts in [-%.4f, +%.4f]\n",
                accN, kXmax, kXmax);
    std::printf("           max abs err = %.3e\n", acc.maxAbs);
    std::printf("           max rel err = %.3e  (limit %.0e)  at x=%+.6f\n",
                acc.maxRel, kRelErrLimit, acc.worstX);
    std::printf("           -> %s\n\n", accOk ? "PASS" : "FAIL");

    // ---- performance ----
    auto runStdTan = [&]() -> double {
        double a = 0.0;
        CHRONOS_NO_VECTORIZE
        for (std::size_t i = 0; i < scalarOps; ++i) {
            a += std::tan(xs[i & mask]);
            doNotOptimize(a);
        }
        return a;
    };
    auto runMmTanScalar = [&]() -> double {
        double a = 0.0;
        CHRONOS_NO_VECTORIZE
        for (std::size_t i = 0; i < scalarOps; ++i) {
            a += mmTanScalar(xs[i & mask]);
            doNotOptimize(a);
        }
        return a;
    };
    auto runMmTanM128 = [&]() -> double {
        M128 a = MM(setzero_ps)();
        for (std::size_t i = 0; i < vecIters; ++i) {
            const M128 v = MM(loadu_ps)(xs4.data() + 4 * (i & mask4));
            a = MM(add_ps)(a, mmTan(v));
            doNotOptimize(MM(cvtss_f32)(a));   // keep a live each iter; prevents hoisting
        }
        float lanes[4];
        MM(storeu_ps)(lanes, a);
        doNotOptimize(lanes[0]);
        return static_cast<double>(lanes[0] + lanes[1] + lanes[2] + lanes[3]);
    };

    double sink = 0.0;
    const double nsStdTan = benchNsPerOp(runStdTan,      scalarOps, reps, sink);
    const double nsScalar = benchNsPerOp(runMmTanScalar, scalarOps, reps, sink);
    const double nsM128   = benchNsPerOp(runMmTanM128,   vecOps,    reps, sink);

    const bool tanPerfOk = nsScalar <= nsStdTan * kPerfRegression;

    std::printf("[tan perf] ns/tan (min of %zu reps):\n", reps);
    std::printf("       std::tan        : %7.3f ns/tan\n", nsStdTan);
    std::printf("       mmTanScalar     : %7.3f ns/tan  (%.2fx vs std::tan)\n",
                nsScalar, nsStdTan / nsScalar);
    std::printf("       mmTan(M128) 4la : %7.3f ns/tan  (%.2fx vs std::tan)\n",
                nsM128, nsStdTan / nsM128);
    std::printf("       -> %s (mmTanScalar <= %.2fx std::tan)\n\n",
                tanPerfOk ? "PASS" : "FAIL", kPerfRegression);

    // ---- SVF throughput: SimdSVF block-ramp self-baseline ----
    // Varying cutoff (slow sine) simulates the 20ms smoothed parameter so the
    // block-ramp deltas are non-zero (realistic).  Informational — no pass/fail
    // since the scalar TwoPoleSVF baseline has been removed; compare against the
    // documented ~6.3 ns/sample on arm64 for regression detection.
    constexpr std::size_t kBlockSize  = 256;
    constexpr std::size_t kSvfSamples = 1'000'000;
    constexpr std::size_t kSvfBlocks  = kSvfSamples / kBlockSize;
    constexpr double kSvfQ      = 0.7071;   // Butterworth, matches processor
    constexpr double kSvfHpfF   = 200.0;
    constexpr double kSvfLpfF   = 8000.0;

    // Precomputed stereo input + per-sample cutoff (sine sweep around base).
    std::vector<float> inL(kSvfSamples), inR(kSvfSamples);
    std::vector<double> cutHpf(kSvfSamples), cutLpf(kSvfSamples);
    for (std::size_t i = 0; i < kSvfSamples; ++i)
    {
        inL[i]    = static_cast<float>(std::sin(2.0 * kPi * 440.0 * static_cast<double>(i) / kFs));
        inR[i]    = static_cast<float>(std::sin(2.0 * kPi * 330.0 * static_cast<double>(i) / kFs));
        cutHpf[i] = kSvfHpfF * (1.0 + 0.1 * std::sin(2.0 * kPi * 0.5 * static_cast<double>(i) / kFs));
        cutLpf[i] = kSvfLpfF * (1.0 + 0.1 * std::sin(2.0 * kPi * 0.3 * static_cast<double>(i) / kFs));
    }

    // SimdSVF, block-ramp setCoeffForBlock + per-sample processBlockStep,
    // stereo packed into lanes 0,1 of a single M128.  Coefficient computation is
    // off the per-sample path; float32 SIMD arithmetic.
    auto runNewSVF = [&]() -> double {
        MarsDSP::Filters::SimdSVF hpf, lpf;
        hpf.reset(); lpf.reset();
        double acc = 0.0;
        for (std::size_t b = 0; b < kSvfBlocks; ++b)
        {
            const std::size_t bs = b * kBlockSize;   // block-start cutoff (realistic)
            hpf.setCoeffForBlock(MarsDSP::Filters::SimdSVF::SVFType::HighPass,
                                 kFs, cutHpf[bs], kSvfQ, 0.0, static_cast<int>(kBlockSize));
            lpf.setCoeffForBlock(MarsDSP::Filters::SimdSVF::SVFType::LowPass,
                                 kFs, cutLpf[bs], kSvfQ, 0.0, static_cast<int>(kBlockSize));
            for (std::size_t i = 0; i < kBlockSize; ++i)
            {
                const std::size_t idx = bs + i;
                const M128 wetV = MM(set_ps)(0.0f, 0.0f, inR[idx], inL[idx]);
                const M128 hpV  = hpf.processBlockStep(wetV);
                const M128 lpV  = lpf.processBlockStep(hpV);
                alignas(16) float out[4];
                MM(storeu_ps)(out, lpV);
                acc += static_cast<double>(out[0] + out[1]);
                doNotOptimize(acc);
            }
        }
        return acc;
    };

    const double nsNewSVF = benchNsPerOp(runNewSVF, kSvfSamples, reps, sink);

    std::printf("[svf perf] ns/sample, stereo HPF+LPF wet path (min of %zu reps):\n", reps);
    std::printf("       SimdSVF (block-ramp, M128 float32) : %7.3f ns/sample  (info: ~6.3 on arm64)\n\n",
                nsNewSVF);

    std::printf("(sink=%f)\n", sink);

    const bool ok = accOk && tanPerfOk;
    std::printf("=== %s ===\n", ok ? "NO REGRESSION" : "REGRESSION DETECTED");
    return ok ? 0 : 1;
}

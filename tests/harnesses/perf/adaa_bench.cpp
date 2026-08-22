/**
 * Throughput microbenchmark and branch histogram for the ADAA saturators.
 * Reports ns per sample for std::tanh, ADAA1<TanhNL>, ADAA2<TanhNL>, and
 * ADAA2<AlgebraicNL>, plus the per-branch sample fractions of the ADAA2
 * branch selection over drives {0, 6, 12, 24, 40} dB. Min-of-5 reps.
 * Informational only.
 */

#include <chrono>
#include <cmath>
#include <cstddef>
#include <print>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "bench_util.h"
#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"

namespace
{
    using MarsDSP::Nonlinear::ADAA1;
    using MarsDSP::Nonlinear::ADAA2;
    using MarsDSP::Nonlinear::AlgebraicNL;
    using MarsDSP::Nonlinear::TanhNL;

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kFs = 48000.0;

    // Mirror of ADAA2<NL>::kEpsInner / kEpsOuter for the branch histogram shadow
    // classifier.  Kept in sync by hand - the histogram is meaningless if these
    // drift from the header.
    constexpr double kEpsInner = 1e-4;
    constexpr double kEpsOuter = 1e-6;

    using Clock = std::chrono::steady_clock;

    // Keep the scalar std::tanh loop scalar so it measures true per-call
    // throughput rather than being auto-vectorized across iterations (which would
    // conflate scalar and 4-wide SIMD tanh throughput).  The ADAA loops are
    // stateful and cannot vectorize across iterations regardless.
#ifdef __clang__
#define CHRONOS_NO_VECTORIZE _Pragma("clang loop vectorize(disable)")
#else
#define CHRONOS_NO_VECTORIZE
#endif

    // Compiler barriers (the Google Benchmark DoNotOptimize technique).  At -O2
    // clang will otherwise hoist the deterministic accumulations above Clock::now()
    // or dead-code-eliminate them, yielding 0 ns/sample.  doNotOptimize(v) emits
    // `asm volatile("" : : "r,m"(v) : "memory")`, which forces v to be materialized
    // each iteration and blocks reordering across the timer.
#if defined(__clang__) || defined(__GNUC__)
    template<class T>
    inline void doNotOptimize(T const &v) noexcept
    {
        asm volatile("" : : "r,m"(v) : "memory");
    }
#else
    template<class T>
    inline void doNotOptimize(T const &v) noexcept
    {
        volatile T sink = v;
        (void) sink;
    }
#endif

    // Run fn (which performs `ops` sample evaluations) `reps` times; return the
    // best (min) ns/sample and sink the accumulators into sinkOut so the loop
    // body stays live.  Min is the most representative throughput sample (least
    // contention).
    template<class Fn>
    double benchNsPerOp(Fn fn, std::size_t ops, std::size_t reps, double &sinkOut)
    {
        double best = std::numeric_limits<double>::infinity();
        double total = 0.0;
        for (std::size_t r = 0; r < reps; ++r)
        {
            const auto t0 = Clock::now();
            const double a = fn();
            const auto t1 = Clock::now();
            total += a;
            best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        sinkOut = total;
        return best / static_cast<double>(ops);
    }

    // Exponential frequency sweep from f0 to f1 over N samples at amplitude amp.
    // The frequency sweeps 20 Hz -> 20 kHz, exercising the full node-spacing
    // range from near-DC (all nodes coincident -> branch (a)) to near-Nyquist
    // (alternation -> branch (b)).
    std::vector<double> makeSineSweep(std::size_t N, double amp, double f0, double f1)
    {
        std::vector<double> xs(N);
        const double ratio = f1 / f0;
        double phase = 0.0;
        for (std::size_t i = 0; i < N; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(N);
            const double f = f0 * std::pow(ratio, t);
            phase += 2.0 * kPi * f / kFs;
            xs[i] = amp * std::sin(phase);
        }
        return xs;
    }

    // Shadow of ADAA2<NL>::process branch selection.  Returns 0=(a), 1=(b),
    // 2=(c), 3=(d).  Depends only on the three input nodes, not on the policy.
    int classifyBranch(double x0, double x1, double x2) noexcept
    {
        const double A = std::fabs(x0 - x1);
        const double B = std::fabs(x1 - x2);
        const double C = std::fabs(x0 - x2);
        if (A < kEpsInner && B < kEpsInner) return 0; // (a) centroid
        if (C < kEpsOuter) return 1; // (b) confluent Hermite
        if (A < kEpsInner) return 2; // (c) inner midpoint
        return 3; // (d) generic
    }
} // namespace

int main(int argc, char **argv)
{
    std::string jsonPath;
    bool provisional = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) jsonPath = argv[++i];
        else if (std::strcmp(argv[i], "--provisional") == 0) provisional = true;
    }

    bench::setFtzDaz();

    constexpr std::size_t tableN = 1u << 18; // 262144 samples (~2 MB)
    constexpr std::size_t mask = tableN - 1;
    constexpr std::size_t ops = 5'000'000;
    constexpr std::size_t reps = 5;
    constexpr double kDriveDb = 12.0;
    const double driveLin = std::pow(10.0, kDriveDb / 20.0);

    const std::vector<double> xs = makeSineSweep(tableN, driveLin, 20.0, 20000.0);

    std::println("=== Chronos ADAA throughput + branch histogram ===");
    std::println("fs={:.0} Hz  sweep 20 Hz -> 20 kHz  drive={:.0} dB (x{:.2})",
                kFs, kDriveDb, driveLin);
    std::println("ops={}  reps={}  (min of {}, sink accumulators keep loops live)\n",
                ops, reps, reps);

    // ---- timing ----
    auto runStdTanh = [&]() -> double
    {
        double a = 0.0;
        CHRONOS_NO_VECTORIZE
        for (std::size_t i = 0; i < ops; ++i)
        {
            a += std::tanh(xs[i & mask]);
            doNotOptimize(a);
        }
        return a;
    };
    auto runADAA1 = [&]() -> double
    {
        double a = 0.0;
        ADAA1<TanhNL> s;
        s.reset();
        for (std::size_t i = 0; i < ops; ++i)
        {
            a += s.process(xs[i & mask]);
            doNotOptimize(a);
        }
        return a;
    };
    auto runADAA2Tanh = [&]() -> double
    {
        double a = 0.0;
        ADAA2<TanhNL> s;
        s.reset();
        for (std::size_t i = 0; i < ops; ++i)
        {
            a += s.process(xs[i & mask]);
            doNotOptimize(a);
        }
        return a;
    };
    auto runADAA2Alg = [&]() -> double
    {
        double a = 0.0;
        ADAA2<AlgebraicNL> s;
        s.reset();
        for (std::size_t i = 0; i < ops; ++i)
        {
            a += s.process(xs[i & mask]);
            doNotOptimize(a);
        }
        return a;
    };

    double sink = 0.0;
    const double nsTanh = benchNsPerOp(runStdTanh, ops, reps, sink);
    const double nsA1 = benchNsPerOp(runADAA1, ops, reps, sink);
    const double nsA2T = benchNsPerOp(runADAA2Tanh, ops, reps, sink);
    const double nsA2A = benchNsPerOp(runADAA2Alg, ops, reps, sink);

    std::vector<bench::Record> records;
    records.push_back({"std::tanh", "", nsTanh});
    records.push_back({"ADAA1<TanhNL>", "", nsA1});
    records.push_back({"ADAA2<TanhNL>", "", nsA2T});
    records.push_back({"ADAA2<AlgebraicNL>", "", nsA2A});

    std::println("[timing] ns/sample (min of {} reps, 48 kHz sweep at {:.0} dB):", reps, kDriveDb);
    std::println("       std::tanh (no ADAA)    : {:7.2} ns/sample", nsTanh);
    std::println("       ADAA1<TanhNL>          : {:7.2} ns/sample  ({:.1}x vs tanh)", nsA1, nsA1 / nsTanh);
    std::println("       ADAA2<TanhNL>          : {:7.2} ns/sample  ({:.1}x vs tanh)", nsA2T, nsA2T / nsTanh);
    std::println("       ADAA2<AlgebraicNL>     : {:7.2} ns/sample  ({:.1}x vs tanh)", nsA2A, nsA2A / nsTanh);
    std::println("       ADAA2 / ADAA1          : {:.2}x\n", nsA2T / nsA1);

    // ---- branch histogram ----
    // One sweep per drive level.  Branch selection is policy-independent
    // (depends only on node spacing), so only TanhNL is run.
    constexpr std::size_t kSweepN = 1'000'000;
    const std::array<double, 5> drives { { 0.0, 6.0, 12.0, 24.0, 40.0 } };
    constexpr int kNDrives = static_cast<int>(sizeof(drives) / sizeof(drives[0]));

    std::println("[branch histogram] exponential sweep 20 Hz -> 20 kHz, {} samples:", kSweepN);
    std::println("       drive  (a) centroid  (b) confluent  (c) inner-mid  (d) generic");
    for (int d = 0; d < kNDrives; ++d)
    {
        const double amp = std::pow(10.0, drives[d] / 20.0);
        const std::vector<double> sweep = makeSineSweep(kSweepN, amp, 20.0, 20000.0);

        ADAA2<TanhNL> s;
        s.reset();
        long counts[4] = {0, 0, 0, 0};
        double xm1 = 0.0;
        double xm2 = 0.0;
        for (std::size_t i = 0; i < kSweepN; ++i)
        {
            const double x = sweep[i];
            (void) s.process(x);
            if (i >= 2)
                ++counts[classifyBranch(x, xm1, xm2)];
            xm2 = xm1;
            xm1 = x;
        }
        const long total = static_cast<long>(kSweepN) - 2;
        const double f0 = 100.0 * static_cast<double>(counts[0]) / static_cast<double>(total);
        const double f1 = 100.0 * static_cast<double>(counts[1]) / static_cast<double>(total);
        const double f2 = 100.0 * static_cast<double>(counts[2]) / static_cast<double>(total);
        const double f3 = 100.0 * static_cast<double>(counts[3]) / static_cast<double>(total);
        std::println("       {:4.0} dB   {:10.4}%   {:10.4}%   {:10.4}%   {:10.4}%",
                    drives[d], f0, f1, f2, f3);
    }

    std::println("\n(sink={})", sink);

    if (!jsonPath.empty())
        bench::writeJson(jsonPath, records, provisional);

    std::println("=== informational only — no pass/fail gate ===");
    return 0;
}

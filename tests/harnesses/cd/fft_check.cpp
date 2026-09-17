/**
 * FFT correctness harness for source/math/FFT.h.
 * Host-free: links SharedCode only. Forced -O2 and -mfma so the
 * power-of-two SoA engine takes its vector stages, the path the
 * spectrum analyser runs.
 *
 * Properties:
 *   1. Complex forward vs a naive O(N^2) DFT, N in {2..4096} pow2 plus
 *      the mixed-radix lengths 6, 12, 15, 36, 100, 1000 (float and double).
 *   2. inv(fwd(x)) == N * x (unnormalised inverse), same lengths.
 *   3. Real forward vs the naive DFT of the real input, N in {4..4096}.
 *   4. Real inverse round trip.
 *   5. Parseval: sum |x|^2 == (1/N) sum |X|^2.
 *   6. A bin-centred tone lands its energy in that bin alone.
 */

#include "math/FFT.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <print>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Deterministic xorshift for reproducible test vectors.
std::uint32_t g_rng = 0x2545F491u;
double uniform()
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return static_cast<double>(g_rng >> 8) * (1.0 / 16777216.0) * 2.0 - 1.0;
}

// Naive DFT in double, the oracle. X[k] = sum x[n] e^{-2 pi i k n / N}.
std::vector<std::complex<double>> naiveDft(const std::vector<std::complex<double>>& x)
{
    const std::size_t n = x.size();
    std::vector<std::complex<double>> out(n);
    for (std::size_t k = 0; k < n; ++k)
    {
        std::complex<double> acc{};
        for (std::size_t i = 0; i < n; ++i)
        {
            const double ph = -2.0 * std::numbers::pi_v<double> * static_cast<double>(k * i % n) / static_cast<double>(n);
            acc += x[i] * std::complex<double>(std::cos(ph), std::sin(ph));
        }
        out[k] = acc;
    }
    return out;
}

// Relative error tolerance scaled by log2(N): each stage adds rounding.
template <typename ST>
double tolFor(std::size_t n)
{
    const double base = std::is_same_v<ST, float> ? 2.0e-6 : 4.0e-14;
    return base * std::max(1.0, std::log2(static_cast<double>(n)));
}

template <typename ST>
void checkComplex(std::size_t n)
{
    using C = std::complex<ST>;
    std::vector<C> x(n);
    std::vector<std::complex<double>> xd(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const double re = uniform(), im = uniform();
        x[i] = C(static_cast<ST>(re), static_cast<ST>(im));
        xd[i] = { static_cast<double>(x[i].real()), static_cast<double>(x[i].imag()) };
    }

    MarsDSP::MathOps::FFT<ST> fft(n);
    std::vector<C> X(n), back(n);
    fft.fwd(x.data(), X.data());

    const auto ref = naiveDft(xd);
    double scale = 0.0;
    for (const auto& r : ref) scale = std::max(scale, std::abs(r));
    scale = std::max(scale, 1.0);
    const double tol = tolFor<ST>(n);
    for (std::size_t k = 0; k < n; ++k)
    {
        const double err = std::abs(std::complex<double>(X[k].real(), X[k].imag()) - ref[k]) / scale;
        if (err > tol)
            FAIL("complex N={} ({}) bin {} err {:.3e} > {:.3e}", n, sizeof(ST) == 4 ? "float" : "double", k, err, tol);
    }

    // Unnormalised inverse: inv(fwd(x)) == N x. The error is relative to
    // the signal scale N * max|x|, the same normalisation as the forward.
    fft.inv(X.data(), back.data());
    double xScale = 0.0;
    for (const auto& v : xd) xScale = std::max(xScale, std::abs(v));
    const double backScale = std::max(1.0, xScale * static_cast<double>(n));
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto want = std::complex<double>(xd[i]) * static_cast<double>(n);
        const double err = std::abs(std::complex<double>(back[i].real(), back[i].imag()) - want) / backScale;
        if (err > tol)
            FAIL("complex round trip N={} sample {} err {:.3e} > {:.3e}", n, i, err, tol);
    }

    // Parseval.
    double lhs = 0.0, rhs = 0.0;
    for (std::size_t i = 0; i < n; ++i) lhs += std::norm(xd[i]);
    for (std::size_t k = 0; k < n; ++k) rhs += std::norm(std::complex<double>(X[k].real(), X[k].imag()));
    rhs /= static_cast<double>(n);
    if (std::abs(lhs - rhs) / std::max(1.0, lhs) > 10.0 * tol)
        FAIL("Parseval N={} lhs {} rhs {}", n, lhs, rhs);
}

template <typename ST>
void checkReal(std::size_t n)
{
    using C = std::complex<ST>;
    std::vector<ST> x(n);
    std::vector<std::complex<double>> xd(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = static_cast<ST>(uniform());
        xd[i] = { static_cast<double>(x[i]), 0.0 };
    }

    MarsDSP::MathOps::RealFFT<ST> rfft(n);
    // The real transform writes N/2 complex bins: out[0] packs DC (re) and
    // Nyquist (im); bins 1..N/2-1 are the ordinary complex values.
    std::vector<C> X(n / 2);
    rfft.forward(x.data(), X.data());

    const auto ref = naiveDft(xd);
    double scale = 0.0;
    for (const auto& r : ref) scale = std::max(scale, std::abs(r));
    scale = std::max(scale, 1.0);
    // The real path is a half-size complex pass plus the split/merge
    // twiddles, and the round trip runs it twice: allow two passes.
    const double tol = 2.0 * tolFor<ST>(n);

    const double dcErr = std::abs(static_cast<double>(X[0].real()) - ref[0].real()) / scale;
    const double nyErr = std::abs(static_cast<double>(X[0].imag()) - ref[n / 2].real()) / scale;
    if (dcErr > tol || nyErr > tol)
        FAIL("real N={} DC err {:.3e} Nyquist err {:.3e} > {:.3e}", n, dcErr, nyErr, tol);
    for (std::size_t k = 1; k < n / 2; ++k)
    {
        const double err = std::abs(std::complex<double>(X[k].real(), X[k].imag()) - ref[k]) / scale;
        if (err > tol)
            FAIL("real N={} bin {} err {:.3e} > {:.3e}", n, k, err, tol);
    }

    // Round trip: inverse(forward(x)) == N x, relative to the signal scale.
    std::vector<ST> back(n);
    rfft.inverse(X.data(), back.data());
    double xScale = 0.0;
    for (const auto& v : xd) xScale = std::max(xScale, std::abs(v));
    const double backScale = std::max(1.0, xScale * static_cast<double>(n));
    for (std::size_t i = 0; i < n; ++i)
    {
        const double want = static_cast<double>(x[i]) * static_cast<double>(n);
        const double err = std::abs(static_cast<double>(back[i]) - want) / backScale;
        if (err > tol)
            FAIL("real round trip N={} sample {} err {:.3e} > {:.3e}", n, i, err, tol);
    }
}

int runAll()
{
    g_section = "complex_pow2";
    for (std::size_t n = 2; n <= 4096; n *= 2)
    {
        checkComplex<float>(n);
        checkComplex<double>(n);
    }
    std::println("complex pow2 2..4096 (float + double, vs naive DFT, round trip, Parseval): PASS");

    g_section = "complex_mixed";
    for (const std::size_t n : { 6uz, 12uz, 15uz, 36uz, 100uz, 1000uz })
    {
        checkComplex<float>(n);
        checkComplex<double>(n);
    }
    std::println("complex mixed radix 6/12/15/36/100/1000: PASS");

    g_section = "real";
    for (std::size_t n = 4; n <= 4096; n *= 2)
    {
        checkReal<float>(n);
        checkReal<double>(n);
    }
    std::println("real 4..4096 (float + double, vs naive DFT, round trip): PASS");

    // A bin-centred tone: all energy in one bin, the rest at the noise floor.
    g_section = "tone_in_bin";
    {
        constexpr std::size_t n = 4096;
        constexpr std::size_t kBin = 85;
        std::vector<float> x(n);
        for (std::size_t i = 0; i < n; ++i)
            x[i] = std::sin(2.0f * std::numbers::pi_v<float> * static_cast<float>(kBin) * static_cast<float>(i) / static_cast<float>(n));
        MarsDSP::MathOps::RealFFT<float> rfft(n);
        std::vector<std::complex<float>> X(n / 2);
        rfft.forward(x.data(), X.data());
        // |X[k]| = N/2 for a unit sine on a bin centre.
        const double peak = std::abs(std::complex<double>(X[kBin].real(), X[kBin].imag()));
        CHECK(std::abs(peak - static_cast<double>(n) / 2.0) < 1.0e-2 * static_cast<double>(n));
        for (std::size_t k = 1; k < n / 2; ++k)
        {
            if (k == kBin) continue;
            const double leak = std::abs(std::complex<double>(X[k].real(), X[k].imag()));
            if (leak > 1.0e-3 * peak)
                FAIL("tone leak at bin {}: {} vs peak {}", k, leak, peak);
        }
        std::println("tone in bin {} of {}: peak {:.1f} (N/2 = {}), leak < -60 dB: PASS", kBin, n, peak, n / 2);
    }

    std::println("fft_check: ALL PASS");
    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos FFT harness ===");
    std::println();
    const int r = runAll();
    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

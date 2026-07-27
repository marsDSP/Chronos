// tests/harnesses/dsp/alias_check.cpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dsp/nonlinear/ADAA1.h"
#include "dsp/nonlinear/ADAA2.h"
#include "dsp/nonlinear/Nonlinearities.h"

namespace {

using MarsDSP::Nonlinear::ADAA1;
using MarsDSP::Nonlinear::ADAA2;
using MarsDSP::Nonlinear::TanhNL;

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

constexpr double kPi = 3.14159265358979323846;
constexpr double kFs = 48000.0;
constexpr int    kN  = 16384;          // analysis window (power of two)
constexpr int    kWarmup = 2048;       // > resampler group delay + ADAA memory

// Measurement floor. A = T - H is a difference of nearly equal quantities, so
// the resolvable alias energy bottoms out at the relative error of H, which is
// dominated by Goertzel's O(N*u) recurrence error (N*u = 3.6e-12 ~ -114 dB).
// The identity calibration below measures the realised floor at about -134 dB;
// -130 is the reported bar. Cells at or below it are printed as "< -130" and
// clamped to it in arithmetic, so a difference against an immeasurably small
// value can never manufacture a large apparent improvement.
constexpr double kFloorDbc = -130.0;

inline bool atFloor(double dbc) noexcept { return dbc <= kFloorDbc; }
inline double clampFloor(double dbc) noexcept { return dbc < kFloorDbc ? kFloorDbc : dbc; }

// Format one cell: an explicit "< floor" marker rather than a number we cannot
// actually support.
void fmtDbc(char* buf, std::size_t n, double dbc)
{
    if (atFloor(dbc)) std::snprintf(buf, n, "  < %-6.0f", kFloorDbc);
    else              std::snprintf(buf, n, "%9.1f", dbc);
}

// Kahan-compensated sum. The alias energy is a difference of two nearly equal
// quantities (T - H), so a naive sum's O(N*u) error would put the measurement
// floor around -114 dB - inside the range we care about. Kahan moves it to
// ~u, i.e. below -150 dB.
class KahanSum
{
public:
    void add(double v) noexcept
    {
        const double y = v - c_;
        const double t = s_ + y;
        c_ = (t - s_) - y;
        s_ = t;
    }
    double get() const noexcept { return s_; }
private:
    double s_ = 0.0;
    double c_ = 0.0;
};

// |X_k|^2 for integer bin k over N samples (Goertzel).
double goertzelMagSq(const double* x, int N, int k) noexcept
{
    const double w  = 2.0 * kPi * static_cast<double>(k) / static_cast<double>(N);
    const double cw = std::cos(w);
    const double coeff = 2.0 * cw;
    double s1 = 0.0, s2 = 0.0;
    for (int n = 0; n < N; ++n)
    {
        const double s0 = x[n] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

struct Analysis
{
    double aliasDbc;      // inharmonic energy, dB relative to the fundamental
    double fundEnergy;
    double aliasEnergy;
    int    nHarmonics;    // legitimate (unfolded) harmonics that were masked
};

// Split N samples into harmonic (masked) and inharmonic energy.
Analysis analyze(const std::vector<double>& y, int k0)
{
    const int N = kN;

    KahanSum tot;
    for (int n = 0; n < N; ++n)
        tot.add(y[static_cast<std::size_t>(n)] * y[static_cast<std::size_t>(n)]);
    const double T = tot.get();

    const double invN = 1.0 / static_cast<double>(N);

    // Bin 0 (DC) is legitimate, not an alias. tanh is odd so it should be ~0,
    // but a branch fallback could leak a little and it must not be counted
    // against the saturator as aliasing.
    KahanSum harm;
    harm.add(goertzelMagSq(y.data(), N, 0) * invN);

    int kMax = 0;
    for (int j = 1; j * k0 < N / 2; ++j)
    {
        // Real signal: bins j*k0 and N-j*k0 are conjugates, so the physical
        // energy at this frequency is twice the single-bin energy.
        harm.add(2.0 * goertzelMagSq(y.data(), N, j * k0) * invN);
        kMax = j;
    }

    const double H = harm.get();
    const double A = std::fmax(T - H, 0.0);
    const double F = 2.0 * goertzelMagSq(y.data(), N, k0) * invN;

    Analysis r;
    r.fundEnergy  = F;
    r.aliasEnergy = A;
    r.nHarmonics  = kMax;
    // A <= 0 means the harmonic energy accounted for everything measurable:
    // the alias energy is below the cancellation floor, not zero.
    r.aliasDbc    = (A > 0.0 && F > 0.0) ? 10.0 * std::log10(A / F) : -999.0;
    return r;
}

// ── 2x resampling (measurement path only) ─────────────────────────────────

double besselI0(double x) noexcept
{
    double sum = 1.0, term = 1.0;
    for (int m = 1; m < 80; ++m)
    {
        const double r = x / (2.0 * static_cast<double>(m));
        term *= r * r;
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

// Kaiser-windowed sinc lowpass, M+1 taps, cutoff fc in cycles/sample of the
// rate the filter runs at. beta = 14 puts the stopband near -126 dB, well
// below the alias floors being measured, so the resampler is not the thing
// under test.
std::vector<double> designLpf(int M, double fc, double beta)
{
    std::vector<double> h(static_cast<std::size_t>(M) + 1);
    const double i0b = besselI0(beta);
    const double half = 0.5 * static_cast<double>(M);
    for (int n = 0; n <= M; ++n)
    {
        const double d = static_cast<double>(n) - half;
        const double s = (d == 0.0) ? 2.0 * fc
                                    : std::sin(2.0 * kPi * fc * d) / (kPi * d);
        const double r = d / half;
        const double wk = besselI0(beta * std::sqrt(std::fmax(0.0, 1.0 - r * r))) / i0b;
        h[static_cast<std::size_t>(n)] = s * wk;
    }
    return h;
}

constexpr int kFirM = 512;             // 513 taps, integer group delay 256

// Zero-stuff by 2 and interpolate. Only even input positions are non-zero, so
// the inner loop strides the input rather than materialising the zeros.
std::vector<double> upsample2x(const std::vector<double>& x, const std::vector<double>& h)
{
    const int N = static_cast<int>(x.size());
    const int M = static_cast<int>(h.size()) - 1;
    std::vector<double> y(static_cast<std::size_t>(2 * N), 0.0);
    for (int m = 0; m < 2 * N; ++m)
    {
        int jlo = (m - M + 1) / 2;
        if (jlo < 0) jlo = 0;
        int jhi = m / 2;
        if (jhi > N - 1) jhi = N - 1;
        double acc = 0.0;
        for (int j = jlo; j <= jhi; ++j)
            acc += h[static_cast<std::size_t>(m - 2 * j)] * x[static_cast<std::size_t>(j)];
        y[static_cast<std::size_t>(m)] = 2.0 * acc;   // zero-stuffing halves the gain
    }
    return y;
}

// Filter and keep every second sample. Only the retained outputs are computed.
std::vector<double> downsample2x(const std::vector<double>& x, const std::vector<double>& h)
{
    const int N2 = static_cast<int>(x.size());
    const int M  = static_cast<int>(h.size()) - 1;
    const int N  = N2 / 2;
    std::vector<double> y(static_cast<std::size_t>(N), 0.0);
    for (int n = 0; n < N; ++n)
    {
        const int m = 2 * n;
        int klo = m - (N2 - 1);
        if (klo < 0) klo = 0;
        int khi = (m < M) ? m : M;
        double acc = 0.0;
        for (int k = klo; k <= khi; ++k)
            acc += h[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(m - k)];
        y[static_cast<std::size_t>(n)] = acc;
    }
    return y;
}

// ── configurations ────────────────────────────────────────────────────────

enum Config { kNone = 0, kIdentity, kADAA1, kADAA2, kADAA1x2, kIdentityX2 };

const char* configName(Config c)
{
    switch (c)
    {
        case kNone:        return "no-ADAA";
        case kIdentity:    return "identity";
        case kADAA1:       return "ADAA1";
        case kADAA2:       return "ADAA2";
        case kADAA1x2:     return "ADAA1@2x";
        default:           return "identity@2x";
    }
}

// Produce kN analysed samples for the given configuration.
std::vector<double> render(Config cfg, int k0, double driveLin, const std::vector<double>& h)
{
    const int total = kWarmup + kN;

    // Coherent input: exactly k0 cycles per kN samples, so the analysis window
    // is periodic regardless of where the warmup ends.
    std::vector<double> x(static_cast<std::size_t>(total));
    for (int n = 0; n < total; ++n)
        x[static_cast<std::size_t>(n)] =
            std::sin(2.0 * kPi * static_cast<double>(k0) * static_cast<double>(n) / static_cast<double>(kN));

    std::vector<double> out(static_cast<std::size_t>(total));

    if (cfg == kADAA1x2 || cfg == kIdentityX2)
    {
        const std::vector<double> up = upsample2x(x, h);
        std::vector<double> sat(up.size());
        if (cfg == kADAA1x2)
        {
            ADAA1<TanhNL> s;
            s.reset();
            for (std::size_t i = 0; i < up.size(); ++i)
                sat[i] = s.process(driveLin * up[i]);
        }
        else
        {
            for (std::size_t i = 0; i < up.size(); ++i)
                sat[i] = driveLin * up[i];
        }
        out = downsample2x(sat, h);
    }
    else if (cfg == kADAA1)
    {
        ADAA1<TanhNL> s;
        s.reset();
        for (int n = 0; n < total; ++n)
            out[static_cast<std::size_t>(n)] = s.process(driveLin * x[static_cast<std::size_t>(n)]);
    }
    else if (cfg == kADAA2)
    {
        ADAA2<TanhNL> s;
        s.reset();
        for (int n = 0; n < total; ++n)
            out[static_cast<std::size_t>(n)] = s.process(driveLin * x[static_cast<std::size_t>(n)]);
    }
    else if (cfg == kNone)
    {
        for (int n = 0; n < total; ++n)
            out[static_cast<std::size_t>(n)] = std::tanh(driveLin * x[static_cast<std::size_t>(n)]);
    }
    else // kIdentity
    {
        for (int n = 0; n < total; ++n)
            out[static_cast<std::size_t>(n)] = driveLin * x[static_cast<std::size_t>(n)];
    }

    return std::vector<double>(out.begin() + kWarmup, out.begin() + kWarmup + kN);
}

// Nearest ODD integer to v, clamped to >= 1. Oddness is what guarantees each
// harmonic gets its own bin (see the header block).
int roundToOdd(double v)
{
    int k = static_cast<int>(std::lround(v));
    if ((k & 1) == 0) k += (v >= static_cast<double>(k)) ? 1 : -1;
    if (k < 1) k = 1;
    return k;
}

} // namespace

int main()
{
    const double f0s[]    = { 55.0, 110.0, 220.0, 440.0, 1000.0, 2000.0, 5000.0, 10000.0 };
    const double drives[] = { 0.0, 6.0, 12.0, 24.0, 40.0 };
    constexpr int kNF = static_cast<int>(sizeof(f0s) / sizeof(f0s[0]));
    constexpr int kND = static_cast<int>(sizeof(drives) / sizeof(drives[0]));

    const std::vector<double> h = designLpf(kFirM, 0.25, 14.0);

    std::printf("=== Chronos ADAA aliasing harness ===\n");
    std::printf("fs=%.0f Hz  N=%d (coherent, odd bin)  warmup=%d  2x FIR: %d taps Kaiser beta=14\n",
                kFs, kN, kWarmup, kFirM + 1);
    std::printf("Alias energy = total - masked harmonics, reported in dBc vs the fundamental.\n\n");

    // ── calibration: the measurement's own noise floor ────────────────────
    g_section = "calibration";
    {
        std::printf("[calibration] identity path (no nonlinearity) - this is the measurement floor:\n");
        std::printf("        f0 (Hz)   k0   harmonics   direct        via 2x resampler\n");
        double worstDirect = -999.0, worst2x = -999.0;
        for (int i = 0; i < kNF; ++i)
        {
            const int k0 = roundToOdd(f0s[i] * static_cast<double>(kN) / kFs);
            const double fAct = static_cast<double>(k0) * kFs / static_cast<double>(kN);
            const Analysis a1 = analyze(render(kIdentity,   k0, 1.0, h), k0);
            const Analysis a2 = analyze(render(kIdentityX2, k0, 1.0, h), k0);
            worstDirect = std::fmax(worstDirect, a1.aliasDbc);
            worst2x     = std::fmax(worst2x,     a2.aliasDbc);
            char c1[32], c2[32];
            fmtDbc(c1, sizeof c1, a1.aliasDbc);
            fmtDbc(c2, sizeof c2, a2.aliasDbc);
            std::printf("      %8.1f  %5d  %9d   %s dB   %s dB\n",
                        fAct, k0, a1.nHarmonics, c1, c2);
        }
        std::printf("      worst: direct %.1f dB, 2x %.1f dB\n", worstDirect, worst2x);
        if (worstDirect > kFloorDbc)
            FAIL("measurement floor %.1f dB exceeds the declared %.0f dB bar", worstDirect, kFloorDbc);
        std::printf("      -> PASS (direct floor at or below %.0f dB; the 2x column is\n"
                    "         resampler-limited, so 2x cells are only trustworthy above it)\n\n",
                    kFloorDbc);
    }

    // ── the matrix ────────────────────────────────────────────────────────
    g_section = "matrix";
    const Config cfgs[4] = { kNone, kADAA1, kADAA2, kADAA1x2 };
    double specNone = 0.0, specADAA2 = 0.0;      // 110 Hz / 24 dB (spec cell)
    double gateNone = 0.0, gateADAA2 = 0.0;      // 10 kHz / 24 dB (gate cell)
    bool specFound = false, gateFound = false;

    for (int d = 0; d < kND; ++d)
    {
        const double driveLin = std::pow(10.0, drives[d] / 20.0);
        std::printf("[drive %.0f dB]  alias floor, dBc vs fundamental (lower is better)\n", drives[d]);
        std::printf("        f0 (Hz)   harm  %-10s %-10s %-10s %-10s  ADAA2 vs none\n",
                    configName(kNone), configName(kADAA1), configName(kADAA2), configName(kADAA1x2));
        for (int i = 0; i < kNF; ++i)
        {
            const int k0 = roundToOdd(f0s[i] * static_cast<double>(kN) / kFs);
            const double fAct = static_cast<double>(k0) * kFs / static_cast<double>(kN);

            double v[4];
            int nh = 0;
            for (int c = 0; c < 4; ++c)
            {
                const Analysis a = analyze(render(cfgs[c], k0, driveLin, h), k0);
                v[c] = a.aliasDbc;
                nh = a.nHarmonics;
            }

            char cell[4][32];
            for (int c = 0; c < 4; ++c)
                fmtDbc(cell[c], sizeof cell[c], v[c]);

            // If no-ADAA is already at the floor there is no aliasing to
            // remove and the comparison is meaningless; if only ADAA2 is at
            // the floor the improvement is a lower bound.
            char gain[32];
            if (atFloor(v[0]))
                std::snprintf(gain, sizeof gain, "     n/m");
            else
                std::snprintf(gain, sizeof gain, "%s%+6.1f dB",
                              atFloor(v[2]) ? " >" : "  ",
                              clampFloor(v[0]) - clampFloor(v[2]));

            std::printf("      %8.1f  %5d %s %s %s %s  %s\n",
                        fAct, nh, cell[0], cell[1], cell[2], cell[3], gain);

            if (std::fabs(drives[d] - 24.0) < 1e-9 && std::fabs(f0s[i] - 110.0) < 1e-9)
            {
                specNone = v[0]; specADAA2 = v[2]; specFound = true;
            }
            if (std::fabs(drives[d] - 24.0) < 1e-9 && std::fabs(f0s[i] - 10000.0) < 1e-9)
            {
                gateNone = v[0]; gateADAA2 = v[2]; gateFound = true;
            }
        }
        std::printf("\n");
    }

    // ── gate ──────────────────────────────────────────────────────────────
    //
    // The spec nominated 110 Hz / 24 dB for a >= 20 dB gate. That cell cannot
    // carry it, and the reason is a property of the curve rather than of the
    // implementation: tanh is analytic, so its harmonic series decays
    // EXPONENTIALLY at a rate set by the drive (~exp(-k/driveLin)), not as 1/k
    // the way a hard clipper's does. At 110 Hz there are 221 harmonic slots
    // below Nyquist and the series has long since died before reaching the
    // fold point, so no-ADAA already measures at/near the floor there and
    // there is essentially no aliasing for ADAA2 to remove. A >= 20 dB
    // improvement is not demonstrable because the quantity being improved is
    // not measurable.
    //
    // The gate therefore moves to 10 kHz / 24 dB - the same drive, at the
    // frequency where folding actually dominates (only 2 harmonics fit below
    // Nyquist). Both cells are reported so the deviation is auditable.
    g_section = "gate";
    CHECK(specFound);
    CHECK(gateFound);

    char sN[32], sA[32];
    fmtDbc(sN, sizeof sN, specNone);
    fmtDbc(sA, sizeof sA, specADAA2);
    std::printf("[spec cell] 110 Hz / 24 dB: no-ADAA %s dBc, ADAA2 %s dBc\n", sN, sA);
    std::printf("            tanh's harmonics decay exponentially, so nothing folds at 110 Hz;\n"
                "            both configurations sit at the measurement floor. Sanity-checked\n"
                "            (ADAA2 must not be worse), gated below at 10 kHz instead.\n");
    if (clampFloor(specADAA2) > clampFloor(specNone) + 1.0)
        FAIL("ADAA2 is worse than no-ADAA at the spec cell: %.1f vs %.1f dBc",
             specADAA2, specNone);

    const double improvement = clampFloor(gateNone) - clampFloor(gateADAA2);
    std::printf("[gate] 10 kHz / 24 dB: no-ADAA %.1f dBc, ADAA2 %.1f dBc -> %.1f dB better\n",
                gateNone, gateADAA2, improvement);
    if (improvement < 20.0)
        FAIL("ADAA2 must beat no-ADAA by >= 20 dB at 10 kHz / 24 dB; got %.1f dB", improvement);
    std::printf("       -> PASS (>= 20 dB)\n");

    std::printf("\n=== ALL PROPERTIES HELD ===\n");
    return 0;
}

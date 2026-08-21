#include "dsp/NestedAllpass.h"
#include "dsp/FracDelayTap.h"
#include "dsp/Pow2RingBuffer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>
#include <vector>

namespace
{
    const char *g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::printf("FAIL [%s] %s:%d: %s\n", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(fmt, ...) \
    do { std::printf("FAIL [%s] " fmt "\n", g_section, ##__VA_ARGS__); std::exit(1); } while (0)

    // Radix-2 in-place FFT for power-of-two size N.
    void fft(std::vector<double> &re, std::vector<double> &im)
    {
        const std::size_t n = re.size();
        for (std::size_t i = 1, j = 0; i < n; ++i)
        {
            std::size_t bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
        for (std::size_t len = 2; len <= n; len <<= 1)
        {
            const double ang = -2.0 * std::numbers::pi / static_cast<double>(len);
            const double wlenRe = std::cos(ang);
            const double wlenIm = std::sin(ang);
            for (std::size_t i = 0; i < n; i += len)
            {
                double wRe = 1.0;
                double wIm = 0.0;
                for (std::size_t j = 0; j < len / 2; ++j)
                {
                    const double uRe = re[i + j];
                    const double uIm = im[i + j];
                    const double vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
                    const double vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
                    re[i + j] = uRe + vRe;
                    im[i + j] = uIm + vIm;
                    re[i + j + len / 2] = uRe - vRe;
                    im[i + j + len / 2] = uIm - vIm;
                    const double nextWRe = wRe * wlenRe - wIm * wlenIm;
                    const double nextWIm = wRe * wlenIm + wIm * wlenRe;
                    wRe = nextWRe;
                    wIm = nextWIm;
                }
            }
        }
    }

    // Measure the allpass magnitude response via an 8192-point FFT.
    void checkAllpassMagnitude()
    {
        g_section = "allpass_magnitude";
        constexpr int kN = 8192;
        constexpr int dOut = 40;
        constexpr int dIn = 25;

        const float testCoeffs[] = {0.3f, 0.6f, 0.78f};
        for (const float g : testCoeffs)
        {
            MarsDSP::Diffusion::NestedAllpass nap;
            nap.prepare(dOut, dIn);
            nap.setCoefficients(g, 0.85f * g);
            nap.setDelays(static_cast<float>(dOut), static_cast<float>(dIn));

            std::vector<float> ir(static_cast<std::size_t>(kN), 0.0f);
            ir[0] = 1.0f;
            nap.processBlock(ir.data(), kN);

            std::vector<double> re(kN, 0.0);
            std::vector<double> im(kN, 0.0);
            for (int i = 0; i < kN; ++i)
                re[static_cast<std::size_t>(i)] = static_cast<double>(ir[static_cast<std::size_t>(i)]);

            fft(re, im);

            double maxErr = 0.0;
            for (int i = 0; i < kN; ++i)
            {
                const double mag = std::sqrt(re[static_cast<std::size_t>(i)] * re[static_cast<std::size_t>(i)]
                                           + im[static_cast<std::size_t>(i)] * im[static_cast<std::size_t>(i)]);
                const double err = std::abs(mag - 1.0);
                if (err > maxErr) maxErr = err;
                CHECK(err < 1e-5);
            }
            std::printf("  coeff %.2f: max |H(w) - 1| = %.2e (gate < 1e-5)\n", static_cast<double>(g), maxErr);
        }
    }

    // Measure the energy arrival centroid of a 200000-sample impulse response.
    void checkEnergyCentroid()
    {
        g_section = "energy_centroid";
        constexpr int kN = 200000;
        constexpr int dOut = 294;
        constexpr int dIn = 182;
        const double expectedCentroid = static_cast<double>(dOut + dIn);

        const float testCoeffs[] = {0.3f, 0.6f, 0.78f};
        for (const float g : testCoeffs)
        {
            MarsDSP::Diffusion::NestedAllpass nap;
            nap.prepare(dOut, dIn);
            nap.setCoefficients(g, 0.85f * g);
            nap.setDelays(static_cast<float>(dOut), static_cast<float>(dIn));
            CHECK(std::abs(nap.centroidSamples() - static_cast<float>(dOut + dIn)) < 1e-6f);

            std::vector<float> ir(static_cast<std::size_t>(kN), 0.0f);
            ir[0] = 1.0f;
            nap.processBlock(ir.data(), kN);

            double energySum = 0.0;
            double weightedTimeSum = 0.0;
            for (int n = 0; n < kN; ++n)
            {
                const double y = static_cast<double>(ir[static_cast<std::size_t>(n)]);
                const double energy = y * y;
                energySum += energy;
                weightedTimeSum += static_cast<double>(n) * energy;
            }
            CHECK(energySum > 0.0);
            const double measuredCentroid = weightedTimeSum / energySum;
            const double delta = std::abs(measuredCentroid - expectedCentroid);
            std::printf("  coeff %.2f: centroid = %.4f samples, expected = %.1f, delta = %.4f (gate <= 0.5)\n",
                        static_cast<double>(g), measuredCentroid, expectedCentroid, delta);
            CHECK(delta <= 0.5);
        }
    }

    // Test stability under 60 s of white noise at coefficients 0.9 and 0.99.
    void checkStability()
    {
        g_section = "stability";
        constexpr int kSampleRate = 48000;
        constexpr int kN = 60 * kSampleRate;
        constexpr int kBlock = 256;
        constexpr int dOut = 200;
        constexpr int dIn = 110;

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> noise(static_cast<std::size_t>(kN));
        for (std::size_t i = 0; i < noise.size(); ++i)
            noise[i] = dist(rng);

        const float testCoeffs[] = {0.9f, 0.99f};
        for (const float g : testCoeffs)
        {
            MarsDSP::Diffusion::NestedAllpass nap;
            nap.prepare(dOut, dIn);
            nap.setCoefficients(g, 0.85f * g);
            nap.setDelays(static_cast<float>(dOut), static_cast<float>(dIn));

            std::vector<float> out = noise;
            for (int off = 0; off < kN; off += kBlock)
            {
                const int m = std::min(kBlock, kN - off);
                nap.processBlock(out.data() + off, m);
            }

            float maxVal = 0.0f;
            double energyLast10s = 0.0;
            constexpr int last10sStart = 50 * kSampleRate;

            for (int i = 0; i < kN; ++i)
            {
                const float val = out[static_cast<std::size_t>(i)];
                CHECK(std::isfinite(val));
                maxVal = std::max(maxVal, std::abs(val));
                if (i >= last10sStart)
                {
                    energyLast10s += static_cast<double>(val) * static_cast<double>(val);
                }
            }
            const double rmsLast10s = std::sqrt(energyLast10s / static_cast<double>(10 * kSampleRate));
            std::printf("  coeff %.2f: peak = %.2f, RMS (final 10 s) = %.4f (all finite, bounded)\n",
                        static_cast<double>(g), static_cast<double>(maxVal), rmsLast10s);
            CHECK(maxVal < 100.0f);
        }
    }

    // Verify bit-exact parity between processRef and processBlock.
    void checkParity()
    {
        g_section = "parity";
        constexpr int kN = 1000000;
        constexpr int dOut = 175;
        constexpr int dIn = 95;

        std::mt19937 rng(67890);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> input(static_cast<std::size_t>(kN));
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = dist(rng);

        const int blockSizes[] = {1, 7, 16, 64};
        for (const int blockSize : blockSizes)
        {
            MarsDSP::Diffusion::NestedAllpass napRef;
            napRef.prepare(dOut, dIn);
            napRef.setCoefficients(0.72f, 0.61f);
            napRef.setDelays(static_cast<float>(dOut), static_cast<float>(dIn));

            MarsDSP::Diffusion::NestedAllpass napBlock;
            napBlock.prepare(dOut, dIn);
            napBlock.setCoefficients(0.72f, 0.61f);
            napBlock.setDelays(static_cast<float>(dOut), static_cast<float>(dIn));

            std::vector<float> outRef = input;
            napRef.processRef(outRef.data(), kN);

            std::vector<float> outBlock = input;
            for (int off = 0; off < kN; off += blockSize)
            {
                const int m = std::min(blockSize, kN - off);
                napBlock.processBlock(outBlock.data() + off, m);
            }

            for (int i = 0; i < kN; ++i)
            {
                if (outRef[static_cast<std::size_t>(i)] != outBlock[static_cast<std::size_t>(i)])
                {
                    FAIL("block size %d mismatch at sample %d: ref=%.8f block=%.8f",
                         blockSize, i, static_cast<double>(outRef[static_cast<std::size_t>(i)]),
                         static_cast<double>(outBlock[static_cast<std::size_t>(i)]));
                }
            }
            std::printf("  block size %2d: %d samples bit-exact\n", blockSize, kN);
        }
    }

    // Count arrivals above -60 dBFS in the first 4800 samples compared to plain allpass.
    double checkDensity()
    {
        g_section = "density";
        constexpr int kN = 4800;
        constexpr int dOut = 777;
        constexpr int dIn = 481;
        constexpr int dPlain = dOut + dIn; // 1258
        constexpr float gOut = 0.78f;
        constexpr float gIn = 0.85f * gOut; // 0.663
        constexpr float thr = 0.001f; // -60 dBFS

        // Nested allpass impulse response.
        MarsDSP::Diffusion::NestedAllpass nap;
        nap.prepare(dOut, dIn);
        nap.setCoefficients(gOut, gIn);
        nap.setDelays(static_cast<float>(dOut), static_cast<float>(dIn));

        std::vector<float> irNested(static_cast<std::size_t>(kN), 0.0f);
        irNested[0] = 1.0f;
        nap.processBlock(irNested.data(), kN);

        int nestedArrivals = 0;
        for (int i = 0; i < kN; ++i)
        {
            if (std::abs(irNested[static_cast<std::size_t>(i)]) >= thr)
                ++nestedArrivals;
        }

        // Plain Schroeder allpass of the same total delay.
        MarsDSP::Delays::Pow2RingBuffer plainRing;
        plainRing.prepare(dPlain + MarsDSP::Delays::Pow2RingBuffer::kTail + 8);
        int plainW = 0;
        std::vector<float> irPlain(static_cast<std::size_t>(kN), 0.0f);
        irPlain[0] = 1.0f;

        for (int s = 0; s < kN; ++s)
        {
            const float x = irPlain[static_cast<std::size_t>(s)];
            const float d = MarsDSP::Delays::FracDelayTap::read(plainRing, plainW, static_cast<float>(dPlain));
            float v = x - gOut * d;
            if (!std::isfinite(v)) v = 0.0f;
            const float y = d + gOut * v;
            plainRing.writeBlock(&v, plainW, 1);
            plainRing.refreshMirror(plainW, 1);
            plainW = (plainW + 1) & plainRing.mask();
            irPlain[static_cast<std::size_t>(s)] = y;
        }

        int plainArrivals = 0;
        for (int i = 0; i < kN; ++i)
        {
            if (std::abs(irPlain[static_cast<std::size_t>(i)]) >= thr)
                ++plainArrivals;
        }

        const double ratio = static_cast<double>(nestedArrivals) / static_cast<double>(std::max(1, plainArrivals));
        std::printf("  arrivals above -60 dBFS in %d samples:\n", kN);
        std::printf("    nested allpass (D_out=%d, D_in=%d): %d\n", dOut, dIn, nestedArrivals);
        std::printf("    plain allpass  (D=%d):             %d\n", dPlain, plainArrivals);
        std::printf("    density ratio: %.2fx (gate >= 3.0x)\n", ratio);

        CHECK(ratio >= 3.0);
        return ratio;
    }
}

int main()
{
    std::printf("=== Chronos nested_allpass_check (S22a) ===\n\n");

    std::printf("1. Allpass Magnitude Check:\n");
    checkAllpassMagnitude();

    std::printf("\n2. Energy Centroid Check:\n");
    checkEnergyCentroid();

    std::printf("\n3. Stability Check:\n");
    checkStability();

    std::printf("\n4. Parity Check:\n");
    checkParity();

    std::printf("\n5. Arrival Density Check:\n");
    const double densityRatio = checkDensity();

    std::printf("\n=== ALL PROPERTIES HELD (Measured Density Ratio = %.2fx) ===\n", densityRatio);
    return 0;
}

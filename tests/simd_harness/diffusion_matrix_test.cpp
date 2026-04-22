// Diffusion mixing-matrix correctness test.
//
// Exercises the two orthogonal mixers used by the diffusion / FDN path:
//
//   - OrthogonalHadamardMixer  : unitary Hadamard (should preserve ||v||).
//   - OrthogonalHouseholderMixer : orthogonal reflection (Householder^2 == I).
//
// Emits a CSV summary to tests/simd_harness/logs/diffusion_matrix.csv for
// the matplotlib dashboard.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "dsp/diffusion/orthogonal_mixing_matrices.h"

namespace fs = std::filesystem;
using namespace MarsDSP::DSP::Diffusion;

namespace
{
    constexpr double kTolerance = 1e-5;
    int g_passes = 0;
    int g_fails  = 0;

    const fs::path kLogDir = "tests/simd_harness/logs";

    void ensureLogDir()
    {
        std::error_code ec;
        fs::create_directories(kLogDir, ec);
    }

    template <std::size_t N>
    double squaredNorm(const std::array<float, N>& channelValues)
    {
        double accumulatedSquaredNorm = 0.0;
        for (std::size_t channelIndex = 0; channelIndex < N; ++channelIndex)
        {
            const double channelValueAsDouble = channelValues[channelIndex];
            accumulatedSquaredNorm += channelValueAsDouble * channelValueAsDouble;
        }
        return accumulatedSquaredNorm;
    }

    template <std::size_t N>
    std::array<float, N> randomVector(std::mt19937& randomEngine)
    {
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::array<float, N> randomVector{};
        for (std::size_t channelIndex = 0; channelIndex < N; ++channelIndex)
            randomVector[channelIndex] = distribution(randomEngine);
        return randomVector;
    }

    template <std::size_t N>
    void testHadamardPreservesNorm(std::ofstream& csvSink,
                                   std::mt19937&  randomEngine)
    {
        auto channelValues = randomVector<N>(randomEngine);
        const double inputSquaredNorm  = squaredNorm(channelValues);

        OrthogonalHadamardMixer<float, N>::applyInPlace(channelValues.data());
        const double outputSquaredNorm = squaredNorm(channelValues);

        const double relativeDelta =
            std::fabs(outputSquaredNorm - inputSquaredNorm)
            / std::max(inputSquaredNorm, 1e-12);

        const bool ok = relativeDelta < kTolerance;
        ok ? ++g_passes : ++g_fails;
        std::cout << "  Hadamard<N=" << N << "> norm preservation: "
                  << (ok ? "PASS" : "FAIL")
                  << "  |dN|/N = " << relativeDelta << "\n";
        csvSink << "Hadamard," << N << ","
                << inputSquaredNorm << "," << outputSquaredNorm << ","
                << relativeDelta << "," << (ok ? 1 : 0) << "\n";
    }

    template <std::size_t N>
    void testHouseholderPreservesNorm(std::ofstream& csvSink,
                                      std::mt19937&  randomEngine)
    {
        auto channelValues = randomVector<N>(randomEngine);
        const double inputSquaredNorm  = squaredNorm(channelValues);

        OrthogonalHouseholderMixer<float, N>::applyInPlace(channelValues.data());
        const double outputSquaredNorm = squaredNorm(channelValues);

        const double relativeDelta =
            std::fabs(outputSquaredNorm - inputSquaredNorm)
            / std::max(inputSquaredNorm, 1e-12);

        const bool ok = relativeDelta < kTolerance;
        ok ? ++g_passes : ++g_fails;
        std::cout << "  Householder<N=" << N << "> norm preservation: "
                  << (ok ? "PASS" : "FAIL")
                  << "  |dN|/N = " << relativeDelta << "\n";
        csvSink << "Householder," << N << ","
                << inputSquaredNorm << "," << outputSquaredNorm << ","
                << relativeDelta << "," << (ok ? 1 : 0) << "\n";
    }

    template <std::size_t N>
    void testHouseholderIsInvolution(std::mt19937& randomEngine)
    {
        // Applying the Householder reflector twice must return the input.
        auto originalChannelValues = randomVector<N>(randomEngine);
        auto workingChannelValues  = originalChannelValues;

        OrthogonalHouseholderMixer<float, N>::applyInPlace(workingChannelValues.data());
        OrthogonalHouseholderMixer<float, N>::applyInPlace(workingChannelValues.data());

        double maxAbsoluteRoundTripError = 0.0;
        for (std::size_t channelIndex = 0; channelIndex < N; ++channelIndex)
        {
            const double roundTripError =
                std::fabs(workingChannelValues[channelIndex]
                          - originalChannelValues[channelIndex]);
            if (roundTripError > maxAbsoluteRoundTripError)
                maxAbsoluteRoundTripError = roundTripError;
        }

        const bool ok = maxAbsoluteRoundTripError < 1e-5;
        ok ? ++g_passes : ++g_fails;
        std::cout << "  Householder<N=" << N << "> involution check: "
                  << (ok ? "PASS" : "FAIL")
                  << "  max|roundtrip| = " << maxAbsoluteRoundTripError << "\n";
    }

    template <std::size_t N>
    void testHadamardIsInvolution(std::mt19937& randomEngine)
    {
        // H * H = I for the unitary Hadamard (H is symmetric and its own
        // inverse up to the 1/sqrt(N) normalisation we've already baked in).
        auto originalChannelValues = randomVector<N>(randomEngine);
        auto workingChannelValues  = originalChannelValues;

        OrthogonalHadamardMixer<float, N>::applyInPlace(workingChannelValues.data());
        OrthogonalHadamardMixer<float, N>::applyInPlace(workingChannelValues.data());

        double maxAbsoluteRoundTripError = 0.0;
        for (std::size_t channelIndex = 0; channelIndex < N; ++channelIndex)
        {
            const double roundTripError =
                std::fabs(workingChannelValues[channelIndex]
                          - originalChannelValues[channelIndex]);
            if (roundTripError > maxAbsoluteRoundTripError)
                maxAbsoluteRoundTripError = roundTripError;
        }

        const bool ok = maxAbsoluteRoundTripError < 1e-5;
        ok ? ++g_passes : ++g_fails;
        std::cout << "  Hadamard<N=" << N << "> involution check:    "
                  << (ok ? "PASS" : "FAIL")
                  << "  max|roundtrip| = " << maxAbsoluteRoundTripError << "\n";
    }
} // namespace

int main()
{
    ensureLogDir();
    std::ofstream csvSink(kLogDir / "diffusion_matrix.csv");
    csvSink << "mixer,channel_count,input_norm_sq,output_norm_sq,relative_delta,passed\n";

    std::mt19937 randomEngine(0xC0FFEE42u);

    std::cout << "\n[diffusion matrix] norm-preservation tests\n";
    testHadamardPreservesNorm<2>  (csvSink, randomEngine);
    testHadamardPreservesNorm<4>  (csvSink, randomEngine);
    testHadamardPreservesNorm<8>  (csvSink, randomEngine);
    testHadamardPreservesNorm<16> (csvSink, randomEngine);

    testHouseholderPreservesNorm<2>  (csvSink, randomEngine);
    testHouseholderPreservesNorm<4>  (csvSink, randomEngine);
    testHouseholderPreservesNorm<8>  (csvSink, randomEngine);
    testHouseholderPreservesNorm<16> (csvSink, randomEngine);

    std::cout << "\n[diffusion matrix] involution (H*H = I) checks\n";
    testHadamardIsInvolution<4>  (randomEngine);
    testHadamardIsInvolution<8>  (randomEngine);
    testHouseholderIsInvolution<4>  (randomEngine);
    testHouseholderIsInvolution<8>  (randomEngine);
    testHouseholderIsInvolution<16> (randomEngine);

    csvSink.close();
    std::cout << "\nSummary:  passed=" << g_passes
              << "  failed=" << g_fails << "\n";
    return g_fails == 0 ? 0 : 1;
}

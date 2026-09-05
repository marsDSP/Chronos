/**
 * Correctness harness for DiffusionModel (rev G7 appendix A).
 * Prepares a real Diffuser, feeds one impulse, collects 2 s, and gates
 * the model sigma1 and transport against the measured energy spread and centroid.
 * Links SharedCode only (DiffusionModel.h is header-only and JUCE-free).
 */

#include "gui/tap/DiffusionModel.h"
#include "dsp/Diffuser.h"
#include "utils/memory/BumpArena.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <print>
#include <vector>

namespace {

const char* g_section = "(startup)";

#define CHECK(cond) \
    do { if (!(cond)) { std::println("FAIL [{}] {}:{}: {}", g_section, __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

#define FAIL(...) \
    do { std::print("FAIL [{}] ", g_section); std::println(__VA_ARGS__); std::exit(1); } while (0)

// Collect the Diffuser impulse response for one channel.
std::vector<float> diffuserImpulse(MarsDSP::Diffusion::Diffuser& d, float size, float coef, int sr)
{
    d.setSize(size);
    d.setDiffusion(coef);
    d.prime();

    constexpr int kSeconds = 2;
    const int n = kSeconds * sr;
    std::vector<float> in(static_cast<std::size_t>(n), 0.0f);
    in[0] = 1.0f;
    std::vector<float> out(static_cast<std::size_t>(n), 0.0f);

    MarsDSP::Memory::BumpArena arena(MarsDSP::Diffusion::Diffuser::ringStorageFloats(static_cast<double>(sr)) * sizeof(float) + 1024);
    d.prepare(static_cast<double>(sr), arena);
    // Mono: left in, left out. Right is null.
    d.processBlock(in.data(), nullptr, n);
    std::copy(in.begin(), in.begin() + n, out.begin());
    return out;
}

// Energy centroid and standard deviation of a response.
struct Stats { float centroid; float sigma; };
Stats energyStats(const std::vector<float>& h, int sr)
{
    double totalE = 0.0, sumT = 0.0, sumT2 = 0.0;
    for (int i = 0; i < static_cast<int>(h.size()); ++i)
    {
        const double e = static_cast<double>(h[static_cast<std::size_t>(i)]) * static_cast<double>(h[static_cast<std::size_t>(i)]);
        totalE += e;
        const double t = static_cast<double>(i) / static_cast<double>(sr);
        sumT  += e * t;
        sumT2 += e * t * t;
    }
    if (totalE < 1e-12)
        return { 0.0f, 0.0f };
    const double mean = sumT / totalE;
    const double var = std::max(sumT2 / totalE - mean * mean, 0.0);
    return { static_cast<float>(mean), static_cast<float>(std::sqrt(var)) };
}

int runAll()
{
    constexpr int sr = 48000;
    MarsDSP::GUI::DiffusionModel model(sr);

    const float diffs[] = { 0.25f, 0.5f, 0.75f, 1.0f };
    const float sizes[] = { 0.0f, 0.5f, 1.0f };

    std::println("diffusion  size   sigma1_model   sigma1_meas   ratio   centroid_model  centroid_meas  centroid_dev%");
    std::println("---------- ----  ------------   ------------   -----   -------------  -------------  -------------");

    for (const float diff : diffs)
    {
        for (const float size : sizes)
        {
            g_section = "table";

            MarsDSP::Diffusion::Diffuser d;
            const auto hL = diffuserImpulse(d, size, MarsDSP::Diffusion::Diffuser::kMaxCoefficient * diff, sr);
            const auto st = energyStats(hL, sr);

            const float sigmaModel = model.sigma1Seconds(diff, size);
            const float transportModel = model.transportSeconds(size);

            // sigma1(0, s) == 0 for every size.
            if (diff <= 0.001f)
            {
                CHECK(sigmaModel == 0.0f);
                CHECK(st.sigma < 1e-4f);
            }

            // The ratio of measured to model sigma1 must be in [0.5, 2.0].
            if (diff > 0.001f && st.sigma > 1e-4f)
            {
                const float ratio = st.sigma / sigmaModel;
                if (ratio < 0.5f || ratio > 2.0f)
                    FAIL("diff={} size={} sigma1 ratio {} outside [0.5, 2.0] (model={} meas={})",
                         diff, size, ratio, sigmaModel, st.sigma);
            }

            // The centroid deviation must be within 10 percent.
            if (diff > 0.001f && transportModel > 1e-4f)
            {
                const float dev = std::fabs(st.centroid - transportModel) / transportModel;
                if (dev > 0.10f)
                    FAIL("diff={} size={} centroid dev {} outside 10% (model={} meas={})",
                         diff, size, dev, transportModel, st.centroid);
            }

            std::println("{:.2f}      {:.2f}   {:.6}      {:.6}      {:.3f}   {:.6}      {:.6}      {:.3f}",
                diff, size, sigmaModel, st.sigma,
                (sigmaModel > 1e-6f ? st.sigma / sigmaModel : 0.0f),
                transportModel, st.centroid,
                (transportModel > 1e-6f ? std::fabs(st.centroid - transportModel) / transportModel : 0.0f));
        }
    }

    // Monotonic non-decreasing in diffusion at fixed size, and in size at fixed diffusion.
    g_section = "monotonic";
    {
        for (const float size : sizes)
        {
            float prev = -1.0f;
            for (const float diff : diffs)
            {
                const float s = model.sigma1Seconds(diff, size);
                if (s < prev - 1e-9f)
                    FAIL("size={}: sigma1 decreased at diff {} ({} < {})", size, diff, s, prev);
                prev = s;
            }
        }
        for (const float diff : diffs)
        {
            float prev = -1.0f;
            for (const float size : sizes)
            {
                const float s = model.sigma1Seconds(diff, size);
                if (s < prev - 1e-9f)
                    FAIL("diff={}: sigma1 decreased at size {} ({} < {})", diff, size, s, prev);
                prev = s;
            }
        }
        std::println("monotonic non-decreasing in diffusion and size: PASS");
    }

    std::println("diffusion_halo_check: ALL PASS");
    return 0;
}

} // namespace

int main()
{
    std::println("=== Chronos DiffusionModel harness ===");
    std::println();
    const int r = runAll();
    std::println();
    std::println("=== {} ===", r == 0 ? "ALL PROPERTIES HELD" : "PROPERTY FAILED");
    return r;
}

#pragma once

#ifndef CHRONOS_DIFFUSER_H
#define CHRONOS_DIFFUSER_H


#include "FracDelayTap.h"
#include "LinearSmoother.h"
#include "Modulation.h"
#include "NestedAllpass.h"
#include "Pow2RingBuffer.h"
#include "simd/Config.h"
#include "utils/memory/BumpArena.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numbers>

namespace MarsDSP::Diffusion
{
    class Diffuser
    {
    public:
        static constexpr int kNumPlainSections = 3;
        static constexpr int kNumNestedSections = 3;
        static constexpr int kNumSections = 6;
        static constexpr float kMaxCoefficient = 0.78f;
        // Per-section coefficient taper. Plain sections 0..2, nested 3..5.
        static constexpr std::array<float, kNumSections> kSectionGain{
            1.00f, 0.97f, 0.94f, 0.91f, 0.88f, 0.85f
        };
        // Per-section rate spread for the plain sections.
        static constexpr std::array<float, kNumPlainSections> kRateSpread{
            1.000f, 0.773f, 1.317f
        };

        static constexpr std::uint64_t kModSeed = 0x9E3779B97F4A7C15uLL;
        static constexpr float kMaxSizeCut = 0.55f; // size 0 cuts the path by 55%; size 1 is the full path
        static constexpr int kChunk = 16; // block-vectorized chunk
        static constexpr float kMinDelay = 32.0f; // MUST exceed kChunk: a chunk's
        // reads must not touch that same
        // chunk's writes.

        static constexpr float kTotalMetersL = 20.95125f;
        static constexpr float kTotalMetersR = 21.07500f;
        static constexpr int kNumDelaysPerBank = 9;

        // Ratio table for acoustic budget partition.
        // Slots 0..2 are plain allpass sections.
        // Slots 3..8 are nested allpass pairs (outer, inner).
        static constexpr std::array<float, kNumDelaysPerBank> kPathRatios{
            0.0396f, // 0: plain
            0.0567f, // 1: plain
            0.0787f, // 2: plain
            0.1002f, // 3: nest 0 outer
            0.0620f, // 4: nest 0 inner
            0.1456f, // 5: nest 1 outer
            0.0883f, // 6: nest 1 inner
            0.2649f, // 7: nest 2 outer
            0.1640f  // 8: nest 2 inner
        };

        void prepare(double sampleRate) noexcept
        {
            prepareImpl_(sampleRate, nullptr);
        }

        void prepare(double sampleRate, Memory::BumpArena &arena) noexcept
        {
            prepareImpl_(sampleRate, &arena);
        }

        static int modHeadroomFor(double sampleRate) noexcept
        {
            return std::max(128, static_cast<int>(std::ceil(0.002 * sampleRate)));
        }

        static std::size_t ringStorageFloats(double sampleRate) noexcept
        {
            const int headroom = modHeadroomFor(sampleRate);

            std::array<int, kNumDelaysPerBank> lenL{};
            std::array<int, kNumDelaysPerBank> lenR{};

            computeSectionLens(sampleRate, lenL.data(), lenR.data());
            std::size_t total = 0;
            for (int i = 0; i < kNumDelaysPerBank; ++i)
            {
                total += Delays::Pow2RingBuffer::arenaFloatsFor(lenL[static_cast<std::size_t>(i)] + headroom + Delays::Pow2RingBuffer::kTail + 8);
                total += Delays::Pow2RingBuffer::arenaFloatsFor(lenR[static_cast<std::size_t>(i)] + headroom + Delays::Pow2RingBuffer::kTail + 8);
            }
            return total;
        }

        void prime() noexcept
        {
            for (int b = 0; b < 2; ++b)
            {
                auto &bank = (b == 0) ? secL_ : secR_;
                for (int i = 0; i < kNumPlainSections; ++i)
                {
                    auto &s = bank.plain[static_cast<std::size_t>(i)];
                    s.ring.clear();
                    s.w = 0;
                    s.ou.reset();

                    const std::uint64_t stream =
                        static_cast<std::uint64_t>(b) * kNumPlainSections +
                        static_cast<std::uint64_t>(i) + 1u;
                    s.rng.seed(kModSeed, stream);
                }
                for (int i = 0; i < kNumNestedSections; ++i)
                {
                    bank.nested[static_cast<std::size_t>(i)].reset();
                }
            }
            sizeSm_.setCurrentAndTargetValue(sizeSm_.getTargetValue());
            coefSm_.setCurrentAndTargetValue(coefSm_.getTargetValue());
            depthSm_.setCurrentAndTargetValue(depthSm_.getTargetValue());
        }

        void reset() noexcept
        {
            prime();
        }

        void setDiffusion(float amount01) noexcept
        {
            coefSm_.setTargetValue(kMaxCoefficient * std::clamp(amount01, 0.0f, 1.0f));
        }

        void setSize(float size01) noexcept
        {
            sizeSm_.setTargetValue(std::clamp(size01, 0.0f, 1.0f));
        }

        void setModDepthSamples(float depth) noexcept
        {
            depthSm_.setTargetValue(std::clamp(depth, 0.0f, static_cast<float>(kModHeadroom_ - 2)));
        }

        void setModRateHz(float hz) noexcept
        {
            const double f = std::clamp(static_cast<double>(hz), 0.0, 8.0);
            for (auto *bank: {&secL_, &secR_})
                for (int i = 0; i < kNumPlainSections; ++i)
                    bank->plain[static_cast<std::size_t>(i)].ou.setRate(
                        sampleRate_, f * static_cast<double>(kRateSpread[static_cast<std::size_t>(i)]));
        }

        void processBlock(float *left, float *right, int n) noexcept
        {
            assert(left != nullptr);
            for (int off = 0; off < n; off += kChunk)
            {
                const int m = std::min(kChunk, n - off);

                for (int j = 0; j < m; ++j)
                {
                    sizeRamp_[static_cast<std::size_t>(j)] = sizeSm_.getNextValue();
                    gRamp_[static_cast<std::size_t>(j)] = std::clamp(coefSm_.getNextValue(), -kMaxCoefficient, kMaxCoefficient);
                    depthRamp_[static_cast<std::size_t>(j)] = depthSm_.getNextValue();
                }

                chunk_(secL_, left + off, m, sectionLenCache_.lenL);
                if (right != nullptr) chunk_(secR_, right + off, m, sectionLenCache_.lenR);
            }
        }

        void processBlockRef(float *left, float *right, int n) noexcept
        {
            assert(left != nullptr);
            for (int s = 0; s < n; ++s)
            {
                const float size = sizeSm_.getNextValue();
                const float g = std::clamp(coefSm_.getNextValue(), -kMaxCoefficient, kMaxCoefficient);
                const float depth = depthSm_.getNextValue();

                left[s] = chain_(secL_, left[s], size, g, depth, sectionLenCache_.lenL);
                if (right != nullptr) right[s] = chain_(secR_, right[s], size, g, depth, sectionLenCache_.lenR);
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept { return 0; }

        [[nodiscard]] int sectionLenL(int i) const noexcept { return sectionLenCache_.lenL[static_cast<std::size_t>(i)]; }
        [[nodiscard]] int sectionLenR(int i) const noexcept { return sectionLenCache_.lenR[static_cast<std::size_t>(i)]; }
        [[nodiscard]] float getSizeCurrent() const noexcept { return sizeSm_.getCurrentValue(); }
        [[nodiscard]] float getCoefCurrent() const noexcept { return coefSm_.getCurrentValue(); }

        static constexpr float sectionSign(int i) noexcept
        {
            return (i & 1) != 0 ? -1.0f : 1.0f;
        }

        // Return the section length in samples for the size control.
        // A size of 0 gives the shortest path. A size of 1 gives the full path.
        [[nodiscard]] static float effLen(float lenF, float size01) noexcept
        {
            return lenF * (1.0f - kMaxSizeCut * (1.0f - size01));
        }

        // Return the largest OU state in sigmas across all sections.
        // The state stays inside kClamp sigmas by construction.
        [[nodiscard]] float ouStateMaxSigma() const noexcept
        {
            float maxSig = 0.0f;
            for (const auto &s: secL_.plain)
                maxSig = std::max(maxSig, static_cast<float>(std::fabs(s.ou.state())));
            for (const auto &s: secR_.plain)
                maxSig = std::max(maxSig, static_cast<float>(std::fabs(s.ou.state())));
            return maxSig;
        }

    private:
        static constexpr double kSpeedOfSoundMps = 343.0;
        static constexpr int kMaxPrimeScan = 1 << 16;
        int kModHeadroom_ = 128;

        static bool isPrime_(int v) noexcept
        {
            if (v < 2) return false;
            if (v % 2 == 0) return v == 2;
            for (int d = 3; d * d <= v; d += 2)
                if (v % d == 0) return false;
            return true;
        }

        // Find the nearest unused prime to the given length.
        static int distinctPrimeNear_(int want, std::bitset<kMaxPrimeScan> &used) noexcept
        {
            want = std::clamp(want, 5, kMaxPrimeScan - 2);
            for (int d = 0; d < kMaxPrimeScan; ++d)
            {
                for (const int cand: {want - d, want + d})
                {
                    if (cand >= 5 && cand < kMaxPrimeScan && isPrime_(cand) && !used.test(static_cast<size_t>(cand)))
                    {
                        used.set(static_cast<size_t>(cand));
                        return cand;
                    }
                }
            }
            return want | 1; // unreachable at sane rates
        }

    public:
        // Compute the section lengths from the acoustic path tables.
        // The prepare path and the arena size query call this function.
        // The cache holds the result so the prime scan runs once per rate.
        static void computeSectionLens(double sampleRate, int *outL, int *outR) noexcept
        {
            if (sectionLenCache_.valid && sectionLenCache_.sr == sampleRate)
            {
                std::copy_n(sectionLenCache_.lenL.data(), kNumDelaysPerBank, outL);
                std::copy_n(sectionLenCache_.lenR.data(), kNumDelaysPerBank, outR);
                return;
            }

            const double samplesPerMeter = sampleRate / kSpeedOfSoundMps;
            sectionLenCache_.used.reset();
            for (int i = 0; i < kNumDelaysPerBank; ++i)
            {
                const auto wantL = static_cast<int>(std::lround(
                    static_cast<double>(kTotalMetersL * kPathRatios[static_cast<std::size_t>(i)]) * samplesPerMeter));
                const auto wantR = static_cast<int>(std::lround(
                    static_cast<double>(kTotalMetersR * kPathRatios[static_cast<std::size_t>(i)]) * samplesPerMeter));
                sectionLenCache_.lenL[static_cast<std::size_t>(i)] = distinctPrimeNear_(wantL, sectionLenCache_.used);
                sectionLenCache_.lenR[static_cast<std::size_t>(i)] = distinctPrimeNear_(wantR, sectionLenCache_.used);
            }
            sectionLenCache_.sr = sampleRate;
            sectionLenCache_.valid = true;
            std::copy_n(sectionLenCache_.lenL.data(), kNumDelaysPerBank, outL);
            std::copy_n(sectionLenCache_.lenR.data(), kNumDelaysPerBank, outR);
        }

    private:

        void prepareImpl_(double sampleRate, Memory::BumpArena *arena) noexcept
        {
            assert(sampleRate > 0.0);
            sampleRate_ = sampleRate;

            int lenL[kNumDelaysPerBank];
            int lenR[kNumDelaysPerBank];
            computeSectionLens(sampleRate, lenL, lenR);

            kModHeadroom_ = modHeadroomFor(sampleRate);
            for (auto *bank: {&secL_, &secR_})
            {
                const auto &len = (bank == &secL_) ? lenL : lenR;
                for (int i = 0; i < kNumPlainSections; ++i)
                {
                    auto &s = bank->plain[static_cast<std::size_t>(i)];
                    s.len = len[i];
                    const int minCap = s.len + kModHeadroom_ + Delays::Pow2RingBuffer::kTail + 8;
                    if (arena != nullptr) s.ring.prepare(minCap, *arena);
                    else s.ring.prepare(minCap);
                }
                for (int i = 0; i < kNumNestedSections; ++i)
                {
                    const int dOut = len[3 + 2 * i];
                    const int dIn = len[4 + 2 * i];
                    const int minCapOut = dOut + kModHeadroom_;
                    const int minCapIn = dIn + kModHeadroom_;
                    auto &nest = bank->nested[static_cast<std::size_t>(i)];
                    if (arena != nullptr) nest.prepare(minCapOut, minCapIn, *arena);
                    else nest.prepare(minCapOut, minCapIn);
                    nest.setDelays(static_cast<float>(dOut), static_cast<float>(dIn));
                }
            }

            sizeSm_.reset(sampleRate, 0.050);
            coefSm_.reset(sampleRate, 0.020);
            depthSm_.reset(sampleRate, 0.050);
            setModRateHz(0.5f);
            reset();
        }

        struct Section
        {
            Delays::Pow2RingBuffer ring;
            Mod::OrnsteinUhlenbeck ou;
            Mod::Pcg32 rng;
            int len = 0;
            int w = 0;
        };

        struct Bank
        {
            std::array<Section, kNumPlainSections> plain{};
            std::array<NestedAllpass, kNumNestedSections> nested{};
        };

    public:
        [[nodiscard]] std::array<float, 2> baseTransportSamplesLR(float size01) const noexcept
        {
            const float s = std::clamp(size01, 0.0f, 1.0f);
            auto sumBank = [&](const Bank &, const std::array<int, kNumDelaysPerBank> &len) noexcept
            {
                float sum = 0.0f;
                for (int i = 0; i < kNumPlainSections; ++i)
                {
                    const auto lenF = static_cast<float>(len[static_cast<std::size_t>(i)]);
                    float eff = effLen(lenF, s);
                    eff = std::nearbyintf(eff);
                    eff = std::clamp(eff, kMinDelay, lenF);
                    sum += eff;
                }
                for (int i = 0; i < kNumNestedSections; ++i)
                {
                    const auto lenOutF = static_cast<float>(len[static_cast<std::size_t>(3 + 2 * i)]);
                    const auto lenInF = static_cast<float>(len[static_cast<std::size_t>(4 + 2 * i)]);
                    float effOut = effLen(lenOutF, s);
                    effOut = std::nearbyintf(effOut);
                    effOut = std::clamp(effOut, kMinDelay, lenOutF);

                    float effIn = effLen(lenInF, s);
                    effIn = std::nearbyintf(effIn);
                    effIn = std::clamp(effIn, kMinDelay, lenInF);

                    sum += (effOut + effIn);
                }
                return sum;
            };
            return {sumBank(secL_, sectionLenCache_.lenL), sumBank(secR_, sectionLenCache_.lenR)};
        }

        [[nodiscard]] float baseTransportSamples(float size01) const noexcept
        {
            const auto lr = baseTransportSamplesLR(size01);
            return 0.5f * (lr[0] + lr[1]);
        }

        [[nodiscard]] float transportSamples() const noexcept { return baseTransportSamples(getSizeCurrent()); }

    private:

        void chunk_(Bank &bank, float *io, int m, const std::array<int, kNumDelaysPerBank> &len) noexcept
        {
            std::memcpy(tmp_.data(), io, static_cast<std::size_t>(m) * sizeof(float));

            for (int i = 0; i < kNumPlainSections; ++i)
            {
                auto &sec = bank.plain[static_cast<std::size_t>(i)];
                const int mask = sec.ring.mask();
                const auto lenF = static_cast<float>(sec.len);
                const float sgn = sectionSign(i);
                const float secGain = kSectionGain[static_cast<std::size_t>(i)];

                for (int j = 0; j < m; ++j)
                {
                    const float gj = sgn * secGain * gRamp_[static_cast<std::size_t>(j)];
                    float eff = effLen(lenF, sizeRamp_[static_cast<std::size_t>(j)]);
                    const float depth = depthRamp_[static_cast<std::size_t>(j)];
                    const float peak = std::min(depth, 0.25f * eff);
                    if (const float mm = peak * sec.ou.next(sec.rng); mm == 0.0f) eff = std::nearbyintf(eff);
                    else eff += mm;
                    eff = std::clamp(eff, kMinDelay, lenF);

                    const float dj = Delays::FracDelayTap::read(sec.ring, sec.w, eff);
                    float vj = tmp_[static_cast<std::size_t>(j)] - gj * dj;
                    if (!std::isfinite(vj)) vj = 0.0f;
                    tmp_[static_cast<std::size_t>(j)] = dj + gj * vj;

                    sec.ring.writeBlock(&vj, sec.w, 1);
                    sec.ring.refreshMirror(sec.w, 1);
                    sec.w = (sec.w + 1) & mask;
                }
            }

            for (int i = 0; i < kNumNestedSections; ++i)
            {
                auto &nest = bank.nested[static_cast<std::size_t>(i)];
                const auto lenOutF = static_cast<float>(len[static_cast<std::size_t>(3 + 2 * i)]);
                const auto lenInF = static_cast<float>(len[static_cast<std::size_t>(4 + 2 * i)]);
                const float sgn = sectionSign(3 + i);
                const float secGain = kSectionGain[static_cast<std::size_t>(3 + i)];

                for (int j = 0; j < m; ++j)
                {
                    const float size = sizeRamp_[static_cast<std::size_t>(j)];
                    float effOut = effLen(lenOutF, size);
                    effOut = std::nearbyintf(effOut);
                    effOut = std::clamp(effOut, kMinDelay, lenOutF);

                    float effIn = effLen(lenInF, size);
                    effIn = std::nearbyintf(effIn);
                    effIn = std::clamp(effIn, kMinDelay, lenInF);

                    nest.setDelays(effOut, effIn);
                    const float gOut = sgn * secGain * gRamp_[static_cast<std::size_t>(j)];
                    const float gIn = 0.85f * gOut;
                    nest.setCoefficients(gOut, gIn);

                    nest.processRef(tmp_.data() + j, 1);
                }
            }

            std::memcpy(io, tmp_.data(), static_cast<std::size_t>(m) * sizeof(float));
        }

        // reference only -- do not optimize, do not delete.
        float chain_(Bank &bank, float x, float size, float g, float depth, const std::array<int, kNumDelaysPerBank> &len) noexcept
        {
            for (int i = 0; i < kNumPlainSections; ++i)
            {
                auto &sec = bank.plain[static_cast<std::size_t>(i)];
                const auto lenF = static_cast<float>(sec.len);
                float eff = effLen(lenF, size);
                const float peak = std::min(depth, 0.25f * eff);
                if (const float mm = peak * sec.ou.next(sec.rng); mm == 0.0f)
                    eff = std::nearbyintf(eff);
                else
                    eff += mm;
                eff = std::clamp(eff, kMinDelay, lenF);

                const float gs = sectionSign(i) * kSectionGain[static_cast<std::size_t>(i)] * g;
                const float d = Delays::FracDelayTap::read(sec.ring, sec.w, eff);
                float v = x - gs * d;
                if (!std::isfinite(v)) v = 0.0f;
                const float y = d + gs * v;

                sec.ring.writeBlock(&v, sec.w, 1);
                sec.ring.refreshMirror(sec.w, 1);
                sec.w = (sec.w + 1) & sec.ring.mask();
                x = y;
            }

            for (int i = 0; i < kNumNestedSections; ++i)
            {
                auto &nest = bank.nested[static_cast<std::size_t>(i)];
                const auto lenOutF = static_cast<float>(len[static_cast<std::size_t>(3 + 2 * i)]);
                const auto lenInF = static_cast<float>(len[static_cast<std::size_t>(4 + 2 * i)]);
                float effOut = effLen(lenOutF, size);
                effOut = std::nearbyintf(effOut);
                effOut = std::clamp(effOut, kMinDelay, lenOutF);

                float effIn = effLen(lenInF, size);
                effIn = std::nearbyintf(effIn);
                effIn = std::clamp(effIn, kMinDelay, lenInF);

                nest.setDelays(effOut, effIn);
                const float gOut = sectionSign(3 + i) * kSectionGain[static_cast<std::size_t>(3 + i)] * g;
                const float gIn = 0.85f * gOut;
                nest.setCoefficients(gOut, gIn);

                nest.processRef(&x, 1);
            }
            return x;
        }

        // chunk scratch, 16-byte aligned for load_ps/store_ps
        alignas(16) std::array<float, kChunk> tmp_{};
        alignas(16) std::array<float, kChunk> gRamp_{};
        alignas(16) std::array<float, kChunk> sizeRamp_{};
        alignas(16) std::array<float, kChunk> depthRamp_{};

        Bank secL_{};
        Bank secR_{};
        double sampleRate_ = 48000.0;

        Smoothers::LinearSmoother<float> sizeSm_;
        Smoothers::LinearSmoother<float> coefSm_;
        Smoothers::LinearSmoother<float> depthSm_;

        // Cache for the section length prime scan. Holds the result per
        // sample rate so the scan runs once. The bitset replaces the old
        // 64 kB stack array. The prepare path is single-threaded.
        struct SectionLenCache
        {
            SectionLenCache() noexcept : sr{0.0}, lenL{}, lenR{}, valid{false} {}
            double sr;
            std::array<int, kNumDelaysPerBank> lenL;
            std::array<int, kNumDelaysPerBank> lenR;
            std::bitset<kMaxPrimeScan> used;
            bool valid;
        };

        static inline SectionLenCache sectionLenCache_{};
    };
}
#endif

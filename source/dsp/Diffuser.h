#pragma once

#ifndef CHRONOS_DIFFUSER_H
#define CHRONOS_DIFFUSER_H

#include "FracDelayTap.h"
#include "LinearSmoother.h"
#include "Modulation.h"
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

namespace MarsDSP::Diffusion {
    class Diffuser {
    public:
        static constexpr int   kNumSections    = 8;
        static constexpr float kMaxCoefficient = 0.78f;
        // Per-section coefficient taper. Each section scales the master
        // coefficient. The effective coefficient stays below the maximum.
        static constexpr std::array<float, kNumSections> kSectionGain{
            1.00f, 0.97f, 0.94f, 0.91f, 0.88f, 0.85f, 0.82f, 0.79f };
        // Per-section rate spread. Each section runs at a distinct fraction
        // of the user rate. This decorrelates the sections.
        static constexpr std::array<float, kNumSections> kRateSpread{
            1.000f, 0.773f, 1.317f, 0.618f, 1.129f, 0.874f, 1.481f, 0.702f };
        static constexpr std::uint64_t kModSeed = 0x9E3779B97F4A7C15ull;
        static constexpr float kMaxSizeCut     = 0.55f; // size 0 cuts the path by 55%; size 1 is the full path
        static constexpr int   kChunk          = 16;    // block-vectorized chunk
        static constexpr float kMinDelay       = 32.0f; // MUST exceed kChunk: a chunk's
                                                        // reads must not touch that same
                                                        // chunk's writes.

        static constexpr std::array<float, kNumSections> kPathMetersL
        {
            4.54125f,
            3.93375f,
            3.19125f,
            2.92875f,
            2.32875f,
            2.01000f,
            1.18875f,
            0.82875f
        };

        static constexpr std::array<float, kNumSections> kPathMetersR
        {
            4.53000f,
            3.92625f,
            3.18375f,
            2.91375f,
            2.33625f,
            1.99875f,
            1.39125f,
            0.79500f
        };

        void prepare(double sampleRate) noexcept
        {
            prepareImpl_(sampleRate, nullptr);
        }

        void prepare(double sampleRate, Memory::BumpArena& arena) noexcept
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
            int lenL[kNumSections];
            int lenR[kNumSections];
            computeSectionLens(sampleRate, lenL, lenR);
            std::size_t total = 0;
            for (int i = 0; i < kNumSections; ++i)
            {
                total += Delays::Pow2RingBuffer::arenaFloatsFor(lenL[i] + headroom + Delays::Pow2RingBuffer::kTail + 8);
                total += Delays::Pow2RingBuffer::arenaFloatsFor(lenR[i] + headroom + Delays::Pow2RingBuffer::kTail + 8);
            }
            return total;
        }

        void prime() noexcept
        {
            for (int b = 0; b < 2; ++b)
            {
                auto& bank = (b == 0) ? secL_ : secR_;
                for (int i = 0; i < kNumSections; ++i)
                {
                    auto& s = bank[static_cast<std::size_t>(i)];
                    s.ring.clear();
                    s.w = 0;
                    s.ou.reset();
                    s.rng.seed(kModSeed, static_cast<std::uint64_t>(b * kNumSections + i + 1));
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
            for (auto* bank : { &secL_, &secR_ })
                for (int i = 0; i < kNumSections; ++i)
                    (*bank)[static_cast<std::size_t>(i)].ou.setRate(
                        sampleRate_, f * static_cast<double>(kRateSpread[static_cast<std::size_t>(i)]));
        }

        void processBlock(float* left, float* right, int n) noexcept
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

                chunk_(secL_, left + off, m);
                if (right != nullptr) chunk_(secR_, right + off, m);
            }
        }

        void processBlockRef(float* left, float* right, int n) noexcept
        {
            assert(left != nullptr);
            for (int s = 0; s < n; ++s)
            {
                const float size  = sizeSm_.getNextValue();
                const float g     = std::clamp(coefSm_.getNextValue(), -kMaxCoefficient, kMaxCoefficient);
                const float depth = depthSm_.getNextValue();

                left[s] = chain_(secL_, left[s], size, g, depth);
                if (right != nullptr) right[s] = chain_(secR_, right[s], size, g, depth);
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept { return 0; }

        [[nodiscard]] float baseTransportSamples(float size01) const noexcept
        {
            const auto lr = baseTransportSamplesLR(size01);
            return 0.5f * (lr[0] + lr[1]);
        }

        [[nodiscard]] std::array<float, 2> baseTransportSamplesLR(float size01) const noexcept
        {
            const float s = std::clamp(size01, 0.0f, 1.0f);
            auto sumBank = [&](const Bank& bank) noexcept -> float
            {
                float sum = 0.0f;
                for (int i = 0; i < kNumSections; ++i)
                {
                    const float lenF = static_cast<float>(bank[static_cast<std::size_t>(i)].len);
                    float eff = effLen(lenF, s);
                    eff = std::nearbyintf(eff);
                    eff = std::clamp(eff, kMinDelay, lenF);
                    sum += eff;
                }
                return sum;
            };
            return { sumBank(secL_), sumBank(secR_) };
        }

        [[nodiscard]] int sectionLenL(int i) const noexcept { return secL_[static_cast<std::size_t>(i)].len; }
        [[nodiscard]] int sectionLenR(int i) const noexcept { return secR_[static_cast<std::size_t>(i)].len; }
        [[nodiscard]] float getSizeCurrent() const noexcept { return sizeSm_.getCurrentValue(); }
        [[nodiscard]] float getCoefCurrent() const noexcept { return coefSm_.getCurrentValue(); }
        [[nodiscard]] float transportSamples() const noexcept { return baseTransportSamples(getSizeCurrent()); }

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
            for (const auto& s : secL_)
                maxSig = std::max(maxSig, static_cast<float>(std::fabs(s.ou.state())));
            for (const auto& s : secR_)
                maxSig = std::max(maxSig, static_cast<float>(std::fabs(s.ou.state())));
            return maxSig;
        }

    private:
        static constexpr double kSpeedOfSoundMps = 343.0;
        static constexpr int    kMaxPrimeScan    = 1 << 16;
        int    kModHeadroom_  = 128;

        // Compute the section lengths from the acoustic path tables.
        // The prepare path and the arena size query call this function.
        // The cache holds the result so the prime scan runs once per rate.
        static void computeSectionLens(double sampleRate,
                                       int* outL, int* outR) noexcept
        {
            if (sectionLenCache_.valid && sectionLenCache_.sr == sampleRate)
            {
                std::copy_n(sectionLenCache_.lenL, kNumSections, outL);
                std::copy_n(sectionLenCache_.lenR, kNumSections, outR);
                return;
            }

            const double samplesPerMeter = sampleRate / kSpeedOfSoundMps;
            sectionLenCache_.used.reset();
            for (int i = 0; i < kNumSections; ++i)
            {
                const int wantL = static_cast<int>(std::lround(static_cast<double>(kPathMetersL[static_cast<std::size_t>(i)]) * samplesPerMeter));
                const int wantR = static_cast<int>(std::lround(static_cast<double>(kPathMetersR[static_cast<std::size_t>(i)]) * samplesPerMeter));
                sectionLenCache_.lenL[i] = distinctPrimeNear_(wantL, sectionLenCache_.used);
                sectionLenCache_.lenR[i] = distinctPrimeNear_(wantR, sectionLenCache_.used);
            }
            sectionLenCache_.sr = sampleRate;
            sectionLenCache_.valid = true;
            std::copy_n(sectionLenCache_.lenL, kNumSections, outL);
            std::copy_n(sectionLenCache_.lenR, kNumSections, outR);
        }

        void prepareImpl_(double sampleRate, Memory::BumpArena* arena) noexcept
        {
            assert(sampleRate > 0.0);
            sampleRate_ = sampleRate;

            int lenL[kNumSections];
            int lenR[kNumSections];
            computeSectionLens(sampleRate, lenL, lenR);
            for (int i = 0; i < kNumSections; ++i)
            {
                secL_[static_cast<std::size_t>(i)].len = lenL[i];
                secR_[static_cast<std::size_t>(i)].len = lenR[i];
            }

            kModHeadroom_ = modHeadroomFor(sampleRate);
            for (auto* bank : { &secL_, &secR_ })
                for (auto& s : *bank)
                {
                    const int minCap = s.len + kModHeadroom_ + Delays::Pow2RingBuffer::kTail + 8;
                    if (arena != nullptr) s.ring.prepare(minCap, *arena);
                    else                  s.ring.prepare(minCap);
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
            int w   = 0;
        };
        using Bank = std::array<Section, kNumSections>;

        static bool isPrime_(int v) noexcept
        {
            if (v < 2) return false;
            if (v % 2 == 0) return v == 2;
            for (int d = 3; d * d <= v; d += 2)
                if (v % d == 0) return false;
            return true;
        }

        // Find the nearest unused prime to the given length.
        static int distinctPrimeNear_(int want, std::bitset<kMaxPrimeScan>& used) noexcept
        {
            want = std::clamp(want, 5, kMaxPrimeScan - 2);
            for (int d = 0; d < kMaxPrimeScan; ++d)
            {
                for (const int cand : { want + d, want - d })
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

        void chunk_(Bank& bank, float* io, int m) noexcept
        {
            std::memcpy(tmp_.data(), io, static_cast<std::size_t>(m) * sizeof(float));

            for (int i = 0; i < kNumSections; ++i)
            {
                auto& sec = bank[static_cast<std::size_t>(i)];
                const int   mask = sec.ring.mask();
                const float lenF = static_cast<float>(sec.len);
                const float sgn = sectionSign(i);
                const float secGain = kSectionGain[static_cast<std::size_t>(i)];

                // Every section takes the exact fractional path. The OU
                // modulates the tap per sample.
                for (int j = 0; j < m; ++j)
                {
                    const float gj = sgn * secGain * gRamp_[static_cast<std::size_t>(j)];
                    float eff = effLen(lenF, sizeRamp_[static_cast<std::size_t>(j)]);
                    const float depth = depthRamp_[static_cast<std::size_t>(j)];
                    const float peak = std::min(depth, 0.25f * eff);
                    const float mm = peak * sec.ou.next(sec.rng);
                    if (mm == 0.0f) eff = std::nearbyintf(eff);
                    else            eff += mm;
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

            std::memcpy(io, tmp_.data(), static_cast<std::size_t>(m) * sizeof(float));
        }

        // reference only -- do not optimize, do not delete.
        float chain_(Bank& bank, float x, float size, float g, float depth) noexcept
        {
            for (int i = 0; i < kNumSections; ++i)
            {
                auto& sec = bank[static_cast<std::size_t>(i)];
                const float lenF = static_cast<float>(sec.len);
                float eff = effLen(lenF, size);
                const float peak = std::min(depth, 0.25f * eff);
                const float mm = peak * sec.ou.next(sec.rng);
                if (mm == 0.0f)
                    eff = std::nearbyintf(eff);
                else
                    eff += mm;
                eff = std::clamp(eff, kMinDelay, lenF);

                // canonical Schroeder: v = x - g*d ; y = d + g*v ; write v.
                const float gs = sectionSign(i) * kSectionGain[static_cast<std::size_t>(i)] * g;
                const float d = Delays::FracDelayTap::read(sec.ring, sec.w, eff);
                float v = x - gs * d;
                if (!std::isfinite(v)) v = 0.0f; // scrub before it recirculates
                const float y = d + gs * v;

                sec.ring.writeBlock(&v, sec.w, 1);
                sec.ring.refreshMirror(sec.w, 1);
                sec.w = (sec.w + 1) & sec.ring.mask();
                x = y;
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
        struct SectionLenCache {
            double sr;
            int lenL[kNumSections];
            int lenR[kNumSections];
            std::bitset<kMaxPrimeScan> used;
            bool valid;
        };
        static inline SectionLenCache sectionLenCache_{};
    };
}
#endif

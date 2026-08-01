#pragma once

#ifndef CHRONOS_DIFFUSER_H
#define CHRONOS_DIFFUSER_H

#include "FracDelayTap.h"
#include "LinearSmoother.h"
#include "Pow2RingBuffer.h"
#include "simd/Config.h"
#include "utils/memory/BumpArena.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numbers>

namespace MarsDSP::Diffusion {

    class Diffuser {
    public:
        static constexpr int   kNumSections    = 8;
        static constexpr float kMaxCoefficient = 0.92f;
        static constexpr float kMaxSizeCut     = 0.90f; // size shortens delay by <= 90%
        static constexpr int   kChunk          = 16;    // block-vectorized chunk
        static constexpr float kMinDelay       = 32.0f; // MUST exceed kChunk: a chunk's
                                                        // reads must not touch that same
                                                        // chunk's writes.
        static constexpr int   kModSectionA    = 2;
        static constexpr int   kModSectionB    = 5;
        static constexpr float kDetuneRatio    = 1.317f;

        // Acoustic path lengths in meters. Short-smear diffusion territory
        // (Dattorro/Schroeder input diffusion): 0.8–4.5 m ≈ 2.4–13 ms per
        // section at 48 kHz, total series delay ≈ 61 ms at size 0 (full
        // length) down to ~7 ms at size 1 (90% cut, floor-clamped). The
        // original tables were 10× longer (8–45 m, 61–613 ms total), which
        // smeared the repeat across a ±300 ms window no tap compensation
        // could hide — the audible arrival drifted with both size and g. At
        // this scale the whole smear is perceptually part of the repeat;
        // the comp (C7c, transportSamples below) anchors the cascade's
        // energy centroid — exactly baseTransportSamples at every g — so
        // repeats stay centroid-locked to the grid at all settings, and the
        // comp clamp (delay < transport) is only reachable below ~65 ms
        // delay times.
        static constexpr std::array<float, kNumSections> kPathMetersL{
            4.54125f, 3.93375f, 3.19125f, 2.92875f, 2.32875f, 2.01000f, 1.18875f, 0.82875f };
        static constexpr std::array<float, kNumSections> kPathMetersR{
            4.53000f, 3.92625f, 3.18375f, 2.91375f, 2.33625f, 1.99875f, 1.39125f, 0.79500f };

        void prepare(double sampleRate) noexcept
        {
            prepareImpl_(sampleRate, nullptr);
        }

        // C9b: carve all 16 section rings from the caller's arena (sized via
        // ringStorageFloats) instead of owning them.
        void prepare(double sampleRate, Memory::BumpArena& arena) noexcept
        {
            prepareImpl_(sampleRate, &arena);
        }

        // floats an arena must supply for all 16 rings: the section lengths
        // come from the same computeSectionLens prepareImpl_ uses, so the
        // carve fits exactly.
        static std::size_t ringStorageFloats(double sampleRate) noexcept
        {
            int lenL[kNumSections], lenR[kNumSections];
            computeSectionLens(sampleRate, lenL, lenR);
            std::size_t total = 0;
            for (int i = 0; i < kNumSections; ++i)
            {
                total += Delays::Pow2RingBuffer::arenaFloatsFor(
                    lenL[i] + kModHeadroom + Delays::Pow2RingBuffer::kTail + 8);
                total += Delays::Pow2RingBuffer::arenaFloatsFor(
                    lenR[i] + kModHeadroom + Delays::Pow2RingBuffer::kTail + 8);
            }
            return total;
        }

        void prime() noexcept
        {
            for (auto* bank : { &secL_, &secR_ })
                for (auto& s : *bank) { s.ring.clear(); s.w = 0; }

            oscAc_ = 1.0; oscAs_ = 0.0;
            oscBc_ = 0.0; oscBs_ = 1.0;
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
            depthSm_.setTargetValue(
                std::clamp(depth, 0.0f, static_cast<float>(kModHeadroom - 2)));
        }

        void setModRateHz(float hz) noexcept
        {
            const double f = std::clamp(static_cast<double>(hz), 0.0, 8.0);
            oscAk_ = 2.0 * std::sin(std::numbers::pi * f / sampleRate_);
            oscBk_ = 2.0 * std::sin(std::numbers::pi * f
                                    * static_cast<double>(kDetuneRatio) / sampleRate_);
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
                    gRamp_[static_cast<std::size_t>(j)] =
                        std::clamp(coefSm_.getNextValue(), -kMaxCoefficient, kMaxCoefficient);
                    const float depth = depthSm_.getNextValue();

                    oscAs_ += oscAk_ * oscAc_;
                    oscAc_ -= oscAk_ * oscAs_;
                    oscBs_ += oscBk_ * oscBc_;
                    oscBc_ -= oscBk_ * oscBs_;

                    modAL_[static_cast<std::size_t>(j)] = depth * static_cast<float>(oscAc_);
                    modBL_[static_cast<std::size_t>(j)] = depth * static_cast<float>(oscBc_);
                    modAR_[static_cast<std::size_t>(j)] = depth * static_cast<float>(oscAs_);
                    modBR_[static_cast<std::size_t>(j)] = depth * static_cast<float>(oscBs_);
                }

                const bool settled =
                    (sizeRamp_[0] == sizeRamp_[static_cast<std::size_t>(m - 1)]);

                chunk_(secL_, left + off, m, settled, modAL_.data(), modBL_.data());
                if (right != nullptr)
                    chunk_(secR_, right + off, m, settled, modAR_.data(), modBR_.data());
            }
        }

        void processBlockRef(float* left, float* right, int n) noexcept
        {
            assert(left != nullptr);
            for (int s = 0; s < n; ++s)
            {
                const float size  = sizeSm_.getNextValue();
                const float g     = std::clamp(coefSm_.getNextValue(),
                                               -kMaxCoefficient, kMaxCoefficient);
                const float depth = depthSm_.getNextValue();

                oscAs_ += oscAk_ * oscAc_;
                oscAc_ -= oscAk_ * oscAs_;
                oscBs_ += oscBk_ * oscBc_;
                oscBc_ -= oscBk_ * oscBs_;

                const float modAL = depth * static_cast<float>(oscAc_);
                const float modBL = depth * static_cast<float>(oscBc_);
                const float modAR = depth * static_cast<float>(oscAs_);
                const float modBR = depth * static_cast<float>(oscBs_);

                left[s] = chain_(secL_, left[s], size, g, modAL, modBL);
                if (right != nullptr)
                    right[s] = chain_(secR_, right[s], size, g, modAR, modBR);
            }
        }

        [[nodiscard]] static constexpr int latencySamples() noexcept { return 0; }

        // base transport — the total delay the 8-section cascade carries
        // at a given size, EXCLUDING modulation (modulation is a per-sample
        // perturbation around this base, not part of the transport). This is
        // the quantity C7c absorbs into the tap position so repeats stay on
        // the tempo grid (see transportSamples: it is the exact energy
        // centroid of the cascade IR at every g). The per-section arithmetic
        // is token-identical to the settled/unmodulated `eff` in chain_
        // (m==0 branch) and chunk_ (fast path): nearbyintf then clamp to
        // [kMinDelay, len]. At diffusion = 0 (g = 0) each section is a pure
        // D-sample delay, so the base transport is the exact series delay;
        // the compensation is sample-exact there.
        //
        // L and R banks differ (~3.8 ms skew at size 0, 48 kHz) because the
        // prime-snapped path lengths differ — intentional decorrelation.
        // baseTransportSamples returns the MEAN of L and R, which preserves the
        // skew (mean-compensation moves the image center, not the L-R offset).
        [[nodiscard]] float baseTransportSamples(float size01) const noexcept
        {
            const auto lr = baseTransportSamplesLR(size01);
            return 0.5f * (lr[0] + lr[1]);
        }

        // Per-bank {L, R} base transport. Each is Σᵢ effᵢ(size) over the 8
        // sections of that bank.
        [[nodiscard]] std::array<float, 2> baseTransportSamplesLR(float size01) const noexcept
        {
            const float s = std::clamp(size01, 0.0f, 1.0f);
            auto sumBank = [&](const Bank& bank) noexcept -> float
            {
                float sum = 0.0f;
                for (int i = 0; i < kNumSections; ++i)
                {
                    const float lenF = static_cast<float>(bank[static_cast<std::size_t>(i)].len);
                    float eff = lenF * (1.0f - kMaxSizeCut * s);
                    eff = std::nearbyintf(eff);
                    eff = std::clamp(eff, kMinDelay, lenF);
                    sum += eff;
                }
                return sum;
            };
            return { sumBank(secL_), sumBank(secR_) };
        }

        // section length accessors (for harness recompute of base
        // transport). Returns the prime-snapped length of section i.
        [[nodiscard]] int sectionLenL(int i) const noexcept { return secL_[static_cast<std::size_t>(i)].len; }
        [[nodiscard]] int sectionLenR(int i) const noexcept { return secR_[static_cast<std::size_t>(i)].len; }

        [[nodiscard]] float getSizeCurrent() const noexcept { return sizeSm_.getCurrentValue(); }

        [[nodiscard]] float getCoefCurrent() const noexcept { return coefSm_.getCurrentValue(); }

        // C7c: per-pass transport for the in-loop/tap compensation. The
        // 8-section cascade's ENERGY centroid is exactly baseTransportSamples
        // at every g (each section's energy centroid is D exactly — the
        // allpass average group delay is g-invariant, and the per-section
        // sign alternation below changes phases, not energies). Anchoring
        // the comp to the exact centroid (rather than the old w = 1-g^8
        // energy-MEDIAN estimate) is what makes the loop period exact per
        // pass at every diffusion setting: repeat n's centroid lands on
        // n*delay for all g, while the blob widens ~sqrt(n) (the wash).
        [[nodiscard]] float transportSamples() const noexcept
        {
            return baseTransportSamples(getSizeCurrent());
        }

        // Per-section Schroeder coefficient sign (signalsmith/Dattorro
        // polarity-flip port): alternating section signs break up the
        // regular phase reinforcement of the cascade (metallic edge) at
        // zero cost. The sign changes each section's phase response but
        // NOT its energy distribution (h -> ±h per arrival), so |H| = 1,
        // stability (|g_i| < 1), and the D-exact energy centroid all hold
        // — the comp above is unaffected.
        static constexpr float sectionSign(int i) noexcept
        {
            return (i & 1) != 0 ? -1.0f : 1.0f;
        }

    private:
        static constexpr double kSpeedOfSoundMps = 343.0;
        static constexpr int    kModHeadroom     = 64;
        static constexpr int    kMaxPrimeScan    = 1 << 16;

        // prime-snapped section lengths from the acoustic path tables.
        // Shared by prepareImpl_ (the rings) and ringStorageFloats (the
        // arena size query) so the two can never drift.
        static void computeSectionLens(double sampleRate,
                                       int* outL, int* outR) noexcept
        {
            const double samplesPerMeter = sampleRate / kSpeedOfSoundMps;
            bool used[kMaxPrimeScan] = {};
            for (int i = 0; i < kNumSections; ++i)
            {
                const int wantL = static_cast<int>(
                    std::lround(static_cast<double>(kPathMetersL[static_cast<std::size_t>(i)]) * samplesPerMeter));
                const int wantR = static_cast<int>(
                    std::lround(static_cast<double>(kPathMetersR[static_cast<std::size_t>(i)]) * samplesPerMeter));
                outL[i] = distinctPrimeNear_(wantL, used);
                outR[i] = distinctPrimeNear_(wantR, used);
            }
        }

        void prepareImpl_(double sampleRate, Memory::BumpArena* arena) noexcept
        {
            assert(sampleRate > 0.0);
            sampleRate_ = sampleRate;

            int lenL[kNumSections], lenR[kNumSections];
            computeSectionLens(sampleRate, lenL, lenR);
            for (int i = 0; i < kNumSections; ++i)
            {
                secL_[static_cast<std::size_t>(i)].len = lenL[i];
                secR_[static_cast<std::size_t>(i)].len = lenR[i];
            }

            for (auto* bank : { &secL_, &secR_ })
                for (auto& s : *bank)
                {
                    const int minCap = s.len + kModHeadroom + Delays::Pow2RingBuffer::kTail + 8;
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

        static int distinctPrimeNear_(int want, bool (&used)[kMaxPrimeScan]) noexcept
        {
            want = std::clamp(want, 5, kMaxPrimeScan - 2);
            for (int d = 0; d < kMaxPrimeScan; ++d)
            {
                for (const int cand : { want + d, want - d })
                {
                    if (cand >= 5 && cand < kMaxPrimeScan
                        && isPrime_(cand) && !used[cand])
                    {
                        used[cand] = true;
                        return cand;
                    }
                }
            }
            return want | 1; // unreachable at sane rates
        }

        void chunk_(Bank& bank, float* io, int m, bool settled,
                    const float* modA, const float* modB) noexcept
        {
            std::memcpy(tmp_.data(), io, static_cast<std::size_t>(m) * sizeof(float));

            for (int i = 0; i < kNumSections; ++i)
            {
                auto& sec = bank[static_cast<std::size_t>(i)];
                const int   mask = sec.ring.mask();
                const float lenF = static_cast<float>(sec.len);
                const bool  isMod = (i == kModSectionA || i == kModSectionB);
                const float sgn = sectionSign(i);

                if (settled && !isMod)
                {
                    // ---- fast path: constant integer tap, 4-wide ----
                    float eff = lenF * (1.0f - kMaxSizeCut * sizeRamp_[0]);
                    eff = std::clamp(std::nearbyintf(eff), kMinDelay, lenF);
                    const int D = static_cast<int>(eff);
                    const int base = (sec.w - D) & mask;

                    const float* d = sec.ring.windowPtr(base, m);
                    if (d == nullptr)
                    {
                        sec.ring.readWindow(rd_.data(), base, m);
                        d = rd_.data();
                    }

                    const M128 sgnv = MM(set1_ps)(sgn);
                    const int mv = m & ~3;
                    for (int j = 0; j < mv; j += 4)
                    {
                        const M128 dv = MM(loadu_ps)(d + j);
                        const M128 gv = MM(mul_ps)(sgnv, MM(load_ps)(gRamp_.data() + j));
                        const M128 xv = MM(load_ps)(tmp_.data() + j);
                        const M128 vv = MM(sub_ps)(xv, MM(mul_ps)(gv, dv));
                        const M128 yv = MM(add_ps)(dv, MM(mul_ps)(gv, vv));
                        MM(store_ps)(wr_.data() + j, vv);
                        MM(store_ps)(tmp_.data() + j, yv);
                    }
                    for (int j = mv; j < m; ++j)
                    {
                        const float dj = d[j];
                        const float gj = sgn * gRamp_[static_cast<std::size_t>(j)];
                        const float vj = tmp_[static_cast<std::size_t>(j)] - gj * dj;
                        wr_[static_cast<std::size_t>(j)] = vj;
                        tmp_[static_cast<std::size_t>(j)] = dj + gj * vj;
                    }
                    for (int j = 0; j < m; ++j)
                        if (!std::isfinite(wr_[static_cast<std::size_t>(j)]))
                            wr_[static_cast<std::size_t>(j)] = 0.0f;

                    sec.ring.writeBlock(wr_.data(), sec.w, m);
                    sec.ring.refreshMirror(sec.w, m);
                    sec.w = (sec.w + m) & mask;
                }
                else
                {
                    // ---- exact path: per-sample fractional read ----
                    for (int j = 0; j < m; ++j)
                    {
                        const float gj = sgn * gRamp_[static_cast<std::size_t>(j)];
                        float eff = lenF * (1.0f - kMaxSizeCut * sizeRamp_[static_cast<std::size_t>(j)]);
                        const float mm = (i == kModSectionA) ? modA[j]
                                       : (i == kModSectionB) ? modB[j]
                                                             : 0.0f;
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
            }

            std::memcpy(io, tmp_.data(), static_cast<std::size_t>(m) * sizeof(float));
        }

        // reference only -- do not optimize, do not delete.
        float chain_(Bank& bank, float x, float size, float g,
                     float modA, float modB) noexcept
        {
            for (int i = 0; i < kNumSections; ++i)
            {
                auto& sec = bank[static_cast<std::size_t>(i)];
                const float lenF = static_cast<float>(sec.len);
                float eff = lenF * (1.0f - kMaxSizeCut * size);

                const float m = (i == kModSectionA) ? modA
                              : (i == kModSectionB) ? modB
                                                    : 0.0f;
                if (m == 0.0f)
                    eff = std::nearbyintf(eff);
                else
                    eff += m;
                eff = std::clamp(eff, kMinDelay, lenF);

                // canonical Schroeder: v = x - g*d ; y = d + g*v ; write v.
                const float gs = sectionSign(i) * g;
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
        alignas(16) std::array<float, kChunk> wr_{};
        alignas(16) std::array<float, kChunk> rd_{};
        alignas(16) std::array<float, kChunk> gRamp_{};
        alignas(16) std::array<float, kChunk> sizeRamp_{};
        alignas(16) std::array<float, kChunk> modAL_{};
        alignas(16) std::array<float, kChunk> modBL_{};
        alignas(16) std::array<float, kChunk> modAR_{};
        alignas(16) std::array<float, kChunk> modBR_{};

        Bank secL_{};
        Bank secR_{};
        double sampleRate_ = 48000.0;

        Smoothers::LinearSmoother<float> sizeSm_;
        Smoothers::LinearSmoother<float> coefSm_;
        Smoothers::LinearSmoother<float> depthSm_;

        // quadrature LFOs: double state (see header note 6), (c, s) pairs.
        double oscAc_ = 1.0;
        double oscAs_ = 0.0;
        double oscAk_ = 0.0;
        double oscBc_ = 0.0;
        double oscBs_ = 1.0;
        double oscBk_ = 0.0;
    };
}
#endif

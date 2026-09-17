#pragma once

#ifndef CHRONOS_FFT_H
#define CHRONOS_FFT_H

// Vendored from FFT2.h (MarsDSP MathOps): a planned complex mixed-radix FFT
// with a power-of-two SoA Stockham engine on the 128-bit MM()/M128 lane
// layer, plus a real FFT packed through one half-size complex transform.
// Unnormalised: inv(fwd(x)) == N * x. GUI-thread use in Chronos (the
// spectrum analyser): setSize allocates, so it is not for the audio path.
// The only local edit is the config header default below.

#include <bit>
#include <cmath>       // std::cos / std::sin in setSize(); the original relied on a transitive include
#include <complex>
#include <vector>
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <numbers>
#include <new>
#include <limits>
#include <type_traits>

// ═══════════════════════════════════════════════════════════════════════════
//  SIMD backend
//  ───────────────────────────────────────────────────────────────────────
//  Was: xsimd::batch<ST>. Now: MarsCore Config.h (MM(x) / M128 / M128D),
//  which resolves to native SSE2..SSE4.1 on x86 and to SIMDe (NEON, WASM,
//  scalar) everywhere else. Vector width is fixed at 128 bits: 4 floats or
//  2 doubles.
//
//  Point this at wherever Config.h actually lives in your tree, e.g.
//      -DMARSDSP_FFT_CONFIG_HEADER="\"marscore/simd/Config.h\""
// ═══════════════════════════════════════════════════════════════════════════
#ifndef MARSDSP_FFT_CONFIG_HEADER
#  define MARSDSP_FFT_CONFIG_HEADER "simd/Config.h"
#endif
#include MARSDSP_FFT_CONFIG_HEADER

// Config.h pulls in emmintrin/pmmintrin/smmintrin only (SSE2/3/4.1). The
// FMA3 intrinsics (_mm_fmadd_pd, _mm_fnmadd_ps, ...) live in immintrin.h,
// so on native x86 we need it explicitly when FMA is on.
#if defined(MARSCORE_SIMD_NATIVE_X86) && (defined(__FMA__) || defined(__AVX2__))
#  include <immintrin.h>
#endif

// ───────────────────────────────────────────────────────────────────────────
//  MARSDSP_FFT_HAS_FMA
//  1 -> the twiddle butterfly uses a single-rounding fused multiply-add.
//  0 -> it uses a rounded product followed by a rounded add.
//  These are NOT bit-identical to each other. This mirrors exactly what
//  xsimd did: xsimd::fma/fnma lower to vfmadd/vfnmadd when the target arch
//  carries FMA and degrade to `x*y+z` / `-x*y+z` when it does not.
//  Define it yourself to override the detection.
// ───────────────────────────────────────────────────────────────────────────
#ifndef MARSDSP_FFT_HAS_FMA
#  if defined(MARSCORE_SIMD_ARM64)
#    define MARSDSP_FFT_HAS_FMA 1   // FMA is unconditional on AArch64
#  elif defined(__FMA__) || defined(SIMDE_X86_FMA_NATIVE)
#    define MARSDSP_FFT_HAS_FMA 1
#  elif defined(_MSC_VER) && defined(__AVX2__)
#    define MARSDSP_FFT_HAS_FMA 1   // MSVC has no __FMA__; /arch:AVX2 implies it
#  else
#    define MARSDSP_FFT_HAS_FMA 0
#  endif
#endif

// ───────────────────────────────────────────────────────────────────────────
//  MARSDSP_FFT_MIN_SIMD_BYTES
//  A stage takes the vector path when Ns >= MARSDSP_FFT_MIN_SIMD_BYTES/sizeof(ST),
//  else it falls through to scalar. The xsimd code spelled this
//  `if (Ns >= B::size)`, so the threshold tracked whatever register width the
//  build had. Default 16 == this backend's native 128 bits.
//
//  Under FMA the vector and scalar butterflies are not bit-identical (one
//  rounding vs two), so *which* stages vectorize is observable. Set this to
//  the register width of the build you are reproducing:
//      16 -> SSE / NEON  (float W=4, double W=2)   [default]
//      32 -> AVX / AVX2  (float W=8, double W=4)
//      64 -> AVX-512     (float W=16, double W=8)
//  Setting 32 makes float output bit-identical to an -mavx2 -mfma xsimd build.
// ───────────────────────────────────────────────────────────────────────────
#ifndef MARSDSP_FFT_MIN_SIMD_BYTES
#  define MARSDSP_FFT_MIN_SIMD_BYTES 16
#endif

// ───────────────────────────────────────────────────────────────────────────
//  MARSDSP_FFT_FUSE_FNMA
//  Whether the real-part butterfly `c - a*b` is fused. Default 1.
//
//  Only reason this is a knob: xsimd (through at least 13.x) specialises
//  fma for NEON -> vfmaq_f32/f64, but never specialises fnma, so on AArch64
//  it silently degraded to the generic `-x*y + z`. The result is an
//  asymmetric butterfly -- fused on the imaginary part, unfused on the real
//  part -- and a real component that carries one extra rounding. On x86 with
//  FMA3, xsimd specialises both, so this only ever mattered on ARM.
//
//  Default 1 makes the two halves of the butterfly agree on every target.
//  Set 0 only to reproduce an AArch64 xsimd build bit-for-bit.
// ───────────────────────────────────────────────────────────────────────────
#ifndef MARSDSP_FFT_FUSE_FNMA
#  define MARSDSP_FFT_FUSE_FNMA 1
#endif

namespace MarsDSP::MathOps {
#if defined(QUOTIENT_FFT_DISABLE_SIMD)
    inline constexpr bool haveSIMD = false;
#else
    inline constexpr bool haveSIMD = true;
#endif
    // Retained so downstream `if constexpr (haveXSIMD)` sites keep compiling.
    // It no longer says anything about xsimd; it means "a vector path exists".
    inline constexpr bool haveXSIMD = haveSIMD;

    namespace detail {
        template <bool conjugateSecond, typename V>
        inline std::complex<V> complexMul(const std::complex<V> &a, const std::complex<V> &b)
        {
            const V ar = a.real();
            const V ai = a.imag();
            const V br = b.real();
            const V bi = conjugateSecond ? -b.imag() : b.imag();
            return std::complex<V>{ar * br - ai * bi, ar * bi + ai * br};
        }

        template <bool flipped, typename V>
        inline std::complex<V> complexAddI(const std::complex<V> &a, const std::complex<V> &b)
        {
            if constexpr (flipped)
                return std::complex<V>{a.real() + b.imag(), a.imag() - b.real()};
            else
                return std::complex<V>{a.real() - b.imag(), a.imag() + b.real()};
        }

        // --------------------------------------------------------------------
        //  Aligned allocator (replaces xsimd::aligned_allocator)
        // --------------------------------------------------------------------
        template <class T, std::size_t Align = 64>
        struct AlignedAllocator {
            static_assert(Align >= alignof(T) && (Align & (Align - 1)) == 0,
                          "Align must be a power of two and at least alignof(T)");
            using value_type = T;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using propagate_on_container_move_assignment = std::true_type;
            using is_always_equal = std::true_type;

            template <class U> struct rebind { using other = AlignedAllocator<U, Align>; };

            AlignedAllocator() noexcept = default;
            template <class U>
            AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

            [[nodiscard]] T *allocate(const std::size_t n)
            {
                if (n == 0) return nullptr;
                if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
                    throw std::bad_array_new_length();
                return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t{Align}));
            }

            void deallocate(T *p, const std::size_t n) noexcept
            {
                ::operator delete(p, n * sizeof(T), std::align_val_t{Align});
            }
        };

        template <class T, class U, std::size_t A>
        bool operator==(const AlignedAllocator<T, A> &, const AlignedAllocator<U, A> &) noexcept { return true; }
        template <class T, class U, std::size_t A>
        bool operator!=(const AlignedAllocator<T, A> &, const AlignedAllocator<U, A> &) noexcept { return false; }

        // --------------------------------------------------------------------
        //  128-bit lane ops over Config.h's MM() / M128 / M128D
        //
        //  Every function below is a literal transcription of the xsimd
        //  kernel it replaces, so the emitted rounding sequence is unchanged:
        //    add/sub/mul  -> the same single IEEE-754 op per lane
        //    neg          -> xor with the sign mask (xsimd_sse2.hpp neg)
        //    fma (a,b,c)  -> vfmadd  : a*b+c one rounding   | else mul then add
        //    fnma(a,b,c)  -> vfnmadd : c-a*b one rounding   | else (-a)*b then add
        //  The unfused fallbacks reproduce xsimd's generic forms `x*y+z` and
        //  `-x*y+z` including the leading negate, which matters for signed zero.
        // --------------------------------------------------------------------
        template <typename T> struct SimdOps { static constexpr bool available = false; };

#ifndef QUOTIENT_FFT_DISABLE_SIMD
        template <> struct SimdOps<float> {
            static constexpr bool available = true;
            static constexpr std::size_t width = 4;
            using vec = M128;

            static inline vec loada(const float *p)          { return MM(load_ps)(p); }
            static inline vec loadu(const float *p)          { return MM(loadu_ps)(p); }
            static inline void storea(float *p, const vec v) { MM(store_ps)(p, v); }

            static inline vec add(const vec a, const vec b) { return MM(add_ps)(a, b); }
            static inline vec sub(const vec a, const vec b) { return MM(sub_ps)(a, b); }
            static inline vec mul(const vec a, const vec b) { return MM(mul_ps)(a, b); }

            static inline vec neg(const vec a)
            {
                return MM(xor_ps)(a, MM(castsi128_ps)(MM(set1_epi32)(static_cast<int>(0x80000000u))));
            }

            // a*b + c
            static inline vec fma(const vec a, const vec b, const vec c)
            {
#if MARSDSP_FFT_HAS_FMA
                return MM(fmadd_ps)(a, b, c);
#else
                return add(mul(a, b), c);
#endif
            }

            // c - a*b
            static inline vec fnma(const vec a, const vec b, const vec c)
            {
#if MARSDSP_FFT_HAS_FMA && MARSDSP_FFT_FUSE_FNMA
                return MM(fnmadd_ps)(a, b, c);
#else
                return add(mul(neg(a), b), c);
#endif
            }
        };

        template <> struct SimdOps<double> {
            static constexpr bool available = true;
            static constexpr std::size_t width = 2;
            using vec = M128D;

            static inline vec loada(const double *p)          { return MM(load_pd)(p); }
            static inline vec loadu(const double *p)          { return MM(loadu_pd)(p); }
            static inline void storea(double *p, const vec v) { MM(store_pd)(p, v); }

            static inline vec add(const vec a, const vec b) { return MM(add_pd)(a, b); }
            static inline vec sub(const vec a, const vec b) { return MM(sub_pd)(a, b); }
            static inline vec mul(const vec a, const vec b) { return MM(mul_pd)(a, b); }

            static inline vec neg(const vec a)
            {
                return MM(xor_pd)(a, MM(castsi128_pd)(MM(setr_epi32)(0, static_cast<int>(0x80000000u),
                                                                     0, static_cast<int>(0x80000000u))));
            }

            static inline vec fma(const vec a, const vec b, const vec c)
            {
#if MARSDSP_FFT_HAS_FMA
                return MM(fmadd_pd)(a, b, c);
#else
                return add(mul(a, b), c);
#endif
            }

            static inline vec fnma(const vec a, const vec b, const vec c)
            {
#if MARSDSP_FFT_HAS_FMA && MARSDSP_FFT_FUSE_FNMA
                return MM(fnmadd_pd)(a, b, c);
#else
                return add(mul(neg(a), b), c);
#endif
            }
        };
#endif // QUOTIENT_FFT_DISABLE_SIMD

        // --------------------------------------------------------------------
        //  Power-of-two SOA Stockham autosort FFT
        // --------------------------------------------------------------------
        template <typename ST>
        class SOAPow2FFT {
        public:
            using Complex = std::complex<ST>;
            [[nodiscard]] std::size_t size() const { return N; }

            void setSize(const std::size_t n)
            {
                N = n;
                aRe.assign(n, ST(0)); aIm.assign(n, ST(0));
                bRe.assign(n, ST(0)); bIm.assign(n, ST(0));
                schedule.clear();
                twRe.clear(); twIm.clear();
                tw4Re.clear(); tw4Im.clear();

                std::size_t Ns = 1;
                while (Ns*4 <= n)
                {
                    const std::size_t off = tw4Re.size();
                    for (auto r {1uz}; r <= 3; ++r)
                    {
                        for (auto k {0uz}; k < Ns; ++k)
                        {
                            const double th = -twoPI * static_cast<double>(r) *
                                                       static_cast<double>(k) /
                                                       static_cast<double>(4*Ns);

                            tw4Re.push_back(static_cast<ST>(std::cos(th)));
                            tw4Im.push_back(static_cast<ST>(std::sin(th)));
                        }
                    }
                    schedule.push_back({4, Ns, off});
                    Ns *= 4;
                }

                if (Ns < n)
                {
                    const std::size_t off = twRe.size();
                    for (auto k {0uz}; k < Ns; ++k)
                    {
                        const double th = -twoPI * static_cast<double>(k) /
                                                   static_cast<double>(2*Ns);

                        twRe.push_back(static_cast<ST>(std::cos(th)));
                        twIm.push_back(static_cast<ST>(std::sin(th)));
                    }
                    schedule.push_back({2, Ns, off});
                }
            }

            void fwd(const Complex *in, Complex *out) { run<false>(in, out); }
            void inv(const Complex *in, Complex *out) { run<true>(in, out); }

        private:
            std::size_t N = 0;
            static constexpr double twoPI = 2.0 * std::numbers::pi_v<double>;

            struct StageInfo
            {
                std::size_t radix;
                std::size_t Ns;
                std::size_t twOff;
            };

            std::vector<StageInfo> schedule;

            template <class U> using Alloc = AlignedAllocator<U, 64>;
            using AVec = std::vector<ST, Alloc<ST>>;

            using Ops = SimdOps<ST>;
            static constexpr bool vectorize = Ops::available;
            static constexpr std::size_t W = vectorize ? Ops::width : 1;
            static constexpr std::size_t minSimdNs =
                (MARSDSP_FFT_MIN_SIMD_BYTES / sizeof(ST)) > W ? (MARSDSP_FFT_MIN_SIMD_BYTES / sizeof(ST)) : W;

            AVec twRe;
            AVec twIm;  // radix-2 cleanup twiddles (single stage, Ns = N/2)
            AVec tw4Re;
            AVec tw4Im; // radix-4 twiddles: per stage [tw1|tw2|tw3] over k
            AVec aRe;
            AVec aIm;
            AVec bRe;
            AVec bIm;

            template <bool inverse>
            void run(const Complex *in, Complex *out)
            {
                for (auto i {0uz}; i < N; ++i)
                {
                    aRe[i] = in[i].real();
                    aIm[i] = in[i].imag();
                }
                ST *sr = aRe.data();
                ST *si = aIm.data();
                ST *dr = bRe.data();
                ST *di = bIm.data();

                for (const StageInfo &st : schedule)
                {
                    if (st.radix == 4) stage4<inverse>(sr, si, dr, di, st.Ns, st.twOff);
                    else stage<inverse>(sr, si, dr, di, st.Ns, st.twOff);
                    std::swap(sr, dr);
                    std::swap(si, di);
                }
                for (auto i {0uz}; i < N; ++i) out[i] = Complex(sr[i], si[i]);
            }

            template <bool inverse>
            void stage(const ST *__restrict sr, const ST *__restrict si, ST *__restrict dr, ST *__restrict di,
                       const std::size_t Ns,    const std::size_t off)
            {
                const std::size_t half = N >> 1;
                const std::size_t nblk = half / Ns;
                const ST *twR = twRe.data();
                const ST *twI = twIm.data();

                if constexpr (vectorize)
                {
                    using V = typename Ops::vec;
                    if (Ns >= minSimdNs)
                    {
                        for (auto blk {0uz}; blk < nblk; ++blk)
                        {
                            const std::size_t jb = blk * Ns;
                            const std::size_t ob = blk * 2 * Ns;
                            for (auto k {0uz}; k < Ns; k += W)
                            {
                                const std::size_t j = jb + k;
                                const V v0r = Ops::loada(sr + j);
                                const V v0i = Ops::loada(si + j);
                                const V ar = Ops::loada(sr + j + half);
                                const V ai = Ops::loada(si + j + half);
                                const V wr = Ops::loadu(twR + off + k);
                                V wi = Ops::loadu(twI + off + k);
                                if constexpr (inverse) wi = Ops::neg(wi);
                                const V v1r = Ops::fnma(ai, wi, Ops::mul(ar, wr));
                                const V v1i = Ops::fma(ai, wr, Ops::mul(ar, wi));
                                Ops::storea(dr + ob + k, Ops::add(v0r, v1r));
                                Ops::storea(di + ob + k, Ops::add(v0i, v1i));
                                Ops::storea(dr + ob + k + Ns, Ops::sub(v0r, v1r));
                                Ops::storea(di + ob + k + Ns, Ops::sub(v0i, v1i));
                            }
                        }
                        return;
                    }
                }
                // scalar
                for (auto blk {0uz}; blk < nblk; ++blk)
                {
                    const std::size_t jb = blk * Ns;
                    const std::size_t ob = blk * 2 * Ns;
                    for (auto k {0uz}; k < Ns; ++k)
                    {
                        const std::size_t j = jb + k;
                        const ST v0r = sr[j];
                        const ST v0i = si[j];
                        const ST ar = sr[j + half];
                        const ST ai = si[j + half];
                        const ST wr = twR[off + k];
                        const ST wi = inverse ? -twI[off + k] : twI[off + k];
                        const ST v1r = ar * wr - ai * wi;
                        const ST v1i = ar * wi + ai * wr;
                        dr[ob + k] = v0r + v1r; di[ob + k] = v0i + v1i;
                        dr[ob + k + Ns] = v0r - v1r; di[ob + k + Ns] = v0i - v1i;
                    }
                }
            }

            template <bool inverse>
            void stage4(const ST *__restrict sr, const ST *__restrict si, ST *__restrict dr, ST *__restrict di,
                        const std::size_t Ns,    const std::size_t off)
            {
                const std::size_t q = N >> 2;                // N/4 read stride
                const std::size_t nblk = q / Ns;
                const ST *w1R = tw4Re.data() + off;          const ST *w1I = tw4Im.data() + off;
                const ST *w2R = tw4Re.data() + off + Ns;     const ST *w2I = tw4Im.data() + off + Ns;
                const ST *w3R = tw4Re.data() + off + 2 * Ns; const ST *w3I = tw4Im.data() + off + 2 * Ns;

                // setSize() emits radix-4 stages at Ns = 1, 4, 16, 64, ... so the
                // only sub-4 stage is Ns == 1. W is 2 (double) or 4 (float), hence
                // every stage that vectorized at W = 2 also vectorizes at W = 4 and
                // vice versa -- no stage changes hands between the two widths.
                if constexpr (vectorize)
                {
                    using V = typename Ops::vec;
                    if (Ns >= minSimdNs)
                    {
                        for (auto blk {0uz}; blk < nblk; ++blk)
                        {
                            const std::size_t jb = blk * Ns;
                            const std::size_t ob = blk * 4 * Ns;
                            for (auto k {0uz}; k < Ns; k += W)
                            {
                                const std::size_t j = jb + k;
                                const V a0r = Ops::loada(sr + j);
                                const V a0i = Ops::loada(si + j);
                                const V a1r = Ops::loada(sr + j + q);
                                const V a1i = Ops::loada(si + j + q);
                                const V a2r = Ops::loada(sr + j + 2 * q), a2i = Ops::loada(si + j + 2 * q);
                                const V a3r = Ops::loada(sr + j + 3 * q), a3i = Ops::loada(si + j + 3 * q);
                                V w1r = Ops::loadu(w1R + k), w1i = Ops::loadu(w1I + k);
                                V w2r = Ops::loadu(w2R + k), w2i = Ops::loadu(w2I + k);
                                V w3r = Ops::loadu(w3R + k), w3i = Ops::loadu(w3I + k);
                                if constexpr (inverse) { w1i = Ops::neg(w1i); w2i = Ops::neg(w2i); w3i = Ops::neg(w3i); }
                                const V v1r = Ops::fnma(a1i, w1i, Ops::mul(a1r, w1r)), v1i = Ops::fma(a1i, w1r, Ops::mul(a1r, w1i));
                                const V v2r = Ops::fnma(a2i, w2i, Ops::mul(a2r, w2r)), v2i = Ops::fma(a2i, w2r, Ops::mul(a2r, w2i));
                                const V v3r = Ops::fnma(a3i, w3i, Ops::mul(a3r, w3r)), v3i = Ops::fma(a3i, w3r, Ops::mul(a3r, w3i));
                                const V s02r = Ops::add(a0r, v2r), s02i = Ops::add(a0i, v2i);
                                const V d02r = Ops::sub(a0r, v2r), d02i = Ops::sub(a0i, v2i);
                                const V s13r = Ops::add(v1r, v3r), s13i = Ops::add(v1i, v3i);
                                const V d13r = Ops::sub(v1r, v3r), d13i = Ops::sub(v1i, v3i);
                                Ops::storea(dr + ob + k, Ops::add(s02r, s13r));
                                Ops::storea(di + ob + k, Ops::add(s02i, s13i));
                                Ops::storea(dr + ob + k + 2 * Ns, Ops::sub(s02r, s13r));
                                Ops::storea(di + ob + k + 2 * Ns, Ops::sub(s02i, s13i));
                                if constexpr (!inverse)
                                {
                                    Ops::storea(dr + ob + k + Ns, Ops::add(d02r, d13i));
                                    Ops::storea(di + ob + k + Ns, Ops::sub(d02i, d13r));
                                    Ops::storea(dr + ob + k + 3 * Ns, Ops::sub(d02r, d13i));
                                    Ops::storea(di + ob + k + 3 * Ns, Ops::add(d02i, d13r));
                                }
                                else
                                {
                                    Ops::storea(dr + ob + k + Ns, Ops::sub(d02r, d13i));
                                    Ops::storea(di + ob + k + Ns, Ops::add(d02i, d13r));
                                    Ops::storea(dr + ob + k + 3 * Ns, Ops::add(d02r, d13i));
                                    Ops::storea(di + ob + k + 3 * Ns, Ops::sub(d02i, d13r));
                                }
                            }
                        }
                        return;
                    }
                }
                // scalar
                for (auto blk {0uz}; blk < nblk; ++blk)
                {
                    const std::size_t jb = blk * Ns;
                    const std::size_t ob = blk * 4 * Ns;
                    for (auto k {0uz}; k < Ns; ++k)
                    {
                        const std::size_t j = jb + k;
                        const ST a0r = sr[j];
                        const ST a0i = si[j];
                        const ST a1r = sr[j + q];
                        const ST a1i = si[j + q];
                        const ST a2r = sr[j + 2 * q];
                        const ST a2i = si[j + 2 * q];
                        const ST a3r = sr[j + 3 * q];
                        const ST a3i = si[j + 3 * q];
                        const ST w1r = w1R[k];
                        const ST w1i = inverse ? -w1I[k] : w1I[k];
                        const ST w2r = w2R[k];
                        const ST w2i = inverse ? -w2I[k] : w2I[k];
                        const ST w3r = w3R[k];
                        const ST w3i = inverse ? -w3I[k] : w3I[k];
                        const ST v1r = a1r * w1r - a1i * w1i;
                        const ST v1i = a1r * w1i + a1i * w1r;
                        const ST v2r = a2r * w2r - a2i * w2i;
                        const ST v2i = a2r * w2i + a2i * w2r;
                        const ST v3r = a3r * w3r - a3i * w3i;
                        const ST v3i = a3r * w3i + a3i * w3r;
                        const ST s02r = a0r + v2r;
                        const ST s02i = a0i + v2i;
                        const ST d02r = a0r - v2r;
                        const ST d02i = a0i - v2i;
                        const ST s13r = v1r + v3r;
                        const ST s13i = v1i + v3i;
                        const ST d13r = v1r - v3r;
                        const ST d13i = v1i - v3i;
                        dr[ob + k] = s02r + s13r;
                        di[ob + k] = s02i + s13i;
                        dr[ob + k + 2 * Ns] = s02r - s13r;
                        di[ob + k + 2 * Ns] = s02i - s13i;
                        if constexpr (!inverse)
                        {
                            dr[ob + k + Ns] = d02r + d13i;
                            di[ob + k + Ns] = d02i - d13r;
                            dr[ob + k + 3 * Ns] = d02r - d13i;
                            di[ob + k + 3 * Ns] = d02i + d13r;
                        }
                        else
                        {
                            dr[ob + k + Ns] = d02r - d13i;
                            di[ob + k + Ns] = d02i + d13r;
                            dr[ob + k + 3 * Ns] = d02r + d13i;
                            di[ob + k + 3 * Ns] = d02i - d13r;
                        }
                    }
                }
            }
        };
    }

    // ------------------------------------------------------------------------
    //  Complex mixed-radix planned FFT
    //
    //  Unnormalised: inv(fwd(x)) == N * x. Any length is accepted; the plan
    //  factorises N and emits radix-2/3/4 stages plus a generic radix-p stage
    //  for whatever is left over. Powers of two are routed to the SoA/SIMD
    //  engine above unless that is disabled at construction.
    // ------------------------------------------------------------------------
    template <typename ST = double>
    class FFT {
    public:
        using Complex = std::complex<ST>;

        FFT() = default;
        explicit FFT(const std::size_t size, const bool preferSimdPow2 = true)
        {
            setSize(size, preferSimdPow2);
        }

        [[nodiscard]] std::size_t size() const { return fftSize; }

        std::size_t setSize(const std::size_t size, const bool preferSimdPow2 = true)
        {
            fftSize = size;
            planLength = size;
            usePow2 = false;
            if (size <= 1)   // nothing to plan; run() handles these directly
            {
                stages.clear(); reorderTable.clear();
                twiddleTable.clear(); rootTable.clear();
                radixFactors.clear(); scratch.clear();
                return fftSize;
            }
            usePow2 = preferSimdPow2 && std::has_single_bit(size);
            if (usePow2) soa.setSize(size);
            else buildPlan();
            return fftSize;
        }

        // Out-of-place, no allocation. in and out must not alias.
        void fwd(const Complex *in, Complex *out) const { run<false>(in, out); }
        void inv(const Complex *in, Complex *out) const { run<true>(in, out); }

        [[nodiscard]] std::vector<Complex> fwd(const std::vector<Complex> &x) const
        {
            assert(x.size() == fftSize && "input size must match FFT size!");
            std::vector<Complex> out(fftSize);
            run<false>(x.data(), out.data());
            return out;
        }

        [[nodiscard]] std::vector<Complex> inv(const std::vector<Complex> &X) const
        {
            assert(X.size() == fftSize && "input size must match FFT size!");
            std::vector<Complex> out(fftSize);
            run<true>(X.data(), out.data());
            return out;
        }

    private:
        // `soa` and `scratch` are working state, not observable state, so the
        // transforms stay const. One instance is therefore NOT safe to share
        // across threads -- give each thread its own.
        mutable detail::SOAPow2FFT<ST> soa;      // power-of-two SoA/SIMD fast path
        bool usePow2 = false;

        enum class StageKind
        {
            mixed, radix2, radix3, radix4
        };

        struct Stage
        {
            StageKind kind;
            std::size_t radix;
            std::size_t offset;
            std::size_t innerCount;
            std::size_t outerCount;
            std::size_t twiddleOffset;
            std::size_t rootOffset;      // into rootTable, generic stages only
        };

        struct ReorderEntry
        {
            std::size_t outputIndex;
            std::size_t inputIndex;
        };

        std::size_t fftSize = 0;
        std::size_t planLength = 0;
        std::vector<std::size_t> radixFactors;
        std::vector<Complex> twiddleTable;
        std::vector<Complex> rootTable;
        std::vector<ReorderEntry> reorderTable;
        std::vector<Stage> stages;
        mutable std::vector<Complex> scratch;

        static constexpr double twoPI = 2 * std::numbers::pi_v<double>;

        template <bool inverse>
        void run(const Complex *in, Complex *out) const
        {
            if (fftSize == 0) return;
            assert(in != out && "FFT is out-of-place; in and out must not alias");
            if (fftSize == 1) { out[0] = in[0]; return; }
            if (usePow2) { if constexpr (inverse) soa.inv(in, out); else soa.fwd(in, out); return; }
            execute<inverse>(in, out);
        }

        void buildPlan()
        {
            radixFactors.clear();
            {
                std::size_t remaining = planLength;
                std::size_t candidate = 2;
                while (remaining > 1)
                {
                    if (remaining % candidate == 0) { radixFactors.push_back(candidate); remaining /= candidate; }
                    else if (candidate * candidate > remaining) { candidate = remaining; }
                    else { ++candidate; }
                }
            }

            stages.clear();
            twiddleTable.clear();
            rootTable.clear();
            if (planLength > 1) appendStages(0, 0, planLength, 1);

            std::size_t maxRadix = 0;
            for (const Stage &s : stages) maxRadix = std::max(maxRadix, s.radix);
            scratch.assign(maxRadix, Complex{});

            reorderTable.clear();
            reorderTable.push_back(ReorderEntry{0, 0});
            std::size_t lowFactorIdx = 0;
            std::size_t highFactorIdx = radixFactors.size();
            std::size_t inStrideLow = planLength;
            std::size_t outStrideLow = 1;
            std::size_t inStrideHigh = 1;
            std::size_t outStrideHigh = planLength;

            while (outStrideLow * inStrideHigh < planLength)
            {
                std::size_t radix;
                std::size_t inStride;
                std::size_t outStride;
                if (outStrideLow <= inStrideHigh)
                {
                    radix = radixFactors[lowFactorIdx++];
                    inStride = (inStrideLow /= radix);
                    outStride = outStrideLow;
                    outStrideLow *= radix;
                }
                else
                {
                    radix = radixFactors[--highFactorIdx];
                    inStride = inStrideHigh;
                    inStrideHigh *= radix;
                    outStrideHigh /= radix;
                    outStride = outStrideHigh;
                }
                const std::size_t prevCount = reorderTable.size();
                for (auto i {1uz}; i < radix; ++i)
                    for (auto j {0uz}; j < prevCount; ++j)
                    {
                        ReorderEntry entry = reorderTable[j];
                        entry.outputIndex += i * inStride;
                        entry.inputIndex += i * outStride;
                        reorderTable.push_back(entry);
                    }
            }
            assert(reorderTable.size() == planLength && "reorder table must cover every sample");
        }

        void appendStages(std::size_t factorIndex, const std::size_t offset, const std::size_t length, const std::size_t repeatCount)
        {
            if (factorIndex >= radixFactors.size()) return;
            std::size_t radix = radixFactors[factorIndex];
            if (factorIndex + 1 < radixFactors.size()
                && radixFactors[factorIndex] == 2
                && radixFactors[factorIndex + 1] == 2)
            {
                ++factorIndex;
                radix = 4;
            }

            const std::size_t subLength = length / radix;
            Stage stage { StageKind::mixed, radix, offset, subLength, repeatCount, twiddleTable.size(), 0 };
            if (radix == 2) stage.kind = StageKind::radix2;
            else if (radix == 3) stage.kind = StageKind::radix3;
            else if (radix == 4) stage.kind = StageKind::radix4;

            bool reusedTwiddles = false;
            for (const Stage &existing : stages)
                if (existing.radix == stage.radix && existing.innerCount == stage.innerCount)
                {
                    // same radix and same subLength implies the same length, so
                    // the twiddle block is genuinely identical
                    stage.twiddleOffset = existing.twiddleOffset;
                    reusedTwiddles = true;
                    break;
                }
            if (!reusedTwiddles)
                for (auto i {0uz}; i < subLength; ++i)
                    for (auto r {0uz}; r < radix; ++r)
                    {
                        const double phase = twoPI * static_cast<double>(i) * static_cast<double>(r) / static_cast<double>(length);
                        twiddleTable.push_back(Complex(static_cast<ST>(std::cos(phase)), static_cast<ST>(-std::sin(phase))));
                    }

            // Generic stages need the radix-p DFT matrix roots. Building them
            // once here replaces the original's two transcendental calls in the
            // innermost loop -- O(radix^2) cos/sin per repetition, which made a
            // large prime factor unusable -- and reduces the argument range, so
            // the roots come out more accurate as well.
            if (stage.kind == StageKind::mixed)
            {
                bool reusedRoots = false;
                for (const Stage &existing : stages)
                    if (existing.kind == StageKind::mixed && existing.radix == stage.radix)
                    {
                        stage.rootOffset = existing.rootOffset;
                        reusedRoots = true;
                        break;
                    }
                if (!reusedRoots)
                {
                    stage.rootOffset = rootTable.size();
                    for (auto m {0uz}; m < radix; ++m)
                    {
                        const double phase = twoPI * static_cast<double>(m) / static_cast<double>(radix);
                        rootTable.push_back(Complex(static_cast<ST>(std::cos(phase)), static_cast<ST>(-std::sin(phase))));
                    }
                }
            }

            appendStages(factorIndex + 1, offset, subLength, repeatCount * radix);
            stages.push_back(stage);
        }

        template <bool inverse>
        void execute(const Complex *in, Complex *out) const
        {
            for (const ReorderEntry &entry : reorderTable) out[entry.outputIndex] = in[entry.inputIndex];
            for (const Stage &stage : stages)
                switch (stage.kind)
                {
                    case StageKind::radix2: butterflyRadix2<inverse>(out + stage.offset, stage); break;
                    case StageKind::radix3: butterflyRadix3<inverse>(out + stage.offset, stage); break;
                    case StageKind::radix4: butterflyRadix4<inverse>(out + stage.offset, stage); break;
                    case StageKind::mixed:  butterflyMixed<inverse>(out + stage.offset, stage);  break;
                }
        }

        // ---- radix-2 ----
        template <bool inverse>
        void butterflyRadix2(Complex *block, const Stage &st) const
        {
            const std::size_t stride = st.innerCount;
            const Complex *twBase = twiddleTable.data() + st.twiddleOffset;
            for (std::size_t o = 0; o < st.outerCount; ++o)
            {
                const Complex *tw = twBase;
                for (Complex *p = block; p < block + stride; ++p)
                {
                    const Complex u = p[0];
                    const Complex v = detail::complexMul<inverse>(p[stride], tw[1]);
                    p[0] = u + v;
                    p[stride] = u - v;
                    tw += 2;
                }
                block += 2 * stride;
            }
        }

        // ---- radix-3 ----
        template <bool inverse>
        void butterflyRadix3(Complex *block, const Stage &st) const
        {
            const Complex root(static_cast<ST>(-0.5),
                           inverse ? static_cast<ST>(0.86602540378443864676)
                                   : static_cast<ST>(-0.86602540378443864676));
            const std::size_t stride = st.innerCount;
            const Complex *twBase = twiddleTable.data() + st.twiddleOffset;
            for (std::size_t o = 0; o < st.outerCount; ++o)
            {
                const Complex *tw = twBase;
                for (Complex *p = block; p < block + stride; ++p)
                {
                    const Complex u = p[0];
                    const Complex v = detail::complexMul<inverse>(p[stride], tw[1]);
                    const Complex w = detail::complexMul<inverse>(p[stride * 2], tw[2]);
                    const Complex axis = u + (v + w) * root.real();
                    const Complex wing = (v - w) * root.imag();
                    p[0] = u + v + w;
                    p[stride]     = detail::complexAddI<false>(axis, wing);
                    p[stride * 2] = detail::complexAddI<true>(axis, wing);
                    tw += 3;
                }
                block += 3 * stride;
            }
        }

        // ---- radix-4 ----
        //  Sub-transforms 1 and 2 arrive swapped: the plan merges two radix-2
        //  factors into one radix-4 stage but the permutation was built from
        //  the unmerged factor list, so the two binary digits land reversed
        //  within the group. Hence tw[2] on p[stride] and tw[1] on p[2*stride].
        template <bool inverse>
        void butterflyRadix4(Complex *block, const Stage &st) const
        {
            const std::size_t stride = st.innerCount;
            const Complex *twBase = twiddleTable.data() + st.twiddleOffset;
            for (std::size_t o = 0; o < st.outerCount; ++o)
            {
                const Complex *tw = twBase;
                for (Complex *p = block; p < block + stride; ++p)
                {
                    const Complex t0 = p[0];
                    const Complex t2 = detail::complexMul<inverse>(p[stride],     tw[2]);
                    const Complex t1 = detail::complexMul<inverse>(p[stride * 2], tw[1]);
                    const Complex t3 = detail::complexMul<inverse>(p[stride * 3], tw[3]);

                    const Complex e0 = t0 + t2;
                    const Complex e1 = t1 + t3;
                    const Complex o0 = t0 - t2;
                    const Complex o1 = t1 - t3;

                    p[0]          = e0 + e1;
                    p[stride]     = detail::complexAddI<!inverse>(o0, o1);
                    p[stride * 2] = e0 - e1;
                    p[stride * 3] = detail::complexAddI<inverse>(o0, o1);
                    tw += 4;
                }
                block += 4 * stride;
            }
        }

        // ---- radix-p ----
        template <bool inverse>
        void butterflyMixed(Complex *block, const Stage &st) const
        {
            Complex *work = scratch.data();
            const std::size_t stride = st.innerCount;
            const std::size_t radix = st.radix;
            const Complex *roots = rootTable.data() + st.rootOffset;
            for (std::size_t o = 0; o < st.outerCount; ++o)
            {
                const Complex *tw = twiddleTable.data() + st.twiddleOffset;
                Complex *col = block;
                for (std::size_t rep = 0; rep < stride; ++rep)
                {
                    for (std::size_t i = 0; i < radix; ++i)
                        work[i] = detail::complexMul<inverse>(col[i * stride], tw[i]);
                    for (std::size_t f = 0; f < radix; ++f)
                    {
                        Complex acc = work[0];
                        for (std::size_t i = 1; i < radix; ++i)
                            acc += detail::complexMul<inverse>(work[i], roots[(f * i) % radix]);
                        col[f * stride] = acc;
                    }
                    ++col;
                    tw += radix;
                }
                block += radix * stride;
            }
        }
    };
    FFT(std::size_t) -> FFT<>;

    // ------------------------------------------------------------------------
    //  Real FFT N = 2M real routed through one half-size complex
    // ------------------------------------------------------------------------
    template <typename ST = double>
    class RealFFT {
    public:
        using Complex = std::complex<ST>;
        RealFFT() = default;
        explicit RealFFT(const std::size_t n)
        {
            setSize(n);
        }

        [[nodiscard]] std::size_t size() const
        {
            return complexEngine.size() * 2;
        }

        std::size_t setSize(const std::size_t n)
        {
            assert(n % 2 == 0 && "RealFFT length must be even!");
            const std::size_t half = n / 2;
            assert(std::has_single_bit(half) && "RealFFT half-size (n/2) must be a power of two!");
            packed.resize(half);
            spectrum.resize(half);

            const std::size_t binCount = half / 2 + 1;
            twiddles.resize(binCount);
            for (auto k {0uz}; k < binCount; ++k)
            {
                const double theta = -twoPI * static_cast<double>(k) / static_cast<double>(n);
                twiddles[k] = Complex(static_cast<ST>(std::cos(theta)), static_cast<ST>(std::sin(theta)));
            }
            complexEngine.setSize(half);
            return n;
        }

        void forward(const ST *in, Complex *out)
        {
            const std::size_t half = complexEngine.size();

            for (auto k {0uz}; k < half; ++k) packed[k] = Complex(in[2 * k], in[2 * k + 1]);

            complexEngine.fwd(packed.data(), spectrum.data());

            const ST re = spectrum[0].real();
            const ST im = spectrum[0].imag();
            out[0] = Complex(re + im, re - im);

            for (auto k {1uz}; k <= half / 2; ++k)
            {
                const std::size_t mirror = half - k;
                const Complex lo = spectrum[k];
                const Complex hi = std::conj(spectrum[mirror]);
                const Complex evenBin = (lo + hi) * static_cast<ST>(0.5);
                const Complex halfDiff = (lo - hi) * static_cast<ST>(0.5);
                const Complex oddBin = Complex(halfDiff.imag(), -halfDiff.real());
                const Complex rotated = detail::complexMul<false>(twiddles[k], oddBin);
                out[k] = evenBin + rotated;
                out[mirror] = std::conj(evenBin - rotated);
            }
        }

        void inverse(const Complex *in, ST *out)
        {
            const std::size_t half = complexEngine.size();
            spectrum[0] = Complex(in[0].real() + in[0].imag(), in[0].real() - in[0].imag());
            for (auto k {1uz}; k <= half / 2; ++k)
            {
                const std::size_t mirror = half - k;
                const Complex sum = in[k] + std::conj(in[mirror]);
                const Complex diff = in[k] - std::conj(in[mirror]);
                const Complex twoOdd = detail::complexMul<true>(diff, twiddles[k]);
                const Complex twoIOdd = Complex(-twoOdd.imag(), twoOdd.real());
                spectrum[k] = sum + twoIOdd;
                spectrum[mirror] = std::conj(sum - twoIOdd);
            }
            complexEngine.inv(spectrum.data(), packed.data());
            for (auto k {0uz}; k < half; ++k)
            {
                out[2*k] = packed[k].real();
                out[2*k+1] = packed[k].imag();
            }
        }

    private:
        static constexpr double twoPI = 2.0 * std::numbers::pi_v<double>;
        detail::SOAPow2FFT<ST> complexEngine;
        std::vector<Complex> packed;
        std::vector<Complex> spectrum;
        std::vector<Complex> twiddles;
    };
    RealFFT(std::size_t) -> RealFFT<>;
}

#endif

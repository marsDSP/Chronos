#pragma once
// ══════════════════════════════════════════════════════════════
#ifndef CHRONOS_SIMD_CONFIG_H
#define CHRONOS_SIMD_CONFIG_H
// ══════════════════════════════════════════════════════════════
// X86 / SSE2 Detection | Native SSE2+ platform support
// ──────────────────────────────────────────────────────────────
#if (defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64) ||                                  \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define MARSCORE_SIMD_NATIVE_X86
#endif
// ══════════════════════════════════════════════════════════════
// ARM64EC Detection | Windows ARM64 Emulation-Compatible ABI
// ──────────────────────────────────────────────────────────────
#if defined(_M_ARM64EC)
#define MARSCORE_SIMD_ARM64EC
#endif
// ══════════════════════════════════════════════════════════════
// ARM64 Detection | native AArch64 / Apple Silicon / etc.
// ──────────────────────────────────────────────────────────────
#if defined(__aarch64__) || defined(__arm64) || defined(__arm64__) || defined(_M_ARM64) ||         \
    defined(_M_ARM64EC)
#define MARSCORE_SIMD_ARM64
#endif
// ══════════════════════════════════════════════════════════════
// Platform-specific headers | ARM64EC on MSVC | Native x86
// ══════════════════════════════════════════════════════════════
#ifdef MARSCORE_SIMD_ARM64EC
#include <intrin.h>             // MSVC compiler intrinsics
#endif

#ifdef MARSCORE_SIMD_NATIVE_X86
#include <emmintrin.h>          // SSE2
#include <pmmintrin.h>          // SSE3
#include <smmintrin.h>          // SSE4.1
#endif
// ══════════════════════════════════════════════════════════════
// SIMDe (SIMD-Everywhere) | (NEON, WASM, or scalar fallback).
// ──────────────────────────────────────────────────────────────
#ifndef SIMDE_UNAVAILABLE
    #ifdef MARSCORE_SIMD_ARM64EC
    #include <cmath>
    #endif
    #ifndef MARSCORE_SIMD_NATIVE_X86
    #ifndef MARSCORE_SIMD_OMIT_NATIVE_ALIASES
    #define SIMDE_ENABLE_NATIVE_ALIASES
    #endif
    #endif
    #include <simde/x86/sse4.2.h>
    #include <simde/x86/fma.h>
#endif
// ══════════════════════════════════════════════════════════════
// Branch A: Native x86 OR SIMDe is unavailable
// ══════════════════════════════════════════════════════════════
// NOTE: SIMDE_UNAVAILABLE is only meaningful on native x86. Branch A
// defines MM(x) as _mm_##x, so a non-x86 target with SIMDE_UNAVAILABLE
// has no _mm_* intrinsics to call and will not compile. Do not attempt
// to make that case work; it is unsupported.
#if defined(MARSCORE_SIMD_NATIVE_X86) || defined(SIMDE_UNAVAILABLE)

// MM(add_ps)  expands to _mm_add_ps
#define MM(x) _mm_##x

// 128-bit vector types (4×float, 4×int32, 2×double)
#define M128  __m128       // 4 × 32-bit float
#define M128I __m128i      // 4 × 32-bit int (or 8×16, 16×8, 2×64…)
#define M128D __m128d      // 2 × 64-bit double

// _MM_SHUFFLE(z,y,x,w) builds an 8-bit immediate for shuffle ops
#define MM_SHUFFLE _MM_SHUFFLE

// FMA: fused multiply-add. simde_mm_fmadd_ps lowers to native vfmadd*ps
// under -mfma on x86_64. Use FMADD(a,b,c) at call sites.
#ifndef SIMDE_UNAVAILABLE
#define FMADD simde_mm_fmadd_ps
#else
// No simde: separate mul+add. NOT bit-identical to the fused path — the
// intermediate product rounds before the add. simd_delay_parity's 1e-5
// tolerance was chosen against the FMA path, so it still holds, but a
// tighter gate could fail on the unfused configuration.
#define FMADD(a,b,c) _mm_add_ps(_mm_mul_ps((a),(b)), (c))
#endif
// ══════════════════════════════════════════════════════════════
// Branch B: Non-x86 with SIMDe available
// ══════════════════════════════════════════════════════════════
#else
#define MM(x) simde_mm_##x
#define M128  simde__m128
#define M128I simde__m128i
#define M128D simde__m128d
#define MM_SHUFFLE SIMDE_MM_SHUFFLE

// FMA: simde_mm_fmadd_ps lowers to native vfmadd on arm64 (FMA is
// unconditional there) or to a mul+add pair on other non-x86 targets.
#define FMADD simde_mm_fmadd_ps
#endif
// ══════════════════════════════════════════════════════════════
// Hard requirement: C++23 or later.
// ══════════════════════════════════════════════════════════════
static_assert(__cplusplus >= 202302L, "You need C++23 to compile this!");
// ══════════════════════════════════════════════════════════════
#endif
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

#endif
// ══════════════════════════════════════════════════════════════
// Branch A: Native x86 OR SIMDe is unavailable
// ══════════════════════════════════════════════════════════════
#if defined(MARSCORE_SIMD_NATIVE_X86) || defined(SIMDE_UNAVAILABLE)

// SIMD_MM(add_ps)  expands to _mm_add_ps
#define SIMD_MM(x) _mm_##x

// 128-bit vector types (4×float, 4×int32, 2×double)
#define SIMD_M128  __m128       // 4 × 32-bit float
#define SIMD_M128I __m128i      // 4 × 32-bit int (or 8×16, 16×8, 2×64…)
#define SIMD_M128D __m128d      // 2 × 64-bit double

// _MM_SHUFFLE(z,y,x,w) builds an 8-bit immediate for shuffle ops
#define SIMD_MM_SHUFFLE _MM_SHUFFLE
// ══════════════════════════════════════════════════════════════
// Branch B: Non-x86 with SIMDe available
// ══════════════════════════════════════════════════════════════
#else
#define SIMD_MM(x) simde_mm_##x
#define SIMD_M128  simde__m128
#define SIMD_M128I simde__m128i
#define SIMD_M128D simde__m128d
#define SIMD_MM_SHUFFLE SIMDE_MM_SHUFFLE
#endif
// ══════════════════════════════════════════════════════════════
// Hard requirement: C++23 or later.
// ══════════════════════════════════════════════════════════════
static_assert(__cplusplus >= 202302L, "You need C++23 to compile this!");
// ══════════════════════════════════════════════════════════════
#endif
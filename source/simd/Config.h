#pragma once

#ifndef CHRONOS_SIMD_CONFIG_H
#define CHRONOS_SIMD_CONFIG_H

/**
 * SIMD abstraction layer over native SSE or SIMDe.
 * Defines the MM(x) and M128 macros plus the FMADD alias.
 * This is the single include of simde/x86/fma.h.
 */

// Detect native x86 SSE2 support.
#if (defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64) ||                                  \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define MARSCORE_SIMD_NATIVE_X86
#endif

// Detect Windows ARM64EC.
#if defined(_M_ARM64EC)
#define MARSCORE_SIMD_ARM64EC
#endif

// Detect native AArch64.
#if defined(__aarch64__) || defined(__arm64) || defined(__arm64__) || defined(_M_ARM64) ||         \
    defined(_M_ARM64EC)
#define MARSCORE_SIMD_ARM64
#endif

#ifdef MARSCORE_SIMD_ARM64EC
#include <intrin.h>
#endif

#ifdef MARSCORE_SIMD_NATIVE_X86
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#endif

// Pull in SIMDe for non-native targets, or when native aliases are requested.
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

// Branch A: native x86, or SIMDe is unavailable.
// On non-x86 with SIMDE_UNAVAILABLE there are no _mm_* intrinsics, so the
// build is unsupported by design.
#if defined(MARSCORE_SIMD_NATIVE_X86) || defined(SIMDE_UNAVAILABLE)

#define MM(x) _mm_##x
#define M128  __m128
#define M128I __m128i
#define M128D __m128d
#define MM_SHUFFLE _MM_SHUFFLE

// FMADD lowers to a fused multiply-add under -mfma on x86_64.
#ifndef SIMDE_UNAVAILABLE
#define FMADD simde_mm_fmadd_ps
#else
// No simde: a mul plus an add. The product rounds before the add, so this is
// not bit-identical to the fused path. Parity tolerances account for it.
#define FMADD(a,b,c) _mm_add_ps(_mm_mul_ps((a),(b)), (c))
#endif

// Branch B: non-x86 with SIMDe available.
#else

#define MM(x) simde_mm_##x
#define M128  simde__m128
#define M128I simde__m128i
#define M128D simde__m128d
#define MM_SHUFFLE SIMDE_MM_SHUFFLE

// FMADD lowers to native vfmadd on arm64 (FMA is unconditional there).
#define FMADD simde_mm_fmadd_ps
#endif

static_assert(__cplusplus >= 202302L, "Chronos requires C++23 or later.");

#endif

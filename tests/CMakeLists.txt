# Chronos test / benchmark harnesses.
#
# Opt-in: this directory is NOT built by default, preserving the documented
# "tests not wired into the build" state. Pass -DBUILD_TEST_HARNESSES=ON to
# enable. Harnesses link only the SharedCode INTERFACE target, so they pull in
# the source/ + libs/simde include paths and cxx_std_23 without dragging in the
# JUCE plugin modules.
option(BUILD_TEST_HARNESSES "Build Chronos test/benchmark harnesses" OFF)

if(BUILD_TEST_HARNESSES)
    # Bilinear pre-warping tan kernel: mmTan (FMA M128) vs std::tan.
    # Accuracy regression check + throughput microbenchmark.
    add_executable(tan_bench harnesses/perf/tan_bench.cpp)
    target_link_libraries(tan_bench PRIVATE SharedCode)
    # A benchmark is meaningless at -O0/Debug: `inline` is not honored there, so
    # the mmTan kernel and every simde intrinsic become real out-of-line calls
    # (hundreds of ns each), while std::tan stays a single libm call.  Force
    # optimization on this target regardless of CMAKE_BUILD_TYPE so the kernel
    # inlines and simde_mm_fmadd_ps lowers to a fused multiply-add.
    if(MSVC)
        target_compile_options(tan_bench PRIVATE /O2)
    else()
        target_compile_options(tan_bench PRIVATE -O2)
    endif()
    if(APPLE)
        # Mirror the plugin target: enable FMA3 on the x86_64 slice so
        # simde_mm_fmadd_ps lowers to vfmadd on Intel. arm64 has FMA
        # unconditionally and needs no flag. See CMakeLists.txt for the same
        # baseline (Haswell/FMA3) note that applies to the x86_64 slice.
        target_compile_options(tan_bench PRIVATE "-Xarch_x86_64" "-mfma")
    endif()

    # Pow2RingBuffer correctness harness: exhaustive window/mirror/contiguity
    # checks against a naive modulo oracle (see harnesses/cd/ring_buffer_check.cpp).
    # Plain main() + exit code + printf, matching tan_bench. Links SharedCode only.
    # No forced -O2: this is a correctness test, and leaving the build unoptimized
    # keeps the header's assert preconditions armed in a Debug configure.
    add_executable(ring_buffer_check harnesses/cd/ring_buffer_check.cpp)
    target_link_libraries(ring_buffer_check PRIVATE SharedCode)
    # Optional sanitizer pass: uncomment to run the two-memcpy
    # split and the mirror region under ASan+UBSan, which catches an off-by-one
    # past the allocation far more directly than a value comparison does.
    # target_compile_options(ring_buffer_check PRIVATE -fsanitize=address,undefined)
    # target_link_options(ring_buffer_check PRIVATE -fsanitize=address,undefined)

    # SimdDelayLine correctness harness: integer impulse, polynomial
    # reproduction (centroid / flaw-B guard), stereo independence, multi-block
    # delay, delay-move settling, ring-wrap endurance, zero-in/zero-out, mono
    # path (see harnesses/cd/simd_delay_check.cpp). Same conventions as
    # ring_buffer_check: plain main() + exit code + printf, links SharedCode
    # only, no forced -O2 so the header's assert preconditions stay armed.
    add_executable(simd_delay_check harnesses/cd/simd_delay_check.cpp)
    target_link_libraries(simd_delay_check PRIVATE SharedCode)

    # Dilogarithm correctness harness: Li2(-t) via Landen-folded series used
    # by TanhNL::F2 (see harnesses/cd/dilog_check.cpp). Zero, Li2(-1)=-pi^2/12,
    # the Landen seam across t=0.5, known values, an independent Simpson-
    # quadrature oracle, and strict monotonicity. Same conventions as
    # ring_buffer_check: plain main() + exit code + printf, links SharedCode
    # only, no forced -O2 so the header's assert preconditions stay armed.
    add_executable(dilog_check harnesses/cd/dilog_check.cpp)
    target_link_libraries(dilog_check PRIVATE SharedCode)

    # Nonlinearity policy correctness harness: TanhNL / AlgebraicNL f/F1/F2
    # (see harnesses/cd/nonlinearity_check.cpp). F1'=f and F2'=F1 by central
    # difference, parity (F1 even, F2 odd, bit-exact), origin values, no
    # overflow at +-700, and the tanh F1/G tail. Same conventions as
    # ring_buffer_check / dilog_check: plain main() + exit code + printf,
    # links SharedCode only, no forced -O2 so assert preconditions stay armed.
    add_executable(nonlinearity_check harnesses/cd/nonlinearity_check.cpp)
    target_link_libraries(nonlinearity_check PRIVATE SharedCode)

    # SimdDelayLine SIMD-vs-scalar parity harness: process() (4-wide SIMD kernel)
    # vs processScalar() (scalar dot6), compared to a tolerance across block
    # sizes, fractional delays, stereo, mono, ring-wrap, and the delay-move
    # crossfade region (see harnesses/simd/simd_delay_parity.cpp). Parity is
    # to a tolerance, not bit-exact, because the MAC tree reassociates. Forced
    # -O2 and -Xarch_x86_64 -mfma (matching tan_bench) so the SIMD kernel
    # inlines and simde_mm_fmadd_ps lowers to a fused multiply-add, matching
    # the plugin target's x86_64 slice.
    add_executable(simd_delay_parity harnesses/simd/simd_delay_parity.cpp)
    target_link_libraries(simd_delay_parity PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(simd_delay_parity PRIVATE /O2)
    else()
        target_compile_options(simd_delay_parity PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(simd_delay_parity PRIVATE "-Xarch_x86_64" "-mfma")
    endif()

    # SimdDelayLine throughput benchmark vs juce::dsp::DelayLine (the old
    # per-sample baseline) + SIMD-vs-scalar + per-mode cost
    # (see harnesses/perf/delay_line_bench.cpp). First harness to link a JUCE
    # module: juce::juce_dsp pulls in juce::dsp::DelayLine; SharedCode
    # propagates the JUCE module config defines (JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED
    # etc.) via INTERFACE, so <juce_dsp/juce_dsp.h> can be included directly.
    # Forced -O2 and -Xarch_x86_64 -mfma (matching tan_bench) so the SIMD kernel
    # inlines and FMADD lowers to a fused multiply-add, matching the plugin's
    # x86_64 slice. Gate: SIMD ns/sample <= 1.05x scalar (regression check);
    # the juce::dsp::DelayLine ratio is informational (machine-dependent).
    add_executable(delay_line_bench harnesses/perf/delay_line_bench.cpp)
    target_link_libraries(delay_line_bench PRIVATE SharedCode juce::juce_dsp)
    if(MSVC)
        target_compile_options(delay_line_bench PRIVATE /O2)
    else()
        target_compile_options(delay_line_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(delay_line_bench PRIVATE "-Xarch_x86_64" "-mfma")
    endif()
endif()

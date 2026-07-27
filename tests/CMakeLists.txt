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

    # HalfSampleFir correctness harness: the symmetric half-sample-delay FIR
    # that compensates ADAA1's 0.5-sample box-kernel wet-path latency
    # (see harnesses/cd/halfsample_fir_check.cpp). Coefficient symmetry
    # (bit-exact), DC gain, impulse-response centroid, flat group delay
    # (< 1e-9 ripple), passband magnitude (gate > -0.1 dB at 15 kHz), Nyquist
    # null, a dormant Lagrange cross-check at kHalfSampleTaps == 6, and reset
    # reproducibility. Coefficients re-derived by scripts/python/halfsample_fir.py
    # (Kaiser-windowed sinc), which doubles as a regression check. Same
    # conventions as ring_buffer_check: plain main() + exit code + printf,
    # links SharedCode only, no forced -O2 so the header's static_assert / any
    # assert preconditions stay armed in a Debug configure.
    add_executable(halfsample_fir_check harnesses/cd/halfsample_fir_check.cpp)
    target_link_libraries(halfsample_fir_check PRIVATE SharedCode)

    # ShortDelay correctness harness: the fixed-capacity integer delay used
    # by SaturatorAlign to pad the wet path to kBudget in Off/ADAA2 modes
    # (see harnesses/cd/short_delay_check.cpp). For every d in [0, MaxDelay]:
    # impulse (exactly 1.0f at sample d, 0 elsewhere in [0, 4*MaxDelay]),
    # ramp (y[n]==x[n-d] bit-exact over 1000 samples, many ring wraps), and
    # reset reproducibility. Tested at MaxDelay=8 (the real kBudget) and
    # MaxDelay=1 (the d in {0,1} edge). Same conventions as ring_buffer_check
    # / halfsample_fir_check: plain main() + exit code + printf, links
    # SharedCode only, no forced -O2 so the header's assert preconditions
    # stay armed in a Debug configure.
    add_executable(short_delay_check harnesses/cd/short_delay_check.cpp)
    target_link_libraries(short_delay_check PRIVATE SharedCode)

    # SaturatorAlign correctness harness: the constant-latency dry/wet
    # alignment layer for the ADAA saturator stage
    # (see harnesses/cd/align_check.cpp). The harness that verifies the
    # property the whole layer exists for: dry is an exact integer delay
    # (I1), wet total group delay equals kBudget in every mode (the ADAA
    # stage kernel is prepended analytically -- box [0.5,0.5] for ADAA1,
    # triangle [0.25,0.5,0.25] for ADAA2), modes 0/2 are bit-exact pure
    # delays, mode 1 is linear phase (I3), dry/wet null (mode 0 bit-
    # identical), cross-mode phase coherence at 1/5/10/15 kHz (the test
    # that would have caught a first-order-allpass design), reset, and
    # mode switching (no NaN). Same conventions as ring_buffer_check /
    # halfsample_fir_check / short_delay_check: plain main() + exit code
    # + printf, links SharedCode only, no forced -O2 so the header's
    # assert preconditions stay armed in a Debug configure.
    add_executable(align_check harnesses/cd/align_check.cpp)
    target_link_libraries(align_check PRIVATE SharedCode)

    # ADAA throughput microbenchmark + branch histogram: ADAA1<TanhNL>,
    # ADAA2<TanhNL>, ADAA2<AlgebraicNL>, and plain std::tanh as the no-
    # antialiasing floor, plus the ADAA2 branch histogram on a 48 kHz sine
    # sweep at several drive levels (see harnesses/perf/adaa_bench.cpp). A
    # benchmark is meaningless at -O0/Debug: `inline` is not honored there, so
    # the ADAA2 template logic and the 50-term dilogarithm series become real
    # out-of-line calls, while std::tanh stays a single libm call. Force
    # optimization on this target regardless of CMAKE_BUILD_TYPE so the kernel
    # inlines. Links SharedCode only; no JUCE. Informational only — no pass/
    # fail gate, cost per sample is not the constraint at this stage.
    add_executable(adaa_bench harnesses/perf/adaa_bench.cpp)
    target_link_libraries(adaa_bench PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(adaa_bench PRIVATE /O2)
    else()
        target_compile_options(adaa_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(adaa_bench PRIVATE "-Xarch_x86_64" "-mfma")
    endif()

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

    # ADAA2 correctness harness: the four-branch second-order ADAA kernel
    # (see harnesses/cd/adaa2_check.cpp). Static curve, the confluent
    # outer-node branch (the Nyquist "a,b,a,b" case), branch-seam continuity,
    # an error surface over x_k = A sin(w(n-k)) that gates the exit code,
    # degenerate node patterns, reset determinism, and parity. The oracle is
    # a Hermite-Genocchi quadrature rather than a long double replay of the
    # same formula: long double is 64-bit on arm64, so the latter would be
    # bit-identical to the code under test. Same conventions as
    # ring_buffer_check / dilog_check / nonlinearity_check: plain main() +
    # exit code + printf, links SharedCode only, no forced -O2 so assert
    # preconditions stay armed.
    add_executable(adaa2_check harnesses/cd/adaa2_check.cpp)
    target_link_libraries(adaa2_check PRIVATE SharedCode)

    # ADAA aliasing measurement (see harnesses/dsp/alias_check.cpp, a new
    # harnesses/dsp/ subdirectory). Feeds a coherent sine, masks the legitimate
    # harmonic bins with Goertzel, and takes the rest of the Parseval energy as
    # alias energy; reports an f0 x drive matrix for no-ADAA / ADAA1 / ADAA2 /
    # ADAA1@2x. Forced -O2: this one does real work (Goertzel over ~860
    # harmonic bins per cell plus 513-tap Kaiser resampling), and at -O0 it
    # takes minutes rather than seconds. Links SharedCode only; no JUCE and no
    # FFT. Gate: ADAA2 beats no-ADAA by >= 20 dB at 110 Hz / 24 dB drive; the
    # rest of the matrix is informational.
    add_executable(alias_check harnesses/dsp/alias_check.cpp)
    target_link_libraries(alias_check PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(alias_check PRIVATE /O2)
    else()
        target_compile_options(alias_check PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(alias_check PRIVATE "-Xarch_x86_64" "-mfma")
    endif()

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

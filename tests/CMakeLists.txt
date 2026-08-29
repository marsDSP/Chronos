# Chronos test / benchmark harnesses.
option(BUILD_TEST_HARNESSES "Build Chronos test/benchmark harnesses" OFF)

if(BUILD_TEST_HARNESSES)
    add_executable(tan_bench harnesses/perf/tan_bench.cpp)
    target_link_libraries(tan_bench PRIVATE SharedCode)

    if(MSVC)
        target_compile_options(tan_bench PRIVATE /O2)
    else()
        target_compile_options(tan_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(tan_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(tan_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(tan_bench PRIVATE "-mfma")
    endif()

    add_executable(ring_buffer_check harnesses/cd/ring_buffer_check.cpp)
    target_link_libraries(ring_buffer_check PRIVATE SharedCode)
    add_executable(halfsample_fir_check harnesses/cd/halfsample_fir_check.cpp)
    target_link_libraries(halfsample_fir_check PRIVATE SharedCode)
    add_executable(short_delay_check harnesses/cd/short_delay_check.cpp)
    target_link_libraries(short_delay_check PRIVATE SharedCode)
    add_executable(align_check harnesses/cd/align_check.cpp)
    target_link_libraries(align_check PRIVATE SharedCode)
    add_executable(latency_null_check harnesses/dsp/latency_null_check.cpp)
    target_link_libraries(latency_null_check PRIVATE SharedCode)
    add_executable(chain_fuzz_check harnesses/dsp/chain_fuzz_check.cpp)
    target_link_libraries(chain_fuzz_check PRIVATE SharedCode)
    add_executable(oversized_block_check harnesses/dsp/oversized_block_check.cpp)
    target_link_libraries(oversized_block_check PRIVATE SharedCode)
    add_executable(engine_skeleton_check harnesses/dsp/engine_skeleton_check.cpp)
    target_link_libraries(engine_skeleton_check PRIVATE SharedCode)
    add_executable(chain_parity harnesses/dsp/chain_parity.cpp)
    target_link_libraries(chain_parity PRIVATE SharedCode)
    add_executable(fb_parity harnesses/dsp/fb_parity.cpp)
    target_link_libraries(fb_parity PRIVATE SharedCode)
    # fb_parity is the heaviest correctness harness (2442 configs x 65536
    # samples x 2 engines). -O2 inlines the per-sample processRef path and the
    # ADAA kernels; without it the loop-bound scalar math stays at -O0 and the
    # test dominates the parallel ctest wall time. -mfma already arrives via
    # SharedCode, so the SIMD numerics match the shipped binary either way.
    # Tolerances (abs 1e-6 OR rel 1e-3, +/−0.1 dB envelope) absorb the ULP-level
    # FMA/reassociation differences between chunked and scalar paths, so -O2
    # cannot manufacture spurious failures. NDEBUG is NOT defined (-O2 on a
    # Debug target leaves asserts armed; this harness uses CHECK/FAIL anyway).
    if(NOT MSVC)
        target_compile_options(fb_parity PRIVATE -O2)
    else()
        target_compile_options(fb_parity PRIVATE /O2)
    endif()
    add_executable(diffuser_toggle_check harnesses/dsp/diffuser_toggle_check.cpp)
    target_link_libraries(diffuser_toggle_check PRIVATE SharedCode)
    add_executable(diffusion_onset_check harnesses/dsp/diffusion_onset_check.cpp)
    target_link_libraries(diffusion_onset_check PRIVATE SharedCode)
    add_executable(diffuser_loop_check harnesses/dsp/diffuser_loop_check.cpp)
    target_link_libraries(diffuser_loop_check PRIVATE SharedCode)
    add_executable(diffuser_transport_check harnesses/dsp/diffuser_transport_check.cpp)
    target_link_libraries(diffuser_transport_check PRIVATE SharedCode)
    add_executable(diffuser_ring_check harnesses/dsp/diffuser_ring_check.cpp)
    target_link_libraries(diffuser_ring_check PRIVATE SharedCode)
    add_executable(diffuser_decorrelation_check harnesses/dsp/diffuser_decorrelation_check.cpp)
    target_link_libraries(diffuser_decorrelation_check PRIVATE SharedCode)
    add_executable(loop_gain_check harnesses/dsp/loop_gain_check.cpp)
    target_link_libraries(loop_gain_check PRIVATE SharedCode)
    add_executable(bypass_null_check harnesses/dsp/bypass_null_check.cpp)
    target_link_libraries(bypass_null_check PRIVATE SharedCode)
    add_executable(crossfade_parity harnesses/simd/crossfade_parity.cpp)
    target_link_libraries(crossfade_parity PRIVATE SharedCode)
    add_executable(dither_check harnesses/dsp/dither_check.cpp)
    target_link_libraries(dither_check PRIVATE SharedCode)
    add_executable(bump_arena_check harnesses/utils/bump_arena_check.cpp)
    target_link_libraries(bump_arena_check PRIVATE SharedCode)
    add_executable(spsc_fifo_check harnesses/cd/spsc_fifo_check.cpp)
    target_link_libraries(spsc_fifo_check PRIVATE SharedCode)
    add_executable(tap_sim_check harnesses/cd/tap_sim_check.cpp
        ${CMAKE_SOURCE_DIR}/source/gui/tap/TapSimulation.cpp)
    target_link_libraries(tap_sim_check PRIVATE SharedCode)

    # State schema harness. Links the JUCE processor modules with SharedCode.
    # The harness builds a real APVTS from the Chronos parameter layout.
    # It needs no JucePlugin macros and no plugin client wrapper.
    add_executable(state_roundtrip_check harnesses/dsp/state_roundtrip_check.cpp)
    target_link_libraries(state_roundtrip_check PRIVATE SharedCode
        juce::juce_audio_processors juce::juce_dsp juce::juce_gui_basics)

    add_executable(long_run_stability_check harnesses/dsp/long_run_stability_check.cpp)
    target_link_libraries(long_run_stability_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(long_run_stability_check PRIVATE -O2)
    else()
        target_compile_options(long_run_stability_check PRIVATE /O2)
    endif()

    add_executable(automation_sweep_check harnesses/dsp/automation_sweep_check.cpp)
    target_link_libraries(automation_sweep_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(automation_sweep_check PRIVATE -O2)
    else()
        target_compile_options(automation_sweep_check PRIVATE /O2)
    endif()

    add_executable(unity_transparency_check harnesses/dsp/unity_transparency_check.cpp)
    target_link_libraries(unity_transparency_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(unity_transparency_check PRIVATE -O2)
    else()
        target_compile_options(unity_transparency_check PRIVATE /O2)
    endif()

    add_executable(prepare_bench harnesses/perf/prepare_bench.cpp)
    target_link_libraries(prepare_bench PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(prepare_bench PRIVATE -O2)
    else()
        target_compile_options(prepare_bench PRIVATE /O2)
    endif()

    add_executable(tempo_sync_check harnesses/dsp/tempo_sync_check.cpp)
    target_link_libraries(tempo_sync_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(tempo_sync_check PRIVATE -O2)
    else()
        target_compile_options(tempo_sync_check PRIVATE /O2)
    endif()

    add_executable(glide_rate_check harnesses/dsp/glide_rate_check.cpp)
    target_link_libraries(glide_rate_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(glide_rate_check PRIVATE -O2)
    else()
        target_compile_options(glide_rate_check PRIVATE /O2)
    endif()

    add_executable(delay_mod_check harnesses/dsp/delay_mod_check.cpp)
    target_link_libraries(delay_mod_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(delay_mod_check PRIVATE -O2)
    else()
        target_compile_options(delay_mod_check PRIVATE /O2)
    endif()

    add_executable(drive_loudness_check harnesses/dsp/drive_loudness_check.cpp)
    target_link_libraries(drive_loudness_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(drive_loudness_check PRIVATE -O2)
    else()
        target_compile_options(drive_loudness_check PRIVATE /O2)
    endif()

    add_executable(loop_dc_check harnesses/dsp/loop_dc_check.cpp)
    target_link_libraries(loop_dc_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(loop_dc_check PRIVATE -O2)
    else()
        target_compile_options(loop_dc_check PRIVATE /O2)
    endif()

    add_executable(crossfeed_power_check harnesses/dsp/crossfeed_power_check.cpp)
    target_link_libraries(crossfeed_power_check PRIVATE SharedCode)
    if(NOT MSVC)
        target_compile_options(crossfeed_power_check PRIVATE -O2)
    else()
        target_compile_options(crossfeed_power_check PRIVATE /O2)
    endif()
    if(MSVC)
        target_compile_options(crossfade_parity PRIVATE /O2)
    else()
        target_compile_options(crossfade_parity PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(crossfade_parity PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(crossfade_parity PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(crossfade_parity PRIVATE "-mfma")
    endif()
    # Unlike the other correctness harnesses, this one deliberately feeds
    # NaN/inf through the DSP headers' precondition asserts (e.g. dilogNeg's
    # t in [0,1]) to measure IEEE propagation and recovery. NDEBUG disarms
    # them for this target only; the harness's own CHECK/FAIL stay live.
    target_compile_definitions(chain_fuzz_check PRIVATE NDEBUG)
    add_executable(adaa_bench harnesses/perf/adaa_bench.cpp)
    target_link_libraries(adaa_bench PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(adaa_bench PRIVATE /O2)
    else()
        target_compile_options(adaa_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(adaa_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(adaa_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(adaa_bench PRIVATE "-mfma")
    endif()

    add_executable(simd_delay_check harnesses/cd/simd_delay_check.cpp)
    target_link_libraries(simd_delay_check PRIVATE SharedCode)

    add_executable(frac_delay_tap_check harnesses/cd/frac_delay_tap_check.cpp)
    target_link_libraries(frac_delay_tap_check PRIVATE SharedCode)

    add_executable(dilog_check harnesses/cd/dilog_check.cpp)
    target_link_libraries(dilog_check PRIVATE SharedCode)

    add_executable(nonlinearity_check harnesses/cd/nonlinearity_check.cpp)
    target_link_libraries(nonlinearity_check PRIVATE SharedCode)

    add_executable(adaa2_check harnesses/cd/adaa2_check.cpp)
    target_link_libraries(adaa2_check PRIVATE SharedCode)

    add_executable(f2_oracle_check harnesses/cd/f2_oracle_check.cpp)
    target_link_libraries(f2_oracle_check PRIVATE SharedCode)

    add_executable(f2_minimax_check harnesses/cd/f2_minimax_check.cpp)
    target_link_libraries(f2_minimax_check PRIVATE SharedCode)

    add_executable(makeup_table_check harnesses/cd/makeup_table_check.cpp)
    target_link_libraries(makeup_table_check PRIVATE SharedCode)

    add_executable(alias_check harnesses/dsp/alias_check.cpp)
    target_link_libraries(alias_check PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(alias_check PRIVATE /O2)
    else()
        target_compile_options(alias_check PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(alias_check PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(alias_check PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(alias_check PRIVATE "-mfma")
    endif()

    add_executable(simd_delay_parity harnesses/simd/simd_delay_parity.cpp)
    target_link_libraries(simd_delay_parity PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(simd_delay_parity PRIVATE /O2)
    else()
        target_compile_options(simd_delay_parity PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(simd_delay_parity PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(simd_delay_parity PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(simd_delay_parity PRIVATE "-mfma")
    endif()

    add_executable(diffuser_lengths_check harnesses/dsp/diffuser_lengths_check.cpp)
    target_link_libraries(diffuser_lengths_check PRIVATE SharedCode)
    add_executable(diffuser_prepare_race_check harnesses/dsp/diffuser_prepare_race_check.cpp)
    target_link_libraries(diffuser_prepare_race_check PRIVATE SharedCode)
    add_executable(diffuser_density_check harnesses/dsp/diffuser_density_check.cpp)
    target_link_libraries(diffuser_density_check PRIVATE SharedCode)
    add_executable(feedback_decay_check harnesses/dsp/feedback_decay_check.cpp)
    target_link_libraries(feedback_decay_check PRIVATE SharedCode)
    add_executable(nested_allpass_check harnesses/dsp/nested_allpass_check.cpp)
    target_link_libraries(nested_allpass_check PRIVATE SharedCode)
    add_executable(diffuser_parity harnesses/simd/diffuser_parity.cpp)
    target_link_libraries(diffuser_parity PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(diffuser_parity PRIVATE /O2)
    else()
        target_compile_options(diffuser_parity PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(diffuser_parity PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(diffuser_parity PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(diffuser_parity PRIVATE "-mfma")
    endif()

    # Golden render harness: deterministic FNV-1a 64 hashes over 12 fixed
    # configurations x 3 fixed inputs, plus audible metrics. Forced -O2 and
    # FMA so the SIMD kernels inline and the numerics match the plugin build.
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/golden")
    add_executable(golden_render_check harnesses/dsp/golden_render_check.cpp)
    target_link_libraries(golden_render_check PRIVATE SharedCode)
    target_compile_definitions(golden_render_check PRIVATE
        CHRONOS_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
    if(MSVC)
        target_compile_options(golden_render_check PRIVATE /O2)
    else()
        target_compile_options(golden_render_check PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(golden_render_check PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(golden_render_check PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(golden_render_check PRIVATE "-mfma")
    endif()

    # Real-time safety harnesses. rt_alloc_check overrides the global
    # allocation functions and arms them around process only. rt_stack_check
    # paints a 256 kB stack canary and measures the high-water mark. Forced
    # -O2 and FMA so the SIMD kernels match the plugin build.
    add_executable(rt_alloc_check harnesses/rt/rt_alloc_check.cpp)
    target_link_libraries(rt_alloc_check PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(rt_alloc_check PRIVATE /O2)
    else()
        target_compile_options(rt_alloc_check PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(rt_alloc_check PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(rt_alloc_check PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(rt_alloc_check PRIVATE "-mfma")
    endif()

    add_executable(rt_stack_check harnesses/rt/rt_stack_check.cpp)
    target_link_libraries(rt_stack_check PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(rt_stack_check PRIVATE /O2)
    else()
        target_compile_options(rt_stack_check PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(rt_stack_check PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(rt_stack_check PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(rt_stack_check PRIVATE "-mfma")
    endif()

    add_executable(delay_line_bench harnesses/perf/delay_line_bench.cpp)
    target_link_libraries(delay_line_bench PRIVATE SharedCode juce::juce_dsp)
    if(MSVC)
        target_compile_options(delay_line_bench PRIVATE /O2)
    else()
        target_compile_options(delay_line_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(delay_line_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(delay_line_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(delay_line_bench PRIVATE "-mfma")
    endif()

    add_executable(chain_bench harnesses/perf/chain_bench.cpp)
    target_link_libraries(chain_bench PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(chain_bench PRIVATE /O2)
    else()
        target_compile_options(chain_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(chain_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(chain_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(chain_bench PRIVATE "-mfma")
    endif()

    add_executable(fb_bench harnesses/perf/fb_bench.cpp)
    target_link_libraries(fb_bench PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(fb_bench PRIVATE /O2)
    else()
        target_compile_options(fb_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(fb_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(fb_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(fb_bench PRIVATE "-mfma")
    endif()

    add_executable(diffuser_bench harnesses/perf/diffuser_bench.cpp)
    target_link_libraries(diffuser_bench PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(diffuser_bench PRIVATE /O2)
    else()
        target_compile_options(diffuser_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(diffuser_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(diffuser_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(diffuser_bench PRIVATE "-mfma")
    endif()

    add_executable(rtype_solver_check harnesses/cd/rtype_solver_check.cpp)
    target_link_libraries(rtype_solver_check PRIVATE SharedCode)

    add_executable(sallen_key_response_check harnesses/dsp/sallen_key_response_check.cpp)
    target_link_libraries(sallen_key_response_check PRIVATE SharedCode)

    add_executable(sallen_key_stability_check harnesses/dsp/sallen_key_stability_check.cpp)
    target_link_libraries(sallen_key_stability_check PRIVATE SharedCode)

    add_executable(filter_mode_check harnesses/dsp/filter_mode_check.cpp)
    target_link_libraries(filter_mode_check PRIVATE SharedCode)

    add_executable(sallen_key_bench harnesses/perf/sallen_key_bench.cpp)
    target_link_libraries(sallen_key_bench PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(sallen_key_bench PRIVATE /O2)
    else()
        target_compile_options(sallen_key_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(sallen_key_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(sallen_key_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(sallen_key_bench PRIVATE "-mfma")
    endif()

    add_executable(bbd_pole_bank_check harnesses/cd/bbd_pole_bank_check.cpp)
    target_link_libraries(bbd_pole_bank_check PRIVATE SharedCode)

    add_executable(bbd_line_check harnesses/dsp/bbd_line_check.cpp)
    target_link_libraries(bbd_line_check PRIVATE SharedCode)

    add_executable(bbd_clock_check harnesses/dsp/bbd_clock_check.cpp)
    target_link_libraries(bbd_clock_check PRIVATE SharedCode)

    add_executable(bbd_loop_check harnesses/dsp/bbd_loop_check.cpp)
    target_link_libraries(bbd_loop_check PRIVATE SharedCode)

    add_executable(bbd_response_check harnesses/dsp/bbd_response_check.cpp)
    target_link_libraries(bbd_response_check PRIVATE SharedCode)

    add_executable(bbd_alias_check harnesses/dsp/bbd_alias_check.cpp)
    target_link_libraries(bbd_alias_check PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(bbd_alias_check PRIVATE /O2)
    else()
        target_compile_options(bbd_alias_check PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(bbd_alias_check PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(bbd_alias_check PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(bbd_alias_check PRIVATE "-mfma")
    endif()

    add_executable(bbd_fuzz_check harnesses/dsp/bbd_fuzz_check.cpp)
    target_link_libraries(bbd_fuzz_check PRIVATE SharedCode)
    target_compile_definitions(bbd_fuzz_check PRIVATE NDEBUG)

    add_executable(bbd_bench harnesses/perf/bbd_bench.cpp)
    target_link_libraries(bbd_bench PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(bbd_bench PRIVATE /O2)
    else()
        target_compile_options(bbd_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(bbd_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(bbd_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(bbd_bench PRIVATE "-mfma")
    endif()

    add_executable(compander_cell_check harnesses/cd/compander_cell_check.cpp)
    target_link_libraries(compander_cell_check PRIVATE SharedCode)

    add_executable(compander_alias_check harnesses/dsp/compander_alias_check.cpp)
    target_link_libraries(compander_alias_check PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(compander_alias_check PRIVATE /O2)
    else()
        target_compile_options(compander_alias_check PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(compander_alias_check PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(compander_alias_check PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(compander_alias_check PRIVATE "-mfma")
    endif()

    add_executable(compander_loop_check harnesses/dsp/compander_loop_check.cpp)
    target_link_libraries(compander_loop_check PRIVATE SharedCode)

    add_executable(compander_bench harnesses/perf/compander_bench.cpp)
    target_link_libraries(compander_bench PRIVATE SharedCode)
    if(MSVC)
        target_compile_options(compander_bench PRIVATE /O2)
    else()
        target_compile_options(compander_bench PRIVATE -O2)
    endif()
    if(APPLE)
        target_compile_options(compander_bench PRIVATE "-Xarch_x86_64" "-mfma")
    elseif(MSVC)
        target_compile_options(compander_bench PRIVATE "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
        target_compile_options(compander_bench PRIVATE "-mfma")
    endif()

    # ── CTest registration ──────────────────────────────────────────────
    #
    # Correctness harnesses: one process each, no shared mutable state, no
    # sleeps, no temp files, no ports -> safe to run fully parallel. (The two
    # that need a second look are fine: diffuser_prepare_race_check's two
    # std::threads touch only local Diffuser instances; golden_render_check
    # writes tests/golden/hashes.txt only under --regen, which the registered
    # test does not pass, so it runs read-only compare.)
    #
    # Benchmarks measure ns/sample, so they must own the machine: RUN_SERIAL
    # and excluded from the parallel pass by the "bench" label. bench_gate
    # re-runs 6 perf harnesses (tan/adaa/delay_line/chain/fb/diffuser) against
    # committed baselines, so it carries "bench;gate": -LE bench keeps it out
    # of the correctness pass, -LE gate keeps it out of the individual-bench
    # pass so those 6 do not execute twice.
    #
    #   ctest --test-dir build-tests -j10 -LE bench    # 64 correctness, parallel
    #   ctest --test-dir build-tests -L bench -LE gate # 10 bench harnesses, serial
    #   ctest --test-dir build-tests -L gate           # bench_gate regression gate
    #
    # No TIMEOUT on correctness: build-tests is Debug, so the harnesses without
    # an explicit -O2 run at -O0 and the heavy ones (f2_minimax_check, fb_parity,
    # chain_parity, bbd_loop_check, ...) can take minutes; a fixed 300s would
    # kill them spuriously. Bench harnesses are -O2 and RUN_SERIAL, so 900s.

    set(CHRONOS_CD_TESTS
        ring_buffer_check halfsample_fir_check short_delay_check align_check
        latency_null_check chain_fuzz_check oversized_block_check
        engine_skeleton_check chain_parity fb_parity diffuser_toggle_check
        diffusion_onset_check diffuser_loop_check diffuser_transport_check
        diffuser_ring_check diffuser_decorrelation_check loop_gain_check
        bypass_null_check crossfade_parity dither_check bump_arena_check
        spsc_fifo_check
        tap_sim_check
        simd_delay_check frac_delay_tap_check dilog_check nonlinearity_check
        adaa2_check f2_oracle_check f2_minimax_check makeup_table_check
        alias_check simd_delay_parity diffuser_parity nested_allpass_check
        diffuser_lengths_check diffuser_prepare_race_check diffuser_density_check
        feedback_decay_check golden_render_check rt_alloc_check rt_stack_check
        state_roundtrip_check long_run_stability_check automation_sweep_check
        unity_transparency_check tempo_sync_check glide_rate_check
        delay_mod_check drive_loudness_check loop_dc_check crossfeed_power_check
        rtype_solver_check sallen_key_response_check sallen_key_stability_check
        filter_mode_check bbd_pole_bank_check bbd_line_check bbd_clock_check
        bbd_loop_check bbd_response_check bbd_alias_check bbd_fuzz_check
        compander_cell_check compander_alias_check compander_loop_check)

    set(CHRONOS_BENCH_TESTS
        tan_bench adaa_bench delay_line_bench chain_bench fb_bench
        diffuser_bench sallen_key_bench bbd_bench compander_bench prepare_bench)

    foreach(t IN LISTS CHRONOS_CD_TESTS CHRONOS_BENCH_TESTS)
        add_test(NAME ${t} COMMAND ${t})
    endforeach()

    set_tests_properties(${CHRONOS_CD_TESTS} PROPERTIES LABELS "cd")

    # RUN_SERIAL: ctest schedules nothing else while these run, so ns/sample
    # stays meaningful even under `ctest -j`.
    set_tests_properties(${CHRONOS_BENCH_TESTS} PROPERTIES
        LABELS "bench"
        RUN_SERIAL TRUE
        TIMEOUT 900)

    # Longest-first scheduling: ctest only reorders by measured time once it
    # has a prior run's Testing/Temporary/CTestCostData.txt. COST seeds that
    # on a cold checkout so the long poles start first under -j.
    set_tests_properties(
        chain_fuzz_check chain_parity fb_parity long_run_stability_check
        f2_minimax_check adaa2_check bbd_loop_check alias_check
        PROPERTIES COST 100)

    # bench_gate: 6 perf harnesses with --json, compared against
    # tests/baselines/*.json by scripts/bench_gate.py. "bench;gate" (see above).
    # CI runs 25% tolerance; locally on a pinned core:
    #   <bench> --json cur.json
    #   python scripts/python/bench_gate.py tests/baselines/<bench>.json cur.json --tolerance 15
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/baselines")
    add_test(NAME bench_gate
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/python/bench_gate.py"
                --ci --baselines "${CMAKE_CURRENT_SOURCE_DIR}/baselines"
                --bindir "${CMAKE_CURRENT_BINARY_DIR}" --tolerance 25)
    set_tests_properties(bench_gate PROPERTIES
        LABELS "bench;gate"
        RUN_SERIAL TRUE
        TIMEOUT 1800)
endif()

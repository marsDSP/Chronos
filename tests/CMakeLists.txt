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
endif()

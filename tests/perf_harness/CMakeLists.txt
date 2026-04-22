cmake_minimum_required(VERSION 3.23.1)
project(PerfHarness)

set(CMAKE_CXX_STANDARD 23)

add_executable(perf_test perf_sin_test.cpp)
add_executable(perf_cos_test perf_cos_test.cpp)
add_executable(perf_tan_test perf_tan_test.cpp)
add_executable(perf_tanh_test perf_tanh_test.cpp)
add_executable(perf_boundtopi_test perf_boundtopi_test.cpp)
add_executable(perf_delay_engine_test perf_delay_engine_test.cpp)

# New session: s-plane filter + aligned buffer.
add_executable(perf_splane_filter_test  perf_splane_filter_test.cpp)
add_executable(perf_aligned_buffer_test perf_aligned_buffer_test.cpp)

add_executable(perf_chronos_reverb_test perf_chronos_reverb_test.cpp)

# Enable optimizations for benchmark even in Debug profile
if(MSVC)
    target_compile_options(perf_test PRIVATE /O2)
    target_compile_options(perf_cos_test PRIVATE /O2)
    target_compile_options(perf_tan_test PRIVATE /O2)
    target_compile_options(perf_tanh_test PRIVATE /O2)
    target_compile_options(perf_boundtopi_test PRIVATE /O2)
    target_compile_options(perf_delay_engine_test PRIVATE /O2)
    target_compile_options(perf_splane_filter_test PRIVATE /O2)
    target_compile_options(perf_aligned_buffer_test PRIVATE /O2)
    target_compile_options(perf_chronos_reverb_test PRIVATE /O2)
else()
    target_compile_options(perf_test PRIVATE -O3)
    target_compile_options(perf_cos_test PRIVATE -O3)
    target_compile_options(perf_tan_test PRIVATE -O3)
    target_compile_options(perf_tanh_test PRIVATE -O3)
    target_compile_options(perf_boundtopi_test PRIVATE -O3)
    target_compile_options(perf_delay_engine_test PRIVATE -O3)
    target_compile_options(perf_splane_filter_test PRIVATE -O3)
    target_compile_options(perf_aligned_buffer_test PRIVATE -O3)
    target_compile_options(perf_chronos_reverb_test PRIVATE -O3)
endif()

# Link against SharedCode from the main project
target_link_libraries(perf_test PRIVATE SharedCode)
target_link_libraries(perf_cos_test PRIVATE SharedCode)
target_link_libraries(perf_tan_test PRIVATE SharedCode)
target_link_libraries(perf_tanh_test PRIVATE SharedCode)
target_link_libraries(perf_boundtopi_test PRIVATE SharedCode)
target_link_libraries(perf_delay_engine_test PRIVATE
    SharedCode
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_core
    juce::juce_dsp
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
)

# perf_splane_filter_test uses juce::dsp::IIR as its baseline, so it
# needs the JUCE audio modules.
target_link_libraries(perf_splane_filter_test PRIVATE
    SharedCode
    juce::juce_audio_basics
    juce::juce_audio_processors
    juce::juce_core
    juce::juce_dsp
    juce::juce_events
)

target_link_libraries(perf_aligned_buffer_test PRIVATE SharedCode)


# ChronosReverb perf is header-only and only needs SharedCode.
target_link_libraries(perf_chronos_reverb_test PRIVATE SharedCode)

# Ensure it's in a convenient location
set_target_properties(perf_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_cos_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_tan_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_tanh_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_boundtopi_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_delay_engine_test      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_splane_filter_test  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_aligned_buffer_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_chronos_reverb_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

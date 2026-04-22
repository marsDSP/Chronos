cmake_minimum_required(VERSION 3.23.1)
project(SimdHarness)

set(CMAKE_CXX_STANDARD 23)

add_executable(simd_test simd_sin_test.cpp)
add_executable(simd_cos_test simd_cos_test.cpp)
add_executable(simd_tan_test simd_tan_test.cpp)
add_executable(simd_tanh_test simd_tanh_test.cpp)
add_executable(simd_boundtopi_test simd_boundtopi_test.cpp)

add_executable(simd_delay_engine_test simd_delay_engine_test.cpp)
add_executable(simd_alignment_delay_test simd_alignment_delay_test.cpp)
add_executable(simd_alignment_math_test simd_alignment_math_test.cpp)
add_executable(delay_functional_test delay_functional_test.cpp)

# Diffusion / FDN test targets (new)
add_executable(diffusion_matrix_test diffusion_matrix_test.cpp)
add_executable(diffusion_chain_test diffusion_chain_test.cpp)
add_executable(fdn_test fdn_test.cpp)

# Link against SharedCode from the main project
target_link_libraries(simd_test PRIVATE SharedCode)
target_link_libraries(simd_cos_test PRIVATE SharedCode)
target_link_libraries(simd_tan_test PRIVATE SharedCode)
target_link_libraries(simd_tanh_test PRIVATE SharedCode)
target_link_libraries(simd_boundtopi_test PRIVATE SharedCode)
target_link_libraries(simd_delay_engine_test PRIVATE 
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

target_link_libraries(simd_alignment_delay_test PRIVATE 
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

target_link_libraries(simd_alignment_math_test PRIVATE SharedCode)

# Diffusion / FDN tests only depend on the new diffusion headers, which
# are part of SharedCode. No JUCE audio modules required.
target_link_libraries(diffusion_matrix_test PRIVATE SharedCode)
target_link_libraries(diffusion_chain_test  PRIVATE SharedCode)
target_link_libraries(fdn_test               PRIVATE SharedCode)

target_link_libraries(delay_functional_test PRIVATE
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

# Ensure it's in a convenient location
set_target_properties(simd_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_cos_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_tan_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_tanh_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_boundtopi_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_delay_engine_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_alignment_delay_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_alignment_math_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(delay_functional_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(diffusion_matrix_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(diffusion_chain_test  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(fdn_test               PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

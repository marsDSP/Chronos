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

# Engine-level unity-gain harness: OU / reverb-mix / delay-mix sweeps
add_executable(unity_gain_test unity_gain_test.cpp)

# ChronosReverb mix-amount stability sweep: regression harness for the
# "cranking the reverb send explodes the output" class of bug that
# killed the previous FDN-based send.
add_executable(chronos_reverb_amount_stability_test
               chronos_reverb_amount_stability_test.cpp)

# ChronosReverb smoke test: drags every new header in
# source/dsp/diffusion/chronos_reverb_*.h into a translation unit and
# runs a short block of noise through ChronosReverbStereoProcessor.
add_executable(chronos_reverb_smoke_test chronos_reverb_smoke_test.cpp)

# OU drift characterisation: stats, autocorrelation, distribution,
# mean-reversion trace across an amount sweep.
add_executable(ou_drift_characterisation_test ou_drift_characterisation_test.cpp)

# ChronosReverb characterisation: IR envelope, T60 vs decayTime,
# frequency response, stereo correlation vs modulation, mix sweep.
add_executable(chronos_reverb_characterisation_test
               chronos_reverb_characterisation_test.cpp)

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

# Unity-gain test drives the full DelayEngine through JUCE buffers, so
# it needs the same JUCE audio modules as the SIMD delay engine test.
target_link_libraries(unity_gain_test PRIVATE
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

# ChronosReverb amount stability harness also drives the full
# DelayEngine and therefore needs the same JUCE audio module stack as
# the other engine-level tests.
target_link_libraries(chronos_reverb_amount_stability_test PRIVATE
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

# ChronosReverb smoke test uses only the new header-only ChronosReverb
# building blocks, so it only needs SharedCode for include paths.
target_link_libraries(chronos_reverb_smoke_test PRIVATE SharedCode)

# OU drift characterisation uses OuDriftProcess (which pulls in JUCE
# dsp types via JuceHeader), so it needs the standard JUCE module
# stack.
target_link_libraries(ou_drift_characterisation_test PRIVATE
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

# ChronosReverb characterisation is header-only and just needs
# SharedCode for include paths.
target_link_libraries(chronos_reverb_characterisation_test PRIVATE SharedCode)

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
set_target_properties(unity_gain_test       PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(chronos_reverb_amount_stability_test
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(chronos_reverb_smoke_test
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(ou_drift_characterisation_test
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(chronos_reverb_characterisation_test
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

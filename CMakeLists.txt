cmake_minimum_required(VERSION 3.23.1)

# Suppress CMP0177 warning about install() DESTINATION paths normalization
if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW)
endif()

# General setup
set(PROJECT_NAME "Chronos")
set(CURRENT_VERSION "0.1.0")
set(PRODUCT_NAME "Chronos")
set(COMPANY_NAME "MarsDSP")
set(BUNDLE_ID "com.marsdsp.Chronos")
set(FORMATS Standalone VST3)
if(APPLE)
    list(APPEND FORMATS AU)
endif()

# Keep CMake project version in sync with CURRENT_VERSION defined above
project(${PROJECT_NAME} VERSION ${CURRENT_VERSION})

option(BUILD_AUDIO_PLUGIN_HOST "Build JUCE AudioPluginHost" OFF)
if(BUILD_AUDIO_PLUGIN_HOST)
    set(JUCE_BUILD_EXTRAS ON CACHE INTERNAL "")
endif()

add_subdirectory(libs/JUCE)

# Project configuration
set(CMAKE_CXX_STANDARD 23)

add_compile_options("$<$<CONFIG:Debug>:-gdwarf-4>")

# Find Python 3 for scripting support
set(Python3_FIND_VIRTUALENV STANDARD)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "Found Python3: ${Python3_EXECUTABLE}")

# macOS-specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum OS X deployment version" FORCE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Architectures")
    set(CMAKE_XCODE_ATTRIBUTE_MACOSX_DEPLOYMENT_TARGET[arch=arm64] "14.0" CACHE STRING "arm 64 minimum deployment target" FORCE)
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE INTERNAL "")
    # MSVC reports __cplusplus = 199711L by default, which fails Config.h's
    # static_assert(__cplusplus >= 202302L). /Zc:__cplusplus makes MSVC report
    # the real value (202302L under /std:c++latest, which cxx_std_23 selects).
    add_compile_options(/Zc:__cplusplus)
endif()

# Define the plugin
juce_add_plugin("${PROJECT_NAME}"
        COMPANY_NAME "${COMPANY_NAME}"
        BUNDLE_ID "${BUNDLE_ID}"
        COPY_PLUGIN_AFTER_BUILD FALSE
        PLUGIN_MANUFACTURER_CODE MDSP
        PLUGIN_CODE CHRO
        FORMATS ${FORMATS}
        PRODUCT_NAME "${PRODUCT_NAME}"
        NEEDS_WEB_BROWSER FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        IS_SYNTH FALSE
)

juce_generate_juce_header("${PROJECT_NAME}")

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_artefacts/JuceLibraryCode")

add_compile_definitions(JUCE_PROJECT_NAME="${PROJECT_NAME}")

# Define SharedCode as an INTERFACE library (no sources required)
add_library(SharedCode INTERFACE)

# Set compile features for SharedCode
target_compile_features(SharedCode INTERFACE cxx_std_23)

# Enable FMA3 on the x86_64 baseline so simde_mm_fmadd_ps lowers to a fused
# vfmadd*ps instead of a separate mul+add pair. arm64 has FMA unconditionally.
# Propagated through SharedCode INTERFACE so EVERY consumer (the plugin AND
# every test harness) gets the same FMA configuration — without this, a
# harness that links SharedCode but omits -mfma compiles FMADD as mul+add on
# x86_64, producing ~1 ULP different float32 results than the shipped binary.
if(APPLE)
    target_compile_options(SharedCode INTERFACE "-Xarch_x86_64" "-mfma")
elseif(MSVC)
    target_compile_options(SharedCode INTERFACE "/arch:AVX2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
    target_compile_options(SharedCode INTERFACE "-mfma")
endif()

# Include directories and compile definitions for SharedCode
cmake_policy(SET CMP0167 NEW)

target_include_directories(SharedCode INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/source"
        "${CMAKE_CURRENT_SOURCE_DIR}/libs"
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/JUCE/modules"
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/simde"
        "${GENERATED_DIR}"
)

# Set JUCE flags for SharedCode
target_compile_definitions(SharedCode INTERFACE
        JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        VERSION="${CURRENT_VERSION}"
        JUCE_DISPLAY_SPLASH_SCREEN=0
        PRODUCT_NAME_WITHOUT_VERSION="${PRODUCT_NAME}"
)

if(TRACY_ENABLE)
    target_compile_definitions(SharedCode INTERFACE TRACY_ENABLE)
endif()

# Add sources to the main project
file(GLOB_RECURSE SourceFiles CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.hpp"
)

# Sources to main project
target_sources("${PROJECT_NAME}" PRIVATE ${SourceFiles})

# Assets setup. Embed the knob artwork only. The raw font download under
# assets/fonts is excluded; label fonts come from the resources/fonts target below.
file(GLOB_RECURSE Assets CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/*.*")
list(FILTER Assets EXCLUDE REGEX "/fonts/")

if(Assets)
    juce_add_binary_data(AudioPluginData SOURCES ${Assets})
    target_link_libraries("${PROJECT_NAME}" PRIVATE AudioPluginData)
endif()

# Label typeface (section 4.5). The repository owner supplies the TTF files.
# Guard on the directory and a nonempty file list so a clean checkout without
# the fonts still configures and builds; the editor falls back to the default
# face when the resource is absent.
set(ChronosFontsDir "${CMAKE_CURRENT_SOURCE_DIR}/resources/fonts")
if(EXISTS "${ChronosFontsDir}" AND IS_DIRECTORY "${ChronosFontsDir}")
    file(GLOB ChronosFontFiles CONFIGURE_DEPENDS "${ChronosFontsDir}/*.ttf")
    if(ChronosFontFiles)
        juce_add_binary_data(ChronosFonts
                NAMESPACE ChronosFonts
                HEADER_NAME ChronosFonts.h
                SOURCES ${ChronosFontFiles})
        target_link_libraries("${PROJECT_NAME}" PRIVATE ChronosFonts)
    endif()
endif()
# Link libraries to main project
target_link_libraries("${PROJECT_NAME}" PUBLIC
        SharedCode
        PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Ensure the main project knows where its sources are
target_include_directories("${PROJECT_NAME}" PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/source"
        "${GENERATED_DIR}"
)

if(APPLE)
    target_compile_options("${PROJECT_NAME}" PRIVATE "-fno-objc-arc")

    file(GLOB_RECURSE JuceObjCSourceFiles "${CMAKE_CURRENT_SOURCE_DIR}/libs/JUCE/modules/*.mm")
    set_source_files_properties(
        ${JuceObjCSourceFiles}
        PROPERTIES COMPILE_FLAGS "-fno-objc-arc"
    )
endif()

# Enable FMA3 on the x86_64 baseline so simde_mm_fmadd_ps in
# source/math/Trigonometry.h lowers to a fused vfmadd*ps instead of a
# separate mul+add pair. arm64 has FMA unconditionally and needs no flag.
#
# This raises the x86_64 baseline from SSE4.2 to Haswell/FMA3 (and, on MSVC,
# AVX2 + FMA3 + BMI via /arch:AVX2). Pre-Haswell x86_64 CPUs — up to and
# including the 2013 Mac Pro (Ivy Bridge-EP), which has AVX but no FMA3 —
# would take a SIGILL; drop the relevant flag below if such hardware must
# keep working. Without these flags the code still builds — SIMDe falls
# back to a mul+add pair.
#
# Apple builds are universal (x86_64 + arm64), so -Xarch_x86_64 scopes
# -mfma to the x86_64 pass only (a bare -mfma would also reach the arm64
# pass and error). Linux x86_64 builds are single-arch, so a bare -mfma is
# correct. MSVC x64 uses /arch:AVX2.
if(APPLE)
    target_compile_options("${PROJECT_NAME}" PRIVATE "-Xarch_x86_64" "-mfma")
elseif(MSVC)
    target_compile_options("${PROJECT_NAME}" PRIVATE "/arch:AVX2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
    target_compile_options("${PROJECT_NAME}" PRIVATE "-mfma")
endif()

# Convenience run targets to launch AudioPluginHost for VST3 and AU debugging on macOS
if (BUILD_AUDIO_PLUGIN_HOST AND APPLE)
    # Path to the built AudioPluginHost app bundle
    # JUCE places the app inside <target_binary_dir>/AudioPluginHost_artefacts/<Config>/AudioPluginHost.app
    set(JUCE_HOST_APP "${CMAKE_BINARY_DIR}/libs/JUCE/extras/AudioPluginHost/AudioPluginHost_artefacts/$<CONFIG>/Audio Plugin Host.app")

    # Helper: print where the plugin artifacts are emitted
    add_custom_target(Print_Plugin_Paths ALL
            COMMAND ${CMAKE_COMMAND} -E echo "VST3: ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_artefacts/$<CONFIG>/VST3/${PRODUCT_NAME}.vst3"
            COMMAND ${CMAKE_COMMAND} -E echo "AU:   ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_artefacts/$<CONFIG>/AU/${PRODUCT_NAME}.component"
            COMMENT "Plugin artifact locations"
    )

    add_custom_target(Run_AudioPluginHost_VST3
            DEPENDS ${PROJECT_NAME}_VST3 AudioPluginHost Print_Plugin_Paths
            COMMAND ${CMAKE_COMMAND} -E echo "Launching AudioPluginHost for VST3..."
            COMMAND open ${JUCE_HOST_APP}
            COMMENT "Open AudioPluginHost.app (scan the VST3 folder shown above)"
    )

    add_custom_target(Run_AudioPluginHost_AU
            DEPENDS ${PROJECT_NAME}_AU AudioPluginHost Print_Plugin_Paths
            COMMAND ${CMAKE_COMMAND} -E echo "Launching AudioPluginHost for AU..."
            COMMAND open ${JUCE_HOST_APP}
            COMMENT "Open AudioPluginHost.app (scan the AU folder shown above)"
    )
endif()

# Test / benchmark harnesses (opt-in; see tests/CMakeLists.txt)
enable_testing()
add_subdirectory(tests)

# Post-build script to copy VST3 bundle to a local test folder (opt-in)
option(COPY_VST3_AFTER_BUILD "Copy built VST3 to a local test folder" OFF)
set(VST3_COPY_DEST "$ENV{HOME}/Desktop/vst test" CACHE STRING "Destination folder for post-build VST3 copy")
if(COPY_VST3_AFTER_BUILD AND TARGET ${PROJECT_NAME}_VST3)
    add_custom_command(TARGET ${PROJECT_NAME}_VST3 POST_BUILD
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bash/copy_vst3.sh" "$<TARGET_BUNDLE_DIR:${PROJECT_NAME}_VST3>" "${VST3_COPY_DEST}" "$<CONFIG>"
            COMMENT "Copying VST3 (config: $<CONFIG>) to ${VST3_COPY_DEST}..."
            VERBATIM
    )
endif()

cmake_minimum_required(VERSION 3.23.1)

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

add_subdirectory(modules/JUCE)
option(BUILD_AUDIO_PLUGIN_HOST "Build the JUCE AudioPluginHost app (optional)" ON)
if(BUILD_AUDIO_PLUGIN_HOST)
    add_subdirectory(modules/JUCE/extras/AudioPluginHost)
endif()

# Project configuration
set(CMAKE_CXX_STANDARD 20)

# macOS-specific settings
if(APPLE)
 set (CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version" FORCE)
 set (CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Architectures" FORCE)
 set (CMAKE_XCODE_ATTRIBUTE_MACOSX_DEPLOYMENT_TARGET[arch=arm64] "11.0" CACHE STRING "arm 64 minimum deployment target" FORCE)
endif()

if (UNIX AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
 find_package(PkgConfig REQUIRED)
 pkg_check_modules(WEBKIT2 REQUIRED webkit2gtk-4.0)

 include_directories(${WEBKIT2_INCLUDE_DIRS})
 link_directories(${WEBKIT2_LIBRARY_DIRS})
 add_definitions(${WEBKIT2_CFLAGS_OTHER})
endif()

# Disable all warnings on non-MSVC compilers
if(NOT MSVC)
    add_definitions(-w)
endif()

# Append custom module paths (if they exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/modules/cmake-include/sudara-cmake")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules/cmake-include/sudara-cmake")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libs/cmake-includes")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/cmake-includes")
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE INTERNAL "")
endif()

# Define the plugin
juce_add_plugin("${PROJECT_NAME}"
        COMPANY_NAME "${COMPANY_NAME}"
        BUNDLE_ID "${BUNDLE_ID}"
        COPY_PLUGIN_AFTER_BUILD FALSE
        PLUGIN_MANUFACTURER_CODE MDSP
        PLUGIN_CODE ChRn
        FORMATS ${FORMATS}
        PRODUCT_NAME "${PRODUCT_NAME}"
        NEEDS_WEB_BROWSER TRUE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        IS_SYNTH FALSE
)

add_compile_definitions(JUCE_PROJECT_NAME="${PROJECT_NAME}")

# Define SharedCode as an INTERFACE library (no sources required)
add_library(SharedCode INTERFACE
        source/Parameters.h
        source/Includes.h
        source/Converters.h
        source/DSP/TapeDelayDSP.h
        source/DSP/ProcessDSP.h
        source/Smoother.h)

# Set compile features for SharedCode
target_compile_features(SharedCode INTERFACE cxx_std_20)

# Include directories and compile definitions for SharedCode
target_include_directories(SharedCode INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/source"
        "${CMAKE_CURRENT_SOURCE_DIR}/modules/JUCE/modules"
        # Add top-level modules include so third-party module headers resolve
        # e.g. allows includes like <melatonin_inspector/...>
        "${CMAKE_CURRENT_SOURCE_DIR}/modules"
)

# Set JUCE flags for SharedCode
target_compile_definitions(SharedCode INTERFACE
        JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
        JUCE_WEB_BROWSER=1
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        VERSION="${CURRENT_VERSION}"
        JUCE_DISPLAY_SPLASH_SCREEN=0
        PRODUCT_NAME_WITHOUT_VERSION="${PRODUCT_NAME}"
)

# Add sources to the main project
file(GLOB_RECURSE SourceFiles CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.h"
)

# Sources to main project
target_sources("${PROJECT_NAME}" PRIVATE ${SourceFiles})

# Ensure melatonin_inspector's precompiled asset sources are linked so
# InspectorBinaryData symbols are available at link time
file(GLOB_RECURSE MelatoninInspectorAssets CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/modules/melatonin_inspector/LatestCompiledAssets/*.cpp")
if (MelatoninInspectorAssets)
    target_sources("${PROJECT_NAME}" PRIVATE ${MelatoninInspectorAssets})
endif()

# Assets setup
file(GLOB_RECURSE Assets "${CMAKE_CURRENT_SOURCE_DIR}/assets/*.*")

# No binary-data target currently; if you need to embed assets, use:
# juce_add_binary_data(AudioPluginData SOURCES ${Assets})
# and then link it to SharedCode and the plugin target.

# Link libraries to main project
 target_link_libraries("${PROJECT_NAME}" PRIVATE
         SharedCode
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
         juce::juce_product_unlocking
         PUBLIC
         juce::juce_recommended_config_flags
         juce::juce_recommended_lto_flags
         juce::juce_recommended_warning_flags
 )

# On Linux, link against webkit2gtk discovered via pkg-config when JUCE's web browser is enabled
if (UNIX AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries("${PROJECT_NAME}" PRIVATE ${WEBKIT2_LIBRARIES})
endif()

# Ensure the main project knows where its sources are
target_include_directories("${PROJECT_NAME}" PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/source"
)

# Convenience run targets to launch AudioPluginHost for VST3 and AU debugging on macOS
if (BUILD_AUDIO_PLUGIN_HOST AND APPLE)
    # Path to the built AudioPluginHost app bundle
    # JUCE places the app inside AudioPluginHost_artefacts/<Config>/AudioPluginHost.app
    set(JUCE_HOST_APP "${CMAKE_BINARY_DIR}/AudioPluginHost_build/AudioPluginHost_artefacts/$<CONFIG>/Audio Plugin Host.app")

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
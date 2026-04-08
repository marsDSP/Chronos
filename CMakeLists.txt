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

add_subdirectory(libs/JUCE)

# Project configuration
set(CMAKE_CXX_STANDARD 23)

# Find Python 3 for scripting support
set(Python3_FIND_VIRTUALENV STANDARD)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "Found Python3: ${Python3_EXECUTABLE}")

#[[
# Add a target to verify Python setup
add_custom_target(CheckPython
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/py3/CheckPy.py"
    COMMENT "Checking Python configuration..."
    VERBATIM
)
]]

# Run LogDumper
add_custom_target(Run_LogDumper
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/py3/LogDumper.py"
    COMMENT "Running LogDumper.py..."
    VERBATIM
)
#[[

# Run CheckCMake
add_custom_target(Run_CheckCMake
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/py3/CheckCMake.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/py3"
    COMMENT "Running CheckCMake.py..."
    VERBATIM
)
]]

# macOS-specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version" FORCE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Architectures" FORCE)
    set(CMAKE_XCODE_ATTRIBUTE_MACOSX_DEPLOYMENT_TARGET[arch=arm64] "11.0" CACHE STRING "arm 64 minimum deployment target" FORCE)
endif()

# Disable all warnings on non-MSVC compilers
if(NOT MSVC)
    add_definitions(-w)
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
add_library(SharedCode INTERFACE
        source/ChronosProcessor.h
        source/utils/helpers/conversion.h)

# Set compile features for SharedCode
target_compile_features(SharedCode INTERFACE cxx_std_23)

# Include directories and compile definitions for SharedCode
cmake_policy(SET CMP0167 NEW)
find_package(Boost REQUIRED)

target_include_directories(SharedCode INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/source"
        "${CMAKE_CURRENT_SOURCE_DIR}/libs"
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/JUCE/modules"
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/xsimd/include"
        "${GENERATED_DIR}"
)

target_link_libraries(SharedCode INTERFACE Boost::headers)

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

# Add sources to the main project
file(GLOB_RECURSE SourceFiles CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.hpp"
)

# Sources to main project
target_sources("${PROJECT_NAME}" PRIVATE ${SourceFiles})

# Assets setup
file(GLOB_RECURSE Assets CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/assets/*.*")

if(Assets)
    juce_add_binary_data(AudioPluginData SOURCES ${Assets})
    target_link_libraries("${PROJECT_NAME}" PRIVATE AudioPluginData)
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

#[[# Doxygen documentation generation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYFILE "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile")
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()]]

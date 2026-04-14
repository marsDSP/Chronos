cmake_minimum_required(VERSION 3.23.1)
project(PerfHarness)

set(CMAKE_CXX_STANDARD 23)

add_executable(perf_test perf_sin_test.cpp)
add_executable(perf_cos_test perf_cos_test.cpp)
add_executable(perf_tan_test perf_tan_test.cpp)

# Enable optimizations for benchmark even in Debug profile
if(MSVC)
    target_compile_options(perf_test PRIVATE /O2)
    target_compile_options(perf_cos_test PRIVATE /O2)
    target_compile_options(perf_tan_test PRIVATE /O2)
else()
    target_compile_options(perf_test PRIVATE -O3)
    target_compile_options(perf_cos_test PRIVATE -O3)
    target_compile_options(perf_tan_test PRIVATE -O3)
endif()

# Link against SharedCode from the main project
target_link_libraries(perf_test PRIVATE SharedCode)
target_link_libraries(perf_cos_test PRIVATE SharedCode)
target_link_libraries(perf_tan_test PRIVATE SharedCode)

# Ensure it's in a convenient location
set_target_properties(perf_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_cos_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(perf_tan_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

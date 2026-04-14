cmake_minimum_required(VERSION 3.23.1)
project(SimdHarness)

set(CMAKE_CXX_STANDARD 23)

add_executable(simd_test simd_sin_test.cpp)
add_executable(simd_cos_test simd_cos_test.cpp)
add_executable(simd_tan_test simd_tan_test.cpp)

# Link against SharedCode from the main project
target_link_libraries(simd_test PRIVATE SharedCode)
target_link_libraries(simd_cos_test PRIVATE SharedCode)
target_link_libraries(simd_tan_test PRIVATE SharedCode)

# Ensure it's in a convenient location
set_target_properties(simd_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_cos_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(simd_tan_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

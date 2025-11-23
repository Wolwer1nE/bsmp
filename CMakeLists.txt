cmake_minimum_required(VERSION 3.18)

project(bsmp LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Compile-time block size configuration
set(BSMP_BLOCK_SIZE 4 CACHE STRING "Compile-time block size for BlockSparseMatrix")
message(STATUS "BSMP_BLOCK_SIZE = ${BSMP_BLOCK_SIZE}")

# CUDA architectures (can be overridden: -DCMAKE_CUDA_ARCHITECTURES=86)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  # Default to a common baseline; adjust to your GPU for faster builds
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Library target
add_library(bsmp STATIC
  src/block_sparse_matrix.cu
  src/triplet_loader.cpp
  src/bicgstab.cu
  src/generalized_eigen.cu
)

target_include_directories(bsmp
  PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(bsmp PUBLIC BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Prefer modern CUDAToolkit package; fall back to CUDA language's cudart if needed
include(CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  target_link_libraries(bsmp PUBLIC CUDA::cudart CUDA::cublas)
else()
  message(STATUS "CUDAToolkit not found via package, relying on CUDA language cudart")
  # When CUDA is enabled as a language, the cudart library is available via cuda runtime library
  # On most systems linking is automatic, but we keep explicit to be safe when available
endif()

set_target_properties(bsmp PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

# Example executable
add_executable(example example.cu)

target_link_libraries(example PRIVATE bsmp)
target_compile_definitions(example PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Eigen example executable
add_executable(example_eigen example_eigen.cu)

target_link_libraries(example_eigen PRIVATE bsmp)
target_compile_definitions(example_eigen PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# BiCGStab example executable
add_executable(example_bicgstab example_bicgstab.cu)

target_link_libraries(example_bicgstab PRIVATE bsmp)
target_compile_definitions(example_bicgstab PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Nice warnings for host compiler (avoid pedantic warnings in CUDA front-end)
if(MSVC)
  target_compile_options(bsmp PRIVATE /W4)
else()
  target_compile_options(bsmp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  )
endif()

# Installation (optional)
install(TARGETS bsmp example
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

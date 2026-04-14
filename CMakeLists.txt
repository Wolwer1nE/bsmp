cmake_minimum_required(VERSION 3.18)

project(bsmp LANGUAGES CXX CUDA)

include("cmake/Setup.cmake")
# Prefer modern CUDAToolkit package; fall back to CUDA language's cudart if needed
include(CheckLanguage)
check_language(CUDA)
find_package(CUDAToolkit QUIET)

# Compile-time block size configuration
set(BSMP_BLOCK_SIZE 3)
message(STATUS "BSMP_BLOCK_SIZE = ${BSMP_BLOCK_SIZE}")

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

if(CUDAToolkit_FOUND)
  target_link_libraries(bsmp PUBLIC CUDA::cudart CUDA::cublas)
else()
  message(STATUS "CUDAToolkit not found via package, relying on CUDA language cudart")
  # When CUDA is enabled as a language, the cudart library is available via cuda runtime library
  # On most systems linking is automatic, but we keep explicit to be safe when available
endif()

# Now all targets should be setup with function from Setup.cmake
bsmp_setup_target(bsmp)

# Example executable
add_executable(example example.cu)
target_link_libraries(example PRIVATE bsmp)
target_compile_definitions(example PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

bsmp_setup_target(example)

# Eigen example executable
add_executable(example_eigen example_eigen.cu)
target_link_libraries(example_eigen PRIVATE bsmp)
target_compile_definitions(example_eigen PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

bsmp_setup_target(example_eigen)

# BiCGStab example executable
add_executable(example_bicgstab example_bicgstab.cu)
target_link_libraries(example_bicgstab PRIVATE bsmp)
target_compile_definitions(example_bicgstab PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

bsmp_setup_target(example_bicgstab)

# Installation (optional)
install(TARGETS bsmp example
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

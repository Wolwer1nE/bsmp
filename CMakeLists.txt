cmake_minimum_required(VERSION 3.18)

project(bsmp LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Compile-time block size configuration
set(BSMP_BLOCK_SIZE 3)
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
  src/amg_preconditioner.cu
  src/bicgstab.cu
  src/gmres.cu
  src/generalized_eigen.cu
  src/piezo_block_system.cu
  src/piezo_scaling.cu
  src/schur_operator.cu
  src/node_layout.cpp
  src/elastic_nullspace.cpp
  src/elastic_regularization.cu
  src/mass_orthogonalization.cu
  src/deflated_pcg_eigensolver.cu
  src/chebyshev_smoother.cu
  src/sa_amg_hierarchy.cpp
  src/sa_amg_preconditioner.cu
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
add_executable(example test/example.cu)

target_link_libraries(example PRIVATE bsmp)
target_compile_definitions(example PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Eigen example executable
add_executable(example_eigen test/example_eigen.cu)

target_link_libraries(example_eigen PRIVATE bsmp)
target_compile_definitions(example_eigen PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Article-style SA-AMG + deflated PCG eigen example executable
add_executable(example_sa_amg_pcg_eigen test/example_sa_amg_pcg_eigen.cu)

target_link_libraries(example_sa_amg_pcg_eigen PRIVATE bsmp)
if(CUDAToolkit_FOUND)
  target_link_libraries(example_sa_amg_pcg_eigen PRIVATE CUDA::cusolver)
endif()
target_compile_definitions(example_sa_amg_pcg_eigen PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# BiCGStab example executable
add_executable(example_bicgstab test/example_bicgstab.cu)

target_link_libraries(example_bicgstab PRIVATE bsmp)
target_compile_definitions(example_bicgstab PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# GMRES example executable
add_executable(example_gmres test/example_gmres.cu)

target_link_libraries(example_gmres PRIVATE bsmp)
target_compile_definitions(example_gmres PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Schur complement smoke test executable
add_executable(example_schur_smoke test/example_schur_smoke.cu)

target_link_libraries(example_schur_smoke PRIVATE bsmp)
target_compile_definitions(example_schur_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Mixed mechanical/electrical tail block smoke test executable
add_executable(example_mixed_block_tail_smoke test/example_mixed_block_tail_smoke.cpp)

target_link_libraries(example_mixed_block_tail_smoke PRIVATE bsmp)
target_compile_definitions(example_mixed_block_tail_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Rigid-body mode smoke test executable
add_executable(example_rbm_smoke test/example_rbm_smoke.cpp)

target_link_libraries(example_rbm_smoke PRIVATE bsmp)
target_compile_definitions(example_rbm_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Elastic regularization smoke test executable
add_executable(example_regularized_elastic_smoke test/example_regularized_elastic_smoke.cu)

target_link_libraries(example_regularized_elastic_smoke PRIVATE bsmp)
target_compile_definitions(example_regularized_elastic_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# SA-AMG scaffold smoke test executable
add_executable(example_sa_amg_scaffold_smoke test/example_sa_amg_scaffold_smoke.cu)

target_link_libraries(example_sa_amg_scaffold_smoke PRIVATE bsmp)
target_compile_definitions(example_sa_amg_scaffold_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# SA-AMG aggregation/prolongator smoke test executable
add_executable(example_sa_amg_aggregation_smoke test/example_sa_amg_aggregation_smoke.cpp)

target_link_libraries(example_sa_amg_aggregation_smoke PRIVATE bsmp)
target_compile_definitions(example_sa_amg_aggregation_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# SA-AMG Galerkin coarse operator smoke test executable
add_executable(example_sa_amg_galerkin_smoke test/example_sa_amg_galerkin_smoke.cpp)

target_link_libraries(example_sa_amg_galerkin_smoke PRIVATE bsmp)
target_compile_definitions(example_sa_amg_galerkin_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# SA-AMG smoothing smoke test executable
add_executable(example_sa_amg_smoothing_smoke test/example_sa_amg_smoothing_smoke.cpp)

target_link_libraries(example_sa_amg_smoothing_smoke PRIVATE bsmp)
target_compile_definitions(example_sa_amg_smoothing_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# SA-AMG two-level V-cycle smoke test executable
add_executable(example_sa_amg_vcycle_smoke test/example_sa_amg_vcycle_smoke.cpp)

target_link_libraries(example_sa_amg_vcycle_smoke PRIVATE bsmp)
target_compile_definitions(example_sa_amg_vcycle_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# SA-AMG Chebyshev smoother smoke test executable
add_executable(example_sa_amg_chebyshev_smoke test/example_sa_amg_chebyshev_smoke.cpp)

target_link_libraries(example_sa_amg_chebyshev_smoke PRIVATE bsmp)
target_compile_definitions(example_sa_amg_chebyshev_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Mass-orthogonalization smoke test executable
add_executable(example_mass_orthogonalization_smoke test/example_mass_orthogonalization_smoke.cpp)

target_link_libraries(example_mass_orthogonalization_smoke PRIVATE bsmp)
target_compile_definitions(example_mass_orthogonalization_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Deflated PCG eigensolver smoke test executable
add_executable(example_deflated_pcg_eigensolver_smoke test/example_deflated_pcg_eigensolver_smoke.cpp)

target_link_libraries(example_deflated_pcg_eigensolver_smoke PRIVATE bsmp)
target_compile_definitions(example_deflated_pcg_eigensolver_smoke PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

# Nice warnings for host compiler (avoid pedantic warnings in CUDA front-end)
if(MSVC)
  target_compile_options(bsmp PRIVATE /W4)
else()
  target_compile_options(bsmp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  )
endif()

# Installation (optional)
install(TARGETS bsmp example example_bicgstab example_gmres example_eigen example_sa_amg_pcg_eigen example_schur_smoke example_mixed_block_tail_smoke example_rbm_smoke example_regularized_elastic_smoke example_sa_amg_scaffold_smoke example_sa_amg_aggregation_smoke example_sa_amg_galerkin_smoke example_sa_amg_smoothing_smoke example_sa_amg_vcycle_smoke example_sa_amg_chebyshev_smoke example_mass_orthogonalization_smoke example_deflated_pcg_eigensolver_smoke
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

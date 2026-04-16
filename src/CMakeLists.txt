# Compile-time block size configuration
set(BSMP_BLOCK_SIZE 3)
message(STATUS "BSMP_BLOCK_SIZE = ${BSMP_BLOCK_SIZE}")

# Library target
add_library(bsmp STATIC
  block_sparse_matrix.cu
  triplet_loader.cpp
  amg_preconditioner.cu
  bicgstab.cu
  gmres.cu
  generalized_eigen.cu
  piezo_block_system.cu
  piezo_scaling.cu
  schur_operator.cu
  node_layout.cpp
  elastic_nullspace.cpp
  elastic_regularization.cu
  mass_orthogonalization.cu
  deflated_pcg_eigensolver.cu
  chebyshev_smoother.cu
  sa_amg_hierarchy.cpp
  sa_amg_preconditioner.cu
)

target_include_directories(bsmp PUBLIC
    ${CMAKE_SOURCE_DIR}/include # for public headers
    ${CMAKE_CURRENT_SOURCE_DIR} # if private headers will be needed
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
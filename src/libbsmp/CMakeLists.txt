# Compile-time block size configuration
set(BSMP_BLOCK_SIZE 3)
message(STATUS "BSMP_BLOCK_SIZE = ${BSMP_BLOCK_SIZE}")

# Library target
add_library(bsmp STATIC
  block_sparse_matrix.cu
  triplet_loader.cpp
  bicgstab.cu
  generalized_eigen.cu
)

target_include_directories(bsmp PUBLIC
    ${CMAKE_SOURCE_DIR}/include # for public headers
    ${CMAKE_CURRENT_SOURCE_DIR} # if private headers will be needed
)

# Treat CUDA headers as system headers to suppress warnings
target_include_directories(bsmp SYSTEM PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
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

find_program(CLANG_TIDY clang-tidy)
set(CLANG_TIDY_COMMAND ${CLANG_TIDY}
        ${IGNORE_SANITIZER}
        -extra-arg=${CPP_STANDARD_ARG}
        -extra-arg=-Wno-unknown-warning-option
        -extra-arg=-Wno-ignored-optimization-argument
        -extra-arg=-Wno-unused-command-line-argument
        -warnings-as-errors=*
        -p)

    set_target_properties(
        ${TARGET_NAME} PROPERTIES
        CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")


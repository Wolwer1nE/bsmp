
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

# Article-style SA-AMG + deflated PCG eigen example executable
add_executable(example_sa_amg_pcg_eigen example_sa_amg_pcg_eigen.cu)

target_link_libraries(example_sa_amg_pcg_eigen PRIVATE bsmp)
if(CUDAToolkit_FOUND)
  target_link_libraries(example_sa_amg_pcg_eigen PRIVATE CUDA::cusolver)
endif()
target_compile_definitions(example_sa_amg_pcg_eigen PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

bsmp_setup_target(example_sa_amg_pcg_eigen)

# BiCGStab example executable
add_executable(example_bicgstab example_bicgstab.cu)

target_link_libraries(example_bicgstab PRIVATE bsmp)
target_compile_definitions(example_bicgstab PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

bsmp_setup_target(example_bicgstab)

# GMRES example executable
add_executable(example_gmres example_gmres.cu)

target_link_libraries(example_gmres PRIVATE bsmp)
target_compile_definitions(example_gmres PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

bsmp_setup_target(example_gmres)

add_custom_target(examples)
add_dependencies(examples
    example
    example_eigen
    example_sa_amg_pcg_eigen
    example_bicgstab
    example_gmres
)
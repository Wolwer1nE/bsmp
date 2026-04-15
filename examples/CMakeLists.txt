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

cmake_minimum_required(VERSION 3.18)

project(bsmp LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

include("cmake/Setup.cmake")
include("cmake/FindCuda.cmake")
include("cmake/AddTest.cmake")
include("cmake/StaticAnalysis.cmake")
include("cmake/Sanitizer.cmake")

add_subdirectory(src)
add_subdirectory(examples)

add_subdirectory(test)

# Installation (optional)
install(TARGETS bsmp example example_bicgstab example_gmres example_eigen example_sa_amg_pcg_eigen example_schur_smoke example_mixed_block_tail_smoke example_rbm_smoke example_regularized_elastic_smoke example_sa_amg_scaffold_smoke example_sa_amg_aggregation_smoke example_sa_amg_galerkin_smoke example_sa_amg_smoothing_smoke example_sa_amg_vcycle_smoke example_sa_amg_chebyshev_smoke example_mass_orthogonalization_smoke example_deflated_pcg_eigensolver_smoke
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

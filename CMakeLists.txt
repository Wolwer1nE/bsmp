cmake_minimum_required(VERSION 3.18)

project(bsmp LANGUAGES CXX CUDA)

include("cmake/Setup.cmake")
include("cmake/FindCuda.cmake")

include_directories(include)

add_subdirectory("src/libbsmp")

add_subdirectory("examples")

# Installation (optional)
# install(TARGETS bsmp examples
#   RUNTIME DESTINATION bin
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   )

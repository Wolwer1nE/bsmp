include(CheckLanguage)
enable_language(CUDA)
check_language(CUDA)
find_package(CUDAToolkit QUIET)

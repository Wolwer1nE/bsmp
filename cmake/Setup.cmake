include("cmake/Warnings.cmake")

function(bsmp_setup_target TARGET_NAME)
    set(options DISABLE_CLANG_TIDY DISABLE_CLANG_TIDY_CUDA DISABLE_MEMCHECK DISABLE_WARNINGS)
    cmake_parse_arguments(BSMP_LOCAL "${options}" "" "" ${ARGN})

    set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )

    if(NOT BSMP_LOCAL_DISABLE_WARNINGS)
        bsmp_target_enable_warnings(${TARGET_NAME})
    endif()

    if(NOT BSMP_LOCAL_DISABLE_CLANG_TIDY)
        bsmp_target_enable_clang_tidy(${TARGET_NAME})
    endif()

    if(NOT BSMP_LOCAL_DISABLE_CLANG_TIDY_CUDA)
        bsmp_target_enable_clang_tidy_cuda(${TARGET_NAME})
    endif()

    if(NOT BSMP_LOCAL_DISABLE_MEMCHECK)
        bsmp_add_memcheck(${TARGET_NAME})
    endif()
endfunction()

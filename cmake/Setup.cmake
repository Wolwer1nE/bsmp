include("cmake/Warnings.cmake")

function(bsmp_setup_target TARGET_NAME)
    bsmp_target_enable_warnings(${TARGET_NAME})

    if (${BSMP_ENABLE_CLANG_TIDY})
        bsmp_target_enable_clang_tidy(${TARGET_NAME})
        bsmp_target_enable_clang_tidy_cuda(${TARGET_NAME})
    endif()
    if (${BSMP_ENABLE_COMPUTE_SANITIZER})
        bsmp_add_memcheck(${TARGET_NAME})
    endif()
endfunction()

include("cmake/Warnings.cmake")

function(bsmp_setup_target TARGET_NAME)
    set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    bsmp_target_enable_warnings(${TARGET_NAME})
    bsmp_target_enable_clang_tidy(${TARGET_NAME})
endfunction()
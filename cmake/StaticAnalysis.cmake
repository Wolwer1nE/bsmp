function(bsmp_target_enable_clang_tidy TARGET_NAME)

   find_program(CLANG_TIDY clang-tidy)
    if(NOT CLANG_TIDY)
        message(FATAL_ERROR "clang-tidy not found")
    endif()

    set(CLANG_TIDY_COMMAND ${CLANG_TIDY}
        ${IGNORE_SANITIZER}
        -extra-arg=-Wno-unknown-warning-option
        -extra-arg=-Wno-ignored-optimization-argument
        -extra-arg=-Wno-unused-command-line-argument
        -extra-arg=--cuda-path=${CUDAToolkit_LIBRARY_ROOT}
        # -extra-arg=--cuda-host-only
        -p)

    get_target_property(CPP_STANDARD ${TARGET_NAME} CXX_STANDARD)
    if("${CMAKE_CXX_CLANG_TIDY_DRIVER_MODE}" STREQUAL "cl")
      # clang does not support /Md + /fsanitize=address on Windows and we do not need it for sanitizer anyways
      set(CLANG_TIDY_COMMAND ${CLANG_TIDY_COMMAND} -extra-arg=/std:c++${CPP_STANDARD} --removed-arg=/fsanitize=address --removed-arg=-gencode)
    else()
      set(CLANG_TIDY_COMMAND ${CLANG_TIDY_COMMAND} -extra-arg=-std=c++${CPP_STANDARD})
    endif()

    set_target_properties(
        ${TARGET_NAME} PROPERTIES
        CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endfunction()

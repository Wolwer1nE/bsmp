function(bsmp_target_enable_clang_tidy TARGET_NAME)

   find_program(CLANG_TIDY NAMES clang-tidy clang-tidy-22)
    if(NOT CLANG_TIDY)
        message(FATAL_ERROR "clang-tidy not found")
    endif()

    set(CLANG_TIDY_COMMAND ${CLANG_TIDY}
        -extra-arg=-Wno-unknown-warning-option
        -extra-arg=-Wno-ignored-optimization-argument
        -extra-arg=-Wno-unused-command-line-argument
        -p)

    get_target_property(CPP_STANDARD ${TARGET_NAME} CXX_STANDARD)
    if("${CMAKE_CXX_CLANG_TIDY_DRIVER_MODE}" STREQUAL "cl")
      # clang does not support /Md + /fsanitize=address on Windows and we do not need it for sanitizer anyways
      set(CLANG_TIDY_COMMAND ${CLANG_TIDY_COMMAND} -extra-arg=/std:c++${CPP_STANDARD} --removed-arg=/fsanitize=address)
    else()
      set(CLANG_TIDY_COMMAND ${CLANG_TIDY_COMMAND} -extra-arg=-std=c++${CPP_STANDARD})
    endif()

    set_target_properties(
        ${TARGET_NAME} PROPERTIES
        CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endfunction()

function(bsmp_target_enable_clang_tidy_cuda TARGET_NAME)
    find_program(CLANG_TIDY clang-tidy)
    if(NOT CLANG_TIDY)
        message(FATAL_ERROR "clang-tidy not found")
    endif()

    message(STATUS "CUDA host compiler detected: ${CMAKE_CXX_COMPILER_ID}")

    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CUDA_HOST_COMPILER "cl")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CUDA_HOST_COMPILER "clang++")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CUDA_HOST_COMPILER "g++")
    else()
        message(FATAL_ERROR "Unsupported CUDA host compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()

    message(STATUS "CUDA COMPILER ${CUDAToolkit_LIBRARY_ROOT}")

    set(CLANG_TIDY_COMMAND ${CLANG_TIDY}
        --extra-arg-before=--driver-mode=${CUDA_HOST_COMPILER}
        --extra-arg-before=--cuda-path=${CUDAToolkit_LIBRARY_ROOT}
        --extra-arg=-Wno-unused-command-line-argument
        --extra-arg=-Wno-unknown-warning-option
        --extra-arg=-Wno-invalid-command-line-argument
        --extra-arg=-w
    )
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # apparently on windows path to ninja compile_commands is somehow different...
        set(CLANG_TIDY_COMMAND ${CLANG_TIDY_COMMAND} -p)
    else()
        set(CLANG_TIDY_COMMAND ${CLANG_TIDY_COMMAND} -p ${CMAKE_BINARY_DIR})
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_CLANG_TIDY "${CLANG_TIDY_COMMAND}"
    )
    message(NOTICE ${CLANG_TIDY_COMMAND})
endfunction()

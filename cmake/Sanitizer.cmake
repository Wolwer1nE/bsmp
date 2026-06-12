function(bsmp_add_memcheck TARGET_NAME)
    find_program(COMPUTE_SANITIZER
        NAMES compute-sanitizer
        HINTS ${CUDAToolkit_BIN_DIR}
    )
     if(NOT COMPUTE_SANITIZER)
        message(FATAL_ERROR "compute-sanitizer not found")
    endif()
    # Only add for executables
    get_target_property(target_type ${TARGET_NAME} TYPE)
    if(NOT target_type STREQUAL "EXECUTABLE")
        message(NOTICE "bsmp_add_memcheck only works for executables, skipping ${TARGET_NAME}")
        return()
    endif()
    
    target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:DEBUG>:-lineinfo>)
    
    if(CMAKE_CUDA_COMPILER)
        set(TEST_NAME "sanitizer.memcheck.${TARGET_NAME}")
        add_test(
            NAME ${TEST_NAME}
            COMMAND
                ${COMPUTE_SANITIZER}
                --tool memcheck
                --error-exitcode=1
                --leak-check=full
                $<TARGET_FILE:${TARGET_NAME}>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
        set_tests_properties(${TEST_NAME} PROPERTIES
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
            TIMEOUT 300
        )
    else()
        message(STATUS "CUDA not available, skipping memcheck for ${TARGET_NAME}")
    endif()
endfunction()

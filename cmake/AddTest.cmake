function(bsmp_add_smoke_test test_name target_executable)
    add_test(
        NAME ${test_name}
        COMMAND ${target_executable}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endfunction()
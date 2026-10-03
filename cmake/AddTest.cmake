include(GoogleTest)

function(bsmp_add_smoke_test test_name target_executable)
    add_test(
        NAME ${test_name}
        COMMAND ${target_executable}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endfunction()

function(bsmp_add_unit_test)
    cmake_parse_arguments(TEST "" "NAME" "SOURCES;LIBS" ${ARGN})

    add_executable(${TEST_NAME} ${TEST_SOURCES})

    target_link_libraries(${TEST_NAME} PRIVATE bsmp)
    target_compile_definitions(${TEST_NAME} PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

    bsmp_setup_target(${TEST_NAME} DISABLE_MEMCHECK)

    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

    target_link_libraries(${TEST_NAME} PRIVATE GTest::gtest_main GTest::gmock)

    if(TEST_LIBS)
        target_link_libraries(${TEST_NAME} PRIVATE ${TEST_LIBS})
    endif()

    gtest_discover_tests(${TEST_NAME}
        PREFIX "unit.")
endfunction()



set(CTEST_OUTPUT_ON_FAILURE TRUE)

file(GLOB SMOKE_TEST_SOURCES "*.cu")

add_custom_target(smoke_tests)

foreach(SMOKE_TEST_SOURCE ${SMOKE_TEST_SOURCES})
    # name without extension (for target name)
    get_filename_component(FILE_BASENAME ${SMOKE_TEST_SOURCE} NAME_WE)

    add_executable(${FILE_BASENAME} ${SMOKE_TEST_SOURCE})

    target_link_libraries(${FILE_BASENAME} PRIVATE bsmp)
    target_compile_definitions(${FILE_BASENAME} PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

    bsmp_setup_target(${FILE_BASENAME})

    # remove example as it is not beautiful
    string(REGEX REPLACE "^example_" "" TEST_NAME "${FILE_BASENAME}")
    bsmp_add_smoke_test("integration.smoke.${TEST_NAME}" "${FILE_BASENAME}")

    add_dependencies(smoke_tests "${FILE_BASENAME}")
endforeach()

set(CMAKE_FOLDER "tests/unit")

file(COPY ${CMAKE_SOURCE_DIR}/data/schemas/v1 DESTINATION
${CMAKE_CURRENT_BINARY_DIR}/data/schemas)

add_custom_target(unit_tests)

bsmp_add_unit_test(NAME test_json_recorder SOURCES test_json_recorder.cpp LIBS bsmp nlohmann_json::nlohmann_json
        nlohmann_json_schema_validator)

add_dependencies(unit_tests test_json_recorder)

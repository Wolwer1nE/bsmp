include(FetchContent)

if(BSMP_DOWNLOAD_THIRDPARTY)
    # Scope CMAKE_FOLDER so it doesn't leak to our own targets.
    set(CMAKE_FOLDER "thirdparty")

    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE)

    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.18.0
        GIT_SHALLOW TRUE)

    FetchContent_Declare(nlohmann_json_schema_validator
        GIT_REPOSITORY https://github.com/pboettch/json-schema-validator
        GIT_TAG 2.4.0
        GIT_SHALLOW TRUE)

    FetchContent_MakeAvailable(
        nlohmann_json spdlog googletest nlohmann_json_schema_validator)

    unset(CMAKE_FOLDER)   # or: set(CMAKE_FOLDER "${PROJECT_NAME}")
else()
    add_subdirectory("thirdparty/nlohmann_json")
    add_subdirectory("thirdparty/spdlog")
    add_subdirectory("thirdparty/googletest")
    add_subdirectory("thirdparty/nlohmann_json_schema_validator")
endif()

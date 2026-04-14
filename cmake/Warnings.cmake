function(bsmp_target_enable_warnings TARGET_NAME)
    target_compile_options(
        ${TARGET_NAME} PRIVATE

        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:
            -Xcompiler=/W4
            -Xcompiler=/permissive-
        >
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:
            /W4
            /permissive-
        >
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
           -Wall
           -Wextra
           -Wpedantic
        >
    )
endfunction()
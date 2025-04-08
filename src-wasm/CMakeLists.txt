if(NOT EMSCRIPTEN)
    set(EMSDK $ENV{EMSDK})
    # this is only for ide code completion - actual builds should use emcmake
    # e.g
    # source "/path/to/emsdk/emsdk_env.sh"
    # emmake cmake --build build-wasm
    include_directories(${EMSDK}/upstream/emscripten/system/include)
endif()

add_executable(crepe_wasm
        crepe.cpp
)

target_link_libraries(crepe_wasm PRIVATE
        crepe_core
)

target_compile_options(crepe_wasm PRIVATE
        -O3
        -flto
        -ffast-math
)

set_target_properties(crepe_wasm PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "-s WASM=1 \
                -s EXPORT_NAME='CrepeModule' \
                -s MODULARIZE=1 \
                -s EXPORTED_FUNCTIONS=[\"_analyse_audio\",\"_malloc\",\"_free\"] \
                -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
                -s ALLOW_MEMORY_GROWTH=1 \
                -s INITIAL_MEMORY=16MB \
                -s MAXIMUM_MEMORY=256MB \
                -s STACK_SIZE=2MB \
                -s TOTAL_MEMORY=32MB \
                -s WASM_MEM_MAX=256MB \
                -msimd128 \
                -O3 \
                -flto \
                -ffast-math"
)
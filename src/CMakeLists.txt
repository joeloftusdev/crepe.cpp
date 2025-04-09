add_library(crepe_core
        crepe.hpp
        inference.cpp
)

target_include_directories(crepe_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${DEPS_DIR}/eigen
)

if(EMSCRIPTEN)
    # wasm build
    target_link_libraries(crepe_core PUBLIC
            "${EXTERNAL_DIR}/onnxruntime/wasm/libonnxruntime_webassembly.a"
            Eigen3::Eigen
            embind
    )
else()
    # native mac build
    target_link_libraries(crepe_core PUBLIC
            "-framework Foundation"
            "-framework CoreML"
            "-framework CoreGraphics"
            "-framework Accelerate"
            "${EXTERNAL_DIR}/onnxruntime/macos-arm64_x86_64/libonnxruntime.a"
            Eigen3::Eigen
    )
endif()

target_sources(crepe_core PRIVATE
        ${MODEL_DIR}/model/model.ort.c
)
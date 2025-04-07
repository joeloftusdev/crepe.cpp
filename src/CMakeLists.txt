add_library(crepe_core
        crepe.hpp
        inference.cpp
)

target_include_directories(crepe_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${DEPS_DIR}/eigen
)

target_link_libraries(crepe_core PUBLIC
        "-framework Foundation"
        "-framework CoreML"
        "-framework CoreGraphics"
        "-framework Accelerate"
        "${EXTERNAL_DIR}/onnxruntime/macos-arm64_x86_64/libonnxruntime.a"
        Eigen3::Eigen
)

target_sources(crepe_core PRIVATE
        ${EXTERNAL_DIR}/model.ort.c
)
cmake_minimum_required(VERSION 3.16)
project(CREPE_ONNX)

set(CMAKE_CXX_STANDARD 20)

set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")

include_directories(${EXTERNAL_DIR}/onnxruntime.xcframework/Headers)

add_executable(CREPE_ONNX src/main.cpp)

target_link_libraries(CREPE_ONNX
        "${EXTERNAL_DIR}/onnxruntime.xcframework/macos-arm64_x86_64/libonnxruntime.a"
        "-framework Foundation"
        "-framework CoreML"
        "-framework CoreGraphics"
        "-framework Accelerate"
)

target_compile_definitions(CREPE_ONNX PRIVATE
        -DNDEBUG
)
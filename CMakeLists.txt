cmake_minimum_required(VERSION 3.16)
project(CREPE_ONNX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -fno-signed-zeros -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -funsafe-math-optimizations -fno-trapping-math -DNDEBUG")

set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")
include_directories(${EXTERNAL_DIR}/onnxruntime/include)

add_subdirectory(${DEPS_DIR}/eigen EXCLUDE_FROM_ALL)
include_directories(${DEPS_DIR}/eigen)
include_directories(${DEPS_DIR}/miniaudio)

add_executable(CREPE_ONNX
        src/main.cpp
        ${EXTERNAL_DIR}/model.ort.c
        src/wavloader.h
        src/wavloader.cpp
        src/crepe.hpp
        src/inference.cpp
)

target_link_libraries(CREPE_ONNX
        "-framework Foundation"
        "-framework CoreML"
        "-framework CoreGraphics"
        "-framework Accelerate"
        "${EXTERNAL_DIR}/onnxruntime/macos-arm64_x86_64/libonnxruntime.a"
        Eigen3::Eigen
)

target_compile_definitions(CREPE_ONNX PRIVATE
)
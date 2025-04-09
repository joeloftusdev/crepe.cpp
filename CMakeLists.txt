cmake_minimum_required(VERSION 3.16)
project(CREPE_ONNX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")
set(MODEL_DIR "${CMAKE_SOURCE_DIR}/crepe-model")

include_directories(${EXTERNAL_DIR}/onnxruntime/include)

add_subdirectory(${DEPS_DIR}/eigen EXCLUDE_FROM_ALL)

add_subdirectory(src)

if(EMSCRIPTEN)
    add_subdirectory(src-wasm)
else()
    add_subdirectory(src-cli)
    add_subdirectory(src-test)
endif()
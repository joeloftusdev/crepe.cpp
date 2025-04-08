add_executable(crepe_test
        crepe.cpp
)

target_include_directories(crepe_test PRIVATE
        ${DEPS_DIR}/miniaudio
)

target_link_libraries(crepe_test PRIVATE
        crepe_core
)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -fno-signed-zeros -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -funsafe-math-optimizations -fno-trapping-math -fno-rtti -DNDEBUG")

target_compile_definitions(crepe_test PRIVATE)

configure_file(
        ${CMAKE_SOURCE_DIR}/external/sweep.wav
        ${CMAKE_CURRENT_BINARY_DIR}/sweep.wav
        COPYONLY
)
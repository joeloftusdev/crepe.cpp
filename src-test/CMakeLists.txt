include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
)

FetchContent_MakeAvailable(Catch2)

add_executable(crepe_test
        crepe.cpp
)

target_sources(crepe_test PRIVATE
        ${DEPS_DIR}/miniaudio/miniaudio.c
)

target_include_directories(crepe_test PRIVATE
        ${DEPS_DIR}/miniaudio
)

target_link_libraries(crepe_test PRIVATE
        crepe_core
        Catch2::Catch2WithMain
)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -fno-signed-zeros -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -funsafe-math-optimizations -fno-trapping-math -fno-rtti -DNDEBUG")

target_compile_definitions(crepe_test PRIVATE)

configure_file(
        ${CMAKE_SOURCE_DIR}/src-test/sweep.wav
        ${CMAKE_CURRENT_BINARY_DIR}/sweep.wav
        COPYONLY
)
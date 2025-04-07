add_executable(crepe_test
        crepe.cpp
)

target_include_directories(crepe_test PRIVATE
        ${DEPS_DIR}/miniaudio
)

target_link_libraries(crepe_test PRIVATE
        crepe_core
)

target_compile_definitions(crepe_test PRIVATE)

configure_file(
        ${CMAKE_SOURCE_DIR}/external/sweep.wav
        ${CMAKE_CURRENT_BINARY_DIR}/sweep.wav
        COPYONLY
)
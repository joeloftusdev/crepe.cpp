add_executable(crepe_cli
        crepe.cpp
)

target_include_directories(crepe_cli PRIVATE
        ${CMAKE_SOURCE_DIR}/deps/readerwriterqueue
        ${DEPS_DIR}/miniaudio
)

target_link_libraries(crepe_cli PRIVATE
        crepe_core
)

target_compile_definitions(crepe_cli PRIVATE
        READERWRITERQUEUE_PATH="${CMAKE_SOURCE_DIR}/deps/queue/readerwriterqueue.h"
)
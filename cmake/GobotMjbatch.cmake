# Build the pinned upstream mjbatch core without its Python/nanobind bindings.
# Apply our native API adaptation to a build-tree copy, never the submodule.
set(GOBOT_MJBATCH_SOURCE "${PROJECT_SOURCE_DIR}/3rdparty/mjbatch/src/mjbatch/csrc")
set(GOBOT_MJBATCH_NATIVE "${CMAKE_BINARY_DIR}/_deps/mjbatch-native")
set(GOBOT_MJBATCH_PATCH "${PROJECT_SOURCE_DIR}/cmake/patches/mjbatch-native.patch")
if(NOT EXISTS "${GOBOT_MJBATCH_SOURCE}/batch.h")
    message(FATAL_ERROR "Initialize mjbatch: git submodule update --init 3rdparty/mjbatch")
endif()
find_program(GOBOT_PATCH_EXECUTABLE patch REQUIRED)
find_package(Threads REQUIRED)
file(MAKE_DIRECTORY "${GOBOT_MJBATCH_NATIVE}")
file(COPY "${GOBOT_MJBATCH_SOURCE}/batch.h" "${GOBOT_MJBATCH_SOURCE}/threadpool.h"
     DESTINATION "${GOBOT_MJBATCH_NATIVE}")
execute_process(COMMAND "${GOBOT_PATCH_EXECUTABLE}" --batch --forward -p1
                       -i "${GOBOT_MJBATCH_PATCH}"
                WORKING_DIRECTORY "${GOBOT_MJBATCH_NATIVE}"
                RESULT_VARIABLE GOBOT_MJBATCH_PATCH_RESULT
                OUTPUT_VARIABLE GOBOT_MJBATCH_PATCH_OUTPUT
                ERROR_VARIABLE GOBOT_MJBATCH_PATCH_ERROR)
if(NOT GOBOT_MJBATCH_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "mjbatch native patch failed: ${GOBOT_MJBATCH_PATCH_OUTPUT}${GOBOT_MJBATCH_PATCH_ERROR}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${GOBOT_MJBATCH_PATCH}" "${GOBOT_MJBATCH_SOURCE}/batch.h"
    "${GOBOT_MJBATCH_SOURCE}/threadpool.h")
add_library(gobot_mjbatch INTERFACE)
target_include_directories(gobot_mjbatch SYSTEM INTERFACE "${GOBOT_MJBATCH_NATIVE}")
target_compile_features(gobot_mjbatch INTERFACE cxx_std_20)
target_link_libraries(gobot_mjbatch INTERFACE ${GOBOT_MUJOCO_TARGET} Threads::Threads)

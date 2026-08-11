include_guard(GLOBAL)

set(GOBOT_LIBUIPC_SOLVER_RELATIVE_PATH
    "gobot/libgobot_libuipc_solver.so")
set(GOBOT_LIBUIPC_LIBRARY_RELATIVE_PATHS
    "gobot/libuipc/Release/bin/libuipc_backend_cuda.so"
    "gobot/libuipc/Release/bin/libuipc_backend_none.so"
    "gobot/libuipc/Release/bin/libuipc_constitution.so"
    "gobot/libuipc/Release/bin/libuipc_core.so"
    "gobot/libuipc/Release/bin/libuipc_geometry.so"
    "gobot/libuipc/Release/bin/libuipc_io.so"
    "gobot/libuipc/Release/bin/libuipc_sanity_check.so")
set(GOBOT_LIBUIPC_LICENSE_RELATIVE_PATHS
    "gobot/licenses/Octree/LICENSE"
    "gobot/licenses/GKlib/LICENSE.txt"
    "gobot/licenses/METIS/LICENSE"
    "gobot/licenses/cppitertools/LICENSE.md"
    "gobot/licenses/cpptrace/LICENSE"
    "gobot/licenses/dylib/LICENSE"
    "gobot/licenses/fmt/LICENSE.rst"
    "gobot/licenses/libigl/LICENSE.GPL"
    "gobot/licenses/libigl/LICENSE.MPL2"
    "gobot/licenses/libuipc/LICENSE"
    "gobot/licenses/libuipc/NOTICE"
    "gobot/licenses/magic_enum/LICENSE"
    "gobot/licenses/muda/LICENSE"
    "gobot/licenses/nlohmann_json/LICENSE.MIT"
    "gobot/licenses/spdlog/LICENSE"
    "gobot/licenses/tinygltf/LICENSE")
set(GOBOT_LIBUIPC_BUNDLE_RELATIVE_PATHS
    "${GOBOT_LIBUIPC_SOLVER_RELATIVE_PATH}"
    ${GOBOT_LIBUIPC_LIBRARY_RELATIVE_PATHS}
    ${GOBOT_LIBUIPC_LICENSE_RELATIVE_PATHS})

function(gobot_validate_libuipc_runpath FILE_PATH EXPECTED_RUNPATH READELF)
    execute_process(
        COMMAND "${READELF}" -d "${FILE_PATH}"
        RESULT_VARIABLE GOBOT_LIBUIPC_READELF_RESULT
        OUTPUT_VARIABLE GOBOT_LIBUIPC_DYNAMIC_SECTION
        ERROR_VARIABLE GOBOT_LIBUIPC_READELF_ERROR)
    if(NOT GOBOT_LIBUIPC_READELF_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to inspect prebuilt libuipc ELF file ${FILE_PATH}: "
            "${GOBOT_LIBUIPC_READELF_ERROR}")
    endif()

    string(REGEX MATCH
        "\\((RPATH|RUNPATH)\\)[^\n]*\\[([^]]*)\\]"
        GOBOT_LIBUIPC_RUNPATH_MATCH
        "${GOBOT_LIBUIPC_DYNAMIC_SECTION}")
    set(GOBOT_LIBUIPC_ACTUAL_RUNPATH "${CMAKE_MATCH_2}")
    if(NOT GOBOT_LIBUIPC_ACTUAL_RUNPATH)
        message(FATAL_ERROR
            "Prebuilt libuipc ELF file has no RUNPATH: ${FILE_PATH}")
    endif()
    if(GOBOT_LIBUIPC_ACTUAL_RUNPATH MATCHES "(^|:)/")
        message(FATAL_ERROR
            "Prebuilt libuipc ELF file has an absolute RUNPATH "
            "(${GOBOT_LIBUIPC_ACTUAL_RUNPATH}): ${FILE_PATH}")
    endif()
    if(NOT "${GOBOT_LIBUIPC_ACTUAL_RUNPATH}" STREQUAL "${EXPECTED_RUNPATH}")
        message(FATAL_ERROR
            "Prebuilt libuipc ELF file has RUNPATH "
            "${GOBOT_LIBUIPC_ACTUAL_RUNPATH}, expected ${EXPECTED_RUNPATH}: "
            "${FILE_PATH}")
    endif()
    if(GOBOT_LIBUIPC_DYNAMIC_SECTION MATCHES
       "Shared library: \\[libpython[^]]*\\]")
        message(FATAL_ERROR
            "Prebuilt libuipc ELF file must not depend on libpython: ${FILE_PATH}")
    endif()
endfunction()

function(gobot_use_libuipc_prebuilt_bundle BUNDLE_ROOT)
    if(NOT IS_DIRECTORY "${BUNDLE_ROOT}")
        message(FATAL_ERROR
            "GOB_LIBUIPC_PREBUILT_ROOT is not a directory: ${BUNDLE_ROOT}")
    endif()

    foreach(GOBOT_LIBUIPC_RELATIVE_PATH IN LISTS
            GOBOT_LIBUIPC_BUNDLE_RELATIVE_PATHS)
        if(NOT EXISTS "${BUNDLE_ROOT}/${GOBOT_LIBUIPC_RELATIVE_PATH}")
            message(FATAL_ERROR
                "Prebuilt libuipc bundle is missing "
                "${GOBOT_LIBUIPC_RELATIVE_PATH}")
        endif()
    endforeach()

    file(GLOB_RECURSE GOBOT_LIBUIPC_ACTUAL_RELATIVE_PATHS
        LIST_DIRECTORIES FALSE
        RELATIVE "${BUNDLE_ROOT}"
        "${BUNDLE_ROOT}/*")
    list(SORT GOBOT_LIBUIPC_ACTUAL_RELATIVE_PATHS)
    set(GOBOT_LIBUIPC_EXPECTED_RELATIVE_PATHS
        ${GOBOT_LIBUIPC_BUNDLE_RELATIVE_PATHS})
    list(SORT GOBOT_LIBUIPC_EXPECTED_RELATIVE_PATHS)
    if(NOT "${GOBOT_LIBUIPC_ACTUAL_RELATIVE_PATHS}" STREQUAL
       "${GOBOT_LIBUIPC_EXPECTED_RELATIVE_PATHS}")
        message(FATAL_ERROR
            "Prebuilt libuipc bundle file list does not match the Release "
            "runtime contract. Found: ${GOBOT_LIBUIPC_ACTUAL_RELATIVE_PATHS}")
    endif()

    find_program(GOBOT_LIBUIPC_READELF NAMES readelf llvm-readelf)
    if(NOT GOBOT_LIBUIPC_READELF)
        message(FATAL_ERROR
            "GOB_LIBUIPC_PREBUILT_ROOT requires readelf or llvm-readelf.")
    endif()
    gobot_validate_libuipc_runpath(
        "${BUNDLE_ROOT}/${GOBOT_LIBUIPC_SOLVER_RELATIVE_PATH}"
        "\$ORIGIN:\$ORIGIN/libuipc/Release/bin"
        "${GOBOT_LIBUIPC_READELF}")
    foreach(GOBOT_LIBUIPC_RELATIVE_PATH IN LISTS
            GOBOT_LIBUIPC_LIBRARY_RELATIVE_PATHS)
        gobot_validate_libuipc_runpath(
            "${BUNDLE_ROOT}/${GOBOT_LIBUIPC_RELATIVE_PATH}"
            "\$ORIGIN"
            "${GOBOT_LIBUIPC_READELF}")
    endforeach()

    set(GOBOT_LIBUIPC_BUILD_PACKAGE_DIR
        "${PROJECT_BINARY_DIR}/python/gobot")
    file(MAKE_DIRECTORY
        "${GOBOT_LIBUIPC_BUILD_PACKAGE_DIR}/libuipc/Release/bin")
    file(COPY
        "${BUNDLE_ROOT}/${GOBOT_LIBUIPC_SOLVER_RELATIVE_PATH}"
        DESTINATION "${GOBOT_LIBUIPC_BUILD_PACKAGE_DIR}")
    foreach(GOBOT_LIBUIPC_RELATIVE_PATH IN LISTS
            GOBOT_LIBUIPC_LIBRARY_RELATIVE_PATHS)
        file(COPY "${BUNDLE_ROOT}/${GOBOT_LIBUIPC_RELATIVE_PATH}"
            DESTINATION
                "${GOBOT_LIBUIPC_BUILD_PACKAGE_DIR}/libuipc/Release/bin")
    endforeach()

    foreach(GOBOT_LIBUIPC_INSTALL_COMPONENT python libuipc_runtime)
        install(DIRECTORY "${BUNDLE_ROOT}/gobot/"
            DESTINATION "gobot"
            USE_SOURCE_PERMISSIONS
            COMPONENT "${GOBOT_LIBUIPC_INSTALL_COMPONENT}")
    endforeach()
endfunction()

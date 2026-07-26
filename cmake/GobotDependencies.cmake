include_guard(GLOBAL)

include(ExternalProject)
include(ProcessorCount)

set(GOBOT_DEPENDENCY_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/3rdparty" CACHE PATH
    "Root containing Gobot's pinned dependency sources")
set(GOBOT_DEPENDENCY_BUILD_ROOT "${CMAKE_SOURCE_DIR}/build" CACHE PATH
    "Root used for isolated dependency builds and installs")

option(GOB_DEPENDENCY_BUILD_LUISA "build the pinned LuisaCompute CUDA SDK" ON)
option(GOB_DEPENDENCY_BUILD_GSPLAT "build the Gobot gsplat CUDA inference SDK" ON)
option(GOB_DEPENDENCY_BUILD_OPENUSD "build the pinned minimal OpenUSD SDK" ON)
option(GOB_LUISA_ALLOW_DIRTY_SOURCE
       "allow an intentionally modified LuisaCompute dependency checkout"
       OFF)

ProcessorCount(GOBOT_DETECTED_PROCESSOR_COUNT)
if(NOT GOBOT_DETECTED_PROCESSOR_COUNT OR GOBOT_DETECTED_PROCESSOR_COUNT LESS 1)
    set(GOBOT_DETECTED_PROCESSOR_COUNT 1)
endif()
set(GOB_DEPENDENCY_JOBS "${GOBOT_DETECTED_PROCESSOR_COUNT}" CACHE STRING
    "Parallel jobs used for LuisaCompute and gsplat dependency builds")
if(GOBOT_DETECTED_PROCESSOR_COUNT GREATER 4)
    set(GOBOT_DEFAULT_OPENUSD_JOBS 4)
else()
    set(GOBOT_DEFAULT_OPENUSD_JOBS "${GOBOT_DETECTED_PROCESSOR_COUNT}")
endif()
set(GOB_OPENUSD_JOBS "${GOBOT_DEFAULT_OPENUSD_JOBS}" CACHE STRING
    "Parallel jobs used for OpenUSD; the conservative default limits peak memory")

set(GOB_LUISA_SOURCE_DIR "${GOBOT_DEPENDENCY_SOURCE_ROOT}/luisa_compute" CACHE PATH
    "Pinned LuisaCompute source checkout")
set(GOB_LUISA_BUILD_DIR "${GOBOT_DEPENDENCY_BUILD_ROOT}/luisa_compute/sdk-build" CACHE PATH
    "LuisaCompute SDK build directory")
set(GOB_LUISA_INSTALL_DIR "${GOBOT_DEPENDENCY_BUILD_ROOT}/luisa_compute/install" CACHE PATH
    "LuisaCompute SDK install prefix")
set(GOB_LUISA_C_COMPILER "" CACHE FILEPATH "C compiler for the isolated LuisaCompute build")
set(GOB_LUISA_CXX_COMPILER "" CACHE FILEPATH "C++ compiler for the isolated LuisaCompute build")

set(GOB_GSPLAT_SOURCE_DIR "${GOBOT_DEPENDENCY_SOURCE_ROOT}/gsplat_inference" CACHE PATH
    "Gobot gsplat inference source directory")
set(GOB_GSPLAT_BUILD_DIR "${GOBOT_DEPENDENCY_BUILD_ROOT}/gsplat_inference/sdk-build" CACHE PATH
    "gsplat inference SDK build directory")
set(GOB_GSPLAT_INSTALL_DIR "${GOBOT_DEPENDENCY_BUILD_ROOT}/gsplat_inference/install" CACHE PATH
    "gsplat inference SDK install prefix")
set(GOBOT_GSPLAT_CUDA_ARCHITECTURES
    "75-real;80-real;86-real;89-real;90-real"
    CACHE STRING "CUDA architectures compiled into the gsplat inference library")

set(GOB_OPENUSD_SOURCE_DIR "${GOBOT_DEPENDENCY_SOURCE_ROOT}/openusd" CACHE PATH
    "Pinned OpenUSD source checkout")
set(GOB_OPENUSD_BUILD_DIR "${GOBOT_DEPENDENCY_BUILD_ROOT}/openusd/sdk-build" CACHE PATH
    "OpenUSD SDK build directory")
set(GOB_OPENUSD_INSTALL_DIR "${GOBOT_DEPENDENCY_BUILD_ROOT}/openusd/install" CACHE PATH
    "OpenUSD SDK install prefix")
set(GOB_ONETBB_SOURCE_DIR "${GOBOT_DEPENDENCY_SOURCE_ROOT}/onetbb" CACHE PATH
    "Pinned oneTBB source checkout")
set(GOB_ONETBB_BUILD_DIR "${GOBOT_DEPENDENCY_BUILD_ROOT}/onetbb/sdk-build" CACHE PATH
    "oneTBB SDK build directory")

function(gobot_require_dependency_file path help_text)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "${help_text}\nMissing: ${path}")
    endif()
endfunction()

add_custom_target(gobot_dependencies ALL)
set(GOBOT_PREVIOUS_DEPENDENCY_TARGET "")

if(GOB_DEPENDENCY_BUILD_LUISA)
    gobot_require_dependency_file(
        "${GOB_LUISA_SOURCE_DIR}/CMakeLists.txt"
        "Initialize the pinned source with: git submodule update --init 3rdparty/luisa_compute")
    foreach(GOBOT_LUISA_NESTED_DEPENDENCY reproc spdlog)
        gobot_require_dependency_file(
            "${GOB_LUISA_SOURCE_DIR}/src/ext/${GOBOT_LUISA_NESTED_DEPENDENCY}/CMakeLists.txt"
            "Initialize LuisaCompute's nested dependencies with: git submodule update --init --recursive 3rdparty/luisa_compute")
    endforeach()

    if(NOT GOB_LUISA_ALLOW_DIRTY_SOURCE AND EXISTS "${GOB_LUISA_SOURCE_DIR}/.git")
        execute_process(
            COMMAND git -C "${GOB_LUISA_SOURCE_DIR}" status --porcelain --untracked-files=all
            OUTPUT_VARIABLE GOBOT_LUISA_DIRTY_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE GOBOT_LUISA_GIT_RESULT)
        if(GOBOT_LUISA_GIT_RESULT EQUAL 0 AND GOBOT_LUISA_DIRTY_STATUS)
            message(FATAL_ERROR
                "LuisaCompute or one of its nested dependencies is dirty:\n"
                "${GOBOT_LUISA_DIRTY_STATUS}\n"
                "Use a clean pinned checkout or set GOB_LUISA_ALLOW_DIRTY_SOURCE=ON.")
        endif()
    endif()

    if(NOT GOB_LUISA_C_COMPILER)
        find_program(GOBOT_GCC_11 NAMES gcc-11)
        if(GOBOT_GCC_11)
            set(GOB_LUISA_C_COMPILER "${GOBOT_GCC_11}")
        else()
            find_program(GOB_LUISA_C_COMPILER NAMES cc gcc REQUIRED)
        endif()
    endif()
    if(NOT GOB_LUISA_CXX_COMPILER)
        find_program(GOBOT_GXX_11 NAMES g++-11)
        if(GOBOT_GXX_11)
            set(GOB_LUISA_CXX_COMPILER "${GOBOT_GXX_11}")
        else()
            find_program(GOB_LUISA_CXX_COMPILER NAMES c++ g++ REQUIRED)
        endif()
    endif()

    ExternalProject_Add(gobot_luisa_sdk
        SOURCE_DIR "${GOB_LUISA_SOURCE_DIR}"
        BINARY_DIR "${GOB_LUISA_BUILD_DIR}"
        INSTALL_DIR "${GOB_LUISA_INSTALL_DIR}"
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        BUILD_ALWAYS TRUE
        EXCLUDE_FROM_ALL TRUE
        CMAKE_ARGS
            "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
            "-DCMAKE_C_COMPILER:FILEPATH=${GOB_LUISA_C_COMPILER}"
            "-DCMAKE_CXX_COMPILER:FILEPATH=${GOB_LUISA_CXX_COMPILER}"
            "-DCMAKE_BUILD_TYPE:STRING=Release"
            "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O2 -DNDEBUG"
            "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
            "-DCMAKE_INSTALL_MESSAGE:STRING=LAZY"
            "-DLUISA_COMPUTE_ENABLE_DSL:BOOL=ON"
            "-DLUISA_COMPUTE_ENABLE_CUDA:BOOL=ON"
            "-DLUISA_COMPUTE_ENABLE_DX:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_METAL:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_HIP:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_VULKAN:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_CPU:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_FALLBACK:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_REMOTE:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_GUI:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_RUST:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_TENSOR:BOOL=OFF"
            "-DLUISA_COMPUTE_ENABLE_CLANG_CXX:BOOL=OFF"
            "-DLUISA_COMPUTE_USE_SYSTEM_STL:BOOL=ON"
            "-DLUISA_COMPUTE_BUILD_TESTS:BOOL=OFF"
            "-DLUISA_COMPUTE_DOWNLOAD_OIDN:BOOL=OFF"
            "-DLUISA_COMPUTE_DOWNLOAD_NVCOMP:BOOL=OFF"
        BUILD_COMMAND
            "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel "${GOB_DEPENDENCY_JOBS}"
        INSTALL_COMMAND
            "${CMAKE_COMMAND}" --install <BINARY_DIR>)
    add_dependencies(gobot_dependencies gobot_luisa_sdk)
    set(GOBOT_PREVIOUS_DEPENDENCY_TARGET gobot_luisa_sdk)
endif()

if(GOB_DEPENDENCY_BUILD_GSPLAT)
    gobot_require_dependency_file(
        "${GOB_GSPLAT_SOURCE_DIR}/CMakeLists.txt"
        "The vendored gsplat inference source is incomplete")
    string(REPLACE ";" "|" GOBOT_GSPLAT_CUDA_ARCHITECTURES_EP
           "${GOBOT_GSPLAT_CUDA_ARCHITECTURES}")
    ExternalProject_Add(gobot_gsplat_sdk
        SOURCE_DIR "${GOB_GSPLAT_SOURCE_DIR}"
        BINARY_DIR "${GOB_GSPLAT_BUILD_DIR}"
        INSTALL_DIR "${GOB_GSPLAT_INSTALL_DIR}"
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        BUILD_ALWAYS TRUE
        EXCLUDE_FROM_ALL TRUE
        LIST_SEPARATOR "|"
        CMAKE_ARGS
            "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
            "-DCMAKE_BUILD_TYPE:STRING=Release"
            "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
            "-DCMAKE_INSTALL_MESSAGE:STRING=LAZY"
            "-DGOBOT_GSPLAT_CUDA_ARCHITECTURES:STRING=${GOBOT_GSPLAT_CUDA_ARCHITECTURES_EP}"
        BUILD_COMMAND
            "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel "${GOB_DEPENDENCY_JOBS}"
        INSTALL_COMMAND
            "${CMAKE_COMMAND}" --install <BINARY_DIR>)
    if(GOBOT_PREVIOUS_DEPENDENCY_TARGET)
        add_dependencies(gobot_gsplat_sdk "${GOBOT_PREVIOUS_DEPENDENCY_TARGET}")
    endif()
    add_dependencies(gobot_dependencies gobot_gsplat_sdk)
    set(GOBOT_PREVIOUS_DEPENDENCY_TARGET gobot_gsplat_sdk)
endif()

if(GOB_DEPENDENCY_BUILD_OPENUSD)
    gobot_require_dependency_file(
        "${GOB_OPENUSD_SOURCE_DIR}/pxr/pxrConfig.cmake.in"
        "Initialize the pinned source with: git submodule update --init --depth=1 3rdparty/openusd")
    gobot_require_dependency_file(
        "${GOB_ONETBB_SOURCE_DIR}/CMakeLists.txt"
        "Initialize the pinned source with: git submodule update --init --depth=1 3rdparty/onetbb")

    ExternalProject_Add(gobot_onetbb_sdk
        SOURCE_DIR "${GOB_ONETBB_SOURCE_DIR}"
        BINARY_DIR "${GOB_ONETBB_BUILD_DIR}"
        INSTALL_DIR "${GOB_OPENUSD_INSTALL_DIR}"
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        BUILD_ALWAYS TRUE
        EXCLUDE_FROM_ALL TRUE
        CMAKE_ARGS
            "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
            "-DCMAKE_BUILD_TYPE:STRING=Release"
            "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
            "-DCMAKE_INSTALL_MESSAGE:STRING=LAZY"
            "-DTBB_TEST:BOOL=OFF"
            "-DTBB_EXAMPLES:BOOL=OFF"
            "-DTBB_STRICT:BOOL=OFF"
        BUILD_COMMAND
            "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel "${GOB_OPENUSD_JOBS}"
        INSTALL_COMMAND
            "${CMAKE_COMMAND}" --install <BINARY_DIR>)
    if(GOBOT_PREVIOUS_DEPENDENCY_TARGET)
        add_dependencies(gobot_onetbb_sdk "${GOBOT_PREVIOUS_DEPENDENCY_TARGET}")
    endif()

    ExternalProject_Add(gobot_openusd_sdk
        SOURCE_DIR "${GOB_OPENUSD_SOURCE_DIR}"
        BINARY_DIR "${GOB_OPENUSD_BUILD_DIR}"
        INSTALL_DIR "${GOB_OPENUSD_INSTALL_DIR}"
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        BUILD_ALWAYS TRUE
        EXCLUDE_FROM_ALL TRUE
        DEPENDS gobot_onetbb_sdk
        CMAKE_ARGS
            "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
            "-DCMAKE_BUILD_TYPE:STRING=Release"
            "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
            "-DCMAKE_INSTALL_MESSAGE:STRING=LAZY"
            "-DBUILD_SHARED_LIBS:BOOL=ON"
            "-DPXR_ENABLE_PYTHON_SUPPORT:BOOL=OFF"
            "-DPXR_BUILD_TESTS:BOOL=OFF"
            "-DPXR_BUILD_EXAMPLES:BOOL=OFF"
            "-DPXR_BUILD_TUTORIALS:BOOL=OFF"
            "-DPXR_BUILD_USD_TOOLS:BOOL=OFF"
            "-DPXR_BUILD_IMAGING:BOOL=OFF"
            "-DPXR_BUILD_USD_IMAGING:BOOL=OFF"
            "-DPXR_BUILD_USD_VALIDATION:BOOL=OFF"
            "-DPXR_BUILD_EXEC:BOOL=OFF"
            "-DPXR_ENABLE_GL_SUPPORT:BOOL=OFF"
            "-DPXR_FIND_TBB_IN_CONFIG:BOOL=ON"
            "-DTBB_DIR:PATH=<INSTALL_DIR>/lib/cmake/TBB"
        BUILD_COMMAND
            "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel "${GOB_OPENUSD_JOBS}"
        INSTALL_COMMAND
            "${CMAKE_COMMAND}" --build <BINARY_DIR> --target install --parallel "${GOB_OPENUSD_JOBS}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "<INSTALL_DIR>/share/licenses/OpenUSD"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${GOB_OPENUSD_SOURCE_DIR}/LICENSE.txt"
                    "<INSTALL_DIR>/share/licenses/OpenUSD/LICENSE.txt"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "<INSTALL_DIR>/share/licenses/oneTBB"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${GOB_ONETBB_SOURCE_DIR}/LICENSE.txt"
                    "<INSTALL_DIR>/share/licenses/oneTBB/LICENSE.txt")
    add_dependencies(gobot_dependencies gobot_openusd_sdk)
    set(GOBOT_PREVIOUS_DEPENDENCY_TARGET gobot_openusd_sdk)
endif()

if(NOT GOB_DEPENDENCY_BUILD_LUISA AND
   NOT GOB_DEPENDENCY_BUILD_GSPLAT AND
   NOT GOB_DEPENDENCY_BUILD_OPENUSD)
    message(WARNING "No dependency SDK was selected; gobot_dependencies has no work.")
endif()

cmake_minimum_required (VERSION 3.16)
project (gobot VERSION 0.1.11)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GOBOT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(GOBOT_VERSION "${PROJECT_VERSION}")
set(GOBOT_VERSION_COMMIT "" CACHE STRING "Git commit or build identifier embedded in Gobot version output")

option(GOB_BUILD_TESTS "build tests" OFF)
option(GOB_BUILD_OPENUSD "build OpenUSD scene importer/exporter support" OFF)
option(GOB_BUILD_ASSIMP "build mesh import support through Assimp" ON)
option(GOB_BUILD_MUJOCO "build MuJoCo CPU physics backend support" ON)
option(GOB_BUILD_EGL "build EGL headless OpenGL rendering support" ON)
option(GOB_BUILD_TASK_LLVM "build LLVM/Clang runtime compiler for RL task kernels" ON)
option(GOB_FETCH_MUJOCO "download MuJoCo from the official GitHub repository when GOB_BUILD_MUJOCO is ON and no local SDK/package is found" ON)
option(GOB_ENABLE_TRACY "enable Tracy profiler instrumentation" OFF)
set(GOB_MUJOCO_ROOT "" CACHE PATH "MuJoCo SDK or install prefix. Supports either a CMake package install or an extracted SDK with include/ and lib/.")
set(GOB_LLVM_ROOT "" CACHE PATH "LLVM/Clang install prefix for Gobot RL task kernel compilation.")
set(GOB_TRACY_ROOT "" CACHE PATH "Tracy install prefix or source checkout used when GOB_ENABLE_TRACY is ON")
set(GOB_MUJOCO_GIT_TAG "3.8.0" CACHE STRING "MuJoCo release tag used when GOB_FETCH_MUJOCO is ON")
set(GOB_ASSIMP_IMPORTERS "COLLADA;OBJ;STL;PLY;GLTF" CACHE STRING "Assimp importers enabled for Gobot mesh import")

if(NOT UNIX OR APPLE)
    message(FATAL_ERROR "Only Linux is supported.")
endif()

include(cmake/CPM.cmake)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

find_package(Eigen3 REQUIRED)

set(GOBOT_EGL_TARGET "")
if(GOB_BUILD_EGL)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GOBOT_EGL QUIET IMPORTED_TARGET egl)
    endif()
    if(TARGET PkgConfig::GOBOT_EGL)
        set(GOBOT_EGL_TARGET PkgConfig::GOBOT_EGL)
    else()
        find_path(GOBOT_EGL_INCLUDE_DIR NAMES EGL/egl.h)
        find_library(GOBOT_EGL_LIBRARY NAMES EGL)
        if(GOBOT_EGL_INCLUDE_DIR AND GOBOT_EGL_LIBRARY)
            add_library(gobot_egl_sdk UNKNOWN IMPORTED GLOBAL)
            set_target_properties(gobot_egl_sdk
                PROPERTIES
                    IMPORTED_LOCATION "${GOBOT_EGL_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GOBOT_EGL_INCLUDE_DIR}"
            )
            set(GOBOT_EGL_TARGET gobot_egl_sdk)
        endif()
    endif()
    if(GOBOT_EGL_TARGET)
        message(STATUS "Gobot EGL headless rendering enabled with target: ${GOBOT_EGL_TARGET}")
    else()
        message(WARNING "GOB_BUILD_EGL is ON, but EGL was not found. Python RGB capture will report unavailable.")
    endif()
endif()

CPMFindPackage(
    NAME nlohmann_json
    VERSION 3.11.3
    GITHUB_REPOSITORY nlohmann/json
    OPTIONS "JSON_BuildTests OFF"
)

if(GOB_BUILD_OPENUSD)
    find_package(pxr REQUIRED CONFIG)
endif()

set(GOBOT_TASK_LLVM_AVAILABLE OFF)
if(GOB_BUILD_TASK_LLVM)
    set(GOBOT_LLVM_ROOT_CANDIDATES)
    if(GOB_LLVM_ROOT)
        list(APPEND GOBOT_LLVM_ROOT_CANDIDATES "${GOB_LLVM_ROOT}")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llvm/lib/cmake/llvm")
        list(APPEND GOBOT_LLVM_ROOT_CANDIDATES "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llvm")
    endif()

    find_package(LLVM CONFIG QUIET PATHS ${GOBOT_LLVM_ROOT_CANDIDATES} NO_DEFAULT_PATH)
    find_package(Clang CONFIG QUIET PATHS ${GOBOT_LLVM_ROOT_CANDIDATES} NO_DEFAULT_PATH)
    if(NOT LLVM_FOUND)
        find_package(LLVM CONFIG QUIET)
    endif()
    if(NOT Clang_FOUND)
        find_package(Clang CONFIG QUIET)
    endif()

    if(LLVM_FOUND AND Clang_FOUND)
        set(GOBOT_TASK_LLVM_AVAILABLE ON)
        message(STATUS "Gobot task LLVM enabled with LLVM ${LLVM_PACKAGE_VERSION}")
    else()
        message(WARNING "GOB_BUILD_TASK_LLVM is ON, but LLVM/Clang was not found. @gobot.rl.kernel LLVM compilation will be unavailable.")
    endif()
endif()

set(GOBOT_TRACY_TARGET "")
if(GOB_ENABLE_TRACY)
    set(GOBOT_TRACY_ROOT_CANDIDATE "${GOB_TRACY_ROOT}")
    if(NOT GOBOT_TRACY_ROOT_CANDIDATE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tracy/public/TracyClient.cpp")
        set(GOBOT_TRACY_ROOT_CANDIDATE "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tracy")
    endif()

    find_package(Tracy CONFIG QUIET)

    if(NOT TARGET Tracy::TracyClient AND GOBOT_TRACY_ROOT_CANDIDATE)
        find_package(Tracy CONFIG QUIET
            PATHS
                "${GOBOT_TRACY_ROOT_CANDIDATE}"
                "${GOBOT_TRACY_ROOT_CANDIDATE}/lib/cmake/Tracy"
                "${GOBOT_TRACY_ROOT_CANDIDATE}/lib64/cmake/Tracy"
            NO_DEFAULT_PATH
        )
    endif()

    if(TARGET Tracy::TracyClient)
        set(GOBOT_TRACY_TARGET Tracy::TracyClient)
    elseif(GOBOT_TRACY_ROOT_CANDIDATE AND EXISTS "${GOBOT_TRACY_ROOT_CANDIDATE}/public/TracyClient.cpp")
        add_library(gobot_tracy_client STATIC "${GOBOT_TRACY_ROOT_CANDIDATE}/public/TracyClient.cpp")
        target_include_directories(gobot_tracy_client PUBLIC "${GOBOT_TRACY_ROOT_CANDIDATE}/public")
        target_compile_definitions(gobot_tracy_client PUBLIC TRACY_ENABLE=1)
        set(GOBOT_TRACY_TARGET gobot_tracy_client)
    endif()

    if(GOBOT_TRACY_TARGET)
        message(STATUS "Gobot Tracy profiling enabled with target: ${GOBOT_TRACY_TARGET}")
    else()
        message(FATAL_ERROR
            "GOB_ENABLE_TRACY is ON, but Tracy was not found.\n"
            "Provide a package/prefix with -DGOB_TRACY_ROOT=/path/to/tracy or -DCMAKE_PREFIX_PATH=/path/to/tracy,\n"
            "or initialize the optional 3rdparty/tracy submodule."
        )
    endif()
endif()

set(GOBOT_MUJOCO_TARGET "")
if(GOB_BUILD_MUJOCO)
    if(GOB_MUJOCO_ROOT)
        find_package(mujoco CONFIG QUIET
            PATHS
                "${GOB_MUJOCO_ROOT}"
                "${GOB_MUJOCO_ROOT}/lib/cmake/mujoco"
                "${GOB_MUJOCO_ROOT}/lib64/cmake/mujoco"
            NO_DEFAULT_PATH
        )
    endif()

    if(NOT mujoco_FOUND)
        find_package(mujoco CONFIG QUIET)
    endif()

    if(TARGET mujoco::mujoco)
        set(GOBOT_MUJOCO_TARGET mujoco::mujoco)
    elseif(TARGET mujoco)
        set(GOBOT_MUJOCO_TARGET mujoco)
    endif()

    if(NOT GOBOT_MUJOCO_TARGET)
        find_path(GOBOT_MUJOCO_INCLUDE_DIR
            NAMES mujoco/mujoco.h
            HINTS
                "${GOB_MUJOCO_ROOT}/include"
                "${GOB_MUJOCO_ROOT}"
        )
        find_library(GOBOT_MUJOCO_LIBRARY
            NAMES mujoco
            HINTS
                "${GOB_MUJOCO_ROOT}/lib"
                "${GOB_MUJOCO_ROOT}/lib64"
                "${GOB_MUJOCO_ROOT}/bin"
                "${GOB_MUJOCO_ROOT}"
        )

        if(GOBOT_MUJOCO_INCLUDE_DIR AND GOBOT_MUJOCO_LIBRARY)
            add_library(gobot_mujoco_sdk SHARED IMPORTED GLOBAL)
            set_target_properties(gobot_mujoco_sdk
                PROPERTIES
                    IMPORTED_LOCATION "${GOBOT_MUJOCO_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GOBOT_MUJOCO_INCLUDE_DIR}"
            )
            set(GOBOT_MUJOCO_TARGET gobot_mujoco_sdk)
            message(STATUS "Gobot MuJoCo backend enabled with SDK library: ${GOBOT_MUJOCO_LIBRARY}")
        endif()
    endif()

    if(NOT GOBOT_MUJOCO_TARGET AND GOB_FETCH_MUJOCO)
        include(FetchContent)
        # MuJoCo examples build sample apps through GLFW. Keep this disabled for
        # wheel builds so packaged Python distributions do not need extra
        # desktop build tools.
        set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)
        set(MUJOCO_BUILD_STUDIO OFF CACHE BOOL "" FORCE)
        set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(MUJOCO_TEST_PYTHON_UTIL OFF CACHE BOOL "" FORCE)
        set(MUJOCO_WITH_USD OFF CACHE BOOL "" FORCE)
        set(MUJOCO_USE_FILAMENT OFF CACHE BOOL "" FORCE)
        set(SUPPORTS_GOLD FALSE CACHE BOOL "Disable gold for fetched MuJoCo; it does not support CMake linker dependency files." FORCE)

        FetchContent_Declare(
            mujoco
            GIT_REPOSITORY https://github.com/google-deepmind/mujoco.git
            GIT_TAG "${GOB_MUJOCO_GIT_TAG}"
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(mujoco)

        if(TARGET mujoco::mujoco)
            set(GOBOT_MUJOCO_TARGET mujoco::mujoco)
        elseif(TARGET mujoco)
            set(GOBOT_MUJOCO_TARGET mujoco)
        endif()
    endif()

    if(GOBOT_MUJOCO_TARGET)
        message(STATUS "Gobot MuJoCo backend enabled with target: ${GOBOT_MUJOCO_TARGET}")
    else()
        message(FATAL_ERROR
            "GOB_BUILD_MUJOCO is ON, but MuJoCo was not found.\n"
            "Provide a local SDK/package with -DGOB_MUJOCO_ROOT=/path/to/mujoco or -DCMAKE_PREFIX_PATH=/path/to/mujoco,\n"
            "or allow CMake to fetch the pinned official release with -DGOB_FETCH_MUJOCO=ON."
        )
    endif()
endif()

set(GOBOT_ASSIMP_TARGET "")
if(GOB_BUILD_ASSIMP)
    set(GOBOT_ASSIMP_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/assimp")
    if(EXISTS "${GOBOT_ASSIMP_SUBMODULE_DIR}/CMakeLists.txt")
        if(CMAKE_VERSION VERSION_LESS 3.22)
            message(WARNING "3rdparty/assimp requires CMake 3.22 or newer; falling back to system assimp package.")
        else()
            set(GOBOT_RESTORE_BUILD_SHARED_LIBS OFF)
            if(DEFINED BUILD_SHARED_LIBS)
                set(GOBOT_RESTORE_BUILD_SHARED_LIBS ON)
                set(GOBOT_PRE_ASSIMP_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
            endif()
            set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build package with shared libraries." FORCE)
            set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
            set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
            set(ASSIMP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
            set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
            set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
            set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
            set(ASSIMP_IGNORE_GIT_HASH ON CACHE BOOL "" FORCE)
            set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
            set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
            set(ASSIMP_BUILD_MINIZIP ON CACHE BOOL "" FORCE)
            set(UNZIP_FOUND OFF CACHE BOOL "" FORCE)
            set(UNZIP_LIBRARIES "" CACHE STRING "" FORCE)
            set(UNZIP_INCLUDE_DIRS "" CACHE STRING "" FORCE)
            set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
            set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
            foreach(GOBOT_ASSIMP_IMPORTER IN LISTS GOB_ASSIMP_IMPORTERS)
                set("ASSIMP_BUILD_${GOBOT_ASSIMP_IMPORTER}_IMPORTER" ON CACHE BOOL "" FORCE)
            endforeach()

            add_subdirectory("${GOBOT_ASSIMP_SUBMODULE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/assimp" EXCLUDE_FROM_ALL)
            if(GOBOT_RESTORE_BUILD_SHARED_LIBS)
                set(BUILD_SHARED_LIBS "${GOBOT_PRE_ASSIMP_BUILD_SHARED_LIBS}" CACHE BOOL "Build package with shared libraries." FORCE)
            else()
                unset(BUILD_SHARED_LIBS CACHE)
            endif()
            if(TARGET assimp::assimp)
                set(GOBOT_ASSIMP_TARGET assimp::assimp)
            elseif(TARGET assimp)
                set(GOBOT_ASSIMP_TARGET assimp)
            endif()
        endif()
    endif()

    if(NOT GOBOT_ASSIMP_TARGET)
        find_package(assimp QUIET)
        if(TARGET assimp::assimp)
            set(GOBOT_ASSIMP_TARGET assimp::assimp)
        elseif(TARGET assimp)
            set(GOBOT_ASSIMP_TARGET assimp)
        endif()
    endif()

    if(GOBOT_ASSIMP_TARGET)
        message(STATUS "Gobot mesh import enabled with Assimp target: ${GOBOT_ASSIMP_TARGET}")
    else()
        message(WARNING "GOB_BUILD_ASSIMP is ON, but Assimp was not found. Mesh import support will be disabled.")
    endif()
endif()

set(GOBOT_RESTORE_BUILD_SHARED_LIBS OFF)
if(DEFINED BUILD_SHARED_LIBS)
    set(GOBOT_RESTORE_BUILD_SHARED_LIBS ON)
    set(GOBOT_PRE_LOGGING_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build package with shared libraries." FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "Build spdlog as a static library." FORCE)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "Build spdlog with position independent code." FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
CPMAddPackage("gh:fmtlib/fmt#10.1.1")

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION 1.13.0
    OPTIONS "SPDLOG_FMT_EXTERNAL ON"
)
if(GOBOT_RESTORE_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${GOBOT_PRE_LOGGING_BUILD_SHARED_LIBS}" CACHE BOOL "Build package with shared libraries." FORCE)
else()
    unset(BUILD_SHARED_LIBS CACHE)
endif()

CPMAddPackage("gh:Neargye/magic_enum@0.9.5")
if(magic_enum_SOURCE_DIR)
    target_include_directories(magic_enum INTERFACE ${magic_enum_SOURCE_DIR}/include)
endif()
CPMAddPackage("gh:jarro2783/cxxopts@3.0.0")

find_package(Python3 QUIET COMPONENTS Interpreter Development.Module Development.Embed)
if(NOT Python3_FOUND)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()
set(GOBOT_PYTHON_LIBRARY "" CACHE FILEPATH
        "Optional default libpython path compiled into gobot_editor; runtime GOBOT_PYTHON_LIBRARY overrides it.")
if(NOT GOBOT_PYTHON_LIBRARY)
    set(GOBOT_INFERRED_PYTHON_LIBRARY "")
    if(TARGET Python3::Python)
        foreach(GOBOT_PYTHON_LOCATION_PROPERTY
                IMPORTED_LOCATION
                IMPORTED_LOCATION_RELEASE
                IMPORTED_LOCATION_RELWITHDEBINFO
                IMPORTED_LOCATION_MINSIZEREL
                IMPORTED_LOCATION_DEBUG)
            if(NOT GOBOT_INFERRED_PYTHON_LIBRARY)
                get_target_property(GOBOT_PYTHON_LOCATION
                        Python3::Python
                        ${GOBOT_PYTHON_LOCATION_PROPERTY})
                if(GOBOT_PYTHON_LOCATION AND EXISTS "${GOBOT_PYTHON_LOCATION}" AND NOT IS_DIRECTORY "${GOBOT_PYTHON_LOCATION}")
                    set(GOBOT_INFERRED_PYTHON_LIBRARY "${GOBOT_PYTHON_LOCATION}")
                endif()
            endif()
        endforeach()
    endif()
    if(NOT GOBOT_INFERRED_PYTHON_LIBRARY AND Python3_LIBRARIES)
        foreach(GOBOT_PYTHON_LOCATION IN LISTS Python3_LIBRARIES)
            if(EXISTS "${GOBOT_PYTHON_LOCATION}" AND NOT IS_DIRECTORY "${GOBOT_PYTHON_LOCATION}")
                set(GOBOT_INFERRED_PYTHON_LIBRARY "${GOBOT_PYTHON_LOCATION}")
                break()
            endif()
        endforeach()
    endif()
    if(NOT GOBOT_INFERRED_PYTHON_LIBRARY AND Python3_LIBRARY_RELEASE)
        if(EXISTS "${Python3_LIBRARY_RELEASE}" AND NOT IS_DIRECTORY "${Python3_LIBRARY_RELEASE}")
            set(GOBOT_INFERRED_PYTHON_LIBRARY "${Python3_LIBRARY_RELEASE}")
        endif()
    endif()
    if(GOBOT_INFERRED_PYTHON_LIBRARY)
        set(GOBOT_PYTHON_LIBRARY "${GOBOT_INFERRED_PYTHON_LIBRARY}" CACHE FILEPATH
                "Optional default libpython path compiled into gobot_editor; runtime GOBOT_PYTHON_LIBRARY overrides it." FORCE)
    endif()
endif()
if(GOBOT_PYTHON_LIBRARY)
    message(STATUS "Gobot editor default Python library: ${GOBOT_PYTHON_LIBRARY}")
    if(IS_DIRECTORY "${GOBOT_PYTHON_LIBRARY}")
        set(GOBOT_PYTHON_LIBRARY_DIR "${GOBOT_PYTHON_LIBRARY}")
    else()
        get_filename_component(GOBOT_PYTHON_LIBRARY_DIR "${GOBOT_PYTHON_LIBRARY}" DIRECTORY)
    endif()
endif()
set(PYBIND11_FINDPYTHON ON)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/CMakeLists.txt")
    add_subdirectory(3rdparty/pybind11)
else()
    CPMAddPackage(
        NAME pybind11
        GITHUB_REPOSITORY pybind/pybind11
        VERSION 2.13.6
        OPTIONS "PYBIND11_TEST OFF"
    )
endif()

INCLUDE(GenerateExportHeader)

add_subdirectory(3rdparty)

set(GOBOT_GENERATED_DIR "${PROJECT_BINARY_DIR}/generated/gobot/generated")
set(GOBOT_EDITOR_ICON_HEADER "${GOBOT_GENERATED_DIR}/editor_icon_svg.hpp")
add_custom_command(
        OUTPUT "${GOBOT_EDITOR_ICON_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GOBOT_GENERATED_DIR}"
        COMMAND xxd -i
                "icon.svg"
                "${GOBOT_EDITOR_ICON_HEADER}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/icon.svg"
        VERBATIM)
add_custom_target(gobot_editor_resources DEPENDS "${GOBOT_EDITOR_ICON_HEADER}")

function(gobot_install_license_files SOURCE_DIR LICENSE_DIR_NAME)
    if(NOT EXISTS "${SOURCE_DIR}")
        return()
    endif()

    set(GOBOT_LICENSE_FILES)
    foreach(GOBOT_LICENSE_FILE
            LICENSE
            LICENSE.MIT
            LICENSE.rst
            LICENSE.txt
            LICENSE.md
            COPYING
            COPYING.txt
            COPYING.md
            NOTICE
            NOTICE.txt
            NOTICE.md)
        if(EXISTS "${SOURCE_DIR}/${GOBOT_LICENSE_FILE}")
            list(APPEND GOBOT_LICENSE_FILES "${SOURCE_DIR}/${GOBOT_LICENSE_FILE}")
        endif()
    endforeach()

    if(GOBOT_LICENSE_FILES)
        install(FILES ${GOBOT_LICENSE_FILES}
                DESTINATION "gobot/licenses/${LICENSE_DIR_NAME}"
                COMPONENT python)
    endif()
endfunction()

file(GLOB_RECURSE SOURCE_CORE CONFIGURE_DEPENDS "src/gobot/core/*.cpp")
file(GLOB_RECURSE SOURCE_SCENE CONFIGURE_DEPENDS "src/gobot/scene/*.cpp")
file(GLOB_RECURSE SOURCE_PHYSICS CONFIGURE_DEPENDS "src/gobot/physics/*.cpp")
file(GLOB_RECURSE SOURCE_SIMULATION CONFIGURE_DEPENDS "src/gobot/simulation/*.cpp")
file(GLOB_RECURSE SOURCE_DRIVERS CONFIGURE_DEPENDS "src/gobot/drivers/*.cpp")
file(GLOB_RECURSE SOURCE_OPENGL_DRIVERS CONFIGURE_DEPENDS "src/gobot/drivers/opengl/*.cpp")
list(REMOVE_ITEM SOURCE_OPENGL_DRIVERS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/drivers/opengl/imgui_renderer.cpp")
foreach(GOBOT_OPENGL_DRIVER_SOURCE IN LISTS SOURCE_OPENGL_DRIVERS)
    list(REMOVE_ITEM SOURCE_DRIVERS "${GOBOT_OPENGL_DRIVER_SOURCE}")
endforeach()
file(GLOB_RECURSE SOURCE_RENDERING CONFIGURE_DEPENDS "src/gobot/rendering/*.cpp")
file(GLOB_RECURSE SOURCE_EDITOR CONFIGURE_DEPENDS "src/gobot/editor/*.cpp")
file(GLOB_RECURSE SOURCE_MAIN CONFIGURE_DEPENDS "src/gobot/main/*.cpp")
set(SOURCE_APP_RUNTIME
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/main/engine_context.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/main/runtime_scene_owner.cpp")
list(REMOVE_ITEM SOURCE_MAIN ${SOURCE_APP_RUNTIME})
set(SOURCE_PYTHON_RUNTIME
        src/gobot/python/python_app_context.cpp)
set(SOURCE_EDITOR_SCENE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/scene/imgui_node.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/scene/imgui_window.cpp")
list(REMOVE_ITEM SOURCE_SCENE ${SOURCE_EDITOR_SCENE})

# https://github.com/bincrafters/community/issues/466#issuecomment-426265654
add_library(gobot SHARED ${SOURCE_CORE} ${SOURCE_RENDERING} ${SOURCE_SCENE} ${SOURCE_PHYSICS} ${SOURCE_SIMULATION} ${SOURCE_OPENGL_DRIVERS} ${SOURCE_APP_RUNTIME} ${SOURCE_PYTHON_RUNTIME})
target_compile_options(gobot PUBLIC "-fno-gnu-unique" "-Wno-volatile" "-Wno-deprecated-enum-enum-conversion")

# https://forum.qt.io/topic/97296/export-qobject-based-class-from-dll/2
GENERATE_EXPORT_HEADER(gobot)

target_include_directories(gobot
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/generated
)

target_link_libraries(gobot
    PUBLIC
        RTTR::Core
        stb
        gli
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        magic_enum::magic_enum
        nlohmann_json::nlohmann_json
        glad
)
target_compile_definitions(gobot PUBLIC
        GOBOT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        GOBOT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        GOBOT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        GOBOT_VERSION="${PROJECT_VERSION}"
        GOBOT_VERSION_COMMIT="${GOBOT_VERSION_COMMIT}")

if(GOB_ENABLE_TRACY AND GOBOT_TRACY_TARGET)
    target_compile_definitions(gobot PUBLIC GOBOT_ENABLE_TRACY=1 TRACY_ENABLE=1)
    target_link_libraries(gobot PUBLIC ${GOBOT_TRACY_TARGET})
endif()

if(GOB_BUILD_OPENUSD)
    target_compile_definitions(gobot PUBLIC GOBOT_HAS_OPENUSD=1)
    target_link_libraries(gobot PUBLIC usd usdGeom usdShade sdf tf vt gf)
endif()

if(GOB_BUILD_ASSIMP AND GOBOT_ASSIMP_TARGET)
    target_compile_definitions(gobot PUBLIC GOBOT_HAS_ASSIMP=1)
    target_link_libraries(gobot PUBLIC ${GOBOT_ASSIMP_TARGET})
endif()

if(GOB_BUILD_MUJOCO)
    target_compile_definitions(gobot PUBLIC GOBOT_HAS_MUJOCO=1)
    target_link_libraries(gobot PUBLIC ${GOBOT_MUJOCO_TARGET})
endif()

add_library(gobot_task_llvm STATIC src/gobot/rl/task_llvm.cpp)
target_include_directories(gobot_task_llvm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/rl
)
target_compile_options(gobot_task_llvm PRIVATE "-fno-gnu-unique")
if(GOBOT_TASK_LLVM_AVAILABLE)
    target_compile_definitions(gobot_task_llvm PRIVATE GOBOT_HAS_TASK_LLVM=1 _GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(gobot_task_llvm SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
    target_link_libraries(gobot_task_llvm PRIVATE
            clangCodeGen
            clangFrontend
            clangSerialization
            clangDriver
            clangParse
            clangSema
            clangAnalysis
            clangAST
            clangLex
            clangBasic
            LLVMOrcJIT
            LLVMJITLink
            LLVMExecutionEngine
            LLVMRuntimeDyld
            LLVMX86CodeGen
            LLVMX86AsmParser
            LLVMX86Desc
            LLVMX86Info
            LLVMCodeGen
            LLVMTarget
            LLVMObject
            LLVMMC
            LLVMCore
            LLVMSupport)
endif()

if(GOB_BUILD_EGL AND GOBOT_EGL_TARGET)
    target_compile_definitions(gobot PUBLIC GOBOT_HAS_EGL=1)
    target_link_libraries(gobot PUBLIC ${GOBOT_EGL_TARGET})
endif()

target_compile_definitions(gobot PUBLIC GOBOT_HAS_PYTHON=1 GOBOT_HAS_PYTHON_EMBED=1)
set(GOBOT_PYTHON_BINDING_SOURCES
        src/gobot/python/python_manual_bindings.cpp
        src/gobot/python/manual_bindings/manual_binding_api.cpp
        src/gobot/python/manual_bindings/manual_binding_app_context.cpp
        src/gobot/python/manual_bindings/manual_binding_common.cpp
        src/gobot/python/manual_bindings/manual_binding_helpers.cpp
        src/gobot/python/manual_bindings/manual_binding_module_functions.cpp
        src/gobot/python/manual_bindings/manual_binding_mujoco_batch.cpp
        src/gobot/python/manual_bindings/manual_binding_robot.cpp
        src/gobot/python/manual_bindings/manual_binding_scene_nodes.cpp
        src/gobot/python/manual_bindings/manual_binding_sensors_terrain.cpp
        src/gobot/python/python_reflected_bindings.cpp
        src/gobot/python/python_variant_bridge.cpp)
target_sources(gobot PRIVATE src/gobot/python/python_script_runner.cpp)
if(TARGET pybind11::embed)
    target_link_libraries(gobot PUBLIC pybind11::embed)
else()
    target_link_libraries(gobot PUBLIC pybind11::pybind11)
    if(TARGET Python3::Python)
        target_link_libraries(gobot PUBLIC Python3::Python)
    endif()
endif()

pybind11_add_module(gobot_python MODULE
        src/gobot/python/python_module.cpp
        ${GOBOT_PYTHON_BINDING_SOURCES}
)
target_link_libraries(gobot_python PRIVATE gobot gobot_task_llvm)
set_target_properties(gobot_python
    PROPERTIES
        OUTPUT_NAME "_core"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/gobot"
        INSTALL_RPATH "$ORIGIN"
)
set_target_properties(gobot
    PROPERTIES
        INSTALL_RPATH "$ORIGIN"
)
set(GOBOT_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python/gobot")
set(GOBOT_PYTHON_CLI_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python/gobot_cli")
add_custom_target(gobot_python_package
        COMMAND ${CMAKE_COMMAND} -E rm -rf
                "${CMAKE_BINARY_DIR}/python/gobot/rl"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${GOBOT_PYTHON_PACKAGE_DIR}"
                "${CMAKE_BINARY_DIR}/python/gobot"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${GOBOT_PYTHON_CLI_PACKAGE_DIR}"
                "${CMAKE_BINARY_DIR}/python/gobot_cli"
        COMMENT "Copying Gobot Python package files"
)
add_dependencies(gobot_python gobot_python_package)
install(TARGETS gobot_python gobot
        RUNTIME DESTINATION "gobot" COMPONENT python
        LIBRARY DESTINATION "gobot" COMPONENT python)
install(DIRECTORY "${GOBOT_PYTHON_PACKAGE_DIR}/"
        DESTINATION "gobot" COMPONENT python
        FILES_MATCHING
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.pyi"
        PATTERN "py.typed")
install(DIRECTORY "${GOBOT_PYTHON_CLI_PACKAGE_DIR}/"
        DESTINATION "gobot_cli" COMPONENT python
        FILES_MATCHING
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.py")
if(GOB_BUILD_MUJOCO AND GOBOT_MUJOCO_TARGET)
    install(FILES "$<TARGET_FILE:${GOBOT_MUJOCO_TARGET}>" DESTINATION "gobot" COMPONENT python)
    get_target_property(GOBOT_MUJOCO_IMPORTED_LOCATION "${GOBOT_MUJOCO_TARGET}" IMPORTED_LOCATION)
    if(GOBOT_MUJOCO_IMPORTED_LOCATION)
        execute_process(
                COMMAND "${CMAKE_OBJDUMP}" -p "${GOBOT_MUJOCO_IMPORTED_LOCATION}"
                OUTPUT_VARIABLE GOBOT_MUJOCO_OBJDUMP
                ERROR_QUIET)
        if(GOBOT_MUJOCO_OBJDUMP MATCHES "SONAME[ \t]+([^ \t\r\n]+)")
            set(GOBOT_MUJOCO_SONAME "${CMAKE_MATCH_1}")
            get_filename_component(GOBOT_MUJOCO_LIBRARY_DIR "${GOBOT_MUJOCO_IMPORTED_LOCATION}" DIRECTORY)
            set(GOBOT_MUJOCO_SONAME_PATH "${GOBOT_MUJOCO_LIBRARY_DIR}/${GOBOT_MUJOCO_SONAME}")
            if(EXISTS "${GOBOT_MUJOCO_SONAME_PATH}" AND
               NOT "${GOBOT_MUJOCO_SONAME_PATH}" STREQUAL "${GOBOT_MUJOCO_IMPORTED_LOCATION}")
                install(FILES "${GOBOT_MUJOCO_SONAME_PATH}" DESTINATION "gobot" COMPONENT python)
            endif()
        endif()
    endif()
endif()
gobot_install_license_files("${CMAKE_CURRENT_SOURCE_DIR}" "gobot")
gobot_install_license_files("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/assimp" "assimp")
gobot_install_license_files("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ImGuiColorTextEdit" "ImGuiColorTextEdit")
gobot_install_license_files("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/imgui" "imgui")
gobot_install_license_files("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/meshoptimizer" "meshoptimizer")
gobot_install_license_files("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11" "pybind11")
gobot_install_license_files("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rttr" "rttr")
gobot_install_license_files("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb" "stb")
if(SDL2_SOURCE_DIR)
    gobot_install_license_files("${SDL2_SOURCE_DIR}" "SDL2")
endif()
if(fmt_SOURCE_DIR)
    gobot_install_license_files("${fmt_SOURCE_DIR}" "fmt")
endif()
if(spdlog_SOURCE_DIR)
    gobot_install_license_files("${spdlog_SOURCE_DIR}" "spdlog")
endif()
if(nlohmann_json_SOURCE_DIR)
    gobot_install_license_files("${nlohmann_json_SOURCE_DIR}" "nlohmann_json")
endif()
if(magic_enum_SOURCE_DIR)
    gobot_install_license_files("${magic_enum_SOURCE_DIR}" "magic_enum")
endif()
if(cxxopts_SOURCE_DIR)
    gobot_install_license_files("${cxxopts_SOURCE_DIR}" "cxxopts")
endif()
if(mujoco_SOURCE_DIR)
    gobot_install_license_files("${mujoco_SOURCE_DIR}" "mujoco")
endif()

set_target_properties(gobot
    PROPERTIES
        CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set(SHADER_COMPILE_TARGET GobotShaderCompile)
add_subdirectory(shader)
# Why doesn't work for include
add_dependencies(gobot ${SHADER_COMPILE_TARGET})
target_include_directories(gobot
        PUBLIC
        shader/generated
        )



set(GOBOT_PYTHON_EMBED_SOURCE src/gobot/python/python_embed_module.cpp)
set(GOBOT_PYTHON_EMBED_BINDING_SOURCES ${GOBOT_PYTHON_BINDING_SOURCES})

file(GLOB_RECURSE SOURCE_PLATFORM CONFIGURE_DEPENDS "src/gobot/platform/linux/*.cpp")
list(REMOVE_ITEM SOURCE_PLATFORM
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/platform/linux/gobot_linux.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/platform/linux/gobot_editor_launcher.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/platform/linux/gobot_editor_runtime.cpp"
)

add_library(gobot_editor_runtime SHARED
        ${SOURCE_DRIVERS}
        ${SOURCE_MAIN}
        ${SOURCE_EDITOR}
        ${SOURCE_EDITOR_SCENE}
        ${SOURCE_PLATFORM}
        ${GOBOT_PYTHON_EMBED_SOURCE}
        ${GOBOT_PYTHON_EMBED_BINDING_SOURCES}
        3rdparty/ImGuiColorTextEdit/TextEditor.cpp
        src/gobot/platform/editor_embedded_resources.cpp
        src/gobot/platform/linux/gobot_editor_runtime.cpp
)
add_dependencies(gobot_editor_runtime gobot_editor_resources gobot_python_package)
target_include_directories(gobot_editor_runtime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ImGuiColorTextEdit)
target_compile_definitions(gobot_editor_runtime PRIVATE
        GOBOT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(gobot_editor_runtime PRIVATE
        gobot
        pybind11::pybind11
        imgui-gobot
        glad
        SDL2::SDL2
        cxxopts::cxxopts)
set_target_properties(gobot_editor_runtime
    PROPERTIES
        OUTPUT_NAME "gobot_editor_runtime"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/gobot"
        INSTALL_RPATH "$ORIGIN"
)
install(TARGETS gobot_editor_runtime
        LIBRARY DESTINATION "gobot" COMPONENT python)

add_executable(gobot_editor
        src/gobot/platform/linux/gobot_editor_launcher.cpp
)
add_dependencies(gobot_editor gobot_editor_runtime)
target_compile_definitions(gobot_editor PRIVATE
        GOBOT_DEFAULT_PYTHON_LIBRARY="${GOBOT_PYTHON_LIBRARY}"
        GOBOT_DEFAULT_PYTHON_VERSION_MAJOR=${Python3_VERSION_MAJOR}
        GOBOT_DEFAULT_PYTHON_VERSION_MINOR=${Python3_VERSION_MINOR})
target_link_libraries(gobot_editor PRIVATE
        ${CMAKE_DL_LIBS})
set_target_properties(gobot_editor
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/gobot"
        BUILD_RPATH "${CMAKE_BINARY_DIR}/python/gobot;${CMAKE_BINARY_DIR};${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN"
)
add_custom_command(TARGET gobot_editor POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "python/gobot/$<TARGET_FILE_NAME:gobot_editor>"
                "${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:gobot_editor>")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
        DESTINATION "gobot/examples" COMPONENT python
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "logs" EXCLUDE)
install(TARGETS gobot_editor
        RUNTIME DESTINATION "gobot" COMPONENT python)

if (GOB_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif ()

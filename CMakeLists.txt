cmake_minimum_required (VERSION 3.16)
project (gobot VERSION 0.1.5)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GOBOT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(GOBOT_VERSION "${PROJECT_VERSION}")

option(GOB_BUILD_TESTS "build tests" OFF)
option(GOB_BUILD_OPENUSD "build OpenUSD scene importer/exporter support" OFF)
option(GOB_BUILD_ASSIMP "build mesh import support through Assimp" ON)
option(GOB_BUILD_MUJOCO "build MuJoCo CPU physics backend support" ON)
option(GOB_FETCH_MUJOCO "download MuJoCo from the official GitHub repository when GOB_BUILD_MUJOCO is ON and no local SDK/package is found" ON)
set(GOB_MUJOCO_ROOT "" CACHE PATH "MuJoCo SDK or install prefix. Supports either a CMake package install or an extracted SDK with include/ and lib/.")
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

CPMFindPackage(
    NAME nlohmann_json
    VERSION 3.11.3
    GITHUB_REPOSITORY nlohmann/json
    OPTIONS "JSON_BuildTests OFF"
)

if(GOB_BUILD_OPENUSD)
    find_package(pxr REQUIRED CONFIG)
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
        # wheel builds so manylinux does not need wayland-scanner/desktop deps.
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

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)
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
file(GLOB_RECURSE SOURCE_RENDERING CONFIGURE_DEPENDS "src/gobot/rendering/*.cpp")
file(GLOB_RECURSE SOURCE_EDITOR CONFIGURE_DEPENDS "src/gobot/editor/*.cpp")
file(GLOB_RECURSE SOURCE_MAIN CONFIGURE_DEPENDS "src/gobot/main/*.cpp")
set(SOURCE_APP_RUNTIME
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/main/engine_context.cpp")
list(REMOVE_ITEM SOURCE_MAIN ${SOURCE_APP_RUNTIME})
set(SOURCE_PYTHON_RUNTIME
        src/gobot/python/python_app_context.cpp)
set(SOURCE_EDITOR_SCENE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/scene/imgui_node.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/scene/imgui_window.cpp")
list(REMOVE_ITEM SOURCE_SCENE ${SOURCE_EDITOR_SCENE})

# https://github.com/bincrafters/community/issues/466#issuecomment-426265654
add_library(gobot SHARED ${SOURCE_CORE} ${SOURCE_RENDERING} ${SOURCE_SCENE} ${SOURCE_PHYSICS} ${SOURCE_SIMULATION} ${SOURCE_APP_RUNTIME} ${SOURCE_PYTHON_RUNTIME})
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
)
target_compile_definitions(gobot PUBLIC
        GOBOT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        GOBOT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        GOBOT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        GOBOT_VERSION="${PROJECT_VERSION}")

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

target_compile_definitions(gobot PUBLIC GOBOT_HAS_PYTHON=1 GOBOT_HAS_PYTHON_EMBED=1)
set(GOBOT_PYTHON_BINDING_SOURCES
        src/gobot/python/python_manual_bindings.cpp
        src/gobot/python/python_reflected_bindings.cpp
        src/gobot/python/python_variant_bridge.cpp)
target_sources(gobot PRIVATE src/gobot/python/python_script_runner.cpp)
target_link_libraries(gobot PUBLIC pybind11::embed)

pybind11_add_module(gobot_python MODULE
        src/gobot/python/python_module.cpp
        ${GOBOT_PYTHON_BINDING_SOURCES}
)
target_link_libraries(gobot_python PRIVATE gobot)
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
add_custom_target(gobot_python_package
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${GOBOT_PYTHON_PACKAGE_DIR}"
                "${CMAKE_BINARY_DIR}/python/gobot"
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
if(GOB_BUILD_MUJOCO AND GOBOT_MUJOCO_TARGET)
    install(FILES "$<TARGET_FILE:${GOBOT_MUJOCO_TARGET}>" DESTINATION "gobot" COMPONENT python)
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
)
add_executable(gobot_editor
        ${SOURCE_DRIVERS}
        ${SOURCE_MAIN}
        ${SOURCE_EDITOR}
        ${SOURCE_EDITOR_SCENE}
        ${SOURCE_PLATFORM}
        ${GOBOT_PYTHON_EMBED_SOURCE}
        ${GOBOT_PYTHON_EMBED_BINDING_SOURCES}
        3rdparty/ImGuiColorTextEdit/TextEditor.cpp
        src/gobot/platform/editor_embedded_resources.cpp
        src/gobot/platform/linux/gobot_linux.cpp
)
add_dependencies(gobot_editor gobot_editor_resources)
target_include_directories(gobot_editor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ImGuiColorTextEdit)
target_compile_definitions(gobot_editor PRIVATE
        GOBOT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(gobot_editor PRIVATE
        gobot
        imgui-gobot
        glad
        SDL2::SDL2
        cxxopts::cxxopts)
set_target_properties(gobot_editor
    PROPERTIES
        INSTALL_RPATH "$ORIGIN"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
        DESTINATION "gobot/examples" COMPONENT python
        FILES_MATCHING
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.jscn"
        PATTERN "*.py")
install(TARGETS gobot_editor
        RUNTIME DESTINATION "gobot" COMPONENT python)

if (GOB_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif ()

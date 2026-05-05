cmake_minimum_required (VERSION 3.16)
project (gobot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GOBOT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

option(GOB_BUILD_EXPERIMENT "using imgui and glfw" OFF)
option(GOB_BUILD_TESTS "build tests" OFF)
option(GOB_BUILD_OPENUSD "build OpenUSD scene importer/exporter support" OFF)
option(GOB_BUILD_ASSIMP "build mesh import support through Assimp" ON)
option(GOB_BUILD_MUJOCO "build MuJoCo CPU physics backend support" OFF)
option(GOB_BUILD_PYTHON_BINDINGS "build Python bindings through pybind11" ON)
option(GOB_FETCH_MUJOCO "download MuJoCo from the official GitHub repository when GOB_BUILD_MUJOCO is ON and no local SDK/package is found" OFF)
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
find_package(PNG REQUIRED)

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
        set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)
        set(MUJOCO_BUILD_STUDIO OFF CACHE BOOL "" FORCE)
        set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(MUJOCO_TEST_PYTHON_UTIL OFF CACHE BOOL "" FORCE)
        set(MUJOCO_WITH_USD OFF CACHE BOOL "" FORCE)
        set(MUJOCO_USE_FILAMENT OFF CACHE BOOL "" FORCE)

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
elseif(GOB_FETCH_MUJOCO)
    message(WARNING "GOB_FETCH_MUJOCO is ON but GOB_BUILD_MUJOCO is OFF; MuJoCo download is skipped.")
endif()

set(GOBOT_ASSIMP_TARGET "")
if(GOB_BUILD_ASSIMP)
    set(GOBOT_ASSIMP_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/assimp")
    if(EXISTS "${GOBOT_ASSIMP_SUBMODULE_DIR}/CMakeLists.txt")
        if(CMAKE_VERSION VERSION_LESS 3.22)
            message(WARNING "3rdparty/assimp requires CMake 3.22 or newer; falling back to system assimp package.")
        else()
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

CPMAddPackage("gh:fmtlib/fmt#10.1.1")

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION 1.13.0
    OPTIONS "SPDLOG_FMT_EXTERNAL ON"
)

CPMAddPackage("gh:Neargye/magic_enum@0.9.5")
if(magic_enum_SOURCE_DIR)
    target_include_directories(magic_enum INTERFACE ${magic_enum_SOURCE_DIR}/include)
endif()
CPMAddPackage("gh:jarro2783/cxxopts@3.0.0")

if(GOB_BUILD_PYTHON_BINDINGS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
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
endif()

INCLUDE(GenerateExportHeader)

add_subdirectory(3rdparty)

file(GLOB_RECURSE SOURCE_CORE CONFIGURE_DEPENDS "src/gobot/core/*.cpp")
file(GLOB_RECURSE SOURCE_SCENE CONFIGURE_DEPENDS "src/gobot/scene/*.cpp")
file(GLOB_RECURSE SOURCE_PHYSICS CONFIGURE_DEPENDS "src/gobot/physics/*.cpp")
file(GLOB_RECURSE SOURCE_SIMULATION CONFIGURE_DEPENDS "src/gobot/simulation/*.cpp")
file(GLOB_RECURSE SOURCE_DRIVERS CONFIGURE_DEPENDS "src/gobot/drivers/*.cpp")
file(GLOB_RECURSE SOURCE_RENDERING CONFIGURE_DEPENDS "src/gobot/rendering/*.cpp")
file(GLOB_RECURSE SOURCE_EDITOR CONFIGURE_DEPENDS "src/gobot/editor/*.cpp")
file(GLOB_RECURSE SOURCE_MAIN CONFIGURE_DEPENDS "src/gobot/main/*.cpp")

# https://github.com/bincrafters/community/issues/466#issuecomment-426265654
add_library(gobot SHARED ${SOURCE_DRIVERS} ${SOURCE_MAIN} ${SOURCE_CORE} ${SOURCE_RENDERING} ${SOURCE_SCENE} ${SOURCE_PHYSICS} ${SOURCE_SIMULATION} ${SOURCE_EDITOR})
target_compile_options(gobot PUBLIC "-fno-gnu-unique" "-Wno-volatile" "-Wno-deprecated-enum-enum-conversion")

# https://forum.qt.io/topic/97296/export-qobject-based-class-from-dll/2
GENERATE_EXPORT_HEADER(gobot)

target_include_directories(gobot
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
)

target_link_libraries(gobot
    PUBLIC
        RTTR::Core
        imgui-gobot
        glad
        SDL2::SDL2
        stb
        gli
        spdlog::spdlog
        fmt::fmt
        PNG::PNG
        Eigen3::Eigen
        magic_enum::magic_enum
        nlohmann_json::nlohmann_json
        cxxopts::cxxopts
)

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

if(GOB_BUILD_PYTHON_BINDINGS)
    pybind11_add_module(gobot_python MODULE
            src/gobot/python/python_module.cpp
            src/gobot/python/python_manual_bindings.cpp
            src/gobot/python/python_reflected_bindings.cpp
            src/gobot/python/python_variant_bridge.cpp
    )
    target_link_libraries(gobot_python PRIVATE gobot)
    set_target_properties(gobot_python
        PROPERTIES
            OUTPUT_NAME "gobot"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
            INSTALL_RPATH "$ORIGIN"
    )
    set_target_properties(gobot
        PROPERTIES
            INSTALL_RPATH "$ORIGIN"
    )
    add_custom_command(
            TARGET gobot_python POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/python/gobot_gym_adapter.py"
                    "${CMAKE_BINARY_DIR}/python/gobot_gym_adapter.py"
    )
    install(TARGETS gobot_python gobot LIBRARY DESTINATION "." COMPONENT python)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/gobot_gym_adapter.py" DESTINATION "." COMPONENT python)
    if(TARGET spdlog)
        get_target_property(GOBOT_SPDLOG_TARGET_TYPE spdlog TYPE)
        if(GOBOT_SPDLOG_TARGET_TYPE STREQUAL "SHARED_LIBRARY")
            install(FILES "$<TARGET_FILE:spdlog>" "$<TARGET_SONAME_FILE:spdlog>" DESTINATION "." COMPONENT python)
        endif()
    endif()
    if(TARGET fmt)
        get_target_property(GOBOT_FMT_TARGET_TYPE fmt TYPE)
        if(GOBOT_FMT_TARGET_TYPE STREQUAL "SHARED_LIBRARY")
            install(FILES "$<TARGET_FILE:fmt>" "$<TARGET_SONAME_FILE:fmt>" DESTINATION "." COMPONENT python)
        endif()
    endif()
    if(TARGET RTTR::Core)
        install(FILES "$<TARGET_FILE:RTTR::Core>" DESTINATION "." COMPONENT python)
    endif()
    if(GOB_BUILD_MUJOCO AND GOBOT_MUJOCO_TARGET)
        install(FILES "$<TARGET_FILE:${GOBOT_MUJOCO_TARGET}>" DESTINATION "." COMPONENT python)
    endif()
endif()

# copy icon to bin
add_custom_command(
        TARGET gobot POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/icon.svg
        ${CMAKE_BINARY_DIR}/bin/icon.svg)

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



file(GLOB_RECURSE SOURCE_PLATFORM CONFIGURE_DEPENDS "src/gobot/platform/linux/*.cpp")
list(REMOVE_ITEM SOURCE_PLATFORM
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/platform/linux/gobot_linux.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gobot/platform/linux/gobot_headless.cpp"
)
add_executable(gobot_editor
        ${SOURCE_PLATFORM}
        src/gobot/platform/linux/gobot_linux.cpp
)
target_link_libraries(gobot_editor PRIVATE gobot)

add_executable(gobot_headless src/gobot/platform/linux/gobot_headless.cpp)
target_link_libraries(gobot_headless PRIVATE gobot)

if (GOB_BUILD_EXPERIMENT)
    add_subdirectory(experiment)
endif ()

if (GOB_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif ()

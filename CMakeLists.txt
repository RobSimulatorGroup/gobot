cmake_minimum_required (VERSION 3.16)
project (gobot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GOBOT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

option(GOB_BUILD_EXPERIMENT "using imgui and glfw" OFF)
option(GOB_BUILD_TESTS "build tests" OFF)
option(GOB_BUILD_OPENUSD "build OpenUSD scene importer/exporter support" OFF)

if(NOT UNIX OR APPLE)
    message(FATAL_ERROR "Only Linux is supported.")
endif()

include(cmake/CPM.cmake)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PNG REQUIRED)

if(GOB_BUILD_OPENUSD)
    find_package(pxr REQUIRED CONFIG)
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

INCLUDE(GenerateExportHeader)

add_subdirectory(3rdparty)

file(GLOB_RECURSE SOURCE_CORE CONFIGURE_DEPENDS "src/gobot/core/*.cpp")
file(GLOB_RECURSE SOURCE_SCENE CONFIGURE_DEPENDS "src/gobot/scene/*.cpp")
file(GLOB_RECURSE SOURCE_DRIVERS CONFIGURE_DEPENDS "src/gobot/drivers/*.cpp")
file(GLOB_RECURSE SOURCE_RENDERING CONFIGURE_DEPENDS "src/gobot/rendering/*.cpp")
file(GLOB_RECURSE SOURCE_EDITOR CONFIGURE_DEPENDS "src/gobot/editor/*.cpp")
file(GLOB_RECURSE SOURCE_MAIN CONFIGURE_DEPENDS "src/gobot/main/*.cpp")

# https://github.com/bincrafters/community/issues/466#issuecomment-426265654
add_library(gobot SHARED ${SOURCE_DRIVERS} ${SOURCE_MAIN} ${SOURCE_CORE} ${SOURCE_RENDERING} ${SOURCE_SCENE} ${SOURCE_EDITOR})
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
add_executable(gobot_editor ${SOURCE_PLATFORM})
target_link_libraries(gobot_editor PRIVATE gobot)

if (GOB_BUILD_EXPERIMENT)
    add_subdirectory(experiment)
endif ()

if (GOB_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif ()

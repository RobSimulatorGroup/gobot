cmake_minimum_required (VERSION 3.16)
project (gobot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

set(GOBOT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

option(GOB_BUILD_EXPERIMENT "using imgui and glfw" OFF)
option(GOB_BUILD_TESTS "build tests" OFF)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

include(cmake/CPM.cmake)


if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
            FATAL_ERROR
            "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
    )
endif()

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(spdlog REQUIRED)
find_package(magic_enum REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PNG REQUIRED)
find_package(cxxopts REQUIRED)

INCLUDE(GenerateExportHeader)

add_subdirectory(3rdparty)

file(GLOB_RECURSE SOURCE_CORE CONFIGURE_DEPENDS "src/gobot/core/*.cpp")
file(GLOB_RECURSE SOURCE_SCENE CONFIGURE_DEPENDS "src/gobot/scene/*.cpp")
file(GLOB_RECURSE SOURCE_DRIVERS CONFIGURE_DEPENDS "src/gobot/drivers/*.cpp")
file(GLOB_RECURSE SOURCE_RENDERING CONFIGURE_DEPENDS "src/gobot/rendering/*.cpp")
file(GLOB_RECURSE SOURCE_EDITOR CONFIGURE_DEPENDS "src/gobot/editor/*.cpp")
file(GLOB_RECURSE SOURCE_MAIN CONFIGURE_DEPENDS "src/gobot/main/*.cpp")
file(GLOB_RECURSE SOURCE_SLAM CONFIGURE_DEPENDS "src/gobot/slam/*cpp")

# https://github.com/bincrafters/community/issues/466#issuecomment-426265654
add_library(gobot SHARED
        ${SOURCE_DRIVERS}
        ${SOURCE_MAIN}
        ${SOURCE_CORE}
        ${SOURCE_RENDERING}
        ${SOURCE_SCENE}
        ${SOURCE_EDITOR}
        ${SOURCE_SLAM}
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(GOBOT_COMPILE_OPTIONS "/Zc:preprocessor,__cplusplus" "/bigobj")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(GOBOT_COMPILE_OPTIONS "-fno-gnu-unique")
endif ()
target_compile_options(gobot PUBLIC "${GOBOT_COMPILE_OPTIONS}")

# https://forum.qt.io/topic/97296/export-qobject-based-class-from-dll/2
GENERATE_EXPORT_HEADER(gobot)

target_include_directories(gobot
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
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
        PNG::PNG
        Eigen3::Eigen
        magic_enum::magic_enum
        nlohmann_json::nlohmann_json
        cxxopts::cxxopts
        Sophus::Sophus
        ${OpenCV_LIBS}
        ${PCL_LIBRARY_DIRS}
        Ceres::ceres
)

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



if (MSVC)
    file(GLOB_RECURSE SOURCE_PLATFORM CONFIGURE_DEPENDS "src/gobot/platform/windows/*.cpp")
    add_executable(gobot_editor ${SOURCE_PLATFORM})
    target_compile_options(gobot_editor PUBLIC "/Zc:preprocessor")
elseif(LINUX)
    file(GLOB_RECURSE SOURCE_PLATFORM CONFIGURE_DEPENDS "src/gobot/platform/linux/*.cpp")
    add_executable(gobot_editor ${SOURCE_PLATFORM})
endif ()
target_link_libraries(gobot_editor
    PRIVATE
    gobot
)

if (GOB_BUILD_EXPERIMENT)
    add_subdirectory(experiment)
endif ()

if (GOB_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif ()

if (MSVC)
    add_custom_command(TARGET gobot POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "$<TARGET_FILE_DIR:RTTR::Core>"
            "$<TARGET_FILE_DIR:gobot>"
        COMMENT "Copying RTTR files..."
    )

endif ()

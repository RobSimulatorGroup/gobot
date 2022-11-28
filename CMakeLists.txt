cmake_minimum_required (VERSION 3.23)
project (gobot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

option(BUILD_EXPERIMENT "using imgui and glfw" OFF)
option(BUILD_TESTS "build tests" OFF)

include(cmake/CPM.cmake)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
            FATAL_ERROR
            "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
    )
endif()

add_subdirectory(3rdpart)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

find_package (Eigen3 3.3 REQUIRED)

set(INCLUDE_DIR
        include
        ${EIGEN3_INCLUDE_DIR})


file(GLOB_RECURSE SOURCE_CORE CONFIGURE_DEPENDS "src/gobot/core/*.cpp")
file(GLOB_RECURSE SOURCE_SCENE CONFIGURE_DEPENDS "src/gobot/scene/*.cpp")
file(GLOB_RECURSE SOURCE_EDITOR CONFIGURE_DEPENDS "src/gobot/editor/*.cpp")

qt5_wrap_cpp(SOURCE_CORE "include/gobot/core/object.hpp")
qt5_wrap_cpp(SOURCE_SCENE "include/gobot/scene/scene_tree.hpp")

add_library(gobot SHARED ${SOURCE_CORE} ${SOURCE_SCENE})
target_link_libraries(gobot
        PUBLIC
        spdlog::spdlog
        Qt5::Core
        RTTR::Core
        nlohmann_json
        intrusive_ptr
)
target_include_directories(gobot PUBLIC ${INCLUDE_DIR})

qt5_wrap_cpp(SOURCE_EDITOR "include/gobot/editor/main_window.hpp")

add_executable(gobot_editor src/main.cpp ${SOURCE_EDITOR})
target_link_libraries(gobot_editor
        PUBLIC
        gobot
        Qt5::Widgets
        )


if (BUILD_EXPERIMENT)
    add_subdirectory(experiment)
endif ()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()




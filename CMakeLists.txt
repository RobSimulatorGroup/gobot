cmake_minimum_required (VERSION 3.16)
project (gobot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

option(BUILD_EXPERIMENT "using imgui and glfw" OFF)
option(BUILD_TESTS "build tests" OFF)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# ---------------------------------------- START OF CONAN SETUP ----------------------------------------
if (NOT EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        message(STATUS "Try to copy conan.cmake to ${CMAKE_BINARY_DIR}")
        file(COPY_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/conan.cmake" "${CMAKE_BINARY_DIR}/conan.cmake")
    endif ()
    include(${CMAKE_BINARY_DIR}/conan.cmake)
    if (CONAN_PROFILE_BUILD AND CONAN_PROFILE_HOST)
        conan_cmake_install(PATH_OR_REFERENCE ${PROJECT_SOURCE_DIR}
                            BUILD missing
                            REMOTE conancenter
                            SETTINGS_HOST build_type=${CMAKE_BUILD_TYPE}
                            SETTINGS_BUILD build_type=${CMAKE_BUILD_TYPE}
                            PROFILE_BUILD ${CONAN_PROFILE_BUILD}
                            PROFILE_HOST ${CONAN_PROFILE_HOST})
                            # ENV_HOST CONAN_CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    else ()
        cmake_host_system_information(RESULT CMAKE_SYSTEM_NAME QUERY OS_NAME)

        conan_cmake_autodetect(settings)

        # This is hacky for detecting os. Wait cmake-conan to have a correct implementation.
        if (MSVC)
            set(settings ${settings} os=Windows)
        else ()
            set(settings ${settings} os=Linux)
        endif ()

        conan_cmake_profile(FILEPATH      "${CMAKE_BINARY_DIR}/profile"
                            INCLUDE       default
                            SETTINGS      ${settings}
                            CONF          tools.system.package_manager:mode=install
                                          tools.system.package_manager:sudo=True
                                          tools.system.package_manager:sudo_askpass=True)

        message(${settings})

        conan_cmake_install(PATH_OR_REFERENCE ${PROJECT_SOURCE_DIR}
                            BUILD missing
                            REMOTE conancenter
                            SETTINGS_HOST build_type=${CMAKE_BUILD_TYPE}
                            SETTINGS_BUILD build_type=${CMAKE_BUILD_TYPE}
                            PROFILE_HOST "${CMAKE_BINARY_DIR}/profile"
                            PROFILE_BUILD "${CMAKE_BINARY_DIR}/profile")
    endif ()
endif ()

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
conan_basic_setup(${PROJECT_NAME} KEEP_RPATHS)
# ---------------------------------------- END OF CONAN SETUP ----------------------------------------

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
            FATAL_ERROR
            "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
    )
endif()

find_package(Qt5 5.15 COMPONENTS Core Gui CONFIG REQUIRED)
find_package (Eigen3 3.4 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

INCLUDE(GenerateExportHeader)

add_subdirectory(3rdparty)

file(GLOB_RECURSE SOURCE_CORE CONFIGURE_DEPENDS "src/gobot/core/*.cpp")
file(GLOB_RECURSE SOURCE_SCENE CONFIGURE_DEPENDS "src/gobot/scene/*.cpp")
file(GLOB_RECURSE SOURCE_DRIVERS CONFIGURE_DEPENDS "src/gobot/drivers/*.cpp")
file(GLOB_RECURSE SOURCE_EDITOR CONFIGURE_DEPENDS "src/gobot/editor/*.cpp")
file(GLOB_RECURSE SOURCE_MAIN CONFIGURE_DEPENDS "src/gobot/main/*.cpp")
file(GLOB_RECURSE SOURCE_GRAPHICS CONFIGURE_DEPENDS "src/gobot/graphics/*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "include/*hpp")

# https://github.com/bincrafters/community/issues/466#issuecomment-426265654
add_library(gobot SHARED ${SOURCE_DRIVERS} ${SOURCE_MAIN} ${SOURCE_CORE} ${SOURCE_SCENE} ${SOURCE_EDITOR} ${SOURCE_GRAPHICS} ${HEADER_FILES})
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
)

target_link_libraries(gobot
    PUBLIC
    RTTR::Core
    OpenGL::GL
    Qt5::Core
    Qt5::Gui
    imgui
    ${CONAN_LIBS}
)

# copy icon to bin
add_custom_command(
        TARGET gobot POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/icon.svg
        ${CMAKE_BINARY_DIR}/bin/icon.svg)

set_target_properties(gobot
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    AUTOMOC ON
    AUTORCC ON
    AUTOUIC ON
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

if (BUILD_EXPERIMENT)
    add_subdirectory(experiment)
endif ()

if (BUILD_TESTS)
    enable_testing()
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

cmake_minimum_required (VERSION 3.16)
project (gobot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(3rdparty)

set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

option(BUILD_EXPERIMENT "using imgui and glfw" OFF)
option(BUILD_TESTS "build tests" OFF)
option(BUILD_WITH_PYBIND11 "build python bindings with pybind11" ON)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        file(DOWNLOAD "https://raw.githubusercontent.com/wuyingnan/cmake-conan/develop/conan.cmake" "${CMAKE_BINARY_DIR}/conan.cmake")
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

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
            FATAL_ERROR
            "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
    )
endif()

find_package(Qt5 COMPONENTS Core Widgets Gui CONFIG REQUIRED)
find_package (Eigen3 3.4 CONFIG REQUIRED)


find_package(pybind11 REQUIRED)
find_package(python3 3.10 COMPONENTS Interpreter Development)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

file(GLOB_RECURSE SOURCE_CORE CONFIGURE_DEPENDS "src/gobot/core/*.cpp")
file(GLOB_RECURSE SOURCE_SCENE CONFIGURE_DEPENDS "src/gobot/scene/*.cpp")
file(GLOB_RECURSE SOURCE_EDITOR CONFIGURE_DEPENDS "src/gobot/editor/*.cpp")

file(GLOB_RECURSE HEAD_FILES CONFIGURE_DEPENDS "include/*hpp")

#https://github.com/bincrafters/community/issues/466#issuecomment-426265654
add_library(gobot SHARED ${SOURCE_CORE} ${SOURCE_SCENE} ${HEAD_FILES})
if (MSVC)
    target_compile_options(gobot PUBLIC "/Zc:preprocessor")
endif ()

target_include_directories(gobot
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gobot
    PUBLIC
    RTTR::Core
    intrusive_ptr
    ${CONAN_LIBS}
)

file(GLOB_RECURSE RTTR_BIND CONFIGURE_DEPENDS "src/binding/core/*.cpp")
add_library(gobot_rttr SHARED ${RTTR_BIND} )
target_link_libraries(gobot_rttr PUBLIC gobot )

if (BUILD_WITH_PYBIND11)
    add_library(gobot_python SHARED ${RTTR_BIND} )
    target_compile_definitions(gobot_python PRIVATE BUILD_WITH_PYBIND11)
    target_link_libraries(gobot_python PUBLIC gobot )
endif()


set_target_properties(gobot
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(gobot_editor src/main.cpp ${SOURCE_EDITOR})
if (MSVC)
    target_compile_options(gobot_editor PUBLIC "/Zc:preprocessor")
endif ()
target_link_libraries(gobot_editor
    PRIVATE
    gobot
    ${CONAN_LIBS}
)

if (BUILD_EXPERIMENT)
    add_subdirectory(experiment)
endif ()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (MSVC)
    # Retrieve the absolute path to qmake and then use that path to find
    # the binaries
    get_target_property(_qmake_executable Qt5::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${_qt_bin_dir}")
    find_program(MACDEPLOYQT_EXECUTABLE macdeployqt HINTS "${_qt_bin_dir}")
    add_custom_command(TARGET gobot_editor POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E
            env PATH="${_qt_bin_dir} ${}" "${WINDEPLOYQT_EXECUTABLE}" --debug --verbose 2
                "$<TARGET_FILE:gobot_editor>"
        COMMENT "Running windeployqt..."
  )
endif ()
